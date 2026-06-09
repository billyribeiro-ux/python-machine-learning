"""
quantlab.research.optimize
==========================

A **generic, honest strategy optimizer** built on Optuna. It optimizes any
:mod:`quantlab.research.templates` strategy over its declared search space — but
unlike a naive grid search, it is engineered to *not lie to you*:

1. **Hold-out discipline.** The series is split chronologically into a *train*
   portion (the optimizer sees only this) and a *hold-out* (touched exactly
   once, at the end). The reported edge is the hold-out number.
2. **Robust objective.** On train, the objective is the *mean minus a penalty
   times the spread* of per-fold Sharpes — rewarding strategies that work across
   several sub-periods, not one lucky regime.
3. **Deflation.** After the search we compute the **Deflated Sharpe Ratio**:
   given how many configurations were tried, how likely is this hold-out Sharpe
   to be more than luck? This is the number that survives peer review.
4. **Multi-objective.** Optionally maximize Sharpe *and* minimize drawdown,
   returning the Pareto front instead of one fragile "best".

The result is a single, self-describing object you can log, render, or gate a
deployment on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quantlab.backtest.engine import backtest_signal
from quantlab.backtest.stats import compute_stats
from quantlab.backtest.strategy import StrategyConfig, build_signal
from .metrics import deflated_sharpe_ratio, per_period_sharpe, returns_skew_kurt
from .walkforward import fold_sharpes, split_index


@dataclass
class OptimizeResult:
    """Everything you need to judge — and trust, or distrust — a search."""

    best_params: dict
    best_config_name: str
    train_stats: dict
    holdout_stats: dict
    fold_sharpes: list[float]
    deflated_sharpe: float
    n_trials: int
    is_oos_gap: float                      # train Sharpe - holdout Sharpe
    holdout_returns: pd.Series
    history: list[float]                   # best-so-far objective per trial
    pareto: list[dict] = field(default_factory=list)   # multi-objective only

    @property
    def trustworthy(self) -> bool:
        """A blunt heuristic gate: positive hold-out Sharpe, a deflated Sharpe
        above 0.6 (unlikely to be luck), and an in-sample→out-of-sample gap that
        isn't catastrophic. Tune to taste — but always *have* a gate."""
        return (self.holdout_stats.get("sharpe", 0) > 0
                and self.deflated_sharpe > 0.6
                and self.is_oos_gap < 1.5)


def _sample(space: dict, trial) -> dict:
    out = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "categorical":
            out[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown param kind {kind!r} for {name!r}")
    return out


def _backtest(ohlcv: pd.DataFrame, cfg: StrategyConfig, ppy: float) -> pd.Series:
    sig = build_signal(ohlcv, cfg)
    return backtest_signal(ohlcv["close"], sig, fee_bps=cfg.fee_bps,
                           periods_per_year=ppy)["returns"]


def optimize_strategy(
    ohlcv: pd.DataFrame,
    build: Callable[[dict], StrategyConfig],
    space: dict,
    invalid: Callable[[dict], bool] | None = None,
    n_trials: int = 60,
    holdout: float = 0.25,
    n_folds: int = 4,
    robust_lambda: float = 0.5,
    multiobjective: bool = False,
    periods_per_year: float = 252,
    seed: int = 42,
    storage: str | None = None,
    study_name: str | None = None,
) -> OptimizeResult:
    """Optimize a strategy template honestly. See module docstring for the method.

    Parameters mirror the discipline: ``holdout`` reserves the untouched test
    period, ``n_folds``/``robust_lambda`` shape the robust train objective,
    ``multiobjective`` switches to Sharpe-vs-drawdown Pareto search, and
    ``storage`` (e.g. ``"sqlite:///study.db"``) persists the study for auditing
    how many trials you really ran.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cut = split_index(len(ohlcv), holdout)
    train_ohlcv = ohlcv.iloc[:cut]

    history: list[float] = []
    best_so_far = -np.inf

    def objective(trial):
        nonlocal best_so_far
        params = _sample(space, trial)
        if invalid is not None and invalid(params):
            raise optuna.TrialPruned()
        cfg = build(params)
        rets = _backtest(train_ohlcv, cfg, periods_per_year)
        # Record per-period train Sharpe for the deflation step later.
        trial.set_user_attr("pp_sharpe", per_period_sharpe(rets))
        if multiobjective:
            s = compute_stats(rets, periods_per_year)
            return s["sharpe"], abs(s["max_drawdown"])
        folds = fold_sharpes(rets, n_folds, periods_per_year)
        score = float(np.mean(folds) - robust_lambda * np.std(folds))
        best_so_far = max(best_so_far, score)
        history.append(best_so_far)
        return score

    sampler = (optuna.samplers.NSGAIISampler(seed=seed) if multiobjective
               else optuna.samplers.TPESampler(seed=seed))
    directions = (["maximize", "minimize"] if multiobjective else None)
    direction = (None if multiobjective else "maximize")
    study = optuna.create_study(
        directions=directions, direction=direction, sampler=sampler,
        storage=storage, study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials)

    # Pick the winning params.
    pareto: list[dict] = []
    if multiobjective:
        # Best trials = the Pareto front; choose the highest-Sharpe knee to report.
        front = sorted(study.best_trials, key=lambda t: -t.values[0])
        pareto = [{"sharpe": t.values[0], "max_drawdown": -t.values[1],
                   "params": t.params} for t in front]
        best_params = front[0].params if front else {}
    else:
        best_params = study.best_params

    best_cfg = build(best_params)

    # Evaluate the winner on the FULL series once, then slice train vs hold-out.
    full_rets = _backtest(ohlcv, best_cfg, periods_per_year)
    train_rets = full_rets.iloc[:cut]
    hold_rets = full_rets.iloc[cut:]
    train_stats = compute_stats(train_rets, periods_per_year)
    hold_stats = compute_stats(hold_rets, periods_per_year)

    # Deflated Sharpe on the hold-out, penalized by the number of trials and the
    # CROSS-TRIAL dispersion of (train) Sharpe estimates — the right quantity
    # for E[max of N trials]. Note the deliberate, conservative asymmetry: the
    # canonical DSR deflates the *in-sample* Sharpe of the selected trial; we
    # apply the same luck benchmark to the *hold-out* Sharpe instead. The
    # hold-out wasn't part of the selection, so it needs less deflation — which
    # makes this a strictly harder bar to clear, never an easier one.
    pp_sharpes = [t.user_attrs.get("pp_sharpe", 0.0)
                  for t in study.trials
                  if t.state.name == "COMPLETE" and "pp_sharpe" in t.user_attrs]
    sr_trials_std = float(np.std(pp_sharpes)) if len(pp_sharpes) > 1 else 1.0
    pp_hold = per_period_sharpe(hold_rets)
    skew, kurt = returns_skew_kurt(hold_rets)
    dsr = deflated_sharpe_ratio(pp_hold, len(hold_rets.dropna()),
                                n_trials=max(len(pp_sharpes), 1),
                                sr_trials_std=max(sr_trials_std, 1e-6),
                                skew=skew, kurt=kurt)

    return OptimizeResult(
        best_params=best_params,
        best_config_name=best_cfg.name,
        train_stats=train_stats,
        holdout_stats=hold_stats,
        fold_sharpes=fold_sharpes(train_rets, n_folds, periods_per_year),
        deflated_sharpe=dsr,
        n_trials=len([t for t in study.trials if t.state.name == "COMPLETE"]),
        is_oos_gap=float(train_stats["sharpe"] - hold_stats["sharpe"]),
        holdout_returns=hold_rets,
        history=history,
        pareto=pareto,
    )
