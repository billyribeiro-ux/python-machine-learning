"""
quantlab.ml.optimize
====================

**Optuna** objectives that optimize the things that actually matter in trading:
not validation accuracy, but *backtested performance* — and not a single metric,
but the trade-off between return and risk.

Two reusable objectives:

* :func:`optimize_combo` — tune the multi-indicator strategy's settings
  (fast/slow/RSI/z thresholds) to maximize walk-forward-ish Sharpe. This is the
  principled version of the Module 9 grid search: a smart sampler instead of
  brute force, with pruning.
* :func:`optimize_combo_multiobjective` — maximize Sharpe *and* minimize max
  drawdown simultaneously, returning a Pareto front instead of one "best".

Optuna is optional; imported lazily.
"""

from __future__ import annotations

import pandas as pd

from quantlab.strategies.combo import ComboParams, run_combo


def optimize_combo(close: pd.Series, n_trials: int = 50,
                   periods_per_year: float = 252, seed: int = 42):
    """Single-objective: find ComboParams that maximize Sharpe.

    Returns the Optuna ``study`` (so you can inspect ``study.best_params`` and
    the full trial history for overfitting diagnostics).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        fast = trial.suggest_int("fast", 5, 50)
        slow = trial.suggest_int("slow", 60, 250)
        rsi_floor = trial.suggest_float("rsi_floor", 30, 55)
        z_entry = trial.suggest_float("z_entry", -1.0, 1.5)
        if fast >= slow:  # enforce a valid trend pair; prune invalid combos
            raise optuna.TrialPruned()
        p = ComboParams(fast=fast, slow=slow, rsi_floor=rsi_floor, z_entry=z_entry)
        res = run_combo(close, p, periods_per_year=periods_per_year)
        return res["stats"]["sharpe"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def optimize_combo_multiobjective(close: pd.Series, n_trials: int = 80,
                                  periods_per_year: float = 252, seed: int = 42):
    """Multi-objective: maximize Sharpe AND minimize |max drawdown|.

    Returns the study; ``study.best_trials`` is the Pareto front — the set of
    configurations where you can't improve one objective without hurting the
    other. Picking from the front is a *judgement* call (how much drawdown can
    you stomach?), which is exactly right: optimization informs, it doesn't
    decide.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        fast = trial.suggest_int("fast", 5, 50)
        slow = trial.suggest_int("slow", 60, 250)
        rsi_floor = trial.suggest_float("rsi_floor", 30, 55)
        z_entry = trial.suggest_float("z_entry", -1.0, 1.5)
        if fast >= slow:
            raise optuna.TrialPruned()
        p = ComboParams(fast=fast, slow=slow, rsi_floor=rsi_floor, z_entry=z_entry)
        s = run_combo(close, p, periods_per_year=periods_per_year)["stats"]
        return s["sharpe"], abs(s["max_drawdown"])

    sampler = optuna.samplers.NSGAIISampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
