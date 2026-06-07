"""
quantlab.research.pbo
=====================

The **Probability of Backtest Overfitting (PBO)** via **Combinatorially
Symmetric Cross-Validation (CSCV)** — Bailey, Borwein, López de Prado & Zhu
(2017). This is the gold-standard answer to the question every backtest begs:

    "I tried N configurations and picked the best. How likely is it that my
     selection process overfits — i.e. the in-sample winner is, out of sample,
     no better than a coin flip?"

The method, in plain terms:

1. Build a performance matrix ``M`` of shape ``(T_observations, N_configs)`` —
   each column is one strategy configuration's per-bar returns.
2. Slice the ``T`` rows into ``S`` contiguous blocks. For **every** way to pick
   ``S/2`` blocks as the in-sample set (the complement is out-of-sample):
     * choose the config that is best in-sample (n*),
     * find n*'s **rank** out-of-sample among all configs,
     * convert that rank to a logit ``λ = ln(ω/(1-ω))`` where ω is the relative
       rank in (0, 1).
3. **PBO = fraction of splits where λ < 0** — i.e. how often the in-sample best
   lands below the out-of-sample median. PBO near 0.5 means your selection is no
   better than chance (severe overfitting risk); PBO near 0 means the in-sample
   winner reliably stays good out of sample.

CSCV is symmetric (in-sample and out-of-sample swap across the C(S, S/2) splits),
which removes the bias of a single arbitrary train/test cut. Alongside PBO we
report **performance degradation** (the slope of OOS vs IS performance) and the
**probability of an out-of-sample loss**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from quantlab.backtest.engine import backtest_signal
from quantlab.backtest.strategy import build_signal


@dataclass
class PBOResult:
    """The overfitting verdict and its supporting evidence."""

    pbo: float                         # probability of backtest overfitting
    logits: np.ndarray                 # λ per CSCV split
    is_performance: np.ndarray         # IS performance of the selected config
    oos_performance: np.ndarray        # OOS performance of the selected config
    degradation_slope: float           # slope of OOS-on-IS regression (want >0, ~1)
    prob_oos_loss: float               # P(selected config loses out of sample)
    n_configs: int
    n_splits: int
    s_blocks: int

    @property
    def verdict(self) -> str:
        if self.pbo <= 0.2:
            return "Low overfitting risk — the in-sample winner generalizes."
        if self.pbo <= 0.5:
            return "Moderate overfitting risk — treat the 'best' config with care."
        return "High overfitting risk — selection is little better than chance."


def _slice_perf(M: np.ndarray, rows: np.ndarray, metric: str) -> np.ndarray:
    """Per-config performance over a set of rows (a 1-D vector of length N)."""
    sub = M[rows]
    if metric == "sharpe":
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = mu / sd
        # Configs with no dispersion can't win on Sharpe; send them to the bottom.
        r[~np.isfinite(r)] = -1e9
        return r
    if metric == "mean":
        return sub.mean(axis=0)
    raise ValueError("metric must be 'sharpe' or 'mean'")


def cscv_pbo(M, s_blocks: int = 12, metric: str = "sharpe") -> PBOResult:
    """Compute PBO from a performance matrix ``M`` (shape (T, N)).

    ``s_blocks`` (even) is the number of contiguous time blocks; the number of
    CSCV splits is ``C(s_blocks, s_blocks/2)``. ``metric`` ranks configs by
    Sharpe (default) or mean return per block-set.
    """
    M = np.asarray(M, dtype="float64")
    if M.ndim != 2:
        raise ValueError("M must be 2-D (T observations x N configs)")
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least 2 configurations to assess overfitting")
    if s_blocks % 2 != 0:
        raise ValueError("s_blocks must be even")
    if T < s_blocks:
        raise ValueError("not enough observations for the requested s_blocks")

    bounds = np.linspace(0, T, s_blocks + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(s_blocks)]
    all_blocks = set(range(s_blocks))

    logits, is_perf_sel, oos_perf_sel = [], [], []
    for is_combo in combinations(range(s_blocks), s_blocks // 2):
        oos_combo = sorted(all_blocks - set(is_combo))
        is_rows = np.concatenate([blocks[i] for i in is_combo])
        oos_rows = np.concatenate([blocks[i] for i in oos_combo])

        is_perf = _slice_perf(M, is_rows, metric)
        oos_perf = _slice_perf(M, oos_rows, metric)

        n_star = int(np.argmax(is_perf))             # in-sample winner
        # Out-of-sample rank of the winner (0 = worst, N-1 = best).
        order = np.argsort(oos_perf, kind="mergesort")
        rank = int(np.where(order == n_star)[0][0])
        omega = (rank + 1) / (N + 1)                  # relative rank in (0,1)
        logits.append(float(np.log(omega / (1.0 - omega))))
        is_perf_sel.append(float(is_perf[n_star]))
        oos_perf_sel.append(float(oos_perf[n_star]))

    logits = np.asarray(logits)
    is_arr = np.asarray(is_perf_sel)
    oos_arr = np.asarray(oos_perf_sel)

    pbo = float(np.mean(logits < 0.0))
    # Performance degradation: regress OOS on IS performance of the winners.
    if np.std(is_arr) > 1e-12:
        slope = float(np.polyfit(is_arr, oos_arr, 1)[0])
    else:
        slope = 0.0
    prob_loss = float(np.mean(oos_arr < 0.0))

    return PBOResult(
        pbo=pbo, logits=logits, is_performance=is_arr, oos_performance=oos_arr,
        degradation_slope=slope, prob_oos_loss=prob_loss,
        n_configs=N, n_splits=len(logits), s_blocks=s_blocks,
    )


# --------------------------------------------------------------------------- #
# Building the performance matrix from strategy configurations
# --------------------------------------------------------------------------- #
def sample_param_sets(space: dict, n: int, invalid=None, seed: int = 0) -> list[dict]:
    """Draw ``n`` valid random parameter sets from a template search space —
    the population of trials whose collective overfitting we assess."""
    rng = np.random.default_rng(seed)
    out, guard = [], 0
    while len(out) < n and guard < n * 50:
        guard += 1
        params = {}
        for name, spec in space.items():
            kind = spec[0]
            if kind == "int":
                params[name] = int(rng.integers(spec[1], spec[2] + 1))
            elif kind == "float":
                params[name] = float(rng.uniform(spec[1], spec[2]))
            elif kind == "categorical":
                params[name] = spec[1][int(rng.integers(0, len(spec[1])))]
        if invalid is not None and invalid(params):
            continue
        out.append(params)
    return out


def returns_matrix(ohlcv: pd.DataFrame, build, param_sets: list[dict],
                   periods_per_year: float = 252) -> np.ndarray:
    """Backtest each config on the full series and stack returns into M (T x N)."""
    cols = []
    for p in param_sets:
        cfg = build(p)
        sig = build_signal(ohlcv, cfg)
        r = backtest_signal(ohlcv["close"], sig, fee_bps=cfg.fee_bps,
                            periods_per_year=periods_per_year)["returns"]
        cols.append(r.fillna(0.0).to_numpy())
    return np.column_stack(cols)


def pbo_for_template(ohlcv: pd.DataFrame, build, space: dict, invalid=None,
                     n_configs: int = 40, s_blocks: int = 12,
                     metric: str = "sharpe", seed: int = 0,
                     periods_per_year: float = 252) -> PBOResult:
    """End-to-end PBO for a strategy template: sample configs, build the returns
    matrix, run CSCV. This is what the dashboard's Overfitting page calls."""
    param_sets = sample_param_sets(space, n_configs, invalid=invalid, seed=seed)
    M = returns_matrix(ohlcv, build, param_sets, periods_per_year)
    return cscv_pbo(M, s_blocks=s_blocks, metric=metric)
