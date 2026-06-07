"""
quantlab.research.walkforward
=============================

Time-series evaluation utilities. The cardinal rule of strategy research: the
number you *report* must come from data the *search* never touched.

* :func:`chronological_split` — carve off a final hold-out the optimizer never
  sees. This is the honest out-of-sample test.
* :func:`fold_sharpes` — split a realized returns series into contiguous
  sub-periods and Sharpe each. Used to build a **robust** objective: a strategy
  that works across several regimes, not one lucky stretch.

These operate on *realized returns* (computed once on the full series so rolling
indicators keep their warmup), then slice — never recomputing signals on tiny
windows, which would corrupt warmup and quietly bias results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.backtest.stats import sharpe as _sharpe


def chronological_split(frame, holdout: float = 0.25):
    """Split a DataFrame/Series into (train, holdout) by time order.

    ``holdout`` is the fraction reserved at the END (the future relative to
    training). No shuffling — that would leak the future into the past.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be in (0, 1)")
    n = len(frame)
    cut = int(n * (1.0 - holdout))
    return frame.iloc[:cut], frame.iloc[cut:]


def split_index(n: int, holdout: float = 0.25) -> int:
    return int(n * (1.0 - holdout))


def fold_sharpes(returns: pd.Series, n_folds: int = 4,
                 periods_per_year: float = 252) -> list[float]:
    """Annualized Sharpe within each of ``n_folds`` contiguous sub-periods.

    A spread of fold Sharpes that are all positive signals a robust edge; one
    huge fold and several negative ones signals a regime-specific fluke. The
    optimizer penalizes that dispersion (see :mod:`quantlab.research.optimize`).
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < n_folds or n_folds < 1:
        return [_sharpe(r, periods_per_year)] if n else [0.0]
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    out = []
    for i in range(n_folds):
        seg = r.iloc[bounds[i]:bounds[i + 1]]
        out.append(_sharpe(seg, periods_per_year) if len(seg) > 1 else 0.0)
    return out
