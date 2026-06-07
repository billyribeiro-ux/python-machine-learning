"""
quantlab.research.cpcv
======================

**Combinatorial Purged Cross-Validation** (López de Prado, *Advances in
Financial Machine Learning*, Ch. 12). Where a single train/test split gives one
backtest path, CPCV gives **many** — and that distribution is what you should
judge a strategy on.

The idea: divide the timeline into ``N`` groups and, instead of testing on one
held-out fold, test on **every combination of ``k`` groups at once**, training on
the rest (purged + embargoed so labels can't leak across the boundary). This
produces ``C(N, k)`` train/test splits and, by recombination,

    φ  =  k · C(N, k) / N

distinct backtest **paths** — so you get a *sampling distribution* of Sharpe,
return, and drawdown rather than a single, fragile number. A strategy whose
worst CPCV paths are still acceptable is far more trustworthy than one with a
great average and a terrible tail.

This module provides the splitter (:class:`CombinatorialPurgedCV`) and a
convenience that turns a realized returns series into the distribution of
out-of-sample Sharpes across all combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from quantlab.backtest.stats import sharpe as _sharpe


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial purged cross-validator.

    Parameters
    ----------
    n_groups:
        Number of contiguous time groups the series is split into.
    n_test_groups:
        How many groups form the test set in each combination (``k``).
    embargo:
        Fraction of the series to embargo immediately after each test block
        (removes serially-correlated training rows).
    label_horizon:
        Bars a label looks forward; training rows within this many bars *before*
        a test block are purged (their outcome overlaps the test period).
    """

    n_groups: int = 10
    n_test_groups: int = 2
    embargo: float = 0.01
    label_horizon: int = 0

    def get_n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def n_paths(self) -> int:
        """Number of distinct backtest paths CPCV reconstructs."""
        return self.n_test_groups * comb(self.n_groups, self.n_test_groups) // self.n_groups

    def split(self, n_samples: int):
        """Yield ``(train_idx, test_idx)`` for every combination of test groups,
        with purging and embargo applied to the training indices."""
        if self.n_test_groups >= self.n_groups:
            raise ValueError("n_test_groups must be < n_groups")
        bounds = np.linspace(0, n_samples, self.n_groups + 1).astype(int)
        groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(self.n_groups)]
        emb = int(n_samples * self.embargo)

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[i] for i in combo])
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False
            # Purge + embargo around EACH contiguous test block.
            for g in combo:
                start, stop = bounds[g], bounds[g + 1]
                purge_lo = max(0, start - self.label_horizon)
                train_mask[purge_lo:start] = False           # purge before
                train_mask[stop:min(n_samples, stop + emb)] = False  # embargo after
            train_idx = np.where(train_mask)[0]
            yield train_idx, test_idx


def cpcv_sharpe_distribution(returns: pd.Series, n_groups: int = 10,
                             n_test_groups: int = 2,
                             periods_per_year: float = 252) -> np.ndarray:
    """Distribution of out-of-sample Sharpe across all CPCV test combinations.

    ``returns`` is a strategy's realized per-bar returns (computed once on the
    full series, so rolling-indicator warmup is intact). For each combination we
    Sharpe the concatenated test groups. Report the median and the 5th percentile
    (the "bad luck" path) — a robust strategy keeps the latter respectable.
    """
    r = pd.Series(returns).fillna(0.0)
    n = len(r)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]
    out = []
    for combo in combinations(range(n_groups), n_test_groups):
        idx = np.concatenate([groups[i] for i in combo])
        out.append(_sharpe(r.iloc[idx], periods_per_year))
    return np.asarray(out)
