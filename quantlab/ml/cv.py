"""
quantlab.ml.cv
==============

Cross-validation that doesn't lie in a time-series setting.

Plain ``KFold`` shuffles rows, so a model can train on Tuesday and Thursday and
"predict" Wednesday — using the future to predict the past. Even ``TimeSeries
Split`` leaks subtly when labels span multiple bars (a triple-barrier label at
*t* peeks up to *t+horizon*, which may sit in the training fold). The fix:

* **Purge** — drop training samples whose label window overlaps the test fold.
* **Embargo** — additionally drop training samples immediately *after* the test
  fold, because they are serially correlated with it.

This ``PurgedKFold`` is scikit-learn-compatible (``split`` yields train/test
index arrays), so it drops straight into ``cross_val_score`` and friends.
"""

from __future__ import annotations

import numpy as np


class PurgedKFold:
    """K-fold for time series with purging and an embargo.

    Parameters
    ----------
    n_splits:
        Number of folds.
    embargo:
        Fraction of the dataset length to embargo after each test fold
        (e.g. 0.01 = 1%).
    label_horizon:
        How many bars forward a label looks (e.g. the triple-barrier horizon).
        Training samples within ``label_horizon`` bars *before* the test fold are
        purged because their outcome overlaps the test period.
    """

    def __init__(self, n_splits: int = 5, embargo: float = 0.01,
                 label_horizon: int = 0):
        self.n_splits = n_splits
        self.embargo = embargo
        self.label_horizon = label_horizon

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        embargo_n = int(n * self.embargo)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]

            # Start with everything not in the test fold.
            train_mask = np.ones(n, dtype=bool)
            train_mask[start:stop] = False

            # Purge: remove training rows whose label window reaches into test.
            purge_start = max(0, start - self.label_horizon)
            train_mask[purge_start:start] = False

            # Embargo: remove training rows just after the test fold.
            embargo_stop = min(n, stop + embargo_n)
            train_mask[stop:embargo_stop] = False

            train_idx = indices[train_mask]
            current = stop
            yield train_idx, test_idx
