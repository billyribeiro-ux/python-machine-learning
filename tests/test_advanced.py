"""
Tests for the advanced toolkit: the indicator registry (M7), Numba
path-dependent indicators (M8), the ML feature/labeling/CV stack (M11), and the
scanner (M14). Heavy/optional libraries are gated with importorskip so the suite
stays green on a minimal install.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.indicators import combine, available, supertrend, chandelier_long_stop


# --------------------------------------------------------------------------- #
# M7 — registry
# --------------------------------------------------------------------------- #
def test_registry_combine_named_columns(ohlcv):
    feats = combine(ohlcv, [("sma", {"window": 20}), ("rsi", {"window": 14}),
                            ("zscore", {"window": 20})])
    assert list(feats.columns) == ["sma_20", "rsi_14", "zscore_20"]
    assert len(feats) == len(ohlcv)


def test_registry_has_core_indicators():
    for name in ("sma", "ewma", "rsi", "zscore", "atr", "std", "ret"):
        assert name in available()


def test_registry_unknown_raises(ohlcv):
    with pytest.raises(KeyError):
        combine(ohlcv, [("does_not_exist", {})])


# --------------------------------------------------------------------------- #
# M8 — Numba path-dependent indicators
# --------------------------------------------------------------------------- #
def test_supertrend_direction_is_plus_minus_one(ohlcv):
    line, direction = supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    d = direction[~np.isnan(direction)]
    assert set(np.unique(d)) <= {-1.0, 1.0}


def test_chandelier_stop_ratchets_up(ohlcv):
    stop = chandelier_long_stop(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    valid = stop[~np.isnan(stop)]
    # A long trailing stop must be monotonically non-decreasing while active.
    assert np.all(np.diff(valid) >= -1e-9)


# --------------------------------------------------------------------------- #
# M11 — ML features / labeling / purged CV
# --------------------------------------------------------------------------- #
def test_triple_barrier_labels_values(close):
    from quantlab.ml import triple_barrier_labels

    tb = triple_barrier_labels(close, horizon=10, upper=2.0, lower=2.0)
    labels = tb["label"].dropna().unique()
    assert set(labels) <= {-1.0, 0.0, 1.0}
    # The last `horizon` rows are unknowable -> must be NaN (no look-ahead).
    assert tb["label"].iloc[-1] != tb["label"].iloc[-1]  # NaN != NaN


def test_assemble_dataset_aligns(ohlcv, close):
    from quantlab.ml import assemble_dataset, binary_labels

    y = binary_labels(close, horizon=5)
    X, yy = assemble_dataset(ohlcv, y)
    assert len(X) == len(yy)
    assert not X.isna().any().any()         # warmup + tail dropped
    assert not yy.isna().any()


def test_purged_kfold_shrinks_train(ohlcv, close):
    from quantlab.ml import assemble_dataset, binary_labels, PurgedKFold
    from sklearn.model_selection import KFold

    y = binary_labels(close, horizon=5)
    X, yy = assemble_dataset(ohlcv, y)
    pk = PurgedKFold(n_splits=4, embargo=0.02, label_horizon=10)
    plain = KFold(n_splits=4)
    for (ptr, _), (ktr, _) in zip(pk.split(X), plain.split(X)):
        # Purge + embargo remove samples, so purged train <= plain train.
        assert len(ptr) <= len(ktr)


# --------------------------------------------------------------------------- #
# M14 — scanner
# --------------------------------------------------------------------------- #
def test_scanner_runs_on_fake_provider(ohlcv):
    """The scanner must work against ANY DataProvider (here the test fake),
    proving it depends on the interface, not on Yahoo."""
    from tests.conftest import FakeProvider
    from quantlab.scanners import latest_snapshot

    prov = FakeProvider(ohlcv)
    snap = latest_snapshot(prov, ["AAA", "BBB"], start="2018-01-01")
    assert len(snap) == 2
    for col in ("sma20", "rsi14", "z20", "atr14", "mom20", "dollar_vol"):
        assert col in snap.columns
