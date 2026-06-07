"""
Tests for the walk-forward ML pipeline (out-of-sample predictions, signal
mapping, feature importances). Deterministic & offline; gradient-boosting
backends are gated with importorskip so the suite stays green on a minimal
install.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.ml import (
    assemble_dataset,
    binary_labels,
    feature_importances,
    fit_full_model,
    make_model,
    oos_signal,
    walk_forward_predict,
)


@pytest.fixture(scope="module")
def dataset(ohlcv):
    close = ohlcv["close"]
    y = binary_labels(close, horizon=5)
    X, y = assemble_dataset(ohlcv, y)
    return X, y


def test_make_model_logistic_fits(dataset):
    X, y = dataset
    m = make_model("logistic")
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    assert np.all((p >= 0) & (p <= 1))


def test_make_model_unknown_raises():
    with pytest.raises(ValueError):
        make_model("does_not_exist")


def test_walk_forward_predict_is_out_of_sample(dataset):
    X, y = dataset
    proba = walk_forward_predict(X, y, kind="logistic", n_splits=5,
                                 embargo=0.01, label_horizon=5)
    assert isinstance(proba, pd.Series)
    assert proba.index.equals(X.index)
    # The first block can't be predicted (no prior training data) -> NaN.
    first_block = len(X) // 6
    assert proba.iloc[:first_block].isna().all()
    # Later blocks are predicted and valid probabilities.
    pred = proba.dropna()
    assert len(pred) > 0
    assert np.all((pred >= 0) & (pred <= 1))


def test_walk_forward_is_causal(dataset):
    """Corrupting FUTURE labels must not change EARLY out-of-sample predictions.

    The early test blocks are produced by models trained only on data before
    them; tampering with the last 20% of labels can only affect later blocks. If
    an early prediction changed, the loop would be peeking at the future."""
    X, y = dataset
    n = len(X)
    fold = n // 6                     # n_splits=5 -> 6 blocks
    base = walk_forward_predict(X, y, kind="logistic", n_splits=5, label_horizon=5)

    y_corrupt = y.copy()
    tail = slice(int(n * 0.8), n)
    y_corrupt.iloc[tail] = 1.0 - y_corrupt.iloc[tail]   # flip the future labels
    corrupted = walk_forward_predict(X, y_corrupt, kind="logistic",
                                     n_splits=5, label_horizon=5)

    # Block 1 (indices [fold : 2*fold]) trains only on data before the tail,
    # so its predictions must be byte-for-byte identical.
    early = base.index[fold:2 * fold]
    a = base.reindex(early).dropna()
    b = corrupted.reindex(early).dropna()
    shared = a.index.intersection(b.index)
    assert len(shared) > 0
    assert np.allclose(a.loc[shared], b.loc[shared], atol=1e-12)


def test_oos_signal_mapping(dataset):
    X, y = dataset
    proba = pd.Series([np.nan, 0.4, 0.6, 0.8, 0.45], index=X.index[:5])
    long_only = oos_signal(proba, long_th=0.55)
    assert long_only.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    long_short = oos_signal(proba, long_th=0.55, short_th=0.45)
    # 0.4 < 0.45 -> short (-1); NaN -> flat
    assert long_short.tolist() == [0.0, -1.0, 1.0, 1.0, 0.0]


def test_feature_importances_logistic(dataset):
    X, y = dataset
    model = fit_full_model(X, y, kind="logistic")
    imp = feature_importances(model, X.columns)
    assert list(imp.index) and imp.is_monotonic_decreasing
    assert (imp >= 0).all() and imp.sum() > 0


def test_walk_forward_xgboost_if_available(dataset):
    pytest.importorskip("xgboost")
    X, y = dataset
    proba = walk_forward_predict(X, y, kind="xgboost", n_splits=4, label_horizon=5)
    pred = proba.dropna()
    assert len(pred) > 0 and np.all((pred >= 0) & (pred <= 1))
