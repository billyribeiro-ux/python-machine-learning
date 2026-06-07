"""
quantlab.ml.pipeline
====================

The honest ML-for-trading pipeline: **walk-forward, purged, out-of-sample**
predictions, optional probability **calibration**, and evaluation by **PnL**, not
accuracy. This is what separates a research result you can trade from a
backtest that secretly read the future.

Why walk-forward (not a single split)? A model trained once and tested once
gives one number on one regime. Walk-forward retrains as time advances and
predicts only the *next* untouched block — so the concatenated predictions span
the whole history and are each genuinely out-of-sample. We additionally **purge**
the bars whose labels overlap the test block and **embargo** the bars right
after it (labels are forward-looking and serially correlated — Module 11).

Everything is model-agnostic via :func:`make_model` (logistic / XGBoost /
LightGBM), and heavy libraries are imported lazily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_model(kind: str = "logistic", params: dict | None = None):
    """Return an unfitted, scikit-learn-compatible classifier.

    * ``"logistic"`` — StandardScaler + LogisticRegression (fast, calibrated,
      a strong honest baseline).
    * ``"xgboost"`` / ``"lightgbm"`` — gradient-boosted trees.
    """
    params = params or {}
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, **params))
    if kind == "xgboost":
        from xgboost import XGBClassifier

        defaults = dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                        tree_method="hist", eval_metric="logloss")
        defaults.update(params)
        return XGBClassifier(**defaults)
    if kind == "lightgbm":
        import lightgbm as lgb

        defaults = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
        defaults.update(params)
        return lgb.LGBMClassifier(**defaults)
    raise ValueError(f"Unknown model kind {kind!r} (logistic/xgboost/lightgbm)")


def walk_forward_predict(X: pd.DataFrame, y: pd.Series, kind: str = "logistic",
                         n_splits: int = 5, embargo: float = 0.01,
                         label_horizon: int = 0, params: dict | None = None,
                         calibrate: bool = False) -> pd.Series:
    """Return an out-of-sample probability series via expanding walk-forward.

    The history is divided into ``n_splits + 1`` contiguous blocks. Block *i+1*
    is predicted by a model trained on blocks ``0..i`` minus a purge of
    ``label_horizon`` bars and an embargo of ``embargo`` of the data — so no
    training label can overlap the block being predicted. The first block has no
    prior data and stays NaN.

    With ``calibrate=True`` the probabilities are isotonically calibrated on a
    slice of the training data (Platt/again-leak-free), which matters a lot when
    you later threshold them into positions.
    """
    n = len(X)
    proba = np.full(n, np.nan)
    if n < (n_splits + 1) * 5:
        return pd.Series(proba, index=X.index, name="proba")

    fold = n // (n_splits + 1)
    emb = int(n * embargo)

    for i in range(1, n_splits + 1):
        test_start = i * fold
        test_end = (i + 1) * fold if i < n_splits else n
        train_end = max(0, test_start - emb - label_horizon)   # purge + embargo
        if train_end < 30:
            continue
        ytr = y.iloc[:train_end]
        if ytr.nunique() < 2:        # need both classes to train a classifier
            continue
        Xtr = X.iloc[:train_end]
        Xte = X.iloc[test_start:test_end]

        model = make_model(kind, params)
        if calibrate:
            from sklearn.calibration import CalibratedClassifierCV

            # Calibrate with an internal time-respecting split of the train set.
            model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        model.fit(Xtr, ytr)
        proba[test_start:test_end] = model.predict_proba(Xte)[:, 1]

    return pd.Series(proba, index=X.index, name="proba")


def fit_full_model(X: pd.DataFrame, y: pd.Series, kind: str = "logistic",
                   params: dict | None = None):
    """Fit one model on ALL data — for feature importances / inspection only
    (never for evaluating performance, which must be the walk-forward OOS run)."""
    model = make_model(kind, params)
    model.fit(X, y)
    return model


def feature_importances(model, feature_names) -> pd.Series:
    """Best-effort feature importances from a fitted model (trees) or absolute
    coefficients (linear), returned sorted descending."""
    est = model
    # Unwrap an sklearn Pipeline to its final estimator.
    if hasattr(model, "named_steps"):
        est = list(model.named_steps.values())[-1]
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype="float64")
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_, dtype="float64").ravel())
    else:
        vals = np.zeros(len(feature_names))
    return pd.Series(vals, index=list(feature_names)).sort_values(ascending=False)


def oos_signal(proba: pd.Series, long_th: float = 0.55,
               short_th: float | None = None) -> pd.Series:
    """Map OOS probabilities to a position: long above ``long_th``; optionally
    short below ``short_th``. NaN (un-predicted) bars are flat."""
    sig = (proba > long_th).astype("float64")
    if short_th is not None:
        sig = sig - (proba < short_th).astype("float64")
    return sig.where(proba.notna(), 0.0)
