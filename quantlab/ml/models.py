"""
quantlab.ml.models
==================

Thin, opinionated wrappers around **XGBoost** and **LightGBM** for trading.
The opinions are the point — they encode the practices that keep tree models
honest on financial data:

* **Time-ordered splits, never random.** A random split lets the model see the
  future; we always split chronologically.
* **Early stopping on a real validation set** to avoid overfitting the trees.
* **Predictions are probabilities**, mapped to positions by the caller (so
  position sizing is a separate, testable decision).

Both libraries are optional; we import lazily with a clear message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3):
    """Chronological train/test split (no shuffling). The last ``test_size``
    fraction is the out-of-sample test set."""
    n = len(X)
    cut = int(n * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def train_xgb(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3,
              params: dict | None = None):
    """Train an XGBoost classifier with a chronological split + early stopping.

    Returns ``(model, info)`` where ``info`` holds the test AUC/accuracy and the
    held-out predictions for backtesting.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost not installed: pip install xgboost") from exc

    Xtr, Xte, ytr, yte = time_split(X, y, test_size)
    defaults = dict(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", early_stopping_rounds=30, n_jobs=-1,
        tree_method="hist",
    )
    defaults.update(params or {})
    model = XGBClassifier(**defaults)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    proba = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index, name="proba")
    return model, _eval_info(yte, proba)


def train_lgbm(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3,
               params: dict | None = None):
    """Train a LightGBM classifier (chronological split + early stopping)."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lightgbm not installed: pip install lightgbm") from exc

    Xtr, Xte, ytr, yte = time_split(X, y, test_size)
    defaults = dict(
        n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
    )
    defaults.update(params or {})
    model = lgb.LGBMClassifier(**defaults)
    model.fit(
        Xtr, ytr, eval_set=[(Xte, yte)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    proba = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index, name="proba")
    return model, _eval_info(yte, proba)


def _eval_info(y_true: pd.Series, proba: pd.Series) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score

    pred = (proba > 0.5).astype(int)
    info = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "proba": proba,
        "y_true": y_true,
    }
    # AUC needs both classes present in the test fold.
    try:
        info["auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        info["auc"] = float("nan")
    return info


def proba_to_signal(proba: pd.Series, long_th: float = 0.55,
                    flat_below: float = 0.5) -> pd.Series:
    """Map predicted up-probabilities to a {0,1} long/flat signal.

    Going long only when the model is *confident* (> ``long_th``), not merely
    >0.5, is a simple, robust way to trade a noisy classifier. The backtest
    engine then applies the one-bar execution delay."""
    return (proba > long_th).astype("float")
