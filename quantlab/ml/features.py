"""
quantlab.ml.features
====================

Turn OHLCV into a **leak-free feature matrix** for supervised learning. We reuse
the indicator registry (Module 7) so features are declared as ``(name, params)``
specs — the same vocabulary as everything else in QuantLab.

The cardinal rule (again): every feature at row *t* uses only information
available at *t*. Indicators here are all backward-looking; the *target* is the
only forward-looking quantity, and it gets NaN'd at the tail so you can never
train on an unknown future (see :mod:`quantlab.ml.labeling`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.indicators.registry import combine

# A sensible default feature set spanning trend, momentum, volatility, and
# mean-reversion — diversified so the model has complementary signals.
DEFAULT_FEATURES: list[tuple[str, dict]] = [
    ("ret", {"periods": 1}),
    ("ret", {"periods": 5}),
    ("ret", {"periods": 20}),
    ("rsi", {"window": 14}),
    ("zscore", {"window": 20}),
    ("atr", {"window": 14}),
    ("sma", {"window": 50}),
    ("std", {"window": 20}),
]


def make_features(ohlcv: pd.DataFrame,
                  specs: list[tuple[str, dict]] | None = None,
                  normalize: bool = True) -> pd.DataFrame:
    """Build a feature DataFrame from OHLCV via the indicator registry.

    Parameters
    ----------
    ohlcv:
        Normalized OHLCV frame.
    specs:
        Indicator specs; defaults to :data:`DEFAULT_FEATURES`.
    normalize:
        If True, scale level-like features (sma, atr) by close so they are
        comparable across price regimes and symbols (a $500 stock and a $20
        stock become directly comparable). Returns/RSI/z-score are already
        scale-free and left as-is.
    """
    specs = specs or DEFAULT_FEATURES
    X = combine(ohlcv, specs)

    if normalize:
        close = ohlcv["close"]
        for col in X.columns:
            # Level features (price-scaled) -> express as a ratio to close.
            if col.startswith(("sma_", "atr_", "std_", "ewma_")):
                X[col] = X[col] / close - (1.0 if col.startswith(("sma_", "ewma_")) else 0.0)
    return X


def assemble_dataset(ohlcv: pd.DataFrame, target: pd.Series,
                     specs: list[tuple[str, dict]] | None = None
                     ) -> tuple[pd.DataFrame, pd.Series]:
    """Join features and a target, dropping warmup/tail NaNs so X and y align.

    Returns ``(X, y)`` ready for scikit-learn / XGBoost. The single
    ``dropna()`` here removes both the indicator warmup at the start and the
    forward-looking label's NaNs at the end — so it is impossible to train on a
    row whose outcome isn't known yet.
    """
    X = make_features(ohlcv, specs)
    data = X.join(target.rename("__target__"))
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    y = data.pop("__target__")
    return data, y
