"""
quantlab.indicators
====================

Indicator implementations. We start in pure NumPy (Module 1) and later add
Polars expression versions (Module 3), TA-Lib wrappers (Module 5), Numba
path-dependent indicators (Module 8), and a unified registry (Module 7).

Everything is re-exported here so callers can simply::

    from quantlab.indicators import sma, ewma, rsi, atr, rolling_zscore
"""

from .numpy_indicators import (
    atr,
    ewma,
    rolling_std,
    rolling_zscore,
    rsi,
    sma,
    true_range,
)

__all__ = [
    "sma",
    "ewma",
    "rolling_std",
    "rolling_zscore",
    "rsi",
    "true_range",
    "atr",
]
