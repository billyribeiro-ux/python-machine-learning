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
from .numba_indicators import chandelier_long_stop, supertrend
from .registry import available, combine, register

__all__ = [
    # from-scratch NumPy (Module 1)
    "sma",
    "ewma",
    "rolling_std",
    "rolling_zscore",
    "rsi",
    "true_range",
    "atr",
    # path-dependent / Numba (Module 8)
    "supertrend",
    "chandelier_long_stop",
    # unified registry (Module 7)
    "combine",
    "register",
    "available",
]
