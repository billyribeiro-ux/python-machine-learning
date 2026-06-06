"""
quantlab.indicators.numba_indicators
====================================

**Path-dependent** indicators — the ones where bar *t*'s value depends on bar
*t-1*'s *output*, not just its input. Supertrend, trailing ATR stops, and
parabolic-style stops can't be written as a single vectorized array op without
O(n²) tricks. The honest implementation is a loop; Numba's ``@njit`` compiles
that loop to machine code so it runs at C speed.

Two engineering points this module makes:

1. **Know when vectorization loses.** Most indicators vectorize beautifully
   (Module 1). A minority are inherently sequential. Recognizing which is which
   — and not contorting a recurrence into a slow "clever" vectorized form — is
   the skill.
2. **Degrade gracefully.** Numba is optional. If it isn't installed, the same
   code runs in pure Python (slower) instead of crashing. We achieve that with a
   tiny ``maybe_njit`` shim, so the library works everywhere and is *fast* where
   Numba is present.
"""

from __future__ import annotations

import numpy as np

try:  # Numba is optional; fall back to a no-op decorator (pure-Python loop).
    from numba import njit as _njit

    def maybe_njit(*args, **kwargs):
        return _njit(*args, **kwargs)

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - environment dependent

    def maybe_njit(*args, **kwargs):
        # Support both @maybe_njit and @maybe_njit(cache=True) usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn):
            return fn

        return deco

    HAS_NUMBA = False


@maybe_njit(cache=True)
def _wilder_atr(high, low, close, period):
    """Wilder ATR as a JIT-friendly loop (used internally by Supertrend)."""
    n = high.shape[0]
    atr = np.empty(n)
    atr[:] = np.nan
    if n == 0:
        return atr
    # True range, bar by bar.
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    if n <= period:
        return atr
    # Seed with simple mean of first `period` TRs, then Wilder-smooth.
    s = 0.0
    for i in range(period):
        s += tr[i]
    prev = s / period
    atr[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        atr[i] = prev
    return atr


@maybe_njit(cache=True)
def _supertrend(high, low, close, period, multiplier):
    """Supertrend: an ATR-banded trailing trend filter.

    Returns (supertrend_line, direction) where direction is +1 (uptrend) or
    -1 (downtrend). The recurrence: the lower band can only ratchet UP while in
    an uptrend and the upper band can only ratchet DOWN in a downtrend; a close
    crossing the active band flips the regime. Each bar depends on the previous
    bar's band and direction — hence the loop.
    """
    n = close.shape[0]
    st = np.empty(n)
    direction = np.empty(n)
    st[:] = np.nan
    direction[:] = 1.0

    atr = _wilder_atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    final_upper = np.empty(n)
    final_lower = np.empty(n)
    final_upper[:] = np.nan
    final_lower[:] = np.nan

    start = period
    if n <= start:
        return st, direction

    final_upper[start] = hl2[start] + multiplier * atr[start]
    final_lower[start] = hl2[start] - multiplier * atr[start]
    direction[start] = 1.0
    st[start] = final_lower[start]

    for i in range(start + 1, n):
        basic_upper = hl2[i] + multiplier * atr[i]
        basic_lower = hl2[i] - multiplier * atr[i]

        # Bands ratchet: tighten toward price, never loosen, until a flip.
        if basic_upper < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower
        else:
            final_lower[i] = final_lower[i - 1]

        # Regime flip logic.
        if close[i] > final_upper[i]:
            direction[i] = 1.0
        elif close[i] < final_lower[i]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

        st[i] = final_lower[i] if direction[i] > 0 else final_upper[i]

    return st, direction


def supertrend(high, low, close, period: int = 10, multiplier: float = 3.0):
    """Public wrapper: accepts array-likes, returns (line, direction) arrays.

    ``direction`` is +1 in an uptrend (line is the trailing support) and -1 in a
    downtrend (line is the trailing resistance). A flip from -1 to +1 is a long
    entry signal; +1 to -1 is an exit/short.
    """
    h = np.asarray(high, dtype="float64")
    l = np.asarray(low, dtype="float64")
    c = np.asarray(close, dtype="float64")
    return _supertrend(h, l, c, int(period), float(multiplier))


@maybe_njit(cache=True)
def _chandelier_long_stop(high, low, close, period, multiplier):
    """Chandelier exit for a long: highest-high minus k*ATR, ratcheting up.

    A classic trailing stop. The stop can only rise while price holds, never
    fall — again a path-dependent recurrence."""
    n = close.shape[0]
    stop = np.empty(n)
    stop[:] = np.nan
    atr = _wilder_atr(high, low, close, period)
    if n <= period:
        return stop
    hh = high[period]
    stop[period] = hh - multiplier * atr[period]
    for i in range(period + 1, n):
        if high[i] > hh:
            hh = high[i]
        raw = hh - multiplier * atr[i]
        # Ratchet: the long stop never moves down.
        stop[i] = max(raw, stop[i - 1]) if not np.isnan(stop[i - 1]) else raw
    return stop


def chandelier_long_stop(high, low, close, period: int = 22, multiplier: float = 3.0):
    """Trailing stop level for a long position (highest high - k*ATR)."""
    h = np.asarray(high, dtype="float64")
    l = np.asarray(low, dtype="float64")
    c = np.asarray(close, dtype="float64")
    return _chandelier_long_stop(h, l, c, int(period), float(multiplier))
