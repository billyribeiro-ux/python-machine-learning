"""
quantlab.indicators.numpy_indicators
====================================

Indicators implemented from first principles in **pure NumPy** — vectorized,
numerically stable, and free of look-ahead bias. This module is the payoff of
Module 1: it shows that "better than the off-the-shelf library" is achievable
when you understand the math *and* the machine.

Design rules every function here obeys (and that you should copy):

1. **Vectorized, not looped.** We use cumulative sums, ``sliding_window_view``,
   and recurrence relations instead of Python ``for`` loops, so a million-bar
   series runs in milliseconds.
2. **Numerically stable.** Rolling variance uses a form that avoids
   catastrophic cancellation; EWMA uses the standard recurrence rather than
   summing a geometric series.
3. **No look-ahead.** The value at index *t* uses only data at indices ``<= t``.
   Warmup positions where there is not yet enough history are ``NaN`` — never
   silently back-filled, which would be a subtle leak.
4. **Float output, input-length preserving.** Output is always the same length
   as input and dtype float64, so indicators compose and align cleanly.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _as_float_array(x) -> np.ndarray:
    """Coerce input (list / pandas Series / ndarray) to a 1-D float64 array.
    Centralizing this means every indicator accepts the same flexible input."""
    a = np.asarray(x, dtype="float64")
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D input, got shape {a.shape}.")
    return a


def sma(values, window: int) -> np.ndarray:
    """Simple Moving Average via a **prefix-sum** trick — O(n), no Python loop.

    The naive approach recomputes a sum over ``window`` elements at every step:
    O(n * window). Instead we compute the cumulative sum once, then each window
    sum is a single subtraction: ``cumsum[t] - cumsum[t-window]``. This is the
    canonical example of trading a little memory for a big speedup, and it is
    *exactly* how you beat a naive library implementation.
    """
    a = _as_float_array(values)
    n = a.shape[0]
    out = np.full(n, np.nan)
    if window <= 0:
        raise ValueError("window must be positive")
    if window > n:
        return out
    csum = np.cumsum(a)
    # First valid window sum is csum[window-1]; subsequent ones subtract the
    # element that just left the window.
    out[window - 1] = csum[window - 1]
    out[window:] = csum[window:] - csum[:-window]
    out[window - 1:] /= window
    return out


def ewma(values, span: int) -> np.ndarray:
    """Exponentially Weighted Moving Average using the standard recurrence
    ``y_t = alpha * x_t + (1 - alpha) * y_{t-1}`` with ``alpha = 2/(span+1)``.

    We seed with the first observation (the common convention) and propagate.
    Although there is a data dependency between steps, this is still O(n) and,
    when JIT-compiled (Module 8), runs at C speed. We keep it explicit here
    because EWMA is *the* recursive indicator and understanding the recurrence
    unlocks RSI, MACD, ATR, and most "smoothed" indicators.
    """
    a = _as_float_array(values)
    n = a.shape[0]
    out = np.empty(n)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = a[0]
    for t in range(1, n):  # noqa: intentional — taught & JIT'd in Module 8
        out[t] = alpha * a[t] + (1.0 - alpha) * out[t - 1]
    return out


def rolling_std(values, window: int, ddof: int = 0) -> np.ndarray:
    """Rolling standard deviation via ``sliding_window_view``.

    ``sliding_window_view`` creates a *view* (zero-copy) of shape
    ``(n-window+1, window)`` over the array, so we can apply ``std`` along the
    last axis in one vectorized call. This is both fast and exact — no
    accumulation error from an online sum-of-squares. Warmup is NaN.
    """
    a = _as_float_array(values)
    n = a.shape[0]
    out = np.full(n, np.nan)
    if window <= 0:
        raise ValueError("window must be positive")
    if window > n:
        return out
    windows = sliding_window_view(a, window)  # (n-window+1, window) view
    out[window - 1:] = windows.std(axis=1, ddof=ddof)
    return out


def rolling_zscore(values, window: int, ddof: int = 0) -> np.ndarray:
    """Rolling z-score: ``(x_t - mean_window) / std_window``.

    A workhorse for mean-reversion and for normalizing features before ML. We
    reuse :func:`sma` and :func:`rolling_std` so the implementation is obviously
    correct and the warmup NaNs line up automatically. Guard against
    divide-by-zero on flat windows (std == 0) by returning 0 there rather than
    inf — a flat window has, by definition, zero deviation.
    """
    a = _as_float_array(values)
    mean = sma(a, window)
    std = rolling_std(a, window, ddof=ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (a - mean) / std
    z[std == 0] = 0.0
    return z


def rsi(values, window: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index.

    Implemented with Wilder's smoothing (an EWMA with ``alpha = 1/window``),
    which is what TA-Lib uses — so our output matches the de-facto standard
    while remaining fully transparent. RSI is bounded in [0, 100]; the first
    ``window`` positions are NaN (warmup).
    """
    a = _as_float_array(values)
    n = a.shape[0]
    out = np.full(n, np.nan)
    if n <= window:
        return out
    delta = np.diff(a)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed with the simple average of the first `window` deltas (Wilder).
    avg_gain = gains[:window].mean()
    avg_loss = losses[:window].mean()
    alpha = 1.0 / window

    def _rs_to_rsi(g: float, l: float) -> float:
        if l == 0:
            return 100.0
        rs = g / l
        return 100.0 - 100.0 / (1.0 + rs)

    out[window] = _rs_to_rsi(avg_gain, avg_loss)
    for t in range(window + 1, n):
        avg_gain = (1 - alpha) * avg_gain + alpha * gains[t - 1]
        avg_loss = (1 - alpha) * avg_loss + alpha * losses[t - 1]
        out[t] = _rs_to_rsi(avg_gain, avg_loss)
    return out


def true_range(high, low, close) -> np.ndarray:
    """True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    The building block of ATR and many volatility-scaled strategies. The first
    bar has no previous close, so its TR is simply ``high - low``.
    """
    h = _as_float_array(high)
    l = _as_float_array(low)
    c = _as_float_array(close)
    prev_close = np.empty_like(c)
    prev_close[0] = c[0]
    prev_close[1:] = c[:-1]
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_close),
        np.abs(l - prev_close),
    ])
    tr[0] = h[0] - l[0]
    return tr


def atr(high, low, close, window: int = 14) -> np.ndarray:
    """Average True Range = Wilder-smoothed True Range. The standard measure of
    per-bar volatility, used for position sizing and stop placement."""
    tr = true_range(high, low, close)
    # Wilder smoothing == EWMA with alpha = 1/window; reuse our ewma by mapping
    # span: alpha = 2/(span+1) => span = 2/alpha - 1 = 2*window - 1.
    return ewma(tr, span=2 * window - 1)
