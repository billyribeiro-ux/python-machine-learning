"""
Property-based tests (Module 17) using Hypothesis.

Example-based tests check the cases you imagined; property tests check
*invariants* over thousands of random inputs and shrink failures to a minimal
counterexample. The properties here must hold for ANY price series — a far
stronger guarantee than "passes on AAPL".
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from quantlab.indicators import atr, rolling_zscore, rsi, sma

# A strategy that generates realistic positive price arrays of varied length.
prices = arrays(
    np.float64,
    st.integers(min_value=60, max_value=300),
    elements=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False,
                       allow_infinity=False),
)


@given(prices)
@settings(max_examples=150, deadline=None)
def test_rsi_always_bounded(x):
    r = rsi(x, 14)
    r = r[~np.isnan(r)]
    assert np.all((r >= 0.0) & (r <= 100.0))


@given(prices)
@settings(max_examples=150, deadline=None)
def test_sma_within_window_bounds(x):
    w = 10
    s = sma(x, w)
    for i in range(w - 1, len(x)):
        window = x[i - w + 1 : i + 1]
        assert window.min() - 1e-9 <= s[i] <= window.max() + 1e-9


@given(prices)
@settings(max_examples=150, deadline=None)
def test_zscore_no_lookahead(x):
    """Truncating the future must not change past z-scores (no peeking)."""
    if len(x) < 30:
        return
    z_full = rolling_zscore(x, 20)
    k = len(x) - 5
    z_trunc = rolling_zscore(x[:k], 20)
    m = ~np.isnan(z_full[:k]) & ~np.isnan(z_trunc)
    assert np.allclose(z_full[:k][m], z_trunc[m], atol=1e-9)


@given(prices)
@settings(max_examples=150, deadline=None)
def test_rsi_no_lookahead(x):
    if len(x) < 40:
        return
    r_full = rsi(x, 14)
    k = len(x) - 5
    r_trunc = rsi(x[:k], 14)
    m = ~np.isnan(r_full[:k]) & ~np.isnan(r_trunc)
    assert np.allclose(r_full[:k][m], r_trunc[m], atol=1e-9)


# ATR needs OHLC; build a consistent high>=close>=low triple from one series.
@given(prices)
@settings(max_examples=120, deadline=None)
def test_atr_non_negative(x):
    high = x * 1.01
    low = x * 0.99
    a = atr(high, low, x, 14)
    a = a[~np.isnan(a)]
    assert np.all(a >= 0.0)
