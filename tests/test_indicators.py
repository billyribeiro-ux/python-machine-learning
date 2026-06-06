"""
Tests for the from-scratch NumPy indicators.

Strategy: prove our indicators match an *independent* reference (pandas), and
where TA-Lib is installed, match the de-facto industry standard too. An
indicator without a correctness test is a liability — a single off-by-one in a
rolling window silently biases every backtest that uses it.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.indicators import (
    atr,
    ewma,
    rolling_std,
    rolling_zscore,
    rsi,
    sma,
    true_range,
)


def _agree(a, b, atol=1e-9):
    """Compare two arrays ignoring positions where either is NaN (warmup)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.allclose(a[m], b[m], atol=atol)


def test_sma_matches_pandas(close):
    assert _agree(sma(close.to_numpy(), 20), close.rolling(20).mean().to_numpy())


def test_sma_prefix_sum_equals_naive(close):
    # Independent O(n*w) reference to catch any prefix-sum bug.
    x = close.to_numpy()
    w = 15
    naive = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        naive[i] = x[i - w + 1 : i + 1].mean()
    assert _agree(sma(x, w), naive)


def test_rolling_std_matches_pandas(close):
    got = rolling_std(close.to_numpy(), 20, ddof=0)
    ref = close.rolling(20).std(ddof=0).to_numpy()
    assert _agree(got, ref)


def test_ewma_matches_pandas(close):
    got = ewma(close.to_numpy(), span=20)
    ref = close.ewm(span=20, adjust=False).mean().to_numpy()
    assert _agree(got, ref)


def test_zscore_is_standardized(close):
    z = rolling_zscore(close.to_numpy(), 50)
    z = z[~np.isnan(z)]
    # A rolling z-score on a TRENDING series is legitimately biased away from 0
    # (price persistently above/below its own trailing mean), so this is a loose
    # sanity bound, not an N(0,1) claim: the mean shouldn't be many sigma off and
    # extremes shouldn't explode.
    assert abs(np.mean(z)) < 1.0
    assert np.percentile(np.abs(z), 99) < 6


def test_zscore_handles_flat_window():
    # A perfectly flat series has zero std; we must return 0, not inf/nan.
    flat = np.full(100, 42.0)
    z = rolling_zscore(flat, 10)
    assert np.all(z[~np.isnan(z)] == 0.0)


def test_rsi_is_bounded(close):
    r = rsi(close.to_numpy(), 14)
    r = r[~np.isnan(r)]
    assert r.min() >= 0.0 and r.max() <= 100.0


def test_true_range_first_bar(ohlcv):
    tr = true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    # First bar has no prior close, so TR = high - low by definition.
    assert tr[0] == pytest.approx(ohlcv["high"].iloc[0] - ohlcv["low"].iloc[0])
    assert (tr >= 0).all()


def test_atr_positive(ohlcv):
    a = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
    assert np.all(a[~np.isnan(a)] >= 0)


# ---- Optional: validate against TA-Lib when it's installed -----------------
def test_rsi_matches_talib_if_available(close):
    talib = pytest.importorskip("talib")  # skips cleanly if C lib not installed
    x = close.to_numpy()
    got = rsi(x, 14)
    ref = talib.RSI(x, timeperiod=14)
    # Wilder smoothing seeding can differ by a hair on the first valid point;
    # compare after a short burn-in to be robust.
    assert _agree(got[20:], ref[20:], atol=1e-6)


def test_sma_matches_talib_if_available(close):
    talib = pytest.importorskip("talib")
    x = close.to_numpy()
    assert _agree(sma(x, 30), talib.SMA(x, timeperiod=30), atol=1e-6)
