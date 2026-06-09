"""
Regression tests for the bugs found in the full-course audit. Each test exists
because a specific defect was found (or a specific behavior was deemed worth
pinning) — if any of these fail, an audited fix has regressed.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.data.yahoo import YahooProvider
from quantlab.data.base import OHLCVRequest


# --------------------------------------------------------------------------- #
# AUDIT FIX 1 (critical): cache staleness for open-ended requests.
# A request with end=None means "through now" — its cache key must move with
# the clock, or the paper-trading engine reads day-one data forever.
# --------------------------------------------------------------------------- #
def test_open_ended_cache_key_rolls_daily(tmp_path, monkeypatch):
    import quantlab.data.yahoo as yahoo_mod

    prov = YahooProvider(cache_dir=tmp_path)
    req = OHLCVRequest("SPY", start="2020-01-01", end=None, timeframe="1d")

    monkeypatch.setattr(yahoo_mod, "_utcnow",
                        lambda: pd.Timestamp("2026-06-08", tz="UTC"))
    day1 = prov._cache_path(req)
    monkeypatch.setattr(yahoo_mod, "_utcnow",
                        lambda: pd.Timestamp("2026-06-09", tz="UTC"))
    day2 = prov._cache_path(req)
    assert day1 != day2                     # tomorrow must MISS yesterday's cache


def test_open_ended_intraday_cache_rolls_hourly(tmp_path, monkeypatch):
    import quantlab.data.yahoo as yahoo_mod

    prov = YahooProvider(cache_dir=tmp_path)
    req = OHLCVRequest("SPY", start=None, end=None, timeframe="5m")
    monkeypatch.setattr(yahoo_mod, "_utcnow",
                        lambda: pd.Timestamp("2026-06-08 14:05", tz="UTC"))
    h1 = prov._cache_path(req)
    monkeypatch.setattr(yahoo_mod, "_utcnow",
                        lambda: pd.Timestamp("2026-06-08 15:05", tz="UTC"))
    h2 = prov._cache_path(req)
    assert h1 != h2                          # intraday: stale within the day too


def test_closed_request_cache_key_is_stable(tmp_path, monkeypatch):
    """A fully-specified historical request never changes -> key must be stable
    across days (that's the whole point of the cache)."""
    import quantlab.data.yahoo as yahoo_mod

    prov = YahooProvider(cache_dir=tmp_path)
    req = OHLCVRequest("SPY", start="2020-01-01", end="2021-01-01", timeframe="1d")
    monkeypatch.setattr(yahoo_mod, "_utcnow",
                        lambda: pd.Timestamp("2026-06-08", tz="UTC"))
    p1 = prov._cache_path(req)
    monkeypatch.setattr(yahoo_mod, "_utcnow",
                        lambda: pd.Timestamp("2026-07-01", tz="UTC"))
    p2 = prov._cache_path(req)
    assert p1 == p2


# --------------------------------------------------------------------------- #
# AUDIT FIX 2: Sharpe/Sortino edge cases. A profitable strategy with zero
# volatility / zero downside is infinitely good, not worthless.
# --------------------------------------------------------------------------- #
def test_sharpe_constant_positive_returns_is_inf():
    from quantlab.backtest.stats import sharpe

    # 0.5 is exactly representable -> std is exactly 0 -> the inf branch.
    assert sharpe(pd.Series(np.full(100, 0.5))) == np.inf
    # 0.001 is NOT representable -> std ~1e-19 -> astronomically large ratio.
    # Either way the strategy can no longer masquerade as "no alpha" (the bug).
    assert sharpe(pd.Series(np.full(100, 0.001))) > 1e6
    assert sharpe(pd.Series(np.zeros(100))) == 0.0      # never trades -> 0


def test_sortino_all_positive_returns_is_inf():
    from quantlab.backtest.stats import sortino

    rng = np.random.default_rng(0)
    r = pd.Series(np.abs(rng.normal(0.001, 0.0005, 100)) + 1e-6)  # no losses
    assert sortino(r) == np.inf
    # Mixed returns still produce a finite ratio.
    r2 = pd.Series(rng.normal(0.0, 0.01, 200))
    assert np.isfinite(sortino(r2))


# --------------------------------------------------------------------------- #
# AUDIT FIX 3: duplicate indicator specs must fail loudly, not produce
# silently-duplicated columns.
# --------------------------------------------------------------------------- #
def test_registry_rejects_duplicate_specs(ohlcv):
    from quantlab.indicators import combine

    with pytest.raises(ValueError, match="Duplicate indicator columns"):
        combine(ohlcv, [("sma", {"window": 20}), ("sma", {"window": 20})])


# --------------------------------------------------------------------------- #
# AUDIT FIX 4: the JIT'd ewma/rsi kernels must be numerically identical to the
# original pure-Python recurrences (pandas is the independent referee).
# --------------------------------------------------------------------------- #
def test_jitted_ewma_still_matches_pandas(close):
    from quantlab.indicators import ewma

    got = ewma(close.to_numpy(), span=20)
    ref = close.ewm(span=20, adjust=False).mean().to_numpy()
    assert np.allclose(got, ref)


def test_jitted_rsi_still_bounded_and_seeded(close):
    from quantlab.indicators import rsi

    r = rsi(close.to_numpy(), 14)
    assert np.isnan(r[:14]).all()                      # warmup unchanged
    valid = r[~np.isnan(r)]
    assert np.all((valid >= 0) & (valid <= 100))
