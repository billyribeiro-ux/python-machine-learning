"""
Shared pytest fixtures for QuantLab.

A core testing principle: **tests must be fast, deterministic, and offline.**
We never hit Yahoo in unit tests — a network call makes a test slow, flaky, and
dependent on the outside world. Instead we synthesize realistic OHLCV with a
seeded generator and expose it both as a DataFrame and through a fake
``DataProvider`` (proving, incidentally, that depending on the *interface*
rather than the vendor is what makes the code testable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.base import DataProvider, OHLCVRequest


def _synthetic_ohlcv(n: int = 500, seed: int = 7, start: str = "2018-01-01") -> pd.DataFrame:
    """Generate a plausible daily OHLCV frame via geometric Brownian motion.

    The output already meets the normalized contract (UTC index named
    'timestamp', lowercase float OHLCV, high>=everything>=low) so it can stand
    in for real data everywhere.
    """
    rng = np.random.default_rng(seed)
    # Business-day index so it looks like a trading calendar.
    idx = pd.bdate_range(start=start, periods=n, tz="UTC")
    idx.name = "timestamp"

    daily_ret = rng.normal(0.0003, 0.012, size=n)      # ~ +0.03%/day, 1.2% vol
    close = 100.0 * np.exp(np.cumsum(daily_ret))

    # Build OHLC around the close path with a sane intrabar range.
    spread = np.abs(rng.normal(0.0, 0.004, size=n)) * close
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]                               # open ~ prior close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype("float64")

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class FakeProvider(DataProvider):
    """A DataProvider backed by a fixed in-memory frame — no network, no cache.

    This is the payoff of the interface design: strategy/indicator code that
    takes a ``DataProvider`` can be unit-tested by injecting this fake.
    """

    name = "fake"

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        df = self._frame
        if req.start is not None:
            df = df[df.index >= pd.Timestamp(req.start, tz="UTC")]
        if req.end is not None:
            df = df[df.index <= pd.Timestamp(req.end, tz="UTC")]
        return df.copy()


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """A single deterministic OHLCV frame reused across the suite."""
    return _synthetic_ohlcv()


@pytest.fixture(scope="session")
def close(ohlcv) -> pd.Series:
    return ohlcv["close"]


@pytest.fixture()
def provider(ohlcv) -> FakeProvider:
    return FakeProvider(ohlcv)
