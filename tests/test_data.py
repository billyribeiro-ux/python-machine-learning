"""
Tests for the data layer: the normalized contract is the spec, so we test it.

These tests prove that any provider feeding our normalization produces a frame
the rest of QuantLab can trust, and that the validator actually rejects bad data
(a validator that never says no is worthless).
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.data import OHLCV_COLUMNS, normalize_ohlcv, validate_ohlcv


def test_provider_returns_normalized_contract(provider):
    df = provider.get_ohlcv("TEST", start="2018-01-01", timeframe="1d")
    # The contract, asserted piece by piece.
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert (df.dtypes == "float64").all()


def test_multi_symbol_is_tidy(provider):
    panel = provider.get_ohlcv_multi(["AAA", "BBB"], start="2018-01-01")
    assert list(panel.columns) == ["symbol", *OHLCV_COLUMNS]
    assert set(panel["symbol"].unique()) == {"AAA", "BBB"}
    # Each symbol keeps the full history (no rows lost in concatenation).
    counts = panel.groupby("symbol").size()
    assert counts["AAA"] == counts["BBB"]


def test_normalize_maps_vendor_aliases():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.0, 2.0],
            "Adj Close": [1.5, 2.5, 3.5],   # alias -> close
            "Volume": [10, 20, 30],
        },
        index=idx,
    )
    out = normalize_ohlcv(raw, symbol="X")
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert out["close"].tolist() == [1.5, 2.5, 3.5]
    assert out.index.tz is not None


def test_validate_rejects_bad_high():
    # high < close is economically impossible; the validator must catch it.
    idx = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    idx.name = "timestamp"
    bad = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],      # high below close on row 2 -> invalid
            "low": [0.5, 0.5],
            "close": [0.9, 2.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    with pytest.raises(AssertionError):
        validate_ohlcv(bad, symbol="BAD")


def test_validate_rejects_unsorted_index():
    idx = pd.to_datetime(["2020-01-02", "2020-01-01"], utc=True)
    idx.name = "timestamp"
    df = pd.DataFrame(
        {c: [1.0, 1.0] for c in OHLCV_COLUMNS}, index=idx
    )
    # Make it economically valid but out of order.
    df["high"] = [2.0, 2.0]; df["low"] = [0.5, 0.5]
    with pytest.raises(AssertionError):
        validate_ohlcv(df)
