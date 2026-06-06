"""
quantlab.data.base
==================

The *provider-agnostic* data layer. This is the single most important design
decision in the whole course: **every example in QuantLab pulls market data
through the ``DataProvider`` interface, never through a vendor SDK directly.**

Why does a principal engineer insist on this?

1. **Swappability.** Yahoo Finance is free and great for learning, but it is
   rate-limited, unofficial, and occasionally wrong. The day you move to
   Alpaca, Polygon, IBKR, Databento, or CCXT (crypto), you implement *one*
   class and every notebook, scanner, backtest, and ML pipeline you ever wrote
   keeps working unchanged. Vendor lock-in is a tax you pay forever; an
   interface is a one-time cost.

2. **A normalized schema.** Different vendors disagree about column names
   (``Adj Close`` vs ``adjusted_close``), index types (naive vs tz-aware),
   timestamp conventions (bar-open vs bar-close), and adjustment policy
   (split/dividend adjusted or raw). If you let those differences leak into
   your strategy code, every strategy silently depends on a vendor quirk. We
   pin the schema *here*, once, so the rest of the codebase is clean.

3. **Testability.** Because strategies depend on an *interface*, tests can
   inject a fake/cached provider and run offline, deterministically, in
   milliseconds. (See ``tests/conftest.py``.)

The Normalized OHLCV Contract
-----------------------------
Every provider MUST return a ``pandas.DataFrame`` with:

* A ``DatetimeIndex`` named ``"timestamp"``, **timezone-aware in UTC**, sorted
  ascending, with no duplicate timestamps.
* Lowercase float columns exactly: ``open, high, low, close, volume``.
* ``close`` is **split- and dividend-adjusted** by default (``adjust=True``),
  so returns computed from ``close`` are total returns. Set ``adjust=False``
  for raw prices (what you'd see on a broker ladder).
* The timestamp marks the bar's **open** time (a 2024-01-02 daily bar covers
  the 2024-01-02 session). This matters enormously for look-ahead bias: a bar
  is only *complete* — and therefore only *tradeable* — at the next timestamp.

Anything that does not meet this contract is a bug in the provider, not in the
code that consumes it. ``validate_ohlcv`` below is the executable spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# The canonical column order. Import this everywhere instead of retyping the
# strings; a single source of truth is how you avoid "open"/"Open" bugs.
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Timeframes we promise to understand. Providers map these to their own native
# strings. Keeping our vocabulary small and explicit beats passing vendor
# strings around (is it "1d", "1day", "D", or "daily"? — here it is "1d").
VALID_TIMEFRAMES: frozenset[str] = frozenset(
    {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo"}
)


@dataclass(frozen=True)
class OHLCVRequest:
    """A value object describing a data request.

    Using a frozen dataclass (rather than a pile of positional arguments) makes
    requests hashable — which means they are trivially cacheable as dict keys —
    and self-documenting in logs. This is a small thing that pays off the moment
    you build a cache or a request queue.
    """

    symbol: str
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    timeframe: str = "1d"
    adjust: bool = True


class DataProvider(ABC):
    """Abstract base class every concrete data source implements.

    To add a new provider you implement exactly two methods —
    :meth:`_fetch_ohlcv` (one symbol) and optionally :meth:`get_options_chain`
    — and you get caching, multi-symbol fetching, validation, and schema
    normalization for free from this base class. That is the whole point: the
    *hard, reusable* logic lives here once.

    A minimal new provider looks like::

        class MyBrokerProvider(DataProvider):
            name = "mybroker"

            def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
                raw = mybroker_sdk.bars(req.symbol, req.timeframe, ...)
                # rename/convert into the normalized contract, then:
                return raw  # base class validates & normalizes for you
    """

    #: Short, stable identifier used by the :func:`quantlab.data.get_provider`
    #: factory (e.g. "yahoo", "alpaca"). Subclasses must override.
    name: str = "abstract"

    # ----- the one method subclasses MUST implement -------------------------
    @abstractmethod
    def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        """Fetch raw bars for a single symbol and return them *close to* the
        normalized contract. The base class will still validate and tidy the
        result, so you do not need to be perfect — but the closer the better.
        """
        raise NotImplementedError

    # ----- the public API the rest of QuantLab actually calls ---------------
    def get_ohlcv(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
        adjust: bool = True,
    ) -> pd.DataFrame:
        """Return normalized, validated OHLCV for one symbol.

        This is the method you will call thousands of times across the course.
        It delegates the vendor-specific part to :meth:`_fetch_ohlcv` and then
        enforces the contract, so callers can *trust* the shape of what they
        get back.
        """
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Unknown timeframe {timeframe!r}. "
                f"Valid: {sorted(VALID_TIMEFRAMES)}"
            )
        req = OHLCVRequest(symbol, start, end, timeframe, adjust)
        raw = self._fetch_ohlcv(req)
        return normalize_ohlcv(raw, symbol=symbol)

    def get_ohlcv_multi(
        self,
        symbols: Iterable[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
        adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch several symbols and return a **long/tidy** DataFrame.

        The result has columns ``[symbol, open, high, low, close, volume]`` and
        a ``timestamp`` index. Long format (rather than a wide multi-index) is
        what Polars, DuckDB, and scikit-learn all want, and it scales to
        thousands of symbols without an explosion of NaNs. Module 3 (Polars)
        and Module 4 (DuckDB) lean on this shape heavily.
        """
        frames = []
        for sym in symbols:
            df = self.get_ohlcv(sym, start, end, timeframe, adjust)
            df = df.assign(symbol=sym)
            frames.append(df)
        if not frames:
            return pd.DataFrame(
                columns=["symbol", *OHLCV_COLUMNS]
            ).rename_axis("timestamp")
        out = pd.concat(frames).reset_index().set_index("timestamp")
        # Put symbol first for readability; keep canonical OHLCV order after it.
        return out[["symbol", *OHLCV_COLUMNS]].sort_values(
            ["symbol", "timestamp"]
        )

    def get_options_chain(
        self, symbol: str, expiry: str | None = None
    ) -> pd.DataFrame:
        """Return the options chain for ``symbol`` (Module 15).

        Optional: providers that have no options data raise
        :class:`NotImplementedError`, which is the honest signal that you need a
        different provider for options rather than a silent empty frame.
        """
        raise NotImplementedError(
            f"{self.name!r} provider does not support options chains."
        )


# --------------------------------------------------------------------------- #
# Normalization & validation: the executable specification of our contract.
# --------------------------------------------------------------------------- #
def normalize_ohlcv(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the normalized contract.

    This function is deliberately forgiving on input (vendors are messy) and
    strict on output (downstream code must be able to trust it). It:

    * lowercases column names and maps common aliases,
    * builds a tz-aware UTC ``DatetimeIndex`` named ``timestamp``,
    * drops duplicate timestamps (keeping the last), sorts ascending,
    * casts OHLCV to float, and
    * runs :func:`validate_ohlcv` as a final assertion.
    """
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for symbol {symbol!r}.")

    df = df.copy()

    # 1) Normalize column names. Map the usual vendor aliases to our schema.
    alias = {
        "adj close": "close",
        "adj_close": "close",
        "adjclose": "close",
        "vol": "volume",
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
    }
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=alias)

    # 2) Establish the timestamp index.
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"

    # 3) Keep only the canonical columns (some vendors add extras).
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{symbol!r}: missing required columns {missing}. "
            f"Got {list(df.columns)}."
        )
    df = df[list(OHLCV_COLUMNS)].astype("float64")

    # 4) De-duplicate and sort. Real feeds occasionally send a bar twice.
    df = df[~df.index.duplicated(keep="last")].sort_index()

    validate_ohlcv(df, symbol=symbol)
    return df


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> None:
    """Assert the normalized contract. Raises ``AssertionError`` on violation.

    Treat this as the spec. If you ever wonder "what exactly does normalized
    mean?", read these assertions — they cannot drift out of date because the
    test suite runs them.
    """
    ctx = f"[{symbol}] " if symbol else ""
    assert isinstance(df.index, pd.DatetimeIndex), f"{ctx}index must be DatetimeIndex"
    assert df.index.tz is not None, f"{ctx}index must be tz-aware (UTC)"
    assert df.index.name == "timestamp", f"{ctx}index must be named 'timestamp'"
    assert df.index.is_monotonic_increasing, f"{ctx}index must be sorted ascending"
    assert not df.index.has_duplicates, f"{ctx}index must have no duplicates"
    assert list(df.columns) == list(OHLCV_COLUMNS), (
        f"{ctx}columns must be exactly {OHLCV_COLUMNS}, got {list(df.columns)}"
    )
    # Basic economic sanity: high is the max, low is the min of the bar.
    eps = 1e-9
    assert (df["high"] + eps >= df[["open", "close", "low"]].max(axis=1)).all(), (
        f"{ctx}high must be >= open/close/low"
    )
    assert (df["low"] - eps <= df[["open", "close", "high"]].min(axis=1)).all(), (
        f"{ctx}low must be <= open/close/high"
    )
    assert (df["volume"] >= 0).all(), f"{ctx}volume must be non-negative"
