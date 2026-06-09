"""
quantlab.data.yahoo
===================

A concrete :class:`~quantlab.data.base.DataProvider` backed by **Yahoo Finance**
via the ``yfinance`` library. This is the default provider for the entire
course because it is free and requires no API key — perfect for learning.

What this module demonstrates (and what *every* provider you write should do):

* Translate our small, explicit timeframe vocabulary ("1d", "5m", ...) into the
  vendor's native strings.
* Deal with the vendor's quirks **here**, in isolation, so they never leak:
  yfinance returns capitalized columns, sometimes a MultiIndex, an "Adj Close"
  column whose meaning depends on the ``auto_adjust`` flag, and a tz-naive or
  tz-localized index depending on the timeframe.
* Provide an on-disk cache so you are not hammering a free, rate-limited,
  unofficial endpoint every time you re-run a notebook. Re-running a cell should
  be instant and offline-friendly.

Production note for the learner
-------------------------------
Yahoo's endpoint is unofficial and can break or rate-limit without notice. For
anything beyond learning, implement ``AlpacaProvider`` / ``PolygonProvider`` /
``DatabentoProvider`` against the *same* base class (see the docstring in
``quantlab/data/__init__.py``). Because your strategies depend only on the
interface, that migration touches exactly one file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .base import OHLCV_COLUMNS, DataProvider, OHLCVRequest

def _utcnow() -> pd.Timestamp:
    """Current UTC time — a seam so tests can simulate the passage of days when
    verifying the cache-freshness behavior below."""
    return pd.Timestamp.now(tz="UTC")


# Map our vocabulary -> yfinance's `interval` strings. They mostly line up, but
# pinning the mapping explicitly means a change in yfinance can't silently
# change our behavior.
_TF_TO_YF: dict[str, str] = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "60m", "1h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo",
}


class YahooProvider(DataProvider):
    """Yahoo Finance provider with a transparent parquet cache.

    Parameters
    ----------
    cache_dir:
        Where to store cached bars as parquet. Pass ``None`` to disable caching
        (useful in tests that want to assert a network call happened).
    """

    name = "yahoo"

    def __init__(self, cache_dir: str | Path | None = "data/cache") -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    def _cache_path(self, req: OHLCVRequest) -> Path | None:
        if self.cache_dir is None:
            return None
        # A stable, collision-resistant filename derived from the request. We
        # hash the request fields so that two different requests can never map
        # to the same file, while the same request always maps to the same one.
        #
        # Freshness: a request with end=None means "through NOW", so its result
        # changes as time passes. If we keyed it on end=None alone, the first
        # fetch would be served forever — the paper-trading engine would read
        # yesterday's bars every day and never notice. We therefore fold a
        # freshness bucket into the key: the current UTC date for daily+ bars,
        # the current UTC hour for intraday. Old buckets simply stop being read
        # (delete data/cache to reclaim space).
        end_key = req.end
        if end_key is None:
            now = _utcnow()
            intraday = req.timeframe not in ("1d", "1wk", "1mo")
            end_key = now.strftime("%Y-%m-%dT%H" if intraday else "%Y-%m-%d")
        key = f"{req.symbol}|{req.start}|{end_key}|{req.timeframe}|{req.adjust}"
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        safe_sym = req.symbol.replace("/", "_").replace("=", "_")
        return self.cache_dir / f"{safe_sym}_{req.timeframe}_{digest}.parquet"

    # --------------------------------------------------------------- fetching
    def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        # 1) Serve from cache if we have it. The base class still normalizes &
        #    validates the result, so a cached frame is held to the same spec.
        path = self._cache_path(req)
        if path is not None and path.exists():
            return pd.read_parquet(path)

        # 2) Import yfinance lazily so simply importing quantlab does not require
        #    the dependency (handy for tests that use only cached data).
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "yfinance is required for YahooProvider. "
                "Install it with: pip install yfinance"
            ) from exc

        interval = _TF_TO_YF[req.timeframe]
        # auto_adjust=req.adjust gives split/dividend-adjusted OHLC when True.
        # We pass group_by='column' and a single ticker to keep the frame flat.
        raw = yf.download(
            tickers=req.symbol,
            start=req.start,
            end=req.end,
            interval=interval,
            auto_adjust=req.adjust,
            progress=False,
            actions=False,
        )
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"Yahoo returned no data for {req.symbol!r} "
                f"({req.timeframe}, {req.start}..{req.end}). "
                "Check the symbol, the date range, and that intraday data is "
                "only available for the last ~60 days on Yahoo."
            )

        df = self._flatten_yf(raw)

        # 3) Write through to cache (store the tidy, pre-normalization frame).
        if path is not None:
            df.to_parquet(path)
        return df

    @staticmethod
    def _flatten_yf(raw: pd.DataFrame) -> pd.DataFrame:
        """Turn yfinance's frame into the columns our base class expects.

        yfinance sometimes returns a column MultiIndex like ``(Open, AAPL)``
        even for a single ticker. We collapse that, lowercase names, and keep
        only OHLCV. The base class handles index tz-awareness and validation.
        """
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            # For a single ticker, the second level is the symbol; drop it.
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        # auto_adjust=True already folds adjustment into close, so an extra
        # "adj close" may or may not be present; base.normalize handles aliases.
        keep = [c for c in OHLCV_COLUMNS if c in df.columns]
        df = df[keep]
        df.index.name = "timestamp"
        return df

    # ---------------------------------------------------------------- options
    def get_options_chain(
        self, symbol: str, expiry: str | None = None
    ) -> pd.DataFrame:
        """Return a tidy options chain (calls + puts) for ``symbol``.

        Used in Module 15. yfinance exposes expiries via ``Ticker.options`` and
        a chain per expiry via ``Ticker.option_chain(expiry)``. We tag each row
        with ``option_type`` and ``expiry`` and concatenate, so the result is a
        single tidy frame you can filter, scan, and compute greeks on.
        """
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("yfinance is required for options data.") from exc

        tk = yf.Ticker(symbol)
        expiries = tk.options
        if not expiries:
            raise ValueError(f"No option expiries available for {symbol!r}.")
        expiry = expiry or expiries[0]
        if expiry not in expiries:
            raise ValueError(
                f"Expiry {expiry!r} not available for {symbol!r}. "
                f"Available: {list(expiries)}"
            )

        chain = tk.option_chain(expiry)
        calls = chain.calls.assign(option_type="call", expiry=expiry)
        puts = chain.puts.assign(option_type="put", expiry=expiry)
        out = pd.concat([calls, puts], ignore_index=True)
        out.columns = [str(c).strip().lower() for c in out.columns]
        return out
