"""
quantlab.data
=============

The data subpackage. Import the factory and ask for a provider **by name** so
that the rest of your code never imports a vendor SDK directly::

    from quantlab.data import get_provider

    provider = get_provider("yahoo")
    spy = provider.get_ohlcv("SPY", start="2020-01-01", timeframe="1d")

Switching data vendors is then a one-string change: ``get_provider("alpaca")``.


How to add ANY data provider (the promise of this course)
---------------------------------------------------------
Every example in QuantLab works with any vendor because adding one is small.
Implement :meth:`~quantlab.data.base.DataProvider._fetch_ohlcv`, register it in
``_PROVIDERS`` below, and you are done — caching, multi-symbol fetching,
validation, and the normalized schema all come from the base class.

Example: an Alpaca provider (sketch)::

    # quantlab/data/alpaca.py
    from .base import DataProvider, OHLCVRequest

    class AlpacaProvider(DataProvider):
        name = "alpaca"

        def __init__(self, key, secret):
            from alpaca.data.historical import StockHistoricalDataClient
            self.client = StockHistoricalDataClient(key, secret)

        def _fetch_ohlcv(self, req: OHLCVRequest):
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            tf = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour,
                  "1m": TimeFrame.Minute}[req.timeframe]
            bars = self.client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=req.symbol, timeframe=tf,
                start=req.start, end=req.end)).df
            # rename columns to open/high/low/close/volume and return;
            # the base class normalizes & validates the rest.
            return bars.rename(columns=str.lower)

Example: a crypto provider via CCXT (sketch)::

    class CCXTProvider(DataProvider):
        name = "ccxt"
        def __init__(self, exchange="binance"):
            import ccxt
            self.ex = getattr(ccxt, exchange)()
        def _fetch_ohlcv(self, req):
            import pandas as pd
            tf = {"1m":"1m","5m":"5m","1h":"1h","1d":"1d"}[req.timeframe]
            ohlcv = self.ex.fetch_ohlcv(req.symbol, timeframe=tf)
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high",
                                              "low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.set_index("timestamp")

Then add it to ``_PROVIDERS`` and call ``get_provider("alpaca")`` anywhere.
"""

from __future__ import annotations

from typing import Callable

from .base import (
    OHLCV_COLUMNS,
    VALID_TIMEFRAMES,
    DataProvider,
    OHLCVRequest,
    normalize_ohlcv,
    validate_ohlcv,
)
from .yahoo import YahooProvider

# Registry of known providers. Lambdas are used so that constructing a provider
# (and importing its heavy SDK) only happens when actually requested.
_PROVIDERS: dict[str, Callable[..., DataProvider]] = {
    "yahoo": YahooProvider,
    "yfinance": YahooProvider,  # friendly alias
}


def get_provider(name: str = "yahoo", **kwargs) -> DataProvider:
    """Return a configured :class:`DataProvider` by name.

    Parameters
    ----------
    name:
        Registered provider key, e.g. ``"yahoo"``. Case-insensitive.
    **kwargs:
        Passed through to the provider constructor (API keys, cache_dir, ...).
    """
    key = name.lower().strip()
    if key not in _PROVIDERS:
        raise KeyError(
            f"Unknown provider {name!r}. Registered: {sorted(_PROVIDERS)}. "
            "Add your own by implementing DataProvider and registering it in "
            "quantlab/data/__init__.py."
        )
    return _PROVIDERS[key](**kwargs)


def register_provider(name: str, factory: Callable[..., DataProvider]) -> None:
    """Register a custom provider at runtime (so learners can add their own
    without editing this file). ``factory`` is anything callable that returns a
    ``DataProvider`` — usually the class itself.
    """
    _PROVIDERS[name.lower().strip()] = factory


__all__ = [
    "get_provider",
    "register_provider",
    "DataProvider",
    "OHLCVRequest",
    "YahooProvider",
    "normalize_ohlcv",
    "validate_ohlcv",
    "OHLCV_COLUMNS",
    "VALID_TIMEFRAMES",
]
