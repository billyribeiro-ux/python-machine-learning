"""
quantlab.scanners.scanner
=========================

A multi-symbol **scanner**: rank and filter a universe for trade setups. This is
where the data layer, the indicator library, and cross-sectional ranking come
together into the tool a trader actually opens every morning.

Design: pull a tidy multi-symbol panel through the provider, compute per-symbol
indicators (grouped so there's no cross-contamination — the Module 2/3 rule),
take the latest row per symbol, then apply filters and a ranking score. The same
scanner serves swing setups (daily bars) and day-trade setups (intraday bars) —
you just change the timeframe.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantlab.indicators import atr, rolling_zscore, rsi, sma


def _per_symbol_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns per symbol on a tidy panel (no leakage)."""
    out = []
    for sym, g in panel.groupby("symbol", group_keys=False):
        g = g.sort_index().copy()
        c = g["close"].to_numpy()
        g["sma20"] = sma(c, 20)
        g["sma50"] = sma(c, 50)
        g["sma200"] = sma(c, 200)
        g["rsi14"] = rsi(c, 14)
        g["z20"] = rolling_zscore(c, 20)
        g["atr14"] = atr(g["high"].to_numpy(), g["low"].to_numpy(), c, 14)
        g["mom20"] = g["close"].pct_change(20)
        g["mom60"] = g["close"].pct_change(60)
        g["dollar_vol"] = (g["close"] * g["volume"]).rolling(20).mean()
        out.append(g)
    return pd.concat(out)


def latest_snapshot(provider, symbols, start="2022-01-01", timeframe="1d") -> pd.DataFrame:
    """Return a one-row-per-symbol snapshot with indicators, ready to filter/rank."""
    panel = provider.get_ohlcv_multi(symbols, start=start, timeframe=timeframe)
    feats = _per_symbol_features(panel)
    snap = feats.groupby("symbol").tail(1).reset_index().set_index("symbol")
    return snap


# --------------------------------------------------------------------------- #
# Reusable, named setups. Each takes the snapshot and returns a boolean mask.
# Compose them to build a screen.
# --------------------------------------------------------------------------- #
def swing_pullback(snap: pd.DataFrame, min_dollar_vol: float = 5e7) -> pd.Series:
    """Uptrend pullback: above the 200-day, fast above slow, RSI cooled into the
    40-55 zone (a dip, not a breakdown), and liquid enough to trade."""
    return (
        (snap["close"] > snap["sma200"])
        & (snap["sma20"] > snap["sma50"])
        & (snap["rsi14"].between(40, 55))
        & (snap["dollar_vol"] > min_dollar_vol)
    )


def momentum_breakout(snap: pd.DataFrame, min_dollar_vol: float = 5e7) -> pd.Series:
    """Strong momentum, stretched above the mean (z>1), still trending up."""
    return (
        (snap["sma20"] > snap["sma50"])
        & (snap["z20"] > 1.0)
        & (snap["mom60"] > 0.10)
        & (snap["dollar_vol"] > min_dollar_vol)
    )


def scan(provider, symbols, setup: Callable[[pd.DataFrame], pd.Series] = swing_pullback,
         rank_by: str = "mom20", ascending: bool = False,
         start="2022-01-01", timeframe="1d", top: int | None = None) -> pd.DataFrame:
    """Run a scan end-to-end: snapshot -> filter by ``setup`` -> rank by ``rank_by``.

    Returns the matching symbols sorted by the ranking column. ``top`` caps the
    result to the N best candidates.
    """
    snap = latest_snapshot(provider, symbols, start=start, timeframe=timeframe)
    hits = snap[setup(snap)].copy()
    cols = ["close", "rsi14", "z20", "mom20", "mom60", "atr14", "dollar_vol"]
    hits = hits[[c for c in cols if c in hits.columns]]
    hits = hits.sort_values(rank_by, ascending=ascending)
    return hits.head(top) if top else hits
