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

Two families of scanner live here:

* **Snapshot scanners** (``swing_pullback``, ``momentum_breakout``,
  ``relative_strength``, ``gap_up``, ``new_high_breakout``) — each is a function
  ``setup(snapshot) -> bool mask`` over the one-row-per-symbol snapshot built by
  :func:`latest_snapshot`. Combine them, rank them, run them with :func:`scan`.
* **Intraday session scanner** (``opening_range_breakout`` via
  :func:`orb_scan`) — needs intra-session structure, so it has its own snapshot
  builder :func:`intraday_orb_features`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantlab.indicators import atr, rolling_zscore, rsi, sma

BENCHMARK_DEFAULT = "SPY"


def _per_symbol_features(panel: pd.DataFrame,
                         bench_close: pd.Series | None = None) -> pd.DataFrame:
    """Add indicator columns per symbol on a tidy panel (no leakage).

    When ``bench_close`` (a benchmark close series, e.g. SPY) is supplied, also
    computes relative-strength columns per symbol. All computations are
    backward-looking and grouped by symbol, so one symbol never bleeds into
    another.
    """
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
        # Overnight gap: today's open vs yesterday's close.
        g["gap"] = g["open"] / g["close"].shift(1) - 1.0
        # Distance below the trailing 52-week high (0 == at a new high).
        roll_hi = g["close"].rolling(252, min_periods=20).max()
        g["pct_from_high"] = g["close"] / roll_hi - 1.0

        if bench_close is not None:
            b = bench_close.reindex(g.index).ffill()
            rs = g["close"] / b                       # relative-strength line
            g["rs"] = rs
            g["rs_mom60"] = rs.pct_change(60)
            rs_hi = rs.rolling(252, min_periods=20).max()
            g["rs_pct_from_high"] = rs / rs_hi - 1.0  # 0 == new RS high vs bench

        out.append(g)
    return pd.concat(out)


def latest_snapshot(provider, symbols, start="2022-01-01", timeframe="1d",
                    benchmark: str | None = BENCHMARK_DEFAULT) -> pd.DataFrame:
    """Return a one-row-per-symbol snapshot with indicators, ready to filter/rank.

    If ``benchmark`` is given, relative-strength columns are added (vs that
    symbol). The benchmark is taken from the panel if present, else fetched via
    the provider; if it can't be obtained, RS columns are simply omitted (and
    RS-based setups return no hits rather than crashing).
    """
    panel = provider.get_ohlcv_multi(symbols, start=start, timeframe=timeframe)

    bench_close = None
    if benchmark:
        try:
            present = set(panel["symbol"].unique())
            if benchmark in present:
                bench_close = panel[panel["symbol"] == benchmark]["close"]
            else:
                bench_close = provider.get_ohlcv(benchmark, start=start,
                                                 timeframe=timeframe)["close"]
            bench_close = bench_close[~bench_close.index.duplicated(keep="last")]
        except Exception:
            bench_close = None

    feats = _per_symbol_features(panel, bench_close=bench_close)
    snap = feats.groupby("symbol").tail(1).reset_index().set_index("symbol")
    return snap


# --------------------------------------------------------------------------- #
# Snapshot setups. Each takes the snapshot and returns a boolean mask.
# Comparisons against NaN yield False, so warmup/missing values never match.
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


def relative_strength(snap: pd.DataFrame, min_dollar_vol: float = 5e7,
                      near_high: float = 0.02) -> pd.Series:
    """Leadership: relative-strength line vs the benchmark at/near a new high,
    with positive RS momentum, in an uptrend. These are the names beating the
    market — the first place a swing trader looks."""
    if "rs_pct_from_high" not in snap.columns:
        return pd.Series(False, index=snap.index)
    return (
        (snap["rs_pct_from_high"] >= -abs(near_high))   # within X% of RS high
        & (snap["rs_mom60"] > 0)                         # RS still improving
        & (snap["sma20"] > snap["sma50"])               # price uptrend too
        & (snap["dollar_vol"] > min_dollar_vol)
    ).fillna(False)


def gap_up(snap: pd.DataFrame, min_gap: float = 0.02,
           min_dollar_vol: float = 5e7) -> pd.Series:
    """Gap-up: opened at least ``min_gap`` above the prior close, while above the
    50-day (a continuation gap, not a dead-cat bounce), and liquid. The bread and
    butter of a gap-and-go day-trade scan."""
    return (
        (snap["gap"] > min_gap)
        & (snap["close"] > snap["sma50"])
        & (snap["dollar_vol"] > min_dollar_vol)
    ).fillna(False)


def new_high_breakout(snap: pd.DataFrame, min_dollar_vol: float = 5e7,
                      tol: float = 0.005) -> pd.Series:
    """52-week-high breakout: price at (within ``tol`` of) its trailing 252-day
    high, with positive medium-term momentum. The classic Darvas/Minervini
    breakout screen."""
    return (
        (snap["pct_from_high"] >= -abs(tol))
        & (snap["mom60"] > 0.05)
        & (snap["dollar_vol"] > min_dollar_vol)
    ).fillna(False)


def scan(provider, symbols, setup: Callable[[pd.DataFrame], pd.Series] = swing_pullback,
         rank_by: str = "mom20", ascending: bool = False,
         start="2022-01-01", timeframe="1d", benchmark: str | None = BENCHMARK_DEFAULT,
         top: int | None = None) -> pd.DataFrame:
    """Run a snapshot scan end-to-end: snapshot -> filter by ``setup`` -> rank.

    Returns the matching symbols sorted by ``rank_by``. ``top`` caps the result.
    """
    snap = latest_snapshot(provider, symbols, start=start, timeframe=timeframe,
                           benchmark=benchmark)
    hits = snap[setup(snap)].copy()
    cols = ["close", "rsi14", "z20", "mom20", "mom60", "gap", "pct_from_high",
            "rs", "rs_mom60", "rs_pct_from_high", "atr14", "dollar_vol"]
    hits = hits[[c for c in cols if c in hits.columns]]
    if rank_by in hits.columns:
        hits = hits.sort_values(rank_by, ascending=ascending)
    return hits.head(top) if top else hits


# --------------------------------------------------------------------------- #
# Intraday opening-range breakout (its own snapshot — needs intra-session bars).
# --------------------------------------------------------------------------- #
def intraday_orb_features(panel: pd.DataFrame, open_bars: int = 6) -> pd.DataFrame:
    """Per-symbol opening-range-breakout features for the latest session.

    The opening range is the high/low of the first ``open_bars`` bars of the most
    recent session (e.g. 6 x 5-minute bars = the first 30 minutes). ``orb_up`` is
    True if price later traded above that opening-range high — a long breakout.
    Returns one row per symbol.
    """
    rows = []
    for sym, g in panel.groupby("symbol", group_keys=False):
        g = g.sort_index()
        sessions = g.index.normalize()                 # date part (tz-aware)
        last = sessions.max()
        day = g[sessions == last]
        if len(day) <= open_bars:
            rows.append({"symbol": sym, "or_high": np.nan, "or_low": np.nan,
                         "last": np.nan, "orb_up": False,
                         "session_dollar_vol": np.nan})
            continue
        opening = day.iloc[:open_bars]
        after = day.iloc[open_bars:]
        or_high = float(opening["high"].max())
        or_low = float(opening["low"].min())
        rows.append({
            "symbol": sym,
            "or_high": or_high,
            "or_low": or_low,
            "last": float(day["close"].iloc[-1]),
            "orb_up": bool(after["high"].max() > or_high),
            "session_dollar_vol": float((day["close"] * day["volume"]).sum()),
        })
    return pd.DataFrame(rows).set_index("symbol")


def opening_range_breakout(snap: pd.DataFrame,
                           min_session_dollar_vol: float = 0.0) -> pd.Series:
    """Setup mask over an ORB snapshot: broke above the opening range, and
    traded enough during the session to be real."""
    return (snap["orb_up"]
            & (snap["session_dollar_vol"] > min_session_dollar_vol)).fillna(False)


def orb_scan(provider, symbols, start=None, timeframe="5m", open_bars: int = 6,
             min_session_dollar_vol: float = 0.0, top: int | None = None) -> pd.DataFrame:
    """Run the opening-range-breakout scan on intraday bars.

    Returns symbols that broke above their opening range in the latest session,
    ranked by session dollar volume (most active first).
    """
    panel = provider.get_ohlcv_multi(symbols, start=start, timeframe=timeframe)
    snap = intraday_orb_features(panel, open_bars=open_bars)
    hits = snap[opening_range_breakout(snap, min_session_dollar_vol)].copy()
    hits = hits.sort_values("session_dollar_vol", ascending=False)
    return hits.head(top) if top else hits
