"""
quantlab.backtest.engine
========================

A tiny, **leak-free** vectorized backtest engine. Module 9 (vectorbt) scales
this to parameter sweeps over thousands of combinations; this module exists so
you understand exactly what a backtest *is* before a library hides it from you.

The core idea in one sentence: a backtest converts a **signal** (what you
believe, computed from data up to bar *t*) into a **position** (what you hold,
which can only change on bar *t+1*), then earns the asset's return while
positioned, minus the cost of changing position.

Everything here honors the golden rule from Module 0: **decide on *t*, act on
*t+1*.** The engine applies the one-bar shift for you, so a correct backtest is
the default and look-ahead is something you'd have to go out of your way to add.
"""

from __future__ import annotations

import pandas as pd

from .stats import compute_stats


def backtest_signal(
    prices: pd.Series,
    signal: pd.Series,
    fee_bps: float = 1.0,
    periods_per_year: float = 252,
) -> dict:
    """Backtest a position signal on a single instrument.

    Parameters
    ----------
    prices:
        Close prices (already adjusted), indexed by timestamp.
    signal:
        Desired position per bar, computed from information available *at that
        bar's close*. Typically in {-1, 0, +1} (short/flat/long) but any float
        weight works. We shift it by one bar internally so you cannot peek.
    fee_bps:
        Round-trip-agnostic transaction cost in basis points, charged on the
        *change* in position (turnover). 1 bp = 0.01%.
    periods_per_year:
        For annualizing the stats (252 daily, 52 weekly, ~98k for 5-minute...).

    Returns
    -------
    dict with the strategy ``returns`` series, the ``positions`` actually held,
    and a ``stats`` dict (Sharpe, max drawdown, CAGR, ...).
    """
    prices = prices.astype("float64")
    asset_ret = prices.pct_change()

    # THE leak guard: the position you hold on bar t was decided on bar t-1.
    position = signal.reindex(prices.index).fillna(0.0).shift(1).fillna(0.0)

    # Turnover = how much the position changed; costs scale with it.
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (fee_bps / 10_000.0)

    strat_ret = position * asset_ret - cost
    strat_ret = strat_ret.fillna(0.0)

    return {
        "returns": strat_ret,
        "positions": position,
        "turnover": float(turnover.sum()),
        "stats": compute_stats(strat_ret, periods_per_year=periods_per_year),
    }
