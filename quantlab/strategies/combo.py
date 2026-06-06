"""
quantlab.strategies.combo
=========================

A configurable strategy that **combines multiple indicators with custom
settings** into a single position signal, then hands it to the leak-free backtest
engine to produce stats. This is the executable version of the user's goal:
"combine multiple indicators with custom settings and come up with the stats."

The design generalizes to the indicator registry of Module 7 and the parameter
sweeps of Module 9, but it stands alone and is fully tested here.

The combo logic (a trend + momentum + mean-reversion blend):

* **Trend gate** — only go long when the fast SMA is above the slow SMA
  (don't fight the trend).
* **Momentum filter** — require RSI above a floor (avoid buying collapsing
  momentum) and below a ceiling (avoid chasing the overbought blow-off).
* **Mean-reversion entry** — within an uptrend, prefer entries when the price
  z-score has dipped (buy the pullback, not the extension).

Every threshold is a parameter with a sensible default, so you can sweep them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from quantlab.indicators import rsi, rolling_zscore, sma
from quantlab.backtest.engine import backtest_signal


@dataclass
class ComboParams:
    """All tunable settings for the combo strategy, in one typed object.

    Using a dataclass (not loose kwargs) means a parameter set is a value you
    can log, hash, store in an Optuna trial, or diff against another — exactly
    what you want when sweeping hundreds of configurations in Module 13.
    """

    fast: int = 20          # fast SMA window (trend)
    slow: int = 100         # slow SMA window (trend)
    rsi_window: int = 14
    rsi_floor: float = 40.0     # require RSI above this (momentum not broken)
    rsi_ceiling: float = 75.0   # but below this (not overbought)
    z_window: int = 20
    z_entry: float = 0.5        # enter when z-score <= this (a pullback)
    fee_bps: float = 1.0


def combo_signal(close: pd.Series, p: ComboParams) -> pd.Series:
    """Build the {0, 1} long/flat position signal from the indicator stack.

    Computed entirely from data up to each bar (no look-ahead); the engine
    applies the one-bar execution delay. Returns a float Series aligned to
    ``close``.
    """
    x = close.to_numpy()

    sma_fast = sma(x, p.fast)
    sma_slow = sma(x, p.slow)
    rsi_v = rsi(x, p.rsi_window)
    z = rolling_zscore(x, p.z_window)

    trend_ok = sma_fast > sma_slow
    mom_ok = (rsi_v > p.rsi_floor) & (rsi_v < p.rsi_ceiling)
    pullback = z <= p.z_entry

    long = trend_ok & mom_ok & pullback
    # NaNs in any indicator (warmup) must not count as a signal.
    sig = np.where(np.nan_to_num(long, nan=0.0).astype(bool), 1.0, 0.0)
    return pd.Series(sig, index=close.index, name="signal")


def run_combo(close: pd.Series, params: ComboParams | None = None,
              periods_per_year: float = 252) -> dict:
    """Run the combo end-to-end: indicators -> signal -> backtest -> stats.

    Returns the backtest result dict (``returns``, ``positions``, ``stats``)
    augmented with the ``params`` used, so a result is fully self-describing.
    """
    params = params or ComboParams()
    sig = combo_signal(close, params)
    result = backtest_signal(
        close, sig, fee_bps=params.fee_bps, periods_per_year=periods_per_year
    )
    result["params"] = asdict(params)
    return result
