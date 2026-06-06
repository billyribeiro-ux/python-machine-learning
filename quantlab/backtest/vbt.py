"""
quantlab.backtest.vbt
=====================

A thin bridge to **vectorbt** for large-scale parameter sweeps. Our own engine
(``quantlab.backtest.engine``) is perfect for understanding and for single
backtests; vectorbt's superpower is running *thousands* of parameter
combinations as vectorized array math and returning a stats grid — the
industrial version of "combine multiple indicators with custom settings and come
up with the stats".

vectorbt is optional; imported lazily.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from quantlab.indicators import rolling_zscore, rsi, sma


def sweep_sma_crossover(close: pd.Series,
                        fast_grid=range(10, 60, 10),
                        slow_grid=range(80, 220, 20),
                        fees: float = 0.0005) -> pd.DataFrame:
    """Backtest every fast/slow SMA crossover combo with vectorbt; return a stats
    table indexed by ``(fast, slow)`` with total return, Sharpe, and max drawdown.

    This shows the whole point of vectorbt: one call evaluates the entire grid.
    """
    try:
        import vectorbt as vbt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vectorbt not installed: pip install vectorbt") from exc

    rows = []
    for fast, slow in itertools.product(fast_grid, slow_grid):
        if fast >= slow:
            continue
        f = pd.Series(sma(close.to_numpy(), fast), index=close.index)
        s = pd.Series(sma(close.to_numpy(), slow), index=close.index)
        entries = (f > s) & (f.shift(1) <= s.shift(1))   # cross up
        exits = (f < s) & (f.shift(1) >= s.shift(1))     # cross down
        pf = vbt.Portfolio.from_signals(
            close, entries.fillna(False), exits.fillna(False), fees=fees, freq="1D"
        )
        rows.append({
            "fast": fast, "slow": slow,
            "total_return": float(pf.total_return()),
            "sharpe": float(pf.sharpe_ratio()),
            "max_drawdown": float(pf.max_drawdown()),
        })
    return pd.DataFrame(rows).set_index(["fast", "slow"]).sort_values(
        "sharpe", ascending=False
    )


def combo_sweep(close: pd.Series,
                rsi_floors=(35, 40, 45, 50),
                z_entries=(-0.5, 0.0, 0.5, 1.0),
                fast: int = 20, slow: int = 100,
                fees: float = 0.0005) -> pd.DataFrame:
    """Sweep the multi-indicator combo (RSI floor x z-entry) with vectorbt.

    Demonstrates combining *several* indicators with custom settings across a
    grid and collecting the stats — the capstone idea, at scale.
    """
    try:
        import vectorbt as vbt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vectorbt not installed: pip install vectorbt") from exc

    x = close.to_numpy()
    trend = pd.Series(sma(x, fast) > sma(x, slow), index=close.index)
    rsi_v = pd.Series(rsi(x, 14), index=close.index)
    z = pd.Series(rolling_zscore(x, 20), index=close.index)

    rows = []
    for rf, ze in itertools.product(rsi_floors, z_entries):
        long = trend & (rsi_v > rf) & (rsi_v < 75) & (z <= ze)
        entries = long & ~long.shift(1, fill_value=False)
        exits = ~long & long.shift(1, fill_value=False)
        pf = vbt.Portfolio.from_signals(
            close, entries.fillna(False), exits.fillna(False), fees=fees, freq="1D"
        )
        rows.append({
            "rsi_floor": rf, "z_entry": ze,
            "total_return": float(pf.total_return()),
            "sharpe": float(pf.sharpe_ratio()),
            "max_drawdown": float(pf.max_drawdown()),
        })
    return pd.DataFrame(rows).set_index(["rsi_floor", "z_entry"]).sort_values(
        "sharpe", ascending=False
    )
