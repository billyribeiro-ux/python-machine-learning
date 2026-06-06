"""
quantlab.backtest.stats
=======================

Performance statistics computed from a **returns series**. Module 10 swaps in
QuantStats for gorgeous tearsheets, but you should never use a metric you can't
compute yourself — so here are the core ones, defined precisely and tested.

Every metric takes a pandas Series of *per-period simple returns* (the strategy
already had costs applied, if any) and a ``periods_per_year`` for annualization.
Keeping annualization explicit means the same code is correct for daily swings
(252) or 5-minute scalps (~98,000) — you just pass the right number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compound returns into a growth-of-$1 curve (starts at 1.0)."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def total_return(returns: pd.Series) -> float:
    return float(equity_curve(returns).iloc[-1] - 1.0)


def cagr(returns: pd.Series, periods_per_year: float = 252) -> float:
    """Compound Annual Growth Rate. The geometric average yearly return — the
    honest 'what did I actually compound at' number, unlike a naive mean."""
    eq = equity_curve(returns)
    n = len(returns.dropna())
    if n == 0:
        return 0.0
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def ann_volatility(returns: pd.Series, periods_per_year: float = 252) -> float:
    """Annualized standard deviation of returns (volatility scales with sqrt
    of time under the i.i.d. assumption)."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: float = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio: excess return per unit of total volatility.

    The single most cited risk-adjusted metric. ``rf`` is the per-period
    risk-free rate (0 is a fine default for short horizons). Returns 0 when
    volatility is 0 to avoid a division blowup."""
    excess = returns.dropna() - rf
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float = 252, rf: float = 0.0) -> float:
    """Like Sharpe, but penalizes only *downside* volatility — because upside
    'risk' isn't risk. Uses downside deviation in the denominator."""
    excess = returns.dropna() - rf
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """The drawdown at each point: how far below the running peak the equity
    curve is (always <= 0). The shape of pain over time."""
    eq = equity_curve(returns)
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """The worst peak-to-trough decline (a negative number). The number that
    tells you whether you'd actually have stuck with the strategy."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def calmar(returns: pd.Series, periods_per_year: float = 252) -> float:
    """CAGR divided by the absolute max drawdown — return per unit of worst-case
    pain. A favorite of trend followers."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / mdd


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return (of the periods in a position,
    i.e. nonzero return)."""
    r = returns.dropna()
    active = r[r != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def compute_stats(returns: pd.Series, periods_per_year: float = 252) -> dict:
    """Return the full metric bundle as a dict — the 'come up with the stats'
    deliverable. One call, every headline number, all from the same returns
    series so they are mutually consistent."""
    r = returns.fillna(0.0)
    return {
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year),
        "ann_volatility": ann_volatility(r, periods_per_year),
        "sharpe": sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r, periods_per_year),
        "win_rate": win_rate(r),
        "n_periods": int(r.shape[0]),
    }
