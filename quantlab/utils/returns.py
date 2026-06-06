"""
quantlab.utils.returns
======================

Numerically careful return calculations. These are three-line functions, so why
a whole module with docstrings? Because *almost every bug in a backtest traces
back to returns* — wrong base, off-by-one alignment (look-ahead!), simple vs log
confusion, or silent NaN propagation. Getting these right once, in one tested
place, removes an entire category of errors from everything downstream.

Definitions (so we never argue about them again)
------------------------------------------------
* **Simple return** at time *t*:  r_t = P_t / P_{t-1} - 1.
* **Log return**     at time *t*:  l_t = ln(P_t / P_{t-1}) = ln P_t - ln P_{t-1}.

Why log returns matter to a quant:
* They are **time-additive**: the log return over N bars is the *sum* of the
  per-bar log returns, which makes resampling and aggregation trivial.
* They are closer to normally distributed, which many models assume.
* Simple returns are **asset-additive** (a portfolio's simple return is the
  weighted sum of constituents' simple returns) — so you convert between the two
  depending on whether you are aggregating across *time* or across *assets*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trading periods per year, by timeframe. Used to annualize Sharpe etc. These
# are conventions, not laws — document the choice rather than hard-coding 252
# in fifty places.
PERIODS_PER_YEAR: dict[str, float] = {
    "1d": 252, "1wk": 52, "1mo": 12,
    "1h": 252 * 6.5, "60m": 252 * 6.5,
    "30m": 252 * 13, "15m": 252 * 26, "5m": 252 * 78, "1m": 252 * 390,
}


def simple_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Simple percentage returns. ``periods`` lets you compute multi-bar
    returns (e.g. 5-day forward/backward) without writing the arithmetic each
    time. The first ``periods`` values are NaN by construction — that is honest,
    not a bug; there is no prior price to compare against."""
    return prices.pct_change(periods=periods)


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns, computed as a difference of logs (more numerically stable
    than ``log(p_t/p_{t-1})`` for very small moves, and faster)."""
    return np.log(prices).diff(periods)


def forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """The return realized **over the next ``horizon`` bars**, aligned to the
    bar where you would have made the decision.

    This is the single most leak-prone calculation in all of quant ML, so it
    gets its own function with a loud docstring. A label of "what happens next"
    must be attached to the bar *before* it happens. We compute the forward
    return and then shift it back so row *t* holds the return from *t* to
    *t+horizon* — usable as a supervised-learning target at time *t* without
    peeking. Module 11 relies on this.
    """
    return prices.pct_change(periods=horizon).shift(-horizon)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compound a series of simple returns into an equity curve starting at 0
    (i.e. +0.10 means +10% since inception). NaNs are treated as 0 so a warmup
    period does not nuke the whole curve."""
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def annualization_factor(timeframe: str) -> float:
    """Periods-per-year for a timeframe, defaulting to 252 (daily) if unknown.
    Centralizing this is what lets the stats engine annualize correctly whether
    you backtest daily swings or 5-minute scalps."""
    return PERIODS_PER_YEAR.get(timeframe, 252.0)
