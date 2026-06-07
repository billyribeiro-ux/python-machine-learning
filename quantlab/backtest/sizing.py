"""
quantlab.backtest.sizing
========================

Turn a *direction* into a *size*. A signal tells you which way to bet; sizing
decides how much — and it is where most of the risk-adjusted return actually
comes from. Four composable, **causal** (no look-ahead) tools:

* :func:`bet_size` — size from a predicted probability of being right
  (linear ``2p-1`` or the López de Prado normal-CDF bet size).
* :func:`kelly_size` — the growth-optimal fraction for a known edge/payoff,
  used fractionally (half-Kelly) because full Kelly is brutally volatile.
* :func:`vol_target_scalar` — scale exposure so realized volatility tracks a
  target; this alone flattens the wild equity swings of a raw signal.
* :func:`drawdown_throttle` — cut exposure after the strategy breaches a max
  drawdown, restoring it once recovered. A simple, brutal capital preserver.

Each returns a per-bar Series you multiply into a base position; the backtest
engine then applies its usual one-bar execution delay, so nothing here peeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def bet_size(proba: pd.Series, method: str = "linear", cap: float = 1.0) -> pd.Series:
    """Map P(bet is correct) in [0,1] to a size in [0, cap].

    ``"linear"``  : ``clip(2·(p-0.5), 0, cap)`` — simple and robust.
    ``"normal"``  : standardize the edge and pass through the normal CDF
                    (López de Prado, AFML Ch. 10) — smoother, concentrates size
                    only when the probability is decisively away from 0.5.
    Probabilities at/below 0.5 give zero size (don't take a coin-flip bet).
    """
    p = pd.Series(proba).astype("float64").clip(1e-6, 1 - 1e-6)
    if method == "linear":
        size = 2.0 * (p - 0.5)
    elif method == "normal":
        z = (p - 0.5) / np.sqrt(p * (1.0 - p))
        size = 2.0 * pd.Series(norm.cdf(z), index=p.index) - 1.0
    else:
        raise ValueError("method must be 'linear' or 'normal'")
    return size.clip(0.0, cap)


def kelly_fraction(p: float, payoff_ratio: float) -> float:
    """Kelly fraction for a binary bet: ``f* = p - (1-p)/b`` where ``b`` is the
    win/loss payoff ratio. Negative means *no edge* (don't bet)."""
    if payoff_ratio <= 0:
        return 0.0
    return float(p - (1.0 - p) / payoff_ratio)


def kelly_size(proba: pd.Series, payoff_ratio: float = 1.0,
               fraction: float = 0.5, cap: float = 1.0) -> pd.Series:
    """Fractional-Kelly size from a probability series. ``fraction`` < 1 (e.g.
    0.5 = half-Kelly) tames Kelly's notorious volatility; ``cap`` limits
    leverage. Sizes are floored at 0 (no negative bets — direction is the side's
    job)."""
    p = pd.Series(proba).astype("float64")
    f = p - (1.0 - p) / max(payoff_ratio, 1e-9)
    return (fraction * f).clip(0.0, cap)


def estimate_payoff_ratio(returns: pd.Series) -> float:
    """Average win / average loss magnitude from realized returns — a data-driven
    ``b`` for :func:`kelly_size`. Returns 1.0 if undefined."""
    r = pd.Series(returns).dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 1.0
    avg_loss = -losses.mean()
    return float(wins.mean() / avg_loss) if avg_loss > 0 else 1.0


def vol_target_scalar(asset_returns: pd.Series, target_ann_vol: float = 0.15,
                      lookback: int = 20, periods_per_year: float = 252,
                      cap: float = 3.0) -> pd.Series:
    """Per-bar multiplier so a unit position's realized vol tracks
    ``target_ann_vol``. Uses **lagged** realized vol (``shift(1)``) so the scalar
    at bar *t* depends only on the past — causal. Capped to limit leverage."""
    realized = (pd.Series(asset_returns).rolling(lookback).std().shift(1)
                * np.sqrt(periods_per_year))
    scalar = (target_ann_vol / realized).replace([np.inf, -np.inf], np.nan)
    return scalar.clip(0.0, cap).fillna(0.0)


def drawdown_throttle(position: pd.Series, asset_returns: pd.Series,
                      max_dd: float = 0.2, throttle: float = 0.5) -> pd.Series:
    """Scale ``position`` down by ``throttle`` after the strategy's drawdown
    breaches ``max_dd``, restoring full size once it recovers above the threshold.

    The drawdown is measured on the *base* strategy's equity using the engine's
    convention (position acts next bar), so the multiplier at *t* depends only on
    realized P&L through *t* — causal, no look-ahead.
    """
    base_ret = (pd.Series(position).shift(1) * pd.Series(asset_returns)).fillna(0.0)
    equity = (1.0 + base_ret).cumprod()
    dd = equity / equity.cummax() - 1.0
    mult = pd.Series(np.where(dd < -abs(max_dd), throttle, 1.0), index=position.index)
    return pd.Series(position).astype("float64") * mult
