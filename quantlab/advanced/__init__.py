"""
quantlab.advanced
=================

The frontier toolkit (Module 18): the techniques that separate a competent
systematic trader from a world-class one —

* **Fractional differentiation** — stationarity without amnesia
  (:mod:`~quantlab.advanced.fracdiff`).
* **Pairs / statistical arbitrage** — cointegration, OU half-life, rolling and
  Kalman hedge ratios, a leak-free spread backtest
  (:mod:`~quantlab.advanced.pairs`).
* **Regime detection** — Gaussian-mixture market states with a strictly causal
  expanding-refit API (:mod:`~quantlab.advanced.regimes`).
* **Portfolio construction** — Hierarchical Risk Parity and inverse-variance,
  with a causal walk-forward portfolio backtest
  (:mod:`~quantlab.advanced.portfolio`).
* **Sample uniqueness** — AFML overlap-aware sample weights for honest ML
  training (:mod:`~quantlab.advanced.uniqueness`).
"""

from .fracdiff import (
    ADF_CRIT,
    adf_tstat,
    fracdiff,
    fracdiff_weights,
    min_fracdiff_order,
)
from .pairs import (
    EG_CRIT,
    KalmanHedge,
    engle_granger,
    half_life,
    kalman_hedge_ratio,
    pairs_backtest,
    rolling_hedge_ratio,
)
from .portfolio import hrp_weights, inverse_variance, portfolio_backtest
from .regimes import causal_regimes, fit_regimes, regime_stats
from .uniqueness import (
    average_uniqueness,
    label_concurrency,
    uniqueness_from_labels,
)

__all__ = [
    # fractional differentiation
    "fracdiff",
    "fracdiff_weights",
    "adf_tstat",
    "min_fracdiff_order",
    "ADF_CRIT",
    # pairs / stat arb
    "rolling_hedge_ratio",
    "kalman_hedge_ratio",
    "KalmanHedge",
    "engle_granger",
    "half_life",
    "pairs_backtest",
    "EG_CRIT",
    # regimes
    "fit_regimes",
    "causal_regimes",
    "regime_stats",
    # portfolio
    "hrp_weights",
    "inverse_variance",
    "portfolio_backtest",
    # sample uniqueness
    "label_concurrency",
    "average_uniqueness",
    "uniqueness_from_labels",
]
