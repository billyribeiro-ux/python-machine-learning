"""
quantlab.research
=================

Overfitting-aware research tooling: honest strategy optimization with hold-out
discipline and the Deflated Sharpe Ratio, time-series evaluation helpers, and
parameterized strategy templates an optimizer can sweep.
"""

from .metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    returns_skew_kurt,
)
from .optimize import OptimizeResult, optimize_strategy
from .templates import TEMPLATES
from .walkforward import chronological_split, fold_sharpes, split_index

__all__ = [
    "optimize_strategy",
    "OptimizeResult",
    "TEMPLATES",
    "chronological_split",
    "fold_sharpes",
    "split_index",
    "per_period_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "returns_skew_kurt",
]
