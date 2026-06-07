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
from .cpcv import CombinatorialPurgedCV, cpcv_sharpe_distribution
from .optimize import OptimizeResult, optimize_strategy
from .report import ReportCheck, StrategyReport, evaluate_gate, strategy_report
from .pbo import (
    PBOResult,
    cscv_pbo,
    pbo_for_template,
    returns_matrix,
    sample_param_sets,
)
from .templates import TEMPLATES
from .walkforward import chronological_split, fold_sharpes, split_index

__all__ = [
    "optimize_strategy",
    "OptimizeResult",
    # end-to-end go/no-go dossier (capstone)
    "strategy_report",
    "StrategyReport",
    "evaluate_gate",
    "ReportCheck",
    # combinatorial purged CV + probability of backtest overfitting (capstone)
    "CombinatorialPurgedCV",
    "cpcv_sharpe_distribution",
    "cscv_pbo",
    "pbo_for_template",
    "returns_matrix",
    "sample_param_sets",
    "PBOResult",
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
