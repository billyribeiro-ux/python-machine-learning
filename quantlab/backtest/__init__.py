"""
quantlab.backtest
=================

A minimal, leak-free backtest engine and a self-contained performance-stats
toolkit. Built in Module 9/10; used by the capstone test that combines multiple
indicators with custom settings and produces verified statistics.
"""

from .engine import backtest_signal
from .strategy import (
    StrategyConfig,
    available_indicators,
    build_signal,
    parse_indicator_specs,
    run_strategy,
    run_strategy_multi,
)
from .stats import (
    cagr,
    calmar,
    compute_stats,
    drawdown_series,
    equity_curve,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)

__all__ = [
    "backtest_signal",
    # configurable strategy builder
    "StrategyConfig",
    "run_strategy",
    "run_strategy_multi",
    "build_signal",
    "available_indicators",
    "parse_indicator_specs",
    # stats
    "compute_stats",
    "equity_curve",
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "drawdown_series",
    "calmar",
    "win_rate",
]
