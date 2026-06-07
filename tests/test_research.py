"""
Tests for the overfitting-aware research layer: deflation metrics, walk-forward
helpers, strategy templates, and the honest optimizer. Deterministic & offline.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.research import (
    chronological_split,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    fold_sharpes,
    optimize_strategy,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
)
from quantlab.research.templates import TEMPLATES


# ----- metrics -------------------------------------------------------------
def test_per_period_sharpe_basic():
    r = pd.Series([0.01, 0.02, 0.0, -0.01, 0.015])
    assert per_period_sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert per_period_sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0  # zero dispersion


def test_psr_increases_with_sample_length():
    a = probabilistic_sharpe_ratio(0.1, n=100)
    b = probabilistic_sharpe_ratio(0.1, n=2000)
    assert 0.5 < a < b <= 1.0           # more data -> more confident it's > 0


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(5) < expected_max_sharpe(50) < expected_max_sharpe(5000)


def test_deflated_sharpe_in_unit_interval_and_penalizes_trials():
    # Same observed Sharpe is less impressive after more trials.
    dsr_few = deflated_sharpe_ratio(0.12, n=1000, n_trials=5, sr_trials_std=0.1)
    dsr_many = deflated_sharpe_ratio(0.12, n=1000, n_trials=5000, sr_trials_std=0.1)
    assert 0.0 <= dsr_many <= dsr_few <= 1.0


# ----- walk-forward helpers ------------------------------------------------
def test_chronological_split_is_time_ordered(ohlcv):
    train, hold = chronological_split(ohlcv, holdout=0.25)
    assert len(train) + len(hold) == len(ohlcv)
    assert train.index.max() < hold.index.min()      # no overlap, no shuffle


def test_fold_sharpes_length(close):
    r = close.pct_change()
    assert len(fold_sharpes(r, n_folds=4)) == 4


# ----- templates -----------------------------------------------------------
def test_templates_build_valid_configs():
    for name, (build, space, invalid) in TEMPLATES.items():
        params = {}
        for k, spec in space.items():
            params[k] = spec[1] if spec[0] != "categorical" else spec[1][0]
        # make a clearly-valid combo for fast/slow templates
        if "fast" in params and "slow" in params:
            params["fast"], params["slow"] = 10, 100
        cfg = build(params)
        assert cfg.rule or cfg.entry or cfg.signal
        assert not invalid(params)


def test_template_invalid_rejects_fast_ge_slow():
    build, space, invalid = TEMPLATES["sma_cross"]
    assert invalid({"fast": 100, "slow": 50})
    assert not invalid({"fast": 20, "slow": 100})


# ----- the optimizer (small, deterministic) --------------------------------
def test_optimize_strategy_holdout_and_deflation(ohlcv):
    build, space, invalid = TEMPLATES["trend_pullback"]
    res = optimize_strategy(ohlcv, build, space, invalid=invalid,
                            n_trials=8, holdout=0.25, n_folds=3, seed=0)
    # Hold-out stats are real and sane.
    assert res.holdout_stats["max_drawdown"] <= 0.0
    assert 0.0 <= res.deflated_sharpe <= 1.0
    assert 0.0 <= res.holdout_stats["win_rate"] <= 1.0
    assert len(res.fold_sharpes) == 3
    assert np.isfinite(res.is_oos_gap)
    # Best params respect the declared search space.
    for k, spec in space.items():
        if spec[0] in ("int", "float"):
            assert spec[1] <= res.best_params[k] <= spec[2]
    assert res.best_params["fast"] < res.best_params["slow"]   # invalid pruned


def test_optimize_multiobjective_returns_pareto(ohlcv):
    build, space, invalid = TEMPLATES["sma_cross"]
    res = optimize_strategy(ohlcv, build, space, invalid=invalid,
                            n_trials=10, multiobjective=True, seed=0)
    assert len(res.pareto) >= 1
    for p in res.pareto:
        assert {"sharpe", "max_drawdown", "params"} <= set(p)


def test_optimizer_reproducible_with_seed(ohlcv):
    build, space, invalid = TEMPLATES["sma_cross"]
    a = optimize_strategy(ohlcv, build, space, invalid=invalid, n_trials=8, seed=123)
    b = optimize_strategy(ohlcv, build, space, invalid=invalid, n_trials=8, seed=123)
    assert a.best_params == b.best_params      # same seed -> same search
