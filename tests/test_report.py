"""
Tests for the end-to-end strategy dossier. The decision GATE is a pure function,
so it's tested deterministically; the full report is smoke-tested on the
synthetic fixture (small search) to confirm every field is populated.
"""

import pandas as pd
import pytest

from quantlab.research import StrategyReport, evaluate_gate, strategy_report
from quantlab.research.templates import TEMPLATES


# --------------------------------------------------------------------------- #
# The decision gate (pure, deterministic)
# --------------------------------------------------------------------------- #
def test_gate_go_when_everything_passes():
    decision, checks = evaluate_gate(holdout_sharpe=1.0, deflated_sharpe=0.9,
                                     pbo=0.1, cpcv_p05=0.2, is_oos_gap=0.3,
                                     cost_sharpe=0.8)
    assert decision == "GO"
    assert all(c.passed for c in checks)
    assert len(checks) == 6


def test_gate_conditional_when_only_noncritical_fails():
    # All three critical checks pass; a non-critical one (CPCV 5th pct) fails.
    decision, checks = evaluate_gate(holdout_sharpe=0.8, deflated_sharpe=0.7,
                                     pbo=0.3, cpcv_p05=-0.1, is_oos_gap=0.2,
                                     cost_sharpe=0.5)
    assert decision == "CONDITIONAL"
    assert all(c.passed for c in checks if c.critical)
    assert not all(c.passed for c in checks)


def test_gate_nogo_when_a_critical_fails():
    # PBO too high -> critical failure -> NO-GO regardless of the rest.
    decision, _ = evaluate_gate(holdout_sharpe=1.0, deflated_sharpe=0.9,
                                pbo=0.7, cpcv_p05=0.5, is_oos_gap=0.1,
                                cost_sharpe=1.0)
    assert decision == "NO-GO"


def test_gate_nogo_on_negative_holdout():
    decision, _ = evaluate_gate(holdout_sharpe=-0.2, deflated_sharpe=0.9,
                                pbo=0.1, cpcv_p05=0.5, is_oos_gap=0.1,
                                cost_sharpe=1.0)
    assert decision == "NO-GO"


# --------------------------------------------------------------------------- #
# Full report end-to-end (small, deterministic-ish)
# --------------------------------------------------------------------------- #
def test_strategy_report_populates_everything(ohlcv):
    build, space, invalid = TEMPLATES["sma_cross"]
    rep = strategy_report(ohlcv, build, space, invalid=invalid,
                          n_trials=8, pbo_configs=12, s_blocks=8,
                          cpcv_groups=8, cpcv_test=2, make_tearsheet=False)
    assert isinstance(rep, StrategyReport)
    assert rep.decision in {"GO", "CONDITIONAL", "NO-GO"}
    assert len(rep.checks) == 6
    # CPCV summary fields present and ordered worst <= p05 <= median.
    assert rep.cpcv["worst"] <= rep.cpcv["p05"] <= rep.cpcv["median"] + 1e-9
    # Cost table has a row per fee in the default grid.
    assert list(rep.cost_table.index) == [0.0, 1.0, 5.0, 10.0, 25.0]
    # Higher fees never increase total return.
    tr = rep.cost_table["total_return"]
    assert tr.is_monotonic_decreasing
    assert len(rep.holdout_returns) > 0
    assert 0.0 <= rep.pbo <= 1.0
    assert isinstance(rep.summary(), str) and rep.decision in rep.summary()


def test_strategy_report_is_reproducible(ohlcv):
    build, space, invalid = TEMPLATES["sma_cross"]
    kw = dict(n_trials=8, pbo_configs=10, s_blocks=8, seed=7)
    a = strategy_report(ohlcv, build, space, invalid=invalid, **kw)
    b = strategy_report(ohlcv, build, space, invalid=invalid, **kw)
    assert a.best_params == b.best_params
    assert a.decision == b.decision
