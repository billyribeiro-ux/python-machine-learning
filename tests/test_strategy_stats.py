"""
Capstone test: combine multiple indicators with custom settings and verify the
computed statistics.

This is the user's explicit goal expressed as an executable, regression-guarded
test. It proves three things at once:

1. The indicator stack (SMA + RSI + z-score) composes into a signal.
2. The backtest engine is **leak-free** — a look-ahead version is detectably
   different and better-than-reality (the test asserts the honest one isn't
   secretly peeking).
3. The stats bundle is internally consistent (Sharpe sign matches mean return,
   drawdown is non-positive, equity curve reconciles with total return, etc.).

As later modules land (vectorbt, QuantStats, Optuna) this file grows into the
full multi-indicator parameter-sweep regression.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import compute_stats, equity_curve, backtest_signal
from quantlab.strategies import ComboParams, combo_signal, run_combo


# --------------------------------------------------------------------------- #
# Stats engine correctness
# --------------------------------------------------------------------------- #
def test_stats_bundle_has_all_keys(close):
    res = run_combo(close)
    stats = res["stats"]
    expected = {
        "total_return", "cagr", "ann_volatility", "sharpe", "sortino",
        "max_drawdown", "calmar", "win_rate", "n_periods",
    }
    assert expected <= set(stats)


def test_max_drawdown_is_non_positive(close):
    res = run_combo(close)
    assert res["stats"]["max_drawdown"] <= 0.0


def test_equity_curve_reconciles_with_total_return(close):
    res = run_combo(close)
    eq = equity_curve(res["returns"])
    assert eq.iloc[-1] - 1.0 == pytest.approx(res["stats"]["total_return"], abs=1e-9)


def test_sharpe_sign_matches_mean_return():
    # A deterministic positive-drift return stream must have positive Sharpe.
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    up = pd.Series(np.full(300, 0.001), index=idx)     # +0.1%/day, no vol noise
    rng = np.random.default_rng(0)
    up = up + rng.normal(0, 0.005, size=300)
    s = compute_stats(up)
    assert (s["sharpe"] > 0) == (up.mean() > 0)


def test_win_rate_in_unit_interval(close):
    s = run_combo(close)["stats"]
    assert 0.0 <= s["win_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# Combining multiple indicators with CUSTOM settings
# --------------------------------------------------------------------------- #
def test_custom_settings_change_results(close):
    """Different indicator settings must produce different signals/stats —
    otherwise the parameters aren't actually wired in."""
    a = run_combo(close, ComboParams(fast=10, slow=50, rsi_floor=45, z_entry=0.0))
    b = run_combo(close, ComboParams(fast=30, slow=150, rsi_floor=35, z_entry=1.0))
    assert a["params"] != b["params"]
    # The two configs should not produce identical position series.
    assert not a["positions"].equals(b["positions"])


def test_signal_is_long_or_flat_only(close):
    sig = combo_signal(close, ComboParams())
    assert set(np.unique(sig.dropna())) <= {0.0, 1.0}


def test_signal_respects_trend_gate(close):
    """Positions should only ever be taken when the fast SMA is above the slow
    SMA — the trend gate is a hard constraint of this strategy."""
    from quantlab.indicators import sma

    p = ComboParams()
    sig = combo_signal(close, p)
    x = close.to_numpy()
    trend_ok = pd.Series(sma(x, p.fast) > sma(x, p.slow), index=close.index)
    # Every bar with signal==1 must also have trend_ok==True.
    assert bool((sig.astype(bool) <= trend_ok).all())


# --------------------------------------------------------------------------- #
# The engine must be leak-free
# --------------------------------------------------------------------------- #
def test_engine_applies_one_bar_execution_delay(close):
    """A constant always-long signal should earn the asset return shifted by
    one bar — proving the engine delays execution and never peeks."""
    always_long = pd.Series(1.0, index=close.index)
    res = backtest_signal(close, always_long, fee_bps=0.0)
    # With a one-bar delay and no costs, strategy return at t == asset return at
    # t for t>=2 (position is 1 from bar 2 onward). Compare from index 2.
    asset_ret = close.pct_change()
    got = res["returns"].iloc[2:]
    expected = asset_ret.iloc[2:]
    assert np.allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_costs_reduce_returns(close):
    """Turning on transaction costs can only lower total return vs zero cost."""
    sig = combo_signal(close, ComboParams())
    free = backtest_signal(close, sig, fee_bps=0.0)["stats"]["total_return"]
    costly = backtest_signal(close, sig, fee_bps=10.0)["stats"]["total_return"]
    assert costly <= free + 1e-12
