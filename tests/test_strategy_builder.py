"""
Tests for the configurable backtester (StrategyConfig / build_signal /
run_strategy). Deterministic and offline — uses tiny hand-built frames and the
synthetic FakeProvider fixture, so behavior is exact and CI-safe.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import (
    StrategyConfig,
    available_indicators,
    build_signal,
    run_strategy,
    run_strategy_multi,
)


def _frame(closes):
    idx = pd.date_range("2022-01-03", periods=len(closes), freq="B", tz="UTC")
    idx.name = "timestamp"
    c = np.array(closes, dtype="float64")
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c,
         "volume": np.full(len(c), 1e6)},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# build_signal — the three modes, with exact expected positions
# --------------------------------------------------------------------------- #
def test_entry_exit_state_machine_exact():
    ohlcv = _frame([10, 11, 12, 9, 8, 9, 13])
    cfg = StrategyConfig(entry="close < 9.5", exit="close > 11.5")
    pos = build_signal(ohlcv, cfg)
    # exit at idx2 (close12), enter idx3-5 (close<9.5), exit idx6 (close13)
    assert pos.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_short_direction_flips_sign():
    ohlcv = _frame([10, 11, 12, 9, 8, 9, 13])
    cfg = StrategyConfig(entry="close < 9.5", exit="close > 11.5", direction="short")
    pos = build_signal(ohlcv, cfg)
    assert pos.tolist() == [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0]


def test_rule_mode_holds_while_true():
    ohlcv = _frame([10, 11, 9, 8, 12])
    pos = build_signal(ohlcv, StrategyConfig(rule="close > 9.5"))
    assert pos.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_rule_and_signal_callable_are_equivalent():
    ohlcv = _frame([10, 12, 8, 9, 15])
    rule_pos = build_signal(ohlcv, StrategyConfig(rule="close > 10"))
    sig_pos = build_signal(
        ohlcv, StrategyConfig(signal=lambda f: (f["close"] > 10).astype(float))
    )
    assert rule_pos.tolist() == sig_pos.tolist()


def test_indicators_are_available_as_columns():
    ohlcv = _frame(list(range(1, 60)))
    cfg = StrategyConfig(indicators=[("sma", {"window": 5})], rule="close > sma_5")
    pos = build_signal(ohlcv, cfg)            # must resolve sma_5 from the registry
    assert set(np.unique(pos.dropna())) <= {0.0, 1.0}


def test_missing_indicator_name_raises_clear_error():
    ohlcv = _frame([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Unknown name"):
        build_signal(ohlcv, StrategyConfig(rule="rsi_99 < 30"))  # never declared


def test_requires_one_of_rule_entry_signal():
    ohlcv = _frame([1, 2, 3])
    with pytest.raises(ValueError, match="needs one of"):
        build_signal(ohlcv, StrategyConfig())


# --------------------------------------------------------------------------- #
# run_strategy — full pipeline through a provider
# --------------------------------------------------------------------------- #
def test_run_strategy_returns_stats(provider):
    cfg = StrategyConfig(indicators=[("sma", {"window": 20}), ("sma", {"window": 50})],
                         rule="sma_20 > sma_50")
    res = run_strategy(provider, "TEST", cfg, start="2018-01-01")
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate"):
        assert k in res["stats"]
    assert res["stats"]["max_drawdown"] <= 0.0
    assert res["config"]["name"] == "custom"
    assert res["signal"].index.equals(res["returns"].index)


def test_timeframe_changes_annualization(provider):
    """Same underlying returns, different timeframe -> Sharpe scales by sqrt(ppy).
    The FakeProvider ignores the timeframe arg (returns the same daily frame), so
    only the annualization factor differs — isolating that effect exactly."""
    cfg = StrategyConfig(rule="close > 0")        # always long
    daily = run_strategy(provider, "TEST", cfg, start="2018-01-01", timeframe="1d")
    weekly = run_strategy(provider, "TEST", cfg, start="2018-01-01", timeframe="1wk")
    ratio = daily["stats"]["sharpe"] / weekly["stats"]["sharpe"]
    assert ratio == pytest.approx(np.sqrt(252 / 52), rel=1e-6)


def test_resample_reduces_rows(provider):
    cfg = StrategyConfig(rule="close > 0")
    weekly = run_strategy(provider, "TEST", cfg, start="2018-01-01", resample="W-FRI")
    daily = run_strategy(provider, "TEST", cfg, start="2018-01-01")
    assert len(weekly["returns"]) < len(daily["returns"])
    assert weekly["timeframe"] == "W-FRI"


def test_always_long_matches_buy_and_hold(provider):
    """An always-true rule with zero fees must equal buy & hold (after the
    engine's one-bar delay) — proving the builder wires into the leak-free engine."""
    cfg = StrategyConfig(rule="close > 0", fee_bps=0.0)
    res = run_strategy(provider, "TEST", cfg, start="2018-01-01")
    close = provider.get_ohlcv("TEST", start="2018-01-01")["close"]
    bh = close.pct_change()
    assert np.allclose(res["returns"].iloc[2:].to_numpy(),
                       bh.iloc[2:].to_numpy(), atol=1e-12)


def test_run_strategy_multi_portfolio(provider):
    cfg = StrategyConfig(indicators=[("rsi", {"window": 14})],
                         entry="rsi_14 < 35", exit="rsi_14 > 65")
    out = run_strategy_multi(provider, ["AAA", "BBB", "CCC"], cfg, start="2018-01-01")
    assert set(out["per_symbol"]) == {"AAA", "BBB", "CCC"}
    assert out["portfolio"] is not None
    assert "sharpe" in out["portfolio"]


def test_available_indicators_lists_registry():
    names = available_indicators()
    assert {"sma", "rsi", "zscore", "atr"} <= set(names)
