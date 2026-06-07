"""
Tests for the paper-trading broker and engine. Deterministic and offline.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.strategy import StrategyConfig
from quantlab.live import PaperBroker, PaperTradingEngine, StrategySlot


# --------------------------------------------------------------------------- #
# PaperBroker mechanics
# --------------------------------------------------------------------------- #
def test_full_invest_buys_and_charges_costs():
    b = PaperBroker(cash=100_000)
    prices = {"AAA": 100.0}
    orders = b.rebalance({"AAA": 1.0}, prices, timestamp="t0",
                         fee_bps=1.0, slippage_bps=1.0)
    assert orders["AAA"] == pytest.approx(1000.0, rel=1e-3)
    assert b.positions["AAA"] == pytest.approx(1000.0, rel=1e-3)
    eq = b.equity(prices)
    assert eq < 100_000                       # paid fees + slippage
    assert eq > 99_900                         # but only a little
    assert b.exposure(prices) == pytest.approx(1.0, rel=1e-3)


def test_mark_to_market_tracks_price():
    b = PaperBroker(cash=100_000)
    b.rebalance({"AAA": 1.0}, {"AAA": 100.0}, fee_bps=0.0, slippage_bps=0.0)
    assert b.equity({"AAA": 100.0}) == pytest.approx(100_000, rel=1e-9)
    assert b.equity({"AAA": 110.0}) == pytest.approx(110_000, rel=1e-9)


def test_leverage_is_capped_at_one():
    b = PaperBroker(cash=100_000)
    prices = {"X": 50.0, "Y": 25.0}
    b.rebalance({"X": 1.5, "Y": 1.0}, prices, fee_bps=0.0, slippage_bps=0.0)
    # Gross target 2.5 -> scaled to 1.0; exposure must not exceed ~1.
    assert b.exposure(prices) <= 1.0 + 1e-6


def test_going_flat_closes_positions():
    b = PaperBroker(cash=100_000)
    b.rebalance({"AAA": 1.0}, {"AAA": 100.0}, fee_bps=0.0, slippage_bps=0.0)
    b.rebalance({"AAA": 0.0}, {"AAA": 105.0}, fee_bps=0.0, slippage_bps=0.0)
    assert "AAA" not in b.positions
    assert b.cash == pytest.approx(105_000, rel=1e-6)   # all profit realized to cash


def test_short_position_and_exposure():
    b = PaperBroker(cash=100_000)
    b.rebalance({"AAA": -1.0}, {"AAA": 100.0}, fee_bps=0.0, slippage_bps=0.0)
    assert b.positions["AAA"] < 0
    assert b.exposure({"AAA": 100.0}) == pytest.approx(1.0, rel=1e-3)


def test_persistence_round_trip(tmp_path):
    b = PaperBroker(cash=100_000)
    b.rebalance({"AAA": 0.5}, {"AAA": 100.0})
    path = tmp_path / "portfolio.json"
    b.save(path)
    b2 = PaperBroker.load(path)
    assert b2.cash == pytest.approx(b.cash)
    assert b2.positions == b.positions
    assert len(b2.fills) == len(b.fills)


def test_load_missing_file_returns_empty(tmp_path):
    b = PaperBroker.load(tmp_path / "nope.json")
    assert b.cash == 100_000 and b.positions == {}


# --------------------------------------------------------------------------- #
# PaperTradingEngine
# --------------------------------------------------------------------------- #
def _slot(symbol, rule="close > 0", weight=1.0):
    return StrategySlot("s", symbol, StrategyConfig(rule=rule), weight)


def test_engine_run_once_invests_and_logs(provider, tmp_path):
    # Always-long on two symbols, half each -> fully invested.
    slots = [_slot("AAA", weight=0.5), _slot("BBB", weight=0.5)]
    eng = PaperTradingEngine(provider, slots, state_dir=tmp_path,
                             lookback_start="2018-01-01")
    snap = eng.run_once()
    assert set(snap) >= {"timestamp", "orders", "positions", "equity", "exposure"}
    assert 0.0 <= snap["exposure"] <= 1.01   # tiny slip/fee overshoot ok
    assert snap["equity"] > 0
    # State files were written.
    assert (tmp_path / "portfolio.json").exists()
    assert (tmp_path / "equity.csv").exists()


def test_engine_second_run_is_stable(provider, tmp_path):
    slots = [_slot("AAA", weight=1.0)]
    eng = PaperTradingEngine(provider, slots, state_dir=tmp_path,
                             lookback_start="2018-01-01")
    eng.run_once()
    snap2 = eng.run_once()           # same data -> already at target
    # Any residual orders are dust (tiny equity drift), not real rebalances.
    assert all(abs(v) < 1.0 for v in snap2["orders"].values())


def test_engine_aggregates_weights_per_symbol(provider, tmp_path):
    # Two long slots on the SAME symbol, 0.6 + 0.6 -> capped to 1.0 exposure.
    slots = [_slot("AAA", weight=0.6), _slot("AAA", weight=0.6)]
    eng = PaperTradingEngine(provider, slots, state_dir=tmp_path,
                             lookback_start="2018-01-01")
    snap = eng.run_once()
    assert snap["exposure"] <= 1.01   # capped gross + small cost overshoot


def test_engine_resumes_from_saved_state(provider, tmp_path):
    slots = [_slot("AAA", weight=1.0)]
    PaperTradingEngine(provider, slots, state_dir=tmp_path,
                       lookback_start="2018-01-01").run_once()
    # A fresh engine should load the persisted broker (non-default cash/positions).
    eng2 = PaperTradingEngine(provider, slots, state_dir=tmp_path,
                              lookback_start="2018-01-01")
    assert eng2.broker.positions or eng2.broker.cash != 100_000
