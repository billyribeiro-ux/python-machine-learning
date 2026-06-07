"""
quantlab.live.engine
====================

A **paper-trading engine** that runs a set of strategies forward on live data.

Give it a provider and a list of strategy "slots" — each a
``(name, symbol, StrategyConfig, weight)`` — typically the strategies that earned
a **GO** from :func:`quantlab.research.strategy_report`. On each rebalance it:

  1. pulls fresh data for every symbol through the provider,
  2. builds each strategy's signal and reads the position it wants at the latest
     bar,
  3. aggregates those into per-symbol **target weights** (capped to no leverage),
  4. hands them to the :class:`PaperBroker`, which fills the implied orders, and
  5. persists the account, blotter, and equity log to disk.

Run it once (``run_once``) or on a schedule (``run_forever``). It is provider-
agnostic, so the same engine paper-trades Yahoo today and a real-time feed
tomorrow by changing one line in ``quantlab.data``.

Caveat: it acts on the **latest available bar**, which intraday may be partially
formed. In production you would align to the last *closed* bar; for a learning
paper-trader this is a documented, deliberate simplification.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.backtest.strategy import StrategyConfig, build_signal
from .broker import PaperBroker


@dataclass
class StrategySlot:
    """One strategy allocation: a config on a symbol with a capital weight."""

    name: str
    symbol: str
    config: StrategyConfig
    weight: float = 1.0


class PaperTradingEngine:
    def __init__(self, provider, slots: list[StrategySlot],
                 broker: PaperBroker | None = None,
                 lookback_start: str = "2015-01-01", timeframe: str = "1d",
                 state_dir: str | Path = "data/paper",
                 fee_bps: float = 1.0, slippage_bps: float = 1.0):
        self.provider = provider
        self.slots = slots
        self.timeframe = timeframe
        self.lookback_start = lookback_start
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.state_dir = Path(state_dir)
        self.broker = broker or PaperBroker.load(self.state_dir / "portfolio.json")

    # ----- compute desired book --------------------------------------------
    def target_weights(self):
        """Return ``(weights, prices, timestamp)`` from the latest signals."""
        weights: dict[str, float] = defaultdict(float)
        prices: dict[str, float] = {}
        timestamp = None
        for slot in self.slots:
            df = self.provider.get_ohlcv(slot.symbol, start=self.lookback_start,
                                         timeframe=self.timeframe)
            sig = build_signal(df, slot.config)
            weights[slot.symbol] += slot.weight * float(sig.iloc[-1])
            prices[slot.symbol] = float(df["close"].iloc[-1])
            timestamp = df.index[-1]
        return dict(weights), prices, timestamp

    # ----- one rebalance ---------------------------------------------------
    def run_once(self) -> dict:
        weights, prices, ts = self.target_weights()
        orders = self.broker.rebalance(weights, prices, timestamp=ts,
                                       fee_bps=self.fee_bps,
                                       slippage_bps=self.slippage_bps)
        self._persist(ts, orders, prices)
        return {
            "timestamp": ts,
            "target_weights": weights,
            "orders": orders,
            "positions": dict(self.broker.positions),
            "equity": self.broker.equity(prices),
            "exposure": self.broker.exposure(prices),
        }

    # ----- scheduled loop --------------------------------------------------
    def run_forever(self, interval_minutes: float = 60.0, max_iter: int | None = None):
        """Rebalance every ``interval_minutes`` (blocking loop). ``max_iter``
        bounds the number of cycles (useful for demos/tests). Ctrl-C to stop."""
        i = 0
        while max_iter is None or i < max_iter:
            snap = self.run_once()
            ts = snap["timestamp"]
            print(f"[{ts}] equity={snap['equity']:.2f} "
                  f"exposure={snap['exposure']:.0%} orders={snap['orders']}")
            i += 1
            if max_iter is not None and i >= max_iter:
                break
            time.sleep(interval_minutes * 60.0)

    # ----- persistence -----------------------------------------------------
    def _persist(self, ts, orders, prices) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.broker.save(self.state_dir / "portfolio.json")
        # Append equity and blotter logs.
        eq_row = pd.DataFrame([{"timestamp": ts, "equity": self.broker.equity(prices),
                                "exposure": self.broker.exposure(prices)}])
        self._append_csv(self.state_dir / "equity.csv", eq_row)
        if orders:
            blot = pd.DataFrame([{"timestamp": ts, "symbol": s, "shares": round(o, 4),
                                  "price": prices[s]} for s, o in orders.items()])
            self._append_csv(self.state_dir / "blotter.csv", blot)

    @staticmethod
    def _append_csv(path: Path, df: pd.DataFrame) -> None:
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)
