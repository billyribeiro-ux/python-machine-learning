"""
quantlab.live.broker
====================

A **paper-trading broker**: a simulated account (cash + positions) that fills
orders at live prices with fees and slippage, tracks an equity curve and a trade
blotter, and persists to disk so it survives restarts.

This is *simulation*, not real money — there is no broker API and no order
routing. It exists to run your strategies forward on fresh data and see what they
*would* do, which is the honest bridge between a backtest and a live deployment.
Wiring a real broker later means implementing the same ``rebalance`` against an
execution API; the portfolio accounting here stays identical.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Fill:
    """A single simulated execution."""

    timestamp: str
    symbol: str
    shares: float        # +buy / -sell
    price: float         # fill price (after slippage)
    fee: float


class PaperBroker:
    """A simulated cash account holding share positions.

    Parameters
    ----------
    cash:
        Starting (and current) cash balance.
    positions:
        ``{symbol: shares}`` mapping (shares may be negative for shorts).
    """

    def __init__(self, cash: float = 100_000.0, positions: dict | None = None):
        self.cash = float(cash)
        self.positions: dict[str, float] = dict(positions or {})
        self.fills: list[Fill] = []
        self.equity_curve: list[tuple[str, float]] = []

    # ----- valuation --------------------------------------------------------
    def equity(self, prices: dict[str, float]) -> float:
        """Total account value = cash + marked-to-market positions."""
        mtm = sum(sh * prices[sym] for sym, sh in self.positions.items()
                  if sym in prices)
        return self.cash + mtm

    def exposure(self, prices: dict[str, float]) -> float:
        """Gross exposure as a fraction of equity (1.0 == fully invested)."""
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        gross = sum(abs(sh * prices[sym]) for sym, sh in self.positions.items()
                    if sym in prices)
        return gross / eq

    # ----- rebalancing ------------------------------------------------------
    def rebalance(self, target_weights: dict[str, float], prices: dict[str, float],
                  timestamp=None, fee_bps: float = 1.0,
                  slippage_bps: float = 1.0) -> dict[str, float]:
        """Move the book to ``target_weights`` (fractions of equity) at ``prices``.

        Leverage is capped at 1.0: if the gross target exceeds it, weights are
        scaled down proportionally. Returns the executed ``{symbol: shares}``
        order map. Fees and slippage are charged in basis points.
        """
        eq = self.equity(prices)
        gross = sum(abs(w) for w in target_weights.values())
        if gross > 1.0:                                   # no leverage by default
            target_weights = {k: v / gross for k, v in target_weights.items()}

        # Desired share count per symbol, then the order = target - current.
        orders: dict[str, float] = {}
        for sym, w in target_weights.items():
            price = prices[sym]
            target_shares = (w * eq) / price if price > 0 else 0.0
            orders[sym] = target_shares - self.positions.get(sym, 0.0)
        # Close any held symbol that's no longer targeted.
        for sym, sh in self.positions.items():
            if sym not in target_weights and abs(sh) > 1e-9 and sym in prices:
                orders[sym] = -sh

        ts = str(timestamp) if timestamp is not None else ""
        for sym, sh in orders.items():
            if abs(sh) < 1e-9:
                continue
            price = prices[sym]
            fill_price = price * (1.0 + (slippage_bps / 1e4) * np.sign(sh))
            notional = sh * fill_price
            fee = abs(notional) * fee_bps / 1e4
            self.cash -= notional + fee
            self.positions[sym] = self.positions.get(sym, 0.0) + sh
            self.fills.append(Fill(ts, sym, float(sh), float(fill_price), float(fee)))

        # Drop dust positions and record equity.
        self.positions = {s: v for s, v in self.positions.items() if abs(v) > 1e-9}
        self.equity_curve.append((ts, float(self.equity(prices))))
        return {s: o for s, o in orders.items() if abs(o) > 1e-9}

    # ----- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"cash": self.cash, "positions": self.positions,
                "fills": [asdict(f) for f in self.fills],
                "equity_curve": self.equity_curve}

    @classmethod
    def from_dict(cls, d: dict) -> "PaperBroker":
        b = cls(cash=d.get("cash", 0.0), positions=d.get("positions", {}))
        b.fills = [Fill(**f) for f in d.get("fills", [])]
        b.equity_curve = [tuple(x) for x in d.get("equity_curve", [])]
        return b

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PaperBroker":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))
