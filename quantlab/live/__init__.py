"""
quantlab.live
=============

Paper trading: run strategies forward on live data with a simulated broker
(cash, positions, fees, slippage), persisted to disk. The honest bridge between
a backtest and a real deployment — no real orders, same accounting.
"""

from .broker import Fill, PaperBroker
from .engine import PaperTradingEngine, StrategySlot

__all__ = ["PaperBroker", "Fill", "PaperTradingEngine", "StrategySlot"]
