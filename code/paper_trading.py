"""Paper-trade a set of strategies forward on live data (simulation, no real
orders). State persists to data/paper/, so you can stop and resume.

Examples
--------
  PYTHONPATH=. python code/paper_trading.py                 # one rebalance now
  PYTHONPATH=. python code/paper_trading.py --interval 60   # every 60 min
  PYTHONPATH=. python code/paper_trading.py --interval 1 --iters 3   # demo loop

Each strategy is a (symbol, StrategyConfig, weight) slot — typically the ones
that earned a GO from quantlab.research.strategy_report. Swap the provider in
quantlab/data to paper-trade a real-time feed instead of Yahoo.
"""
from __future__ import annotations

import argparse

from quantlab.data import get_provider
from quantlab.backtest.strategy import StrategyConfig
from quantlab.live import PaperTradingEngine, StrategySlot

# A small, equal-weight book of trend strategies across liquid ETFs/stocks.
SLOTS = [
    StrategySlot("spy_trend", "SPY",
                 StrategyConfig(indicators=[("sma", {"window": 50}), ("sma", {"window": 200})],
                                rule="sma_50 > sma_200"), weight=0.34),
    StrategySlot("qqq_trend", "QQQ",
                 StrategyConfig(indicators=[("sma", {"window": 50}), ("sma", {"window": 200})],
                                rule="sma_50 > sma_200"), weight=0.33),
    StrategySlot("aapl_pullback", "AAPL",
                 StrategyConfig(indicators=[("sma", {"window": 20}), ("sma", {"window": 100}),
                                            ("rsi", {"window": 14}), ("zscore", {"window": 20})],
                                rule="(sma_20 > sma_100) & (rsi_14 > 40) & (zscore_20 <= 0.5)"),
                 weight=0.33),
]


def main():
    ap = argparse.ArgumentParser(description="QuantLab paper trader")
    ap.add_argument("--interval", type=float, default=None,
                    help="minutes between rebalances (omit for a single run)")
    ap.add_argument("--iters", type=int, default=None,
                    help="max number of loop iterations (with --interval)")
    args = ap.parse_args()

    engine = PaperTradingEngine(get_provider("yahoo"), SLOTS,
                                lookback_start="2010-01-01", timeframe="1d")

    if args.interval is None:
        snap = engine.run_once()
        print(f"timestamp : {snap['timestamp']}")
        print(f"equity    : ${snap['equity']:,.2f}")
        print(f"exposure  : {snap['exposure']:.0%}")
        print(f"target wts: {{ {', '.join(f'{k}:{v:+.2f}' for k, v in snap['target_weights'].items())} }}")
        print(f"orders    : {{ {', '.join(f'{k}:{v:+.1f}' for k, v in snap['orders'].items())} }}")
        print(f"positions : {snap['positions']}")
        print("\nState persisted to data/paper/ (portfolio.json, blotter.csv, equity.csv).")
        print("This is PAPER trading — a simulation, not real orders.")
    else:
        print(f"Looping every {args.interval} min (Ctrl-C to stop)...")
        engine.run_forever(interval_minutes=args.interval, max_iter=args.iters)


if __name__ == "__main__":
    main()
