"""Module 14 — multi-symbol scanners for swing & momentum setups.

Run:  python code/m14_scanners.py
"""
from quantlab.data import get_provider
from quantlab.scanners import scan, swing_pullback, momentum_breakout

provider = get_provider("yahoo")

universe = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
            "JPM", "XOM", "WMT", "KO", "TSLA", "AVGO", "COST", "HD", "PG", "UNH"]

print("=== Swing pullbacks (uptrend, RSI cooled to 40-55, liquid) ===")
sw = scan(provider, universe, setup=swing_pullback, rank_by="mom20", start="2022-01-01")
print(sw.round(3).to_string() if len(sw) else "  (no matches in current regime)")

print("\n=== Momentum breakouts (stretched & trending) ===")
mo = scan(provider, universe, setup=momentum_breakout, rank_by="mom60", start="2022-01-01")
print(mo.round(3).to_string() if len(mo) else "  (no matches in current regime)")

print("\nSwap timeframe='5m' (last ~60 days on Yahoo) to turn this into an "
      "intraday day-trade scanner — the same code, finer bars.")
