"""Module 9 — vectorbt parameter sweeps: combine indicators with custom settings
and collect a stats grid, all at once.

Run:  python code/m9_vectorbt.py     (requires vectorbt: pip install vectorbt)
"""
from quantlab.data import get_provider
from quantlab.backtest.vbt import combo_sweep, sweep_sma_crossover

close = get_provider("yahoo").get_ohlcv("SPY", start="2010-01-01")["close"]

# 1) Sweep every SMA fast/slow crossover combo -> stats table, ranked by Sharpe.
print("=== SMA crossover sweep (top 5 by Sharpe) ===")
grid = sweep_sma_crossover(close)
print(grid.head(5).round(3))

# 2) Sweep a MULTI-indicator combo: RSI floor x z-entry (trend fixed).
print("\n=== Combo sweep: RSI floor x z-entry (top 5 by Sharpe) ===")
combo = combo_sweep(close)
print(combo.head(5).round(3))

print("\nNote: the single best cell is likely overfit — Module 10 covers how to "
      "tell luck from edge, and Module 13 optimizes this with Optuna + pruning.")
