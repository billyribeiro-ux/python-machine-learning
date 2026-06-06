"""Module 10 — backtest stats + a QuantStats tearsheet.

Run:  python code/m10_quantstats.py    (writes tearsheet.html if quantstats present)
"""
from quantlab.data import get_provider
from quantlab.strategies import ComboParams, run_combo
from quantlab.backtest.stats import compute_stats
from quantlab.backtest.tearsheet import html_report, metrics

df = get_provider("yahoo").get_ohlcv("SPY", start="2010-01-01")
close = df["close"]

# Run the combo strategy and show our from-scratch stats.
res = run_combo(close, ComboParams(fast=20, slow=100, rsi_floor=40, z_entry=0.5))
print("QuantLab stats bundle:")
for k, v in res["stats"].items():
    print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")

# Benchmark = buy & hold SPY.
bench = close.pct_change()
print(f"\nBuy & hold Sharpe: {compute_stats(bench)['sharpe']:.3f}")

# Full professional tearsheet (HTML) comparing strategy vs benchmark.
try:
    out = html_report(res["returns"], benchmark=bench, output="tearsheet.html")
    print(f"\nWrote tearsheet -> {out} (open it in a browser)")
except ImportError as e:
    print("\nquantstats not installed; our metrics above are the fallback.", e)
