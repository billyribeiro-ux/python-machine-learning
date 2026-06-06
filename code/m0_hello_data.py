"""Module 0 — verify your install and meet the data layer.

Run:  python code/m0_hello_data.py
"""
from quantlab.data import get_provider

# Ask for a provider BY NAME. Swapping vendors later = change this one string.
provider = get_provider("yahoo")           # later: get_provider("alpaca"), etc.

# Daily bars for an ETF (SPY) and a stock (AAPL). The frame you get back is
# ALWAYS the same normalized shape, no matter the vendor:
#   - tz-aware UTC DatetimeIndex named "timestamp"
#   - float columns: open, high, low, close, volume
#   - split/dividend-ADJUSTED close by default
spy = provider.get_ohlcv("SPY", start="2020-01-01", timeframe="1d")
aapl = provider.get_ohlcv("AAPL", start="2020-01-01", timeframe="1d")

print("SPY rows:", len(spy), "| range:", spy.index.min().date(), "->", spy.index.max().date())
print(spy.tail(3))

# A first, honest sanity check: daily returns should be small and centered ~0.
ret = spy["close"].pct_change()
print(f"\nSPY daily return  mean={ret.mean():.4%}  std={ret.std():.4%}")
print(f"Worst day: {ret.min():.2%}  Best day: {ret.max():.2%}")

# Multi-symbol pull returns a tidy (long) frame — the shape Polars/DuckDB/ML want.
panel = provider.get_ohlcv_multi(["SPY", "QQQ", "AAPL"], start="2023-01-01")
print("\nTidy panel shape:", panel.shape)
print(panel.groupby("symbol")["close"].last())
