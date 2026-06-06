"""Module 5 — TA-Lib through the QuantLab adapter, validated against our own.

Run:  python code/m5_talib.py     (requires TA-Lib; see Module 0 to install)
"""
import numpy as np
from quantlab.data import get_provider
from quantlab.indicators import rsi as our_rsi
from quantlab.indicators.talib_wrap import add_indicators, list_functions, talib_func

df = get_provider("yahoo").get_ohlcv("AAPL", start="2018-01-01")

# Discover what's available, grouped by category.
groups = list_functions()
print("TA-Lib categories:", list(groups)[:6], "...")
print("Momentum indicators:", groups["Momentum Indicators"][:8], "...")

# Compute a basket of indicators with custom settings in one call.
feats = add_indicators(df, [
    ("RSI",    {"timeperiod": 14}),
    ("MACD",   {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
    ("ATR",    {"timeperiod": 14}),
    ("BBANDS", {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0}),
])
new_cols = [c for c in feats.columns if c not in df.columns]
print("\nAdded columns:", new_cols)
print(feats[new_cols].tail(3).round(2))

# Validate: TA-Lib's RSI should match our from-scratch RSI (after warmup).
tl_rsi = talib_func("RSI", df, timeperiod=14)["RSI_real"].to_numpy()
mine = our_rsi(df["close"].to_numpy(), 14)
m = ~np.isnan(tl_rsi) & ~np.isnan(mine)
print(f"\nOur RSI matches TA-Lib (after burn-in): "
      f"{np.allclose(tl_rsi[20:][m[20:]], mine[20:][m[20:]], atol=1e-6)}")
