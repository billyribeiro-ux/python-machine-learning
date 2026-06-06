"""Module 1 — from-scratch indicators, proven against pandas/known values.

Run:  python code/m1_indicators.py
"""
import numpy as np
import pandas as pd

from quantlab.indicators import sma, ewma, rolling_std, rolling_zscore, rsi
from quantlab.data import get_provider

# Real data through the provider-agnostic layer.
close = get_provider("yahoo").get_ohlcv("AAPL", start="2018-01-01")["close"]
x = close.to_numpy()

# --- our implementations ---
our_sma = sma(x, 20)
our_std = rolling_std(x, 20, ddof=0)
our_ewma = ewma(x, span=20)
our_z = rolling_zscore(x, 20)
our_rsi = rsi(x, 14)

# --- references (pandas) to PROVE correctness ---
ref_sma = close.rolling(20).mean().to_numpy()
ref_std = close.rolling(20).std(ddof=0).to_numpy()
ref_ewma = close.ewm(span=20, adjust=False).mean().to_numpy()


# np.allclose ignores the NaN warmup region by masking it out.
def agree(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.allclose(a[m], b[m])


print("SMA   matches pandas:", agree(our_sma, ref_sma))
print("STD   matches pandas:", agree(our_std, ref_std))
print("EWMA  matches pandas:", agree(our_ewma, ref_ewma))
print(f"RSI(14) last value : {our_rsi[-1]:.1f}  (0-100, bounded -> sane)")
print(f"z-score last value : {our_z[-1]:+.2f}  (std devs from 20d mean)")

# A z-score regime signal: |z| > 2 flags a stretched market (mean-reversion cue).
stretched = np.abs(our_z) > 2
print(f"Days >2 sigma from mean: {np.nansum(stretched)} of {len(x)}")
