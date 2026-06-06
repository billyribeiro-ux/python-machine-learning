"""A clean, leak-free feature frame — the template for ML modules.

Run:  python code/m2_features.py
"""
import numpy as np
import pandas as pd
from quantlab.data import get_provider
from quantlab.utils import forward_returns

px = get_provider("yahoo").get_ohlcv("SPY", start="2010-01-01")
c, h, l, v = px["close"], px["high"], px["low"], px["volume"]

X = pd.DataFrame(index=px.index)
X["ret1"] = c.pct_change()
X["ret5"] = c.pct_change(5)
X["vol20"] = X["ret1"].rolling(20).std()
X["sma_gap"] = c / c.rolling(50).mean() - 1            # distance from trend
X["rng"] = (h - l) / c                                 # daily range, normalized
X["dvol"] = (c * v).rolling(20).mean()                 # dollar volume (liquidity)

# Target: does price rise over the NEXT 5 days? (classification label)
y = (forward_returns(c, 5) > 0).astype("float")

data = X.join(y.rename("up_5d")).dropna()   # dropna aligns + removes warmup/tail
print("feature matrix:", data.shape)
print("base rate (P[up in 5d]):", f"{data['up_5d'].mean():.1%}")
print(data.tail(3).round(4))
