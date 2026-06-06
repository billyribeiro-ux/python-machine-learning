"""Signals decide on bar t and act on bar t+1. Prove the gap with real data.

Run:  python code/m2_lookahead.py
"""
import pandas as pd
from quantlab.data import get_provider
from quantlab.utils import forward_returns

px = get_provider("yahoo").get_ohlcv("QQQ", start="2016-01-01")
c = px["close"]

# A simple regime signal known only at each day's close.
signal = (c > c.rolling(100).mean()).astype(int)
ret = c.pct_change()

leaky = (signal * ret).fillna(0)             # WRONG: same-bar -> peeks
honest = (signal.shift(1) * ret).fillna(0)   # RIGHT: act next bar

print(f"Leaky cumulative   : {(1+leaky).prod()-1:7.1%}  <- not achievable")
print(f"Honest cumulative  : {(1+honest).prod()-1:7.1%}  <- tradeable")

# For ML, the LABEL is the forward return (future), attached to today's row.
px["target_5d"] = forward_returns(c, horizon=5)
print("\nLabel (forward 5d return) is NaN at the tail (future unknown):")
print(px["target_5d"].tail(6).round(4))
