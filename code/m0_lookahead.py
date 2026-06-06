"""The single most expensive bug in quant, shown in 12 lines.

Run:  python code/m0_lookahead.py
"""
import numpy as np
from quantlab.data import get_provider

px = get_provider("yahoo").get_ohlcv("SPY", start="2015-01-01")
close = px["close"]
sma50 = close.rolling(50).mean()

# A naive "buy when price > 50-day average" signal.
signal = (close > sma50).astype(int)

# WRONG: act on today's signal using today's return (you couldn't have known
# the signal until the close, yet you're crediting yourself today's move).
wrong = signal * close.pct_change()

# RIGHT: you learn the signal at today's close, so you can only earn the NEXT
# day's return. shift(1) enforces that one-bar delay.
right = signal.shift(1) * close.pct_change()

print(f"Look-ahead 'edge' (FAKE): {(1+wrong.fillna(0)).prod()-1:.1%}")
print(f"Honest result:            {(1+right.fillna(0)).prod()-1:.1%}")
print("The gap between them is pure self-deception.")
