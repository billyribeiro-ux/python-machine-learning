"""Module 8 — Numba path-dependent indicators (Supertrend) + a speed demo.

Run:  python code/m8_numba.py
"""
import time
import numpy as np
from quantlab.data import get_provider
from quantlab.indicators import chandelier_long_stop, supertrend
from quantlab.indicators.numba_indicators import HAS_NUMBA

df = get_provider("yahoo").get_ohlcv("NVDA", start="2018-01-01")
h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()

print("Numba available:", HAS_NUMBA, "(without it, the same code runs in pure Python)")

# Supertrend: a path-dependent trailing trend filter. direction flips are signals.
st_line, direction = supertrend(h, l, c, period=10, multiplier=3.0)
flips = int((direction[1:] != direction[:-1]).sum())
print(f"Supertrend: last line={st_line[-1]:.2f}  dir={int(direction[-1])}  flips={flips}")

# Chandelier exit: a long trailing stop that only ratchets up.
stop = chandelier_long_stop(h, l, c, period=22, multiplier=3.0)
print(f"Chandelier long stop (latest): {stop[-1]:.2f}  (price {c[-1]:.2f})")

# Speed: the first call pays JIT compilation; subsequent calls are C-fast.
t0 = time.perf_counter(); supertrend(h, l, c); t1 = time.perf_counter()
supertrend(h, l, c); t2 = time.perf_counter()
print(f"\nFirst call (incl. compile): {(t1-t0)*1000:.1f} ms")
print(f"Second call (compiled):     {(t2-t1)*1000:.3f} ms")
