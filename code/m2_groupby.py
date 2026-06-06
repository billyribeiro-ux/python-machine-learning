"""Per-symbol indicators on a tidy multi-asset panel (no cross-contamination).

Run:  python code/m2_groupby.py
"""
import pandas as pd
from quantlab.data import get_provider

panel = get_provider("yahoo").get_ohlcv_multi(
    ["SPY", "QQQ", "AAPL", "MSFT"], start="2022-01-01"
)

# WRONG (commented): a global rolling mean mixes symbols at their boundaries.
# panel["sma20"] = panel["close"].rolling(20).mean()   # <-- bug!

# RIGHT: group by symbol, THEN roll. transform() returns a same-length column.
g = panel.groupby("symbol", group_keys=False)
panel["sma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
panel["ret"] = g["close"].transform(lambda s: s.pct_change())
panel["vol20"] = g["ret"].transform(lambda s: s.rolling(20).std())
panel["mom20"] = g["close"].transform(lambda s: s.pct_change(20))

# Cross-sectional view: rank symbols by 20-day momentum on the latest day.
latest = panel.groupby("symbol").tail(1).copy()
print(
    latest[["symbol", "close", "sma20", "vol20", "mom20"]]
    .round(3)
    .to_string(index=False)
)
