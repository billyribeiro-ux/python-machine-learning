"""Resample OHLCV the RIGHT way (open=first, high=max, low=min, close=last).

Run:  python code/m2_resample.py
"""
import pandas as pd
from quantlab.data import get_provider

daily = get_provider("yahoo").get_ohlcv("SPY", start="2022-01-01")

# The canonical OHLCV aggregation map — memorize this.
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# Daily -> weekly bars (label by week-ending Friday).
weekly = daily.resample("W-FRI").agg(OHLCV_AGG).dropna()
print("daily rows :", len(daily), "-> weekly rows:", len(weekly))
print(weekly.tail(3))

# Sanity: a weekly bar's high must be >= all its daily highs in that week.
wk = weekly.index[-1]
days_in_wk = daily.loc[(daily.index <= wk) & (daily.index > weekly.index[-2])]
assert weekly["high"].iloc[-1] + 1e-9 >= days_in_wk["high"].max()
print("OHLC aggregation verified: weekly high >= constituent daily highs")
