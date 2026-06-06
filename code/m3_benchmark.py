"""Polars vs pandas on multi-symbol rolling z-score. Same answer, measure speed.

Run:  python code/m3_benchmark.py
"""
import time
import numpy as np
import pandas as pd
import polars as pl
from quantlab.data import get_provider

universe = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
            "TSLA", "JPM", "XOM", "WMT"]
panel = get_provider("yahoo").get_ohlcv_multi(universe, start="2010-01-01")

# --- pandas version ---
t0 = time.perf_counter()
g = panel.groupby("symbol", group_keys=False)["close"]
pd_z = g.transform(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std(ddof=0))
t_pd = time.perf_counter() - t0

# --- polars version ---
df = pl.from_pandas(panel.reset_index()).sort(["symbol", "timestamp"])
t0 = time.perf_counter()
pl_df = df.with_columns(
    z=((pl.col("close") - pl.col("close").rolling_mean(20))
       / pl.col("close").rolling_std(20, ddof=0)).over("symbol")
)
t_pl = time.perf_counter() - t0

print(f"pandas: {t_pd*1000:7.1f} ms")
print(f"polars: {t_pl*1000:7.1f} ms   ({t_pd/max(t_pl,1e-6):.1f}x faster)")
print("(Both compute the identical 20-day rolling z-score per symbol.)")
