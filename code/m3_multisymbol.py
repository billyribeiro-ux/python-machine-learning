"""Per-symbol indicators across a whole universe in a single parallel pass.

Run:  python code/m3_multisymbol.py     (requires polars: pip install polars)
"""
import polars as pl
from quantlab.data import get_provider

universe = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
pdf = get_provider("yahoo").get_ohlcv_multi(universe, start="2021-01-01").reset_index()
df = pl.from_pandas(pdf).sort(["symbol", "timestamp"])

# .over("symbol") = "compute this rolling/shift WITHIN each symbol". This is the
# leak-free, multi-asset equivalent of pandas groupby().transform() — but the
# groups run in parallel and the syntax is one line per feature.
feat = df.with_columns(
    ret=pl.col("close").pct_change().over("symbol"),
    sma20=pl.col("close").rolling_mean(20).over("symbol"),
    sma50=pl.col("close").rolling_mean(50).over("symbol"),
    mom20=pl.col("close").pct_change(20).over("symbol"),
    vol20=pl.col("close").pct_change().rolling_std(20).over("symbol"),
)

# Cross-sectional momentum ranking on the latest date — a scanner in embryo.
latest_date = feat.select(pl.col("timestamp").max()).item()
snapshot = (
    feat.filter(pl.col("timestamp") == latest_date)
    .select("symbol", "close", "mom20", "vol20")
    .sort("mom20", descending=True)
)
print("Momentum leaderboard (20-day) as of", str(latest_date)[:10])
print(snapshot)
