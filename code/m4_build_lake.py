"""Build a partitioned parquet 'data lake' and query it with DuckDB.

Run:  python code/m4_build_lake.py
"""
from quantlab.data import get_provider
from quantlab.data.cache import MarketDataLake

provider = get_provider("yahoo")
lake = MarketDataLake(root="data/lake")     # data/lake/timeframe=1d/symbol=XXX/

# Ingest a universe once; thereafter queries are instant and offline.
universe = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
rows = lake.ingest(provider, universe, start="2015-01-01", timeframe="1d")
print(f"Ingested {rows:,} bars across {len(universe)} symbols.")

# Query the WHOLE lake with one SQL statement. The macro 'lake' expands to a
# partitioned read_parquet(...) under the hood (see quantlab/data/cache.py).
latest = lake.query("""
    SELECT symbol, last(close ORDER BY timestamp) AS last_close,
           count(*) AS bars
    FROM lake
    GROUP BY symbol
    ORDER BY symbol
""")
print(latest)
