"""Relative strength vs SPY via a SQL join over the lake.

Run:  python code/m4_join_benchmark.py  (after m4_build_lake.py)
"""
from quantlab.data.cache import MarketDataLake

lake = MarketDataLake(root="data/lake")

sql = """
WITH px AS (SELECT symbol, timestamp, close FROM lake),
     bench AS (SELECT timestamp, close AS spy FROM lake WHERE symbol = 'SPY')
SELECT p.symbol,
       p.timestamp,
       p.close / b.spy AS rs            -- relative strength ratio vs SPY
FROM px p
JOIN bench b USING (timestamp)
WHERE p.symbol <> 'SPY'
QUALIFY row_number() OVER (PARTITION BY p.symbol ORDER BY p.timestamp DESC) = 1
ORDER BY rs DESC
"""
print("Relative strength vs SPY (latest):")
print(lake.query(sql))
