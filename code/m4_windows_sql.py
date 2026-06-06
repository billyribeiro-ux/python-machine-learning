"""Rolling indicators for an entire universe, expressed once in SQL.

Run:  python code/m4_windows_sql.py  (after m4_build_lake.py)
"""
from quantlab.data.cache import MarketDataLake

lake = MarketDataLake(root="data/lake")

sql = """
WITH feats AS (
    SELECT
        symbol, timestamp, close,
        -- 20-day SMA, per symbol, backward-looking window (no look-ahead):
        avg(close) OVER w20  AS sma20,
        -- 20-day population stddev for a z-score:
        stddev_pop(close) OVER w20 AS sd20,
        -- 20-day momentum: close / close 20 bars ago - 1
        close / lag(close, 20) OVER (PARTITION BY symbol ORDER BY timestamp) - 1
                                AS mom20
    FROM lake
    WINDOW w20 AS (PARTITION BY symbol ORDER BY timestamp
                   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
)
SELECT symbol, timestamp, close, sma20,
       (close - sma20) / nullif(sd20, 0) AS zscore,   -- nullif guards /0
       mom20
FROM feats
QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY timestamp DESC) = 1
ORDER BY mom20 DESC
"""
print("Latest per-symbol snapshot, ranked by 20-day momentum:")
print(lake.query(sql))
