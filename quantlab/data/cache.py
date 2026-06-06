"""
quantlab.data.cache
===================

A small **DuckDB + parquet** local market-data warehouse. Module 4 builds the
mental model; this module is the reusable artifact.

The idea: instead of scattering one-parquet-per-request files around (fine for a
single notebook, painful for a 3,000-symbol scanner), we keep OHLCV in a
columnar **parquet data lake** partitioned by timeframe and symbol, and we run
analytical SQL over it with DuckDB. DuckDB reads parquet directly — no import
step, no server, no copy into a database — so the parquet files *are* the
database. This is exactly how modern quant data platforms are built, just
shrunk to your laptop.

Why DuckDB instead of "just pandas"?

* It queries parquet **larger than RAM** by streaming, with a real query
  optimizer and parallel execution.
* SQL window functions (``OVER (PARTITION BY symbol ORDER BY timestamp)``)
  express per-symbol rolling logic cleanly and run in C++.
* The same SQL works whether the data is 1 symbol or 10,000.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import OHLCV_COLUMNS


class MarketDataLake:
    """A parquet-backed OHLCV store you can query with DuckDB SQL.

    Layout on disk (Hive-style partitioning so DuckDB can prune partitions)::

        <root>/timeframe=1d/symbol=AAPL/data.parquet
        <root>/timeframe=1d/symbol=SPY/data.parquet
        <root>/timeframe=5m/symbol=AAPL/data.parquet
    """

    def __init__(self, root: str | Path = "data/lake") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- writing
    def write(self, df: pd.DataFrame, symbol: str, timeframe: str = "1d") -> Path:
        """Persist one symbol's normalized OHLCV into the lake.

        We store ``timestamp`` as a real column (not just the index) because SQL
        engines work with columns, and we upsert by rewriting the partition —
        simple and correct for append-mostly daily data.
        """
        part = self.root / f"timeframe={timeframe}" / f"symbol={symbol}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / "data.parquet"

        out = df.copy()
        if out.index.name == "timestamp":
            out = out.reset_index()
        # If a partition already exists, merge to avoid losing history.
        if path.exists():
            existing = pd.read_parquet(path)
            out = (
                pd.concat([existing, out])
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp")
            )
        cols = ["timestamp", *OHLCV_COLUMNS]
        out[cols].to_parquet(path, index=False)
        return path

    def ingest(self, provider, symbols, start=None, end=None, timeframe="1d"):
        """Pull symbols from a :class:`DataProvider` and store them. Returns the
        number of rows written. This is the bridge from the data layer to the
        warehouse — the function a nightly cron would call."""
        rows = 0
        for sym in symbols:
            df = provider.get_ohlcv(sym, start, end, timeframe)
            self.write(df, sym, timeframe)
            rows += len(df)
        return rows

    # --------------------------------------------------------------- reading
    def connect(self):
        """Return a DuckDB connection. Lazy import keeps duckdb optional."""
        import duckdb

        return duckdb.connect()

    def glob(self, timeframe: str = "1d") -> str:
        """The parquet glob for a timeframe, for use in SQL ``read_parquet``."""
        return str(self.root / f"timeframe={timeframe}" / "*" / "*.parquet")

    def query(self, sql: str, timeframe: str = "1d") -> pd.DataFrame:
        """Run SQL against the lake. Inside ``sql`` just reference the table
        ``lake`` — we register it as a DuckDB view over the partitioned parquet,
        so you write ordinary SQL with no string substitution magic.

        Example::

            lake.query('''
                SELECT symbol, timestamp, close,
                       AVG(close) OVER (PARTITION BY symbol ORDER BY timestamp
                                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                       AS sma20
                FROM lake
                ORDER BY symbol, timestamp
            ''')
        """
        con = self.connect()
        glob = self.glob(timeframe)
        # A view named `lake` pointing at the partitioned parquet scan. Now any
        # SQL referencing `lake` works exactly like a real table — including
        # joins, CTEs, and window functions — with no fragile text replacement.
        con.execute(
            "CREATE OR REPLACE VIEW lake AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=1)"
        )
        return con.execute(sql).df()
