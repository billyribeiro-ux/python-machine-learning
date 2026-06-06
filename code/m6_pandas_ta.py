"""Module 6 — pandas-ta (with a graceful fallback to the QuantLab registry).

pandas-ta currently publishes wheels only for Python >= 3.12; on 3.11 it may not
install. This script demonstrates the pandas-ta API when available and otherwise
shows the equivalent using our own registry — proving the concepts transfer.

Run:  python code/m6_pandas_ta.py
"""
import pandas as pd
from quantlab.data import get_provider

df = get_provider("yahoo").get_ohlcv("AAPL", start="2020-01-01")

try:
    import pandas_ta as ta  # noqa

    # pandas-ta attaches a .ta accessor to DataFrames. It expects capitalized
    # OHLCV, so map our normalized columns up for the call.
    pdf = df.rename(columns=str.capitalize)
    pdf["RSI_14"] = pdf.ta.rsi(length=14)
    macd = pdf.ta.macd(fast=12, slow=26, signal=9)
    pdf = pd.concat([pdf, macd], axis=1)
    print("pandas-ta columns:", [c for c in pdf.columns if c not in
                                  ("Open", "High", "Low", "Close", "Volume")])
    print(pdf.tail(3).round(2))
except ImportError:
    print("pandas-ta not installed on this interpreter — using QuantLab registry "
          "to compute the SAME indicators (concepts are identical):\n")
    from quantlab.indicators import combine

    feats = combine(df, [("rsi", {"window": 14}),
                         ("ewma", {"span": 12}), ("ewma", {"span": 26})])
    feats["macd"] = feats["ewma_12"] - feats["ewma_26"]
    print(feats[["rsi_14", "macd"]].tail(3).round(3))
