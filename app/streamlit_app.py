"""
QuantLab Streamlit dashboard (Module 16)
========================================

A single app that ties the whole course together:

  • Scanner       — rank a universe for swing / momentum setups
  • Backtester    — tune the multi-indicator combo live and see equity + stats
  • Indicators    — chart price with indicators from the registry

Run:  streamlit run app/streamlit_app.py

Every page is driven by the same ``quantlab`` package the course built, through
the provider-agnostic data layer — so swapping Yahoo for a paid feed is one line
in quantlab/data, and this app instantly uses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.data import get_provider
from quantlab.indicators import combine, sma, supertrend
from quantlab.strategies import ComboParams, run_combo
from quantlab.backtest.stats import equity_curve, drawdown_series
from quantlab.scanners import scan, swing_pullback, momentum_breakout

st.set_page_config(page_title="QuantLab", layout="wide")

# st.cache_data memoizes the network fetch so sliders don't re-download data.
@st.cache_data(show_spinner=False)
def load(symbol: str, start: str, timeframe: str) -> pd.DataFrame:
    return get_provider("yahoo").get_ohlcv(symbol, start=start, timeframe=timeframe)


st.sidebar.title("QuantLab")
page = st.sidebar.radio("Page", ["Backtester", "Scanner", "Indicators"])

DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META",
                    "GOOGL", "JPM", "XOM", "WMT", "KO", "TSLA", "AVGO", "COST"]


# --------------------------------------------------------------------------- #
if page == "Backtester":
    st.header("Strategy Backtester — multi-indicator combo")
    c1, c2, c3 = st.columns(3)
    symbol = c1.text_input("Symbol", "SPY")
    start = c2.text_input("Start date", "2015-01-01")
    fee = c3.number_input("Fee (bps)", 0.0, 50.0, 1.0)

    st.subheader("Indicator settings")
    s1, s2, s3, s4 = st.columns(4)
    fast = s1.slider("Fast SMA", 5, 60, 20)
    slow = s2.slider("Slow SMA", 60, 250, 100)
    rsi_floor = s3.slider("RSI floor", 20, 60, 40)
    z_entry = s4.slider("Z-entry (pullback)", -2.0, 2.0, 0.5, 0.1)

    df = load(symbol, start, "1d")
    params = ComboParams(fast=fast, slow=slow, rsi_floor=float(rsi_floor),
                         z_entry=float(z_entry), fee_bps=float(fee))
    res = run_combo(df["close"], params)
    stats = res["stats"]

    m = st.columns(5)
    m[0].metric("Total return", f"{stats['total_return']:.1%}")
    m[1].metric("CAGR", f"{stats['cagr']:.1%}")
    m[2].metric("Sharpe", f"{stats['sharpe']:.2f}")
    m[3].metric("Max DD", f"{stats['max_drawdown']:.1%}")
    m[4].metric("Win rate", f"{stats['win_rate']:.1%}")

    eq = equity_curve(res["returns"])
    bench = equity_curve(df["close"].pct_change())
    st.line_chart(pd.DataFrame({"strategy": eq, "buy&hold": bench}))
    st.area_chart(drawdown_series(res["returns"]).rename("drawdown"))


# --------------------------------------------------------------------------- #
elif page == "Scanner":
    st.header("Universe Scanner")
    setup_name = st.selectbox("Setup", ["swing_pullback", "momentum_breakout"])
    universe = st.text_area("Universe (comma-separated)",
                            ", ".join(DEFAULT_UNIVERSE)).replace(" ", "").split(",")
    setup = {"swing_pullback": swing_pullback,
             "momentum_breakout": momentum_breakout}[setup_name]
    with st.spinner("Scanning..."):
        hits = scan(get_provider("yahoo"), universe, setup=setup, start="2022-01-01")
    st.write(f"**{len(hits)}** matches")
    st.dataframe(hits.style.format("{:.3f}"))


# --------------------------------------------------------------------------- #
elif page == "Indicators":
    st.header("Indicator Explorer")
    symbol = st.text_input("Symbol", "AAPL")
    start = st.text_input("Start date", "2022-01-01")
    df = load(symbol, start, "1d")
    feats = combine(df, [("sma", {"window": 20}), ("sma", {"window": 50})],
                    join_input=True)
    st_line, st_dir = supertrend(df["high"], df["low"], df["close"])
    feats["supertrend"] = st_line
    st.line_chart(feats[["close", "sma_20", "sma_50", "supertrend"]])
    st.caption("Price with SMAs and the Numba-computed Supertrend trailing line.")
