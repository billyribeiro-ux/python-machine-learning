"""
QuantLab Streamlit dashboard (Module 16)
========================================

A single app that ties the whole course together:

  • Custom Strategy — build ANY indicator strategy on ANY timeframe and test it
  • Backtester     — tune the multi-indicator combo live and see equity + stats
  • Scanner        — rank a universe for swing / momentum / RS / gap / breakout
  • Indicators     — chart price with indicators from the registry

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
from quantlab.indicators import combine, supertrend
from quantlab.strategies import ComboParams, run_combo
from quantlab.backtest import (
    StrategyConfig,
    available_indicators,
    parse_indicator_specs,
    run_strategy,
)
from quantlab.backtest.stats import drawdown_series, equity_curve
from quantlab.scanners import (
    gap_up,
    momentum_breakout,
    new_high_breakout,
    orb_scan,
    relative_strength,
    scan,
    swing_pullback,
)

st.set_page_config(page_title="QuantLab", layout="wide")

INTRADAY = {"1h", "60m", "30m", "15m", "5m", "1m"}


@st.cache_data(show_spinner=False)
def load(symbol: str, start, timeframe: str) -> pd.DataFrame:
    """Cached fetch so widgets don't re-download data on every interaction."""
    return get_provider("yahoo").get_ohlcv(symbol, start=start, timeframe=timeframe)


st.sidebar.title("QuantLab")
page = st.sidebar.radio(
    "Page", ["Custom Strategy", "Backtester", "Scanner", "Indicators"]
)

DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META",
                    "GOOGL", "JPM", "XOM", "WMT", "KO", "TSLA", "AVGO", "COST"]


# --------------------------------------------------------------------------- #
if page == "Custom Strategy":
    st.header("Custom Strategy Builder")
    st.caption("Declare any indicators + rules, on any symbol and timeframe — "
               "backtested leak-free (decide on bar t, act on t+1).")

    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.text_input("Symbol", "SPY")
    start = c2.text_input("Start date", "2010-01-01")
    timeframe = c3.selectbox("Timeframe", ["1d", "1wk", "1h", "30m", "15m", "5m"])
    fee = c4.number_input("Fee (bps)", 0.0, 50.0, 1.0)

    cc1, cc2 = st.columns([1, 2])
    direction = cc1.radio("Direction", ["long", "short"], horizontal=True)
    specs_text = cc2.text_area(
        "Indicators (one per line, `name:value`)",
        "sma:50\nsma:200\nrsi:14",
        height=110,
    )
    specs, cols, spec_errors = parse_indicator_specs(specs_text)
    for e in spec_errors:
        st.warning(e)
    st.caption(f"Available indicators: {available_indicators()}")
    if cols:
        st.caption(f"Columns you can reference in rules: "
                   f"`{'`, `'.join(cols)}`, plus `open/high/low/close/volume`")

    mode = st.radio("Logic", ["rule (hold while true)", "entry / exit"],
                    horizontal=True)
    if mode.startswith("rule"):
        rule = st.text_input("Rule", "sma_50 > sma_200")
        entry = exit_ = None
    else:
        rule = None
        e1, e2 = st.columns(2)
        entry = e1.text_input("Entry when", "(rsi_14 < 35) & (close > sma_200)")
        exit_ = e2.text_input("Exit when", "rsi_14 > 65")

    st.caption("Combine conditions with `&` `|` `~` (vectorized) — not "
               "`and`/`or`/`not`.")

    if st.button("Run backtest", type="primary"):
        cfg = StrategyConfig(indicators=specs, rule=rule, entry=entry, exit=exit_,
                             direction=direction, fee_bps=float(fee), name="dashboard")
        start_arg = None if timeframe in INTRADAY else start
        try:
            with st.spinner("Backtesting..."):
                res = run_strategy(get_provider("yahoo"), symbol, cfg,
                                   start=start_arg, timeframe=timeframe)
                bench = load(symbol, start_arg, timeframe)["close"].pct_change()
        except Exception as exc:  # surface a friendly error, don't crash the app
            st.error(f"Could not run this strategy: {exc}")
        else:
            s = res["stats"]
            m = st.columns(6)
            m[0].metric("Total return", f"{s['total_return']:.1%}")
            m[1].metric("CAGR", f"{s['cagr']:.1%}")
            m[2].metric("Sharpe", f"{s['sharpe']:.2f}")
            m[3].metric("Sortino", f"{s['sortino']:.2f}")
            m[4].metric("Max DD", f"{s['max_drawdown']:.1%}")
            m[5].metric("Win rate", f"{s['win_rate']:.1%}")
            st.caption(f"≈ {int(res['turnover']) // 2} round-trip trades · "
                       f"timeframe {res['timeframe']}")

            eq = equity_curve(res["returns"])
            st.subheader("Equity curve vs buy & hold")
            st.line_chart(pd.DataFrame(
                {"strategy": eq, "buy & hold": equity_curve(bench)}))
            st.subheader("Drawdown")
            st.area_chart(drawdown_series(res["returns"]).rename("drawdown"))
            st.subheader("Recent positions")
            st.dataframe(res["signal"].rename("position").tail(20).to_frame())


# --------------------------------------------------------------------------- #
elif page == "Backtester":
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
    SNAPSHOT_SETUPS = {
        "swing_pullback": swing_pullback,
        "momentum_breakout": momentum_breakout,
        "relative_strength": relative_strength,
        "gap_up": gap_up,
        "new_high_breakout": new_high_breakout,
    }
    setup_name = st.selectbox(
        "Setup", list(SNAPSHOT_SETUPS) + ["opening_range_breakout (intraday)"])
    universe = st.text_area("Universe (comma-separated)",
                            ", ".join(DEFAULT_UNIVERSE)).replace(" ", "").split(",")

    with st.spinner("Scanning..."):
        if setup_name.startswith("opening_range_breakout"):
            hits = orb_scan(get_provider("yahoo"), universe, timeframe="5m")
        else:
            hits = scan(get_provider("yahoo"), universe,
                        setup=SNAPSHOT_SETUPS[setup_name], start="2022-01-01")
    st.write(f"**{len(hits)}** matches")
    if len(hits):
        st.dataframe(hits.style.format("{:.3f}"))
    else:
        st.info("No matches in the current market regime (a valid result).")


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
