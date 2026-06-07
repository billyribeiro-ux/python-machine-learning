"""
QuantLab Streamlit dashboard (Module 16)
========================================

A single app that ties the whole course together:

  • Custom Strategy — build ANY indicator strategy on ANY timeframe and test it
  • Optimize        — honest Optuna search (hold-out + Deflated Sharpe)
  • ML              — walk-forward XGBoost/LightGBM, OOS predictions -> PnL
  • Backtester      — tune the multi-indicator combo live and see equity + stats
  • Scanner         — rank a universe for swing / momentum / RS / gap / breakout
  • Indicators      — chart price with indicators from the registry

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
from quantlab.backtest import backtest_signal
from quantlab.backtest.stats import drawdown_series, equity_curve
from quantlab.research import optimize_strategy
from quantlab.research.templates import TEMPLATES
from quantlab.ml import (
    assemble_dataset,
    feature_importances,
    fit_full_model,
    make_features,
    oos_signal,
    triple_barrier_labels,
    walk_forward_predict,
)
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
    "Page", ["Custom Strategy", "Optimize", "ML", "Backtester", "Scanner",
             "Indicators"]
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
elif page == "Optimize":
    st.header("Honest Strategy Optimizer")
    st.caption("Optuna search with a held-out test period and the Deflated "
               "Sharpe Ratio — so the 'best' result has to beat what luck alone "
               "would produce across all the trials you ran.")

    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.text_input("Symbol", "SPY")
    start = c2.text_input("Start date", "2008-01-01")
    template_name = c3.selectbox("Strategy template", list(TEMPLATES))
    n_trials = c4.slider("Trials", 10, 150, 40, 10)

    d1, d2, d3, d4 = st.columns(4)
    holdout = d1.slider("Hold-out fraction", 0.1, 0.5, 0.25, 0.05)
    n_folds = d2.slider("Robustness folds", 2, 8, 4)
    multiobjective = d3.checkbox("Multi-objective (Sharpe vs drawdown)")
    fee = d4.number_input("Fee (bps)", 0.0, 50.0, 1.0)

    if st.button("Optimize", type="primary"):
        build, space, invalid = TEMPLATES[template_name]
        # bake the fee into the template via a wrapper
        def build_fee(params, _b=build, _f=float(fee)):
            cfg = _b(params); cfg.fee_bps = _f; return cfg
        try:
            with st.spinner(f"Running {n_trials} trials with hold-out validation..."):
                ohlcv = load(symbol, start, "1d")
                res = optimize_strategy(
                    ohlcv, build_fee, space, invalid=invalid, n_trials=int(n_trials),
                    holdout=float(holdout), n_folds=int(n_folds),
                    multiobjective=multiobjective)
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
        else:
            st.subheader("Best parameters")
            st.json({k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in res.best_params.items()})

            tr, ho = res.train_stats, res.holdout_stats
            cols = st.columns(4)
            cols[0].metric("Train Sharpe (in-sample)", f"{tr['sharpe']:.2f}")
            cols[1].metric("Hold-out Sharpe (OOS)", f"{ho['sharpe']:.2f}",
                           delta=f"{-res.is_oos_gap:.2f} vs IS")
            cols[2].metric("Deflated Sharpe", f"{res.deflated_sharpe:.2f}")
            cols[3].metric("Hold-out max DD", f"{ho['max_drawdown']:.1%}")

            verdict = ("✅ Trustworthy — beats the luck threshold and holds out of "
                       "sample." if res.trustworthy else
                       "⚠️ Treat with skepticism — likely overfit (low deflated "
                       "Sharpe or large in-sample→out-of-sample gap).")
            (st.success if res.trustworthy else st.warning)(verdict)

            st.caption(f"Per-fold train Sharpes (robustness across regimes): "
                       f"{[round(f, 2) for f in res.fold_sharpes]}  ·  "
                       f"{res.n_trials} completed trials")

            st.subheader("Hold-out equity curve (the honest result)")
            st.line_chart(equity_curve(res.holdout_returns).rename("hold-out equity"))

            if res.history:
                st.subheader("Optimization progress (best objective so far)")
                st.line_chart(pd.Series(res.history, name="best objective"))

            if res.pareto:
                st.subheader("Pareto front (Sharpe vs drawdown)")
                st.dataframe(pd.DataFrame(res.pareto).head(10))


# --------------------------------------------------------------------------- #
elif page == "ML":
    st.header("Walk-Forward Machine Learning")
    st.caption("Purged, out-of-sample predictions (no peeking), evaluated by "
               "PnL — not accuracy. AUC near 0.5 is normal and honest.")

    c1, c2, c3, c4 = st.columns(4)
    symbol = c1.text_input("Symbol", "QQQ")
    start = c2.text_input("Start date", "2008-01-01")
    kind = c3.selectbox("Model", ["logistic", "xgboost", "lightgbm"])
    n_splits = c4.slider("Walk-forward folds", 3, 10, 5)

    d1, d2, d3, d4 = st.columns(4)
    horizon = d1.slider("Label horizon (bars)", 3, 30, 10)
    barrier = d2.slider("Barrier (× vol)", 1.0, 4.0, 2.0, 0.5)
    long_th = d3.slider("Long threshold", 0.50, 0.70, 0.55, 0.01)
    fee = d4.number_input("Fee (bps)", 0.0, 50.0, 1.0)

    if st.button("Train & backtest", type="primary"):
        try:
            with st.spinner("Building features, labeling, walk-forward training..."):
                df = load(symbol, start, "1d")
                close = df["close"]
                lab = triple_barrier_labels(close, horizon=int(horizon),
                                            upper=float(barrier), lower=float(barrier))
                y = (lab["label"] > 0).astype(float).where(lab["label"].notna())
                X, y = assemble_dataset(df, y)
                proba = walk_forward_predict(X, y, kind=kind, n_splits=int(n_splits),
                                             embargo=0.01, label_horizon=int(horizon))
                sig = oos_signal(proba, long_th=float(long_th))
                bt = backtest_signal(close.loc[sig.index], sig, fee_bps=float(fee))
        except Exception as exc:
            st.error(f"Training failed: {exc}")
        else:
            from sklearn.metrics import roc_auc_score
            mask = proba.notna()
            auc = (roc_auc_score(y[mask], proba[mask])
                   if mask.any() and y[mask].nunique() > 1 else float("nan"))
            s = bt["stats"]
            m = st.columns(5)
            m[0].metric("OOS AUC", f"{auc:.3f}")
            m[1].metric("OOS return", f"{s['total_return']:.1%}")
            m[2].metric("OOS Sharpe", f"{s['sharpe']:.2f}")
            m[3].metric("Max DD", f"{s['max_drawdown']:.1%}")
            m[4].metric("Bars traded", f"{int(mask.sum())}")

            st.subheader("Out-of-sample equity curve")
            st.line_chart(pd.DataFrame({
                "ML strategy": equity_curve(bt["returns"]),
                "buy & hold": equity_curve(close.loc[sig.index].pct_change()),
            }))

            st.subheader("Feature importance (model fit on full history)")
            model = fit_full_model(X, y, kind=kind)
            st.bar_chart(feature_importances(model, X.columns))
            st.caption("AUC ≈ 0.52–0.56 on liquid daily data is a realistic edge. "
                       "Beware anyone reporting 0.9 — that's almost always leakage.")


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
