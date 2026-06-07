# QuantLab — Master Quantitative Trading in Python

An in-depth, **narrated** course that takes you from the memory layout of a
NumPy array to tuned, walk-forward machine-learning trading strategies with a
Streamlit dashboard — explained in plain English at a principal-engineer level
of depth. Every example is copy-pasteable and runs for real, and you learn to
build tools that are **better than the off-the-shelf libraries**.

It is two things at once:

1. **An interactive HTML course** (`index.html` + `lessons/`) with Monaco code
   editors, Copy / Download buttons, a sidebar table of contents, and
   Previous / Next navigation. Your progress is saved in your browser.
2. **A reusable Python package** (`quantlab/`) that the course builds
   incrementally — a provider-agnostic data layer, a from-scratch indicator
   library, a DuckDB market-data warehouse, and a leak-free backtest + stats
   engine — all covered by a `pytest` suite.

Libraries taught: **NumPy, pandas, Polars, DuckDB, TA-Lib, pandas-ta, vectorbt,
Numba, scikit-learn, XGBoost/LightGBM, Optuna, QuantStats, Streamlit.**

---

## Quick start

```bash
# 1) Create an environment (conda recommended — it installs TA-Lib painlessly)
conda create -n quantlab python=3.11 -y
conda activate quantlab
conda install -c conda-forge ta-lib -y          # the one tricky dependency

# 2) Install the course toolkit + the quantlab package (editable)
pip install -r requirements.txt
pip install -e .                                  # makes `import quantlab` work anywhere

# 3) Open the course (serve so relative paths + Monaco load correctly)
python -m http.server 8000
#   then visit http://localhost:8000/index.html
```

> Prefer `pip`/`venv`? See **Module 0** in the course for the TA-Lib C-library
> install per OS. If you skip TA-Lib, everything else still works and the
> TA-Lib tests skip cleanly.

---

## Running the examples

Every lesson's key example is mirrored as a runnable script in `code/` (one per
module, `m0`–`m15`):

```bash
python code/m1_indicators.py   # from-scratch indicators, proven vs pandas
python code/m4_build_lake.py   # build the DuckDB/parquet data lake
python code/m5_talib.py        # TA-Lib via the adapter, validated vs ours
python code/m7_registry.py     # combine indicators by name + settings
python code/m8_numba.py        # Supertrend / trailing stops (path-dependent)
python code/m9_vectorbt.py     # parameter sweeps -> stats grid
python code/m10_quantstats.py  # stats bundle + a full HTML tearsheet
python code/m11_sklearn.py     # leak-free ML: triple-barrier + purged CV
python code/m12_boosting.py    # XGBoost/LightGBM -> backtested signal
python code/m13_optuna.py      # optimize strategy settings (single + Pareto)
python code/m14_scanners.py    # multi-symbol swing/momentum scanners
python code/m15_options.py     # Black-Scholes, Greeks, implied vol
python code/honest_research.py # hold-out optimize + Deflated Sharpe + PBO/CPCV
python code/meta_and_sizing.py # meta-labeling + Kelly/vol-target/DD-throttle
python code/strategy_dossier.py# the full GO/NO-GO report in one call
python code/paper_trading.py   # paper-trade strategies forward (simulation)
streamlit run app/streamlit_app.py   # the interactive dashboard (Module 16)
```

If you didn't run `pip install -e .`, prefix with the repo root on the path:
`PYTHONPATH=. python code/m1_indicators.py`.

---

## The dashboard, page by page

```bash
streamlit run app/streamlit_app.py     # opens http://localhost:8501
```

Nine pages, each a thin shell over the tested `quantlab` package. A natural tour:

1. **Strategy Report** — the headline. Pick a symbol + strategy template, hit
   *Run full report*, and get a **GO / CONDITIONAL / NO-GO** verdict backed by
   hold-out Sharpe, **Deflated Sharpe**, **PBO**, the **CPCV** Sharpe fan, a
   cost-sensitivity curve, the hold-out equity curve, and a downloadable
   QuantStats tearsheet. Start here — it's the whole pipeline in one click.
2. **Custom Strategy** — build any strategy by hand: list indicators as
   `name:value`, type a `rule` (or `entry`/`exit`), choose timeframe/direction,
   and see live metrics, equity vs buy-&-hold, drawdown, and recent positions.
3. **Optimize** — Optuna search with hold-out discipline; shows best params,
   in-sample vs out-of-sample Sharpe, the Deflated Sharpe, and a trustworthy/
   over-fit badge.
4. **Overfitting** — PBO via CSCV with a red/amber/green verdict, the λ-logit
   histogram, an in-sample-vs-out-of-sample scatter, and the CPCV fan.
5. **ML** — walk-forward, out-of-sample XGBoost/LightGBM/logistic: OOS AUC, PnL
   vs buy-&-hold, and feature importances.
6. **Sizing & Meta** — meta-label a primary strategy, then size it (probability,
   half-Kelly, volatility target, drawdown throttle); compares primary vs
   engineered.
7. **Backtester** — slider-driven multi-indicator combo with live equity/stats.
8. **Scanner** — rank a universe with any of the six scanners (incl. intraday
   opening-range breakout).
9. **Indicators** — chart price with SMAs and the Numba Supertrend line.

Every button is exercised headlessly in CI-style checks with Streamlit's
`AppTest`, so the app is tested software, not a demo.

---

## Paper trading (simulation)

Run your strategies *forward* on fresh data with a simulated broker — the honest
bridge between a backtest and a live deployment. **No real orders, no broker
API, no money at risk.**

```bash
PYTHONPATH=. python code/paper_trading.py                # one rebalance now
PYTHONPATH=. python code/paper_trading.py --interval 60  # rebalance every 60 min
```

The `PaperTradingEngine` pulls fresh data through the provider, builds each
strategy's latest signal, aggregates them into per-symbol target weights (capped
to no leverage), and routes the implied orders to a `PaperBroker` that fills with
fees + slippage. State (cash, positions, blotter, equity curve) persists to
`data/paper/` so you can stop and resume. Because it's provider-agnostic, the
same engine paper-trades a real-time feed by swapping the provider in
`quantlab/data`.

---

## Running the tests

```bash
pytest -q
```

The suite validates the data-layer contract, the indicators (against pandas and,
when installed, TA-Lib via `importorskip`), and a **capstone**
(`tests/test_strategy_stats.py`) that combines multiple indicators with custom
settings, runs the leak-free backtest, and asserts the computed statistics are
correct and internally consistent. Tests use synthetic, seeded data so they run
**offline, deterministically, and fast** — no network required.

---

## The data layer (read this once)

Every example pulls market data through a provider-agnostic interface, **never**
a vendor SDK directly:

```python
from quantlab.data import get_provider

provider = get_provider("yahoo")          # swap to "alpaca", "polygon", ... later
spy = provider.get_ohlcv("SPY", start="2020-01-01", timeframe="1d")
```

You always get the same **normalized contract**: a tz-aware UTC `DatetimeIndex`
named `timestamp`, lowercase float columns `open/high/low/close/volume`, with
split/dividend-adjusted close by default — validated on every fetch. To use any
other data provider, implement one method on `DataProvider` and register it;
see `quantlab/data/__init__.py` for Alpaca and CCXT (crypto) sketches.

---

## Repository layout

```
index.html              # course home (curriculum + how-to)
lessons/                # one HTML lesson per module (Modules 0–4 complete)
assets/                 # shared CSS + JS (Monaco, sidebar, prev/next, progress)
quantlab/               # the reusable package the course builds
  data/                 #   provider-agnostic data layer + DuckDB data lake
  indicators/           #   NumPy + TA-Lib adapter + Numba + a unified registry
  backtest/             #   leak-free engine, stats, vectorbt + QuantStats bridges
  strategies/           #   multi-indicator combo strategy (custom settings)
  ml/                   #   features, triple-barrier labels, purged CV, models, optuna
  options/              #   Black-Scholes pricing, Greeks, implied vol
  scanners/             #   multi-symbol swing/momentum scanners
  utils/                #   returns / log-returns / forward returns
app/streamlit_app.py    # the interactive dashboard (Module 16)
code/                   # runnable scripts mirroring each lesson's key example
tests/                  # pytest: contract, indicators, options, properties, capstone
requirements.txt        # full course deps   |   pyproject.toml: package + extras
```

---

## Course status

**All 18 modules (0–17) are complete, written, and runnable** — Setup, NumPy,
pandas, Polars, DuckDB, TA-Lib, pandas-ta, the unified indicator library, Numba,
vectorbt, QuantStats, scikit-learn, XGBoost/LightGBM, Optuna, scanners, options,
Streamlit, and testing. Every module has a narrated lesson, a runnable `code/`
script (or the Streamlit app), and supporting tested code in `quantlab/`.

> Note on `pandas-ta`: it currently publishes wheels only for Python ≥ 3.12, so
> on 3.11 it may not install. Module 6 demonstrates its API and falls back to the
> QuantLab registry to compute the same indicators, so the lesson works either
> way.
