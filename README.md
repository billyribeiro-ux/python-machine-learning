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

Each lesson's key example is mirrored as a runnable script in `code/`:

```bash
python code/m0_hello_data.py      # pull + sanity-check data through the provider
python code/m1_indicators.py      # from-scratch indicators, proven vs pandas
python code/m2_features.py        # leak-free ML feature frame
python code/m3_multisymbol.py     # per-symbol indicators in one Polars pass
python code/m4_build_lake.py      # build the DuckDB/parquet data lake
python code/m4_windows_sql.py     # per-symbol indicators via SQL window functions
```

If you didn't run `pip install -e .`, prefix with the repo root on the path:
`PYTHONPATH=. python code/m1_indicators.py`.

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
  indicators/           #   from-scratch NumPy indicators
  backtest/             #   leak-free engine + performance stats
  strategies/           #   multi-indicator combo strategy (custom settings)
  utils/                #   returns / log-returns / forward returns
code/                   # runnable scripts mirroring each lesson's key example
tests/                  # pytest: contract + indicators + capstone stats
requirements.txt        # full course deps   |   pyproject.toml: package + extras
```

---

## Course status

Modules **0–4** (Setup, NumPy, pandas, Polars, DuckDB) are fully written and
runnable. Modules **5–17** (TA-Lib, pandas-ta, custom indicator library, Numba,
vectorbt, QuantStats, scikit-learn, XGBoost/LightGBM, Optuna, scanners, options,
Streamlit, testing) are navigable with detailed outlines and are being filled in
module-by-module; their underlying `quantlab` building blocks already exist and
are tested.
