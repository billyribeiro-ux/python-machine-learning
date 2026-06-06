/* =========================================================================
   QuantLab lesson manifest — THE single source of truth for course order.

   Every lesson appears here exactly once, in teaching order. This one array
   drives:
     - the sidebar table of contents (grouped by `module`)
     - the Prev / Next buttons on each lesson
     - progress tracking (which lessons are "done")

   To add a lesson: drop a new HTML file in /lessons and add one entry here.
   Nothing else needs to change. `status: "ready"` lessons are fully written;
   "soon" lessons are navigable stubs with detailed outlines.
   ========================================================================= */

window.QUANTLAB_MANIFEST = [
  // module label, lesson id (== filename without .html), title, status
  { module: "Getting Started", id: "00-setup",      title: "0 · Setup & the Principal-Engineer Mindset", status: "ready" },

  { module: "Data Foundations", id: "01-numpy",     title: "1 · NumPy to the Metal",                     status: "ready" },
  { module: "Data Foundations", id: "02-pandas",    title: "2 · pandas for Market Data",                 status: "ready" },
  { module: "Data Foundations", id: "03-polars",    title: "3 · Polars for Speed & Scale",               status: "ready" },
  { module: "Data Foundations", id: "04-duckdb",    title: "4 · DuckDB: Your Market-Data Warehouse",     status: "ready" },

  { module: "Indicators",       id: "05-talib",     title: "5 · TA-Lib Mastery",                         status: "ready" },
  { module: "Indicators",       id: "06-pandas-ta", title: "6 · pandas-ta & Custom Studies",             status: "ready" },
  { module: "Indicators",       id: "07-indicator-lib", title: "7 · Build a Better Indicator Library",   status: "ready" },
  { module: "Indicators",       id: "08-numba",     title: "8 · Numba: Path-Dependent Speed",            status: "ready" },

  { module: "Backtesting",      id: "09-vectorbt",  title: "9 · vectorbt: Combine Indicators at Scale",  status: "ready" },
  { module: "Backtesting",      id: "10-quantstats",title: "10 · Backtesting & QuantStats Tearsheets",   status: "ready" },

  { module: "Machine Learning", id: "11-sklearn",   title: "11 · scikit-learn for Trading",              status: "ready" },
  { module: "Machine Learning", id: "12-boosting",  title: "12 · XGBoost & LightGBM Alpha",              status: "ready" },
  { module: "Machine Learning", id: "13-optuna",    title: "13 · Optuna: Optimize Everything",           status: "ready" },

  { module: "Putting It Together", id: "14-scanners",  title: "14 · Multi-Symbol Scanners",              status: "ready" },
  { module: "Putting It Together", id: "15-options",   title: "15 · Options: Chains, IV & Greeks",       status: "ready" },
  { module: "Putting It Together", id: "16-streamlit", title: "16 · Streamlit Trading Dashboard",        status: "ready" },
  { module: "Putting It Together", id: "17-testing",   title: "17 · Testing & Quality at Scale",         status: "ready" },
];
