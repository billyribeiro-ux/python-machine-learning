"""Module 18 — the frontier toolkit on real data: fractional differentiation,
cointegration + pairs, causal regime detection, HRP, and sample uniqueness.

Run:  PYTHONPATH=. python code/m18_frontier.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from quantlab.data import get_provider
from quantlab.advanced import (
    adf_tstat, min_fracdiff_order, fracdiff,
    engle_granger, half_life, pairs_backtest,
    fit_regimes, causal_regimes, regime_stats,
    hrp_weights, inverse_variance, portfolio_backtest,
    uniqueness_from_labels,
)

prov = get_provider("yahoo")

# ----- 1) Fractional differentiation: stationarity without amnesia ---------
close = prov.get_ohlcv("SPY", start="2010-01-01")["close"]
print("=== Fractional differentiation (SPY) ===")
print(f"ADF(prices)  = {adf_tstat(close):6.2f}   (stationary needs < -2.86)")
print(f"ADF(returns) = {adf_tstat(close.pct_change().dropna()):6.2f}")
d_star, table = min_fracdiff_order(close)
print(f"minimal d* for stationarity: {d_star}")
print(table.round(3).to_string())

# ----- 2) Pairs / stat-arb: QQQ vs SPY --------------------------------------
print("\n=== Cointegration & pairs (QQQ vs SPY) ===")
y = prov.get_ohlcv("QQQ", start="2015-01-01")["close"]
x = prov.get_ohlcv("SPY", start="2015-01-01")["close"]
eg = engle_granger(y, x)
print(f"beta={eg['beta']:.3f}  ADF(resid)={eg['adf_t']:.2f}  "
      f"(EG 5% crit {eg['crit']['5%']})  cointegrated={eg['cointegrated_5pct']}")
print(f"spread half-life: {half_life(eg['residual']):.1f} bars")
res = pairs_backtest(y, x, hedge="rolling", window=250, z_window=30)
s = res["stats"]
print(f"spread strategy: ret={s['total_return']:.1%} sharpe={s['sharpe']:.2f} "
      f"maxDD={s['max_drawdown']:.1%}  (an honest result either way — most "
      "index pairs are arbitraged thin)")

# ----- 3) Regime detection ---------------------------------------------------
print("\n=== Regime detection (SPY, 3 states, vol-sorted) ===")
ret = close.pct_change()
labels, _ = fit_regimes(ret, n_regimes=3)
print(regime_stats(ret, labels).round(3).to_string())
live = causal_regimes(ret, n_regimes=3, min_train=252, refit_every=63)
print(f"causal labels: {int(live.notna().sum())} bars labeled, "
      f"current regime = {int(live.dropna().iloc[-1])}")

# ----- 4) HRP portfolio ------------------------------------------------------
print("\n=== Portfolio construction (6 ETFs) ===")
panel = prov.get_ohlcv_multi(["SPY", "QQQ", "TLT", "GLD", "XLE", "XLF"],
                             start="2015-01-01")
prices = panel.pivot_table(index="timestamp", columns="symbol", values="close")
rets = prices.pct_change().dropna()
print("HRP weights:", hrp_weights(rets).round(3).to_dict())
for name, fn in [("HRP", hrp_weights), ("inverse-variance", inverse_variance)]:
    r = portfolio_backtest(rets, weight_fn=fn, lookback=252, rebalance_every=21)
    print(f"  {name:17s} sharpe={r['stats']['sharpe']:5.2f} "
          f"maxDD={r['stats']['max_drawdown']:6.1%}")

# ----- 5) Sample uniqueness --------------------------------------------------
print("\n=== Sample uniqueness (SPY triple-barrier labels) ===")
from quantlab.ml import triple_barrier_labels
lab = triple_barrier_labels(close, horizon=10, upper=2.0, lower=2.0)
u = uniqueness_from_labels(lab, horizon=10)
print(f"events={len(u)}  mean uniqueness={u.mean():.2f}  "
      f"-> effective independent samples ≈ {int(len(u) * u.mean())}")
print("Pass `sample_weight=uniqueness` to model.fit() — same data, honest weighting.")
