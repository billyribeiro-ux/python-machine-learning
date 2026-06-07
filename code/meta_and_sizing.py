"""Meta-labeling + position sizing: turn a raw signal into a risk-managed system.

Primary strategy gives the SIDE; a walk-forward meta-model decides whether to
take the bet; sizing (probability / fractional Kelly / vol-target / drawdown
throttle) decides how much. Judge on risk-adjusted terms.

Run:  PYTHONPATH=. python code/meta_and_sizing.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from quantlab.data import get_provider
from quantlab.backtest import (backtest_signal, build_signal, bet_size, kelly_size,
                               estimate_payoff_ratio, vol_target_scalar,
                               drawdown_throttle)
from quantlab.research.templates import trend_pullback
from quantlab.ml import (make_features, triple_barrier_meta, meta_dataset,
                         apply_meta, walk_forward_predict)

ohlcv = get_provider("yahoo").get_ohlcv("SPY", start="2008-01-01")
close = ohlcv["close"]

# 1) Primary: a rule-based side (long/flat).
side = build_signal(ohlcv, trend_pullback(
    {"fast": 20, "slow": 150, "rsi_w": 14, "rsi_floor": 40, "z_w": 20, "z_entry": 0.5}))

# 2) Meta-labels + walk-forward OOS meta-probabilities (P[this bet wins]).
meta = triple_barrier_meta(close, side, horizon=10, pt=1.0, sl=1.0)
X, y = meta_dataset(make_features(ohlcv), meta)
print(f"primary bets: {len(y)}   primary win rate: {y.mean():.1%}")
proba = walk_forward_predict(X, y, kind="lightgbm", n_splits=5,
                             label_horizon=10).reindex(close.index)

ar = close.pct_change()
b = estimate_payoff_ratio(backtest_signal(close, side)["returns"])

variants = {
    "primary (raw)":        side,
    "meta-filtered":        apply_meta(side, proba, threshold=0.5),
    "meta-sized (normal)":  apply_meta(side, proba, size=bet_size(proba, "normal")),
    "meta + half-Kelly":    apply_meta(side, proba, size=kelly_size(proba, b, 0.5)),
    "vol-targeted 12%":     side * vol_target_scalar(ar, target_ann_vol=0.12),
    "drawdown-throttled":   drawdown_throttle(side, ar, max_dd=0.15, throttle=0.5),
}

print(f"\n{'variant':22s} {'return':>9s} {'Sharpe':>7s} {'Sortino':>8s} {'maxDD':>8s}")
for name, sig in variants.items():
    s = backtest_signal(close.loc[sig.index], sig, fee_bps=1.0)["stats"]
    print(f"{name:22s} {s['total_return']:9.1%} {s['sharpe']:7.2f} "
          f"{s['sortino']:8.2f} {s['max_drawdown']:8.1%}")

print("\nSizing rarely maximizes raw return — it maximizes RISK-ADJUSTED return: "
      "watch how Sortino rises and max drawdown shrinks.")
