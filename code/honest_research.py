"""Honest research: optimize a strategy WITHOUT fooling yourself, then run a
walk-forward ML model whose every prediction is out-of-sample.

This is the principal-engineer workflow: search hard, then prove the result is
more than luck (hold-out + Deflated Sharpe), and never let a model see its own
test data.

Run:  PYTHONPATH=. python code/honest_research.py   (requires optuna)
"""
import warnings
warnings.filterwarnings("ignore")

from quantlab.data import get_provider
from quantlab.research import optimize_strategy
from quantlab.research.templates import TEMPLATES

provider = get_provider("yahoo")

# ----- 1) Honest strategy optimization ------------------------------------
ohlcv = provider.get_ohlcv("SPY", start="2008-01-01")
build, space, invalid = TEMPLATES["trend_pullback"]

res = optimize_strategy(ohlcv, build, space, invalid=invalid,
                        n_trials=60, holdout=0.25, n_folds=4)
print("=== Honest optimization (trend_pullback on SPY) ===")
print("best params  :", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in res.best_params.items()})
print(f"train  Sharpe: {res.train_stats['sharpe']:.2f}")
print(f"HOLD-OUT Sharpe (the honest number): {res.holdout_stats['sharpe']:.2f}")
print(f"in-sample -> out-of-sample gap     : {res.is_oos_gap:.2f}")
print(f"per-fold train Sharpes (robustness): {[round(f, 2) for f in res.fold_sharpes]}")
print(f"Deflated Sharpe ({res.n_trials} trials): {res.deflated_sharpe:.3f}")
print(f"verdict: {'TRUSTWORTHY' if res.trustworthy else 'LIKELY OVERFIT'}\n")

# ----- 2) Walk-forward, out-of-sample machine learning --------------------
from sklearn.metrics import roc_auc_score
from quantlab.ml import (assemble_dataset, triple_barrier_labels,
                         walk_forward_predict, oos_signal)
from quantlab.backtest import backtest_signal

df = provider.get_ohlcv("QQQ", start="2008-01-01")
close = df["close"]
lab = triple_barrier_labels(close, horizon=10, upper=2.0, lower=2.0)
y = (lab["label"] > 0).astype(float).where(lab["label"].notna())
X, y = assemble_dataset(df, y)

print("=== Walk-forward ML (QQQ, triple-barrier labels) ===")
for kind in ["logistic", "xgboost", "lightgbm"]:
    proba = walk_forward_predict(X, y, kind=kind, n_splits=5, label_horizon=10)
    m = proba.notna()
    auc = roc_auc_score(y[m], proba[m]) if y[m].nunique() > 1 else float("nan")
    sig = oos_signal(proba, long_th=0.55)
    s = backtest_signal(close.loc[sig.index], sig, fee_bps=1.0)["stats"]
    print(f"  {kind:9s} OOS AUC={auc:.3f}  OOS ret={s['total_return']:6.1%}  "
          f"Sharpe={s['sharpe']:.2f}")
print("\nAUC ~0.5 is honest — markets are hard. Edge shows up in PnL with "
      "confidence-gated sizing, not in a flashy accuracy number.")

# ----- 3) Capstone: Probability of Backtest Overfitting (CSCV) + CPCV ------
import numpy as np
from quantlab.research import pbo_for_template, cpcv_sharpe_distribution
from quantlab.backtest import build_signal, backtest_signal
from quantlab.research.templates import trend_pullback

print("\n=== Probability of Backtest Overfitting (trend_pullback on SPY) ===")
pbo = pbo_for_template(ohlcv, build, space, invalid=invalid, n_configs=40, s_blocks=10)
print(f"PBO = {pbo.pbo:.1%}  (lower is better; ~50% = selection is luck)")
print(f"P(out-of-sample loss) = {pbo.prob_oos_loss:.1%}  "
      f"degradation slope = {pbo.degradation_slope:.2f}")
print("verdict:", pbo.verdict)

print("\n=== CPCV: distribution of out-of-sample Sharpe across paths ===")
cfg = trend_pullback({"fast": 20, "slow": 150, "rsi_w": 14,
                      "rsi_floor": 40, "z_w": 20, "z_entry": 0.5})
rets = backtest_signal(ohlcv["close"], build_signal(ohlcv, cfg))["returns"]
dist = cpcv_sharpe_distribution(rets, n_groups=10, n_test_groups=2)
print(f"OOS Sharpe across {len(dist)} paths: median={np.median(dist):.2f}  "
      f"5th pct (bad luck)={np.percentile(dist, 5):.2f}  worst={dist.min():.2f}")
print("\nJudge a strategy by its bad paths, not its average. PBO is the final "
      "gate before risking real capital.")
