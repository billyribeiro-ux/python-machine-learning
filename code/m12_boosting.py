"""Module 12 — gradient-boosted alpha (XGBoost + LightGBM) and predictions->PnL.

Run:  python code/m12_boosting.py   (requires xgboost + lightgbm)
"""
import numpy as np
from quantlab.data import get_provider
from quantlab.ml import assemble_dataset, triple_barrier_labels
from quantlab.ml.models import train_xgb, train_lgbm, proba_to_signal
from quantlab.backtest.engine import backtest_signal

df = get_provider("yahoo").get_ohlcv("QQQ", start="2010-01-01")
close = df["close"]

lab = triple_barrier_labels(close, horizon=10, upper=2.0, lower=2.0)
y = (lab["label"] > 0).astype(float).where(lab["label"].notna())
X, y = assemble_dataset(df, y)

for name, trainer in [("XGBoost", train_xgb), ("LightGBM", train_lgbm)]:
    model, info = trainer(X, y, test_size=0.3)
    print(f"\n=== {name} ===  accuracy={info['accuracy']:.3f}  auc={info['auc']:.3f}")

    # Turn confident up-predictions into a long/flat signal, then BACKTEST it
    # (accuracy is not edge — PnL is). The engine applies the one-bar delay.
    signal = proba_to_signal(info["proba"], long_th=0.55)
    bt = backtest_signal(close.loc[signal.index], signal, fee_bps=1.0)
    s = bt["stats"]
    print(f"  out-of-sample: return={s['total_return']:.1%}  "
          f"sharpe={s['sharpe']:.2f}  maxDD={s['max_drawdown']:.1%}")

# Feature importance from the last model (LightGBM).
imp = sorted(zip(X.columns, model.feature_importances_), key=lambda t: -t[1])
print("\nTop features:", [f"{n}({v})" for n, v in imp[:5]])
