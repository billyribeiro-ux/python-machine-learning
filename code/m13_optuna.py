"""Module 13 — Optuna: optimize strategy settings (single + multi-objective).

Run:  python code/m13_optuna.py     (requires optuna: pip install optuna)
"""
from quantlab.data import get_provider
from quantlab.ml.optimize import optimize_combo, optimize_combo_multiobjective

close = get_provider("yahoo").get_ohlcv("SPY", start="2010-01-01")["close"]

# 1) Single-objective: maximize Sharpe over the combo's settings.
print("=== Single-objective (maximize Sharpe) ===")
study = optimize_combo(close, n_trials=60)
print("Best Sharpe:", round(study.best_value, 3))
print("Best params:", {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in study.best_params.items()})

# 2) Multi-objective: maximize Sharpe AND minimize drawdown -> Pareto front.
print("\n=== Multi-objective (Sharpe up, |drawdown| down) ===")
mstudy = optimize_combo_multiobjective(close, n_trials=80)
front = sorted(mstudy.best_trials, key=lambda t: -t.values[0])[:5]
print("Pareto front (top 5 by Sharpe):")
for t in front:
    print(f"  sharpe={t.values[0]:.3f}  maxDD={t.values[1]:.3f}  "
          f"fast={t.params['fast']} slow={t.params['slow']}")
print("\nPick from the front by your risk tolerance — optimization informs, "
      "judgement decides. Always confirm the choice truly out-of-sample.")
