"""End-to-end strategy dossier: one call runs the WHOLE honesty pipeline and
prints a go/no-go verdict.

Optimization (hold-out) -> Deflated Sharpe -> PBO -> CPCV fan -> cost
sensitivity -> decision. This is the final gate before risking capital.

Run:  PYTHONPATH=. python code/strategy_dossier.py
"""
import warnings
warnings.filterwarnings("ignore")

from quantlab.data import get_provider
from quantlab.research import strategy_report
from quantlab.research.templates import TEMPLATES

ohlcv = get_provider("yahoo").get_ohlcv("SPY", start="2008-01-01")
build, space, invalid = TEMPLATES["trend_pullback"]

rep = strategy_report(
    ohlcv, build, space, invalid=invalid,
    n_trials=60, pbo_configs=40, s_blocks=10,
    make_tearsheet=False,          # set True to also write tearsheet.html
)

print("=" * 64)
print(rep.summary())
print("=" * 64)
print("\nBest params:", {k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in rep.best_params.items()})
print("\nCost sensitivity:")
print(rep.cost_table.round(3))
print(f"\nFull-period stats: return={rep.full_stats['total_return']:.1%} "
      f"Sharpe={rep.full_stats['sharpe']:.2f} maxDD={rep.full_stats['max_drawdown']:.1%}")
print("\nThe verdict combines six checks; a single critical failure (negative "
      "hold-out Sharpe, low Deflated Sharpe, or high PBO) forces NO-GO — exactly "
      "what you want a research process to do.")
