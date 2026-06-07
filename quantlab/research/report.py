"""
quantlab.research.report
========================

The capstone: **one function that runs a strategy through the entire pipeline**
and emits a single go/no-go dossier. This is what you'd put in front of a
risk committee — every honesty check the course built, applied at once, with an
explicit verdict and the reasons behind it.

:func:`strategy_report` orchestrates:

1. **Honest optimization** — Optuna over the template's space with a held-out
   test period and a robust per-fold objective (:func:`optimize_strategy`).
2. **Deflated Sharpe** — is the hold-out Sharpe more than luck across all trials?
3. **PBO** — Combinatorially Symmetric CV probability of backtest overfitting.
4. **CPCV fan** — the distribution of out-of-sample Sharpe across many paths;
   we report the median and the 5th-percentile "bad luck" path.
5. **Cost sensitivity** — does the edge survive realistic fees?
6. **Tearsheet** — an optional QuantStats HTML report on the hold-out returns.

The **decision gate** (:func:`evaluate_gate`) is a *pure* function of the
headline numbers, so it is deterministic and unit-tested independently of the
(stochastic) search.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.backtest.engine import backtest_signal
from quantlab.backtest.stats import compute_stats
from quantlab.backtest.strategy import build_signal
from .cpcv import cpcv_sharpe_distribution
from .optimize import optimize_strategy
from .pbo import pbo_for_template


@dataclass
class ReportCheck:
    """One pass/fail gate criterion with the value behind it."""

    name: str
    passed: bool
    value: float
    detail: str
    critical: bool = False


@dataclass
class StrategyReport:
    """The full dossier — a self-describing, loggable go/no-go record."""

    decision: str                       # "GO" | "CONDITIONAL" | "NO-GO"
    checks: list[ReportCheck]
    best_params: dict
    full_stats: dict
    holdout_stats: dict
    deflated_sharpe: float
    pbo: float
    is_oos_gap: float
    cpcv: dict                          # {median, p05, worst, paths}
    cost_table: pd.DataFrame
    holdout_returns: pd.Series
    optimize_history: list = field(default_factory=list)
    tearsheet_path: str | None = None

    def summary(self) -> str:
        lines = [f"DECISION: {self.decision}",
                 f"  hold-out Sharpe : {self.holdout_stats['sharpe']:.2f}",
                 f"  deflated Sharpe : {self.deflated_sharpe:.3f}",
                 f"  PBO             : {self.pbo:.1%}",
                 f"  IS->OOS gap     : {self.is_oos_gap:.2f}",
                 f"  CPCV Sharpe     : median {self.cpcv['median']:.2f}, "
                 f"5th pct {self.cpcv['p05']:.2f}, worst {self.cpcv['worst']:.2f}",
                 "  checks:"]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            flag = " (critical)" if c.critical else ""
            lines.append(f"    [{mark}] {c.name}{flag}: {c.detail}")
        return "\n".join(lines)


def evaluate_gate(holdout_sharpe: float, deflated_sharpe: float, pbo: float,
                  cpcv_p05: float, is_oos_gap: float, cost_sharpe: float,
                  *, dsr_min: float = 0.6, pbo_max: float = 0.5,
                  gap_max: float = 1.0) -> tuple[str, list[ReportCheck]]:
    """Pure decision logic: map headline numbers to checks + a verdict.

    Critical checks (a failure forces NO-GO): positive hold-out Sharpe, Deflated
    Sharpe above ``dsr_min`` (unlikely to be luck), and PBO below ``pbo_max``
    (selection better than chance). Non-critical checks (failures downgrade GO to
    CONDITIONAL): a contained in-sample→out-of-sample gap, a non-negative
    5th-percentile CPCV path, and an edge that survives realistic costs.
    """
    checks = [
        ReportCheck("Hold-out Sharpe > 0", holdout_sharpe > 0, holdout_sharpe,
                    f"{holdout_sharpe:.2f}", critical=True),
        ReportCheck(f"Deflated Sharpe > {dsr_min}", deflated_sharpe > dsr_min,
                    deflated_sharpe, f"{deflated_sharpe:.3f}", critical=True),
        ReportCheck(f"PBO < {pbo_max:.0%}", pbo < pbo_max, pbo,
                    f"{pbo:.1%}", critical=True),
        ReportCheck(f"IS->OOS gap < {gap_max}", is_oos_gap < gap_max, is_oos_gap,
                    f"{is_oos_gap:.2f}"),
        ReportCheck("CPCV 5th-pct Sharpe >= 0", cpcv_p05 >= 0.0, cpcv_p05,
                    f"{cpcv_p05:.2f}"),
        ReportCheck("Cost-robust Sharpe > 0", cost_sharpe > 0, cost_sharpe,
                    f"{cost_sharpe:.2f}"),
    ]
    critical_ok = all(c.passed for c in checks if c.critical)
    all_ok = all(c.passed for c in checks)
    decision = "GO" if all_ok else ("CONDITIONAL" if critical_ok else "NO-GO")
    return decision, checks


def strategy_report(ohlcv: pd.DataFrame, build, space: dict, invalid=None, *,
                    n_trials: int = 60, holdout: float = 0.25, n_folds: int = 4,
                    s_blocks: int = 10, pbo_configs: int = 40, fee_bps: float = 1.0,
                    periods_per_year: float = 252, cpcv_groups: int = 10,
                    cpcv_test: int = 2, cost_grid=(0.0, 1.0, 5.0, 10.0, 25.0),
                    make_tearsheet: bool = False, tearsheet_path: str = "tearsheet.html",
                    benchmark: pd.Series | None = None, seed: int = 42) -> StrategyReport:
    """Run a strategy template through the whole honesty pipeline → a dossier."""

    # Bake the fee into the template so optimization, PBO, and costs are consistent.
    def build_fee(params, _b=build, _f=float(fee_bps)):
        cfg = _b(params)
        cfg.fee_bps = _f
        return cfg

    # 1) Honest optimization (hold-out + deflated Sharpe).
    opt = optimize_strategy(ohlcv, build_fee, space, invalid=invalid,
                            n_trials=n_trials, holdout=holdout, n_folds=n_folds,
                            periods_per_year=periods_per_year, seed=seed)
    best_cfg = build_fee(opt.best_params)

    # 2) Full-period returns of the chosen config.
    full_ret = backtest_signal(ohlcv["close"], build_signal(ohlcv, best_cfg),
                               fee_bps=fee_bps, periods_per_year=periods_per_year)["returns"]
    full_stats = compute_stats(full_ret, periods_per_year)

    # 3) PBO across the search space.
    pbo_res = pbo_for_template(ohlcv, build_fee, space, invalid=invalid,
                               n_configs=pbo_configs, s_blocks=s_blocks, seed=seed,
                               periods_per_year=periods_per_year)

    # 4) CPCV out-of-sample Sharpe distribution for the chosen config.
    dist = cpcv_sharpe_distribution(full_ret, n_groups=cpcv_groups,
                                    n_test_groups=cpcv_test,
                                    periods_per_year=periods_per_year)
    cpcv = {"median": float(np.median(dist)), "p05": float(np.percentile(dist, 5)),
            "worst": float(dist.min()), "paths": dist}

    # 5) Cost sensitivity.
    rows = []
    for fee in cost_grid:
        r = backtest_signal(ohlcv["close"], build_signal(ohlcv, best_cfg),
                            fee_bps=fee, periods_per_year=periods_per_year)["stats"]
        rows.append({"fee_bps": fee, "total_return": r["total_return"],
                     "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"]})
    cost_table = pd.DataFrame(rows).set_index("fee_bps")
    # Sharpe at ~10bps (or the nearest grid point) for the cost-robustness check.
    ref_fee = min(cost_grid, key=lambda f: abs(f - 10.0))
    cost_sharpe = float(cost_table.loc[ref_fee, "sharpe"])

    # 6) Decision gate (pure).
    decision, checks = evaluate_gate(
        opt.holdout_stats["sharpe"], opt.deflated_sharpe, pbo_res.pbo,
        cpcv["p05"], opt.is_oos_gap, cost_sharpe)

    # 7) Optional QuantStats tearsheet on the hold-out returns.
    ts_path = None
    if make_tearsheet:
        try:
            from quantlab.backtest.tearsheet import html_report
            ts_path = html_report(opt.holdout_returns, benchmark=benchmark,
                                  output=tearsheet_path, title="QuantLab Strategy Report")
        except Exception:
            ts_path = None

    return StrategyReport(
        decision=decision, checks=checks, best_params=opt.best_params,
        full_stats=full_stats, holdout_stats=opt.holdout_stats,
        deflated_sharpe=opt.deflated_sharpe, pbo=pbo_res.pbo,
        is_oos_gap=opt.is_oos_gap, cpcv=cpcv, cost_table=cost_table,
        holdout_returns=opt.holdout_returns, optimize_history=opt.history,
        tearsheet_path=ts_path,
    )
