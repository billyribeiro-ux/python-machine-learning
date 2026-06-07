"""
quantlab.research.metrics
=========================

Statistics for telling **luck from edge** — the numbers a principal-level quant
reaches for before believing any backtest.

The problem: if you try N strategy variants and report the best Sharpe, you have
run N hypothesis tests and kept the luckiest. Its in-sample Sharpe is an
upward-biased estimate of the truth. These tools quantify and correct for that.

* **Probabilistic Sharpe Ratio (PSR)** — the probability that the *true* Sharpe
  exceeds a benchmark, given the observed Sharpe, sample length, and the
  non-normality (skew/kurtosis) of returns. (Bailey & López de Prado, 2012.)
* **Deflated Sharpe Ratio (DSR)** — PSR where the benchmark is the *expected
  maximum* Sharpe you'd see from N independent lucky trials. A strategy only
  passes if it beats what randomness alone would have produced across your search.

All Sharpes here are **per-period** (not annualized): SR = mean(r) / std(r).
Annualization is a presentation choice and cancels inside these formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def per_period_sharpe(returns: pd.Series) -> float:
    """Per-period Sharpe = mean/std of returns (0 if no dispersion)."""
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd) or len(r) < 2:
        return 0.0
    return float(r.mean() / sd)


def probabilistic_sharpe_ratio(sr: float, n: int, skew: float = 0.0,
                               kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``).

    ``kurt`` is the *non-excess* kurtosis (3 for a normal distribution). Longer
    samples and lower benchmark make PSR rise; fat tails and negative skew make
    it fall. Returns a probability in [0, 1].
    """
    if n < 2:
        return 0.0
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_trials_std: float = 1.0) -> float:
    """Expected maximum of ``n_trials`` i.i.d. N(0, sr_trials_std²) Sharpe
    estimates — i.e. the best per-period Sharpe pure luck would hand you across
    that many trials. Uses the standard extreme-value approximation."""
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    e = 0.5772156649015329  # Euler–Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_trials_std * ((1.0 - e) * z1 + e * z2))


def deflated_sharpe_ratio(sr: float, n: int, n_trials: int,
                          sr_trials_std: float, skew: float = 0.0,
                          kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio: PSR against the expected-max-Sharpe benchmark.

    A DSR near 1.0 means the strategy's Sharpe is very unlikely to be a fluke of
    having searched ``n_trials`` configurations; near 0.5 or below means "this is
    about what luck would produce — don't trust it." This is the single most
    important number to report after any parameter search.
    """
    sr0 = expected_max_sharpe(n_trials, sr_trials_std)
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)


def returns_skew_kurt(returns: pd.Series) -> tuple[float, float]:
    """Sample skewness and (non-excess) kurtosis of a returns series, with safe
    fallbacks to the normal values (0, 3) when undefined."""
    r = pd.Series(returns).dropna()
    if len(r) < 4:
        return 0.0, 3.0
    sk = float(r.skew())
    ku = float(r.kurt()) + 3.0  # pandas .kurt() is excess; convert to non-excess
    if not np.isfinite(sk):
        sk = 0.0
    if not np.isfinite(ku):
        ku = 3.0
    return sk, ku
