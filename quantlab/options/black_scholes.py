"""
quantlab.options.black_scholes
==============================

Black-Scholes pricing, the Greeks, and implied volatility — implemented from
scratch in vectorized NumPy. No external options library: understanding these
closed-form expressions is what lets you build options scanners and risk tools
that you actually trust.

Conventions
-----------
* ``S`` spot, ``K`` strike, ``T`` time to expiry in *years*, ``r`` risk-free
  rate (annual, continuously compounded), ``sigma`` annualized volatility,
  ``q`` continuous dividend yield.
* All functions are vectorized: pass scalars or NumPy arrays (a whole chain).
* ``option_type`` is ``"call"`` or ``"put"``.

These are the textbook formulas; the value is in having them tested, vectorized,
and wired to the provider's options chain (Module 15).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm  # SciPy ships with scikit-learn's stack


def _d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    # Guard against T<=0 / sigma<=0 producing nan/inf in downstream math.
    T = np.maximum(T, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes(-Merton) fair value of a European option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> dict:
    """Return delta, gamma, vega, theta, rho — the option's risk sensitivities.

    * **delta** — sensitivity to spot (hedge ratio).
    * **gamma** — sensitivity of delta to spot (convexity).
    * **vega**  — sensitivity to volatility (per 1.00 vol; divide by 100 for 1%).
    * **theta** — time decay per year (divide by 365 for per-day).
    * **rho**   — sensitivity to interest rates.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T = map(np.asarray, (S, K, T))
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                tol=1e-6, max_iter=100):
    """Solve for the volatility that reproduces a market ``price``.

    Uses bisection — slower than Newton-Raphson but rock-solid (it can't diverge,
    and vega->0 deep ITM/OTM breaks Newton). For a scanner over thousands of
    contracts, robustness beats a few iterations of speed. Returns NaN if the
    price is outside no-arbitrage bounds.
    """
    price = np.asarray(price, dtype="float64")
    lo = np.full_like(price, 1e-6)
    hi = np.full_like(price, 5.0)  # 500% vol upper bound

    # Vectorized bisection: invariant is bs_price(lo) <= target <= bs_price(hi).
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = bs_price(S, K, T, r, mid, option_type, q)
        too_low = val < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
        if np.all(hi - lo < tol):
            break
    iv = 0.5 * (lo + hi)
    # Mark non-invertible prices (below intrinsic / above bound) as NaN.
    iv = np.where((iv <= 1e-6 + 1e-9) | (iv >= 5.0 - 1e-9), np.nan, iv)
    return iv
