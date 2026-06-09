"""
quantlab.advanced.pairs
=======================

**Statistical arbitrage on pairs** — the cleanest "real quant" strategy family:
find two assets whose *spread* is stationary (cointegration), bet on the spread
mean-reverting, stay market-neutral. Everything here is built from first
principles in NumPy so you can see exactly where each number comes from:

* :func:`rolling_hedge_ratio` — causal rolling-OLS β (the simple baseline).
* :class:`KalmanHedge` / :func:`kalman_hedge_ratio` — a 2-state Kalman filter
  ``y_t = β_t·x_t + α_t + ε`` with random-walk states. The Kalman filter is the
  optimal online estimator for a drifting hedge ratio: it adapts instantly,
  needs no window length, and is strictly causal by construction.
* :func:`engle_granger` — the two-step cointegration check: OLS of y on x, then
  an ADF test on the residual. (Residual-based critical values are *stricter*
  than plain ADF because β was itself estimated — a subtlety most retail
  implementations miss.)
* :func:`half_life` — Ornstein-Uhlenbeck mean-reversion half-life of the
  spread: how long a divergence takes to close. The single most useful number
  for choosing holding periods and z-score windows.
* :func:`pairs_backtest` — a leak-free spread backtest: positions decided at
  *t* earn the spread change from *t* to *t+1* with β frozen at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fracdiff import adf_tstat

# Engle-Granger residual-test critical values (2 variables, constant;
# MacKinnon). Stricter than plain ADF because the cointegrating vector was
# estimated from the same data.
EG_CRIT = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


# --------------------------------------------------------------------------- #
# Hedge-ratio estimation
# --------------------------------------------------------------------------- #
def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int = 60) -> pd.Series:
    """Causal rolling-OLS β of y on x (with intercept): β_t fitted on the
    trailing ``window`` bars ending at t. NaN during warmup."""
    y = pd.Series(y).astype("float64")
    x = pd.Series(x).reindex(y.index).astype("float64")
    # Rolling covariance/variance form of OLS slope — vectorized, no loop.
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return (cov / var).rename("beta")


@dataclass
class KalmanHedge:
    """Online Kalman estimate of ``y_t = β_t x_t + α_t + ε_t``.

    State s_t = [β_t, α_t] follows a random walk with covariance Q; the
    observation noise variance is R. ``delta`` controls how fast β may drift
    (Q = delta/(1-delta)·I): larger delta = faster adaptation, noisier β.

    This is the production-grade replacement for rolling OLS: no window
    parameter, instant adaptation after regime shifts, and an exact one-bar-
    ahead prediction error (the innovation) that itself is a tradeable signal.
    """

    delta: float = 1e-4
    r: float = 1e-3

    def fit(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        y = pd.Series(y).astype("float64")
        x = pd.Series(x).reindex(y.index).astype("float64")
        n = len(y)
        beta = np.zeros(n)
        alpha = np.zeros(n)
        innovation = np.zeros(n)

        s = np.zeros(2)                       # [beta, alpha]
        P = np.eye(2) * 1e3                   # diffuse prior: we know nothing
        Q = np.eye(2) * (self.delta / (1.0 - self.delta))
        R = self.r

        yv, xv = y.to_numpy(), x.to_numpy()
        for t in range(n):
            # Predict: random-walk state, so the mean is unchanged.
            P = P + Q
            H = np.array([xv[t], 1.0])
            # Innovation: how wrong our one-step-ahead prediction of y was.
            e = yv[t] - H @ s
            S = H @ P @ H + R                 # innovation variance
            K = (P @ H) / S                   # Kalman gain
            s = s + K * e                     # update state toward the surprise
            P = P - np.outer(K, H @ P)        # shrink uncertainty
            beta[t], alpha[t] = s
            innovation[t] = e
        return pd.DataFrame({"beta": beta, "alpha": alpha,
                             "innovation": innovation}, index=y.index)


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, delta: float = 1e-4,
                       r: float = 1e-3) -> pd.Series:
    """Convenience wrapper: just the Kalman β series."""
    return KalmanHedge(delta=delta, r=r).fit(y, x)["beta"]


# --------------------------------------------------------------------------- #
# Cointegration & mean-reversion diagnostics
# --------------------------------------------------------------------------- #
def engle_granger(y: pd.Series, x: pd.Series, lags: int = 1) -> dict:
    """Two-step Engle-Granger cointegration check.

    Step 1: OLS ``y = α + β·x`` on the full sample (the cointegrating vector).
    Step 2: ADF t-stat on the residual. Compare against :data:`EG_CRIT` —
    NOT the plain ADF table, because estimating β biases the residual toward
    stationarity. Returns the verdict and everything behind it.
    """
    y = pd.Series(y).astype("float64")
    x = pd.Series(x).reindex(y.index).astype("float64")
    mask = y.notna() & x.notna()
    yv, xv = y[mask].to_numpy(), x[mask].to_numpy()
    X = np.column_stack([np.ones(len(xv)), xv])
    (a, b), *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = pd.Series(yv - (a + b * xv), index=y[mask].index)
    t = adf_tstat(resid, lags=lags)
    return {
        "alpha": float(a), "beta": float(b), "adf_t": t,
        "crit": dict(EG_CRIT),
        "cointegrated_5pct": t < EG_CRIT["5%"],
        "residual": resid,
    }


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion, in bars.

    Fit Δs_t = c + γ·s_{t-1} + ε; the OU speed is θ = −γ and the half-life is
    ln(2)/θ. Infinite (returned as ``inf``) if the spread doesn't mean-revert
    (γ ≥ 0). Rule of thumb: set the z-score window ≈ the half-life, and don't
    trade pairs whose half-life is longer than your patience.
    """
    s = pd.Series(spread).dropna().to_numpy(dtype="float64")
    if len(s) < 20:
        return float("inf")
    ds = np.diff(s)
    lag = s[:-1]
    X = np.column_stack([np.ones(len(lag)), lag])
    (c, gamma), *_ = np.linalg.lstsq(X, ds, rcond=None)
    if gamma >= 0:
        return float("inf")
    return float(np.log(2.0) / -gamma)


# --------------------------------------------------------------------------- #
# The strategy
# --------------------------------------------------------------------------- #
def pairs_backtest(y: pd.Series, x: pd.Series, hedge: str = "rolling",
                   window: int = 250, z_window: int = 30,
                   entry: float = 2.0, exit: float = 0.5,
                   fee_bps: float = 1.0, delta: float = 1e-7) -> dict:
    """Backtest a z-score mean-reversion strategy on the spread of two assets.

    **The estimation lesson baked into the defaults** (we measured this, you
    should too): hedge-ratio error multiplies *price levels*, so a β off by
    ±0.02 on a $100 stock injects ±$2 of noise into a spread whose true
    oscillation may be smaller than that — your z-score then trades estimation
    noise, not mean reversion. Two consequences:

    * With a *stable* relationship, prefer **rolling OLS with a LONG window**
      (250+ bars): on simulated cointegrated pairs, window=60 trades at a loss
      while window=500 earns Sharpe > 1 on the identical spread.
    * Reserve the **Kalman** hedge for genuinely *drifting* betas, and slow it
      down (``delta`` of 1e-6…1e-8): a fast filter (1e-4) absorbs the spread
      into its own state — it "explains away" the very signal you trade.

    Mechanics (all causal):
      * β_t from rolling OLS or the Kalman filter (information through t only).
      * spread s_t = y_t − β_t·x_t; z_t = rolling z-score of s over ``z_window``.
      * Enter short-spread at z > ``entry`` (short y, long β·x), long-spread at
        z < −``entry``; exit when |z| < ``exit``. Decisions at t act at t+1.
      * Per-bar PnL holds β FROZEN at decision time:
        pnl_t = pos_{t−1} · [(y_t − y_{t−1}) − β_{t−1}(x_t − x_{t−1})],
        normalized by the gross notional |y|+|β|·|x| at t−1 → a true return.
      * Costs charged on position changes (both legs).

    Returns dict with ``returns``, ``positions``, ``spread``, ``zscore``,
    ``beta`` and the standard ``stats`` bundle.
    """
    from quantlab.backtest.stats import compute_stats

    y = pd.Series(y).astype("float64")
    x = pd.Series(x).reindex(y.index).astype("float64")

    if hedge == "kalman":
        beta = kalman_hedge_ratio(y, x, delta=delta)
        warm = max(z_window, 10)
        beta.iloc[:warm] = np.nan        # let the diffuse prior settle
    elif hedge == "rolling":
        beta = rolling_hedge_ratio(y, x, window=window)
    else:
        raise ValueError("hedge must be 'kalman' or 'rolling'")

    spread = y - beta * x
    mu = spread.rolling(z_window).mean()
    sd = spread.rolling(z_window).std(ddof=0)
    z = ((spread - mu) / sd).replace([np.inf, -np.inf], np.nan)

    # Stateful entry/exit on the z-score: -1 short spread, +1 long spread.
    zv = z.to_numpy()
    pos = np.zeros(len(zv))
    state = 0.0
    for t in range(len(zv)):
        if np.isnan(zv[t]):
            pos[t] = 0.0
            state = 0.0
            continue
        if state == 0.0:
            if zv[t] > entry:
                state = -1.0
            elif zv[t] < -entry:
                state = 1.0
        elif abs(zv[t]) < exit:
            state = 0.0
        pos[t] = state
    position = pd.Series(pos, index=y.index, name="position")

    # PnL with β and notional frozen at decision time (no rebalancing leak).
    dy = y.diff()
    dx = x.diff()
    beta_prev = beta.shift(1)
    pnl = position.shift(1) * (dy - beta_prev * dx)
    gross_prev = (y.abs() + beta.abs() * x.abs()).shift(1)
    ret = (pnl / gross_prev).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    ret = ret - turnover * (fee_bps / 1e4)

    return {
        "returns": ret,
        "positions": position,
        "spread": spread,
        "zscore": z,
        "beta": beta,
        "stats": compute_stats(ret),
    }
