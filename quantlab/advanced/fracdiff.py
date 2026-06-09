"""
quantlab.advanced.fracdiff
==========================

**Fractional differentiation** (López de Prado, *Advances in Financial Machine
Learning*, Ch. 5) — the answer to a dilemma every ML-for-markets pipeline hits:

* Prices are non-stationary → models trained on them chase a moving target.
* Returns (first difference, d=1) are stationary → but differencing erases the
  *memory* (the level information) that carries most of the predictive signal.

Integer differencing forces an all-or-nothing choice. Fractional differencing
lets you difference by a real-valued amount ``d ∈ (0, 1)``: just enough to pass
a stationarity test while preserving maximal memory. Empirically, many liquid
price series become stationary around ``d ≈ 0.3–0.6`` — keeping far more signal
than returns do.

The math: the fractional difference operator ``(1-B)^d`` (B = backshift)
expands into an infinite series of weights

    w_0 = 1,   w_k = -w_{k-1} · (d - k + 1) / k

so the differenced series is ``x̃_t = Σ_k w_k · x_{t-k}``. For 0<d<1 the weights
decay slowly (long memory); we use the **fixed-width window** method — truncate
the weights once |w_k| < ``threshold`` — so every output value uses the same
number of lags (no variable look-back drift) and the transform is strictly
causal (uses only past values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fracdiff_weights(d: float, threshold: float = 1e-5,
                     max_width: int = 10_000) -> np.ndarray:
    """Weights of (1-B)^d, truncated where |w_k| drops below ``threshold``.

    Returned oldest-lag-last (w_0 first). For d=0 this is [1] (identity); for
    d=1 it is [1, -1] (the plain first difference) — both useful sanity anchors.
    """
    if d < 0:
        raise ValueError("d must be >= 0")
    w = [1.0]
    k = 1
    while k < max_width:
        w_k = -w[-1] * (d - k + 1.0) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.asarray(w, dtype="float64")


def fracdiff(series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Fixed-width-window fractional differentiation of a series.

    Strictly causal: x̃_t depends only on x_{t-K+1..t}. The first K-1 values are
    NaN (honest warmup). Input may be a pandas Series (index preserved) or any
    1-D array-like.
    """
    s = pd.Series(series).astype("float64")
    w = fracdiff_weights(d, threshold)
    K = len(w)
    x = s.to_numpy()
    n = x.shape[0]
    out = np.full(n, np.nan)
    if n >= K:
        # Convolve: out[t] = sum_k w[k] * x[t-k]. Flip the weights so a sliding
        # dot-product implements the sum directly — vectorized, no Python loop.
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(x, K)          # (n-K+1, K), oldest..newest
        out[K - 1:] = windows @ w[::-1]
    return pd.Series(out, index=s.index, name=f"fracdiff_{d}")


def adf_tstat(series, lags: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic (constant, no trend) via OLS.

    Regress Δx_t on [1, x_{t-1}, Δx_{t-1..t-lags}] and return the t-stat of the
    x_{t-1} coefficient. More negative = stronger evidence of stationarity.
    Reference critical values (constant case): 1%≈−3.43, 5%≈−2.86, 10%≈−2.57.

    This is a teaching implementation (fixed lags, no automatic lag selection,
    no exact p-values). For production inference use ``statsmodels.tsa.adfuller``
    — but knowing the regression behind the test is what lets you trust it.
    """
    x = pd.Series(series).dropna().to_numpy(dtype="float64")
    dx = np.diff(x)
    n = len(dx)
    if n <= lags + 2:
        return 0.0
    # Build the design matrix: const, lagged level, lagged differences.
    y = dx[lags:]
    cols = [np.ones(n - lags), x[lags:-1]]
    for i in range(1, lags + 1):
        cols.append(dx[lags - i:-i])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[1, 1])
    return float(beta[1] / se) if se > 0 else 0.0


# Critical values for quick verdicts (constant, no trend). MacKinnon (1994).
ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def min_fracdiff_order(series, d_grid=None, threshold: float = 1e-4,
                       crit: float = -2.86) -> tuple[float, pd.DataFrame]:
    """Find the smallest d that makes the series pass the ADF test at ``crit``.

    Returns ``(d_star, table)`` where ``table`` has the ADF t-stat and the
    correlation with the original series (the memory retained) for each d. This
    is THE plot from AFML Ch. 5: stationarity rises with d while memory falls —
    you pick the corner.
    """
    s = pd.Series(series).astype("float64")
    d_grid = list(d_grid) if d_grid is not None else [round(0.1 * i, 1) for i in range(11)]
    rows = []
    d_star = None
    for d in d_grid:
        fd = fracdiff(s, d, threshold).dropna()
        if len(fd) < 50:
            continue
        t = adf_tstat(fd)
        corr = float(s.reindex(fd.index).corr(fd)) if d > 0 else 1.0
        rows.append({"d": d, "adf_t": t, "memory_corr": corr})
        if d_star is None and t < crit:
            d_star = d
    table = pd.DataFrame(rows).set_index("d")
    return (d_star if d_star is not None else 1.0), table
