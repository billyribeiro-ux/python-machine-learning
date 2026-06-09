"""
quantlab.advanced.portfolio
===========================

**Portfolio construction** — turning many strategies/assets into one book.

The classical answer (Markowitz mean-variance) is famously fragile: it inverts
the covariance matrix, and small estimation errors in correlations explode into
wild, concentrated weights. Two robust alternatives that practitioners actually
use:

* :func:`inverse_variance` — ignore correlations entirely; weight each asset by
  1/variance. Simple, surprisingly hard to beat, and the building block of HRP.
* :func:`hrp_weights` — **Hierarchical Risk Parity** (López de Prado, 2016).
  Instead of inverting Σ, HRP (1) clusters assets by correlation distance,
  (2) reorders the covariance matrix so similar assets sit together
  (quasi-diagonalization), and (3) splits risk top-down through the cluster
  tree (recursive bisection). No matrix inversion → no error amplification;
  out-of-sample it routinely beats mean-variance and equal weight on risk.

Both take a returns DataFrame (columns = assets) and produce long-only weights
that sum to 1. Estimation is on whatever window you pass — for live use, pass
*trailing* returns only (the usual causality rule applies to portfolio
construction too).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def inverse_variance(returns: pd.DataFrame) -> pd.Series:
    """Inverse-variance weights: w_i ∝ 1/σ_i². The 'ignore correlations'
    baseline that anchors HRP's within-cluster allocation."""
    var = returns.var(ddof=1)
    iv = 1.0 / var.replace(0.0, np.nan)
    w = iv / iv.sum()
    return w.fillna(0.0).rename("weight")


def _cluster_variance(cov: pd.DataFrame, members: list) -> float:
    """Variance of a cluster under inverse-variance weights of its members —
    the risk measure recursive bisection splits between sibling clusters."""
    sub = cov.loc[members, members]
    iv = 1.0 / np.diag(sub)
    w = iv / iv.sum()
    return float(w @ sub.to_numpy() @ w)


def hrp_weights(returns: pd.DataFrame, method: str = "single") -> pd.Series:
    """Hierarchical Risk Parity weights from a returns panel.

    Steps (the three-act structure of the 2016 paper):

    1. **Tree clustering** — distance d_ij = sqrt((1 − ρ_ij)/2) turns
       correlation into a metric (0 = identical, 1 = uncorrelated, √1 ≈
       anti-correlated); hierarchical ``linkage`` builds the asset tree.
    2. **Quasi-diagonalization** — ``leaves_list`` reads the tree's leaf order,
       placing similar assets adjacently so covariance is near-block-diagonal.
    3. **Recursive bisection** — start with weight 1 on the whole list; split
       it in half, allocate between halves *inversely to their cluster
       variances* (α = 1 − v₁/(v₁+v₂)), recurse until single assets remain.

    No inversion of Σ anywhere — only diagonals and small quadratic forms — so
    the weights are stable under estimation noise. Long-only, sums to 1.
    """
    rets = returns.dropna(how="any")
    if rets.shape[1] < 2:
        return pd.Series(1.0, index=returns.columns, name="weight")

    corr = rets.corr()
    cov = rets.cov()

    # 1) Correlation -> distance -> hierarchical tree.
    dist = np.sqrt(0.5 * (1.0 - corr)).clip(lower=0.0)
    condensed = squareform(dist.to_numpy(), checks=False)
    link = linkage(condensed, method=method)

    # 2) Quasi-diagonal ordering: similar assets end up adjacent.
    order = [corr.columns[i] for i in leaves_list(link)]

    # 3) Recursive bisection over the ordered list.
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left, right = cl[:mid], cl[mid:]
            v_l = _cluster_variance(cov, left)
            v_r = _cluster_variance(cov, right)
            alpha = 1.0 - v_l / (v_l + v_r) if (v_l + v_r) > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt

    return weights.reindex(returns.columns).fillna(0.0).rename("weight")


def portfolio_backtest(returns: pd.DataFrame, weight_fn=hrp_weights,
                       lookback: int = 252, rebalance_every: int = 21) -> dict:
    """Walk-forward portfolio backtest: every ``rebalance_every`` bars, compute
    weights from the TRAILING ``lookback`` window only (causal), hold them for
    the next block. Returns the portfolio return series and the standard stats
    bundle — so you can compare HRP vs inverse-variance vs equal weight on the
    same data honestly.
    """
    from quantlab.backtest.stats import compute_stats

    rets = returns.fillna(0.0)
    n = len(rets)
    port = pd.Series(0.0, index=rets.index)
    weights_log = {}

    t = lookback
    while t < n:
        window = rets.iloc[t - lookback:t]
        w = weight_fn(window)
        stop = min(t + rebalance_every, n)
        block = rets.iloc[t:stop]
        port.iloc[t:stop] = block.to_numpy() @ w.reindex(rets.columns).fillna(0.0).to_numpy()
        weights_log[rets.index[t]] = w
        t = stop

    return {"returns": port, "weights": weights_log, "stats": compute_stats(port)}
