"""
quantlab.advanced.regimes
=========================

**Regime detection**: markets alternate between identifiable states — calm
uptrends, choppy ranges, volatile crashes — and a strategy's edge is rarely
constant across them. Detecting the current regime lets you switch strategies,
scale risk, or stand aside.

We use a **Gaussian Mixture Model** over (return, volatility) features: each
regime is a Gaussian cluster in that space, and the model assigns every bar a
regime probability. (A Hidden Markov Model adds explicit transition dynamics;
the GMM is the right first tool because it needs no extra dependency, trains in
milliseconds, and captures most of the value — the lesson discusses when to
upgrade to an HMM.)

The trap — and the reason this module has two APIs — is **look-ahead**: a model
fitted on the full history "knows" 2020 was a crash when labeling 2019. So:

* :func:`fit_regimes` — fit/label on the whole sample. For *analysis and
  charts only*; loudly not causal.
* :func:`causal_regimes` — expanding-window refit: the label at bar *t* comes
  from a model trained strictly on bars < *t*. Safe to feed into strategies.

Regimes are relabeled so **0 = calmest … k-1 = most volatile** (sorted by each
component's volatility), making labels stable and interpretable across refits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _features(returns: pd.Series, vol_window: int = 20) -> pd.DataFrame:
    """The (return, rolling-vol) feature pair the mixture is fitted on."""
    r = pd.Series(returns).astype("float64")
    feats = pd.DataFrame({
        "ret": r,
        "vol": r.rolling(vol_window).std(),
    })
    return feats.dropna()


def _sorted_by_vol(gmm) -> np.ndarray:
    """Map raw component ids -> ids sorted by component volatility (ascending).

    GMM component numbering is arbitrary (it depends on initialization); sorting
    by the vol-dimension mean makes regime 0 'calm' and regime k-1 'turbulent'
    every time — labels you can write strategy rules against.
    """
    order = np.argsort(gmm.means_[:, 1])          # sort by mean of the vol feature
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap


def fit_regimes(returns: pd.Series, n_regimes: int = 3, vol_window: int = 20,
                seed: int = 42):
    """Fit a GMM on the FULL sample and label every bar. NOT causal.

    Use for understanding and visualization (e.g. shading a price chart by
    regime), never as a strategy input — the fit has seen the future. Returns
    ``(labels, model)`` where labels are vol-sorted ints aligned to ``returns``
    (NaN during the vol warmup).
    """
    from sklearn.mixture import GaussianMixture

    feats = _features(returns, vol_window)
    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                          random_state=seed, n_init=3)
    raw = gmm.fit_predict(feats.to_numpy())
    remap = _sorted_by_vol(gmm)
    labels = pd.Series(remap[raw], index=feats.index, name="regime", dtype="float64")
    return labels.reindex(pd.Series(returns).index), gmm


def causal_regimes(returns: pd.Series, n_regimes: int = 3, vol_window: int = 20,
                   min_train: int = 252, refit_every: int = 63,
                   seed: int = 42) -> pd.Series:
    """Causal regime labels: the label at bar t comes from a model fitted on
    bars strictly before the current refit block.

    Expanding-window protocol: starting after ``min_train`` bars, refit every
    ``refit_every`` bars on all *prior* data and label only the next block.
    The first ``min_train`` bars are NaN (no model existed yet). This is the
    series you may feed into a strategy without lying to yourself.
    """
    from sklearn.mixture import GaussianMixture

    feats = _features(returns, vol_window)
    n = len(feats)
    out = np.full(n, np.nan)
    X = feats.to_numpy()

    start = min_train
    while start < n:
        stop = min(start + refit_every, n)
        gmm = GaussianMixture(n_components=n_regimes, covariance_type="full",
                              random_state=seed, n_init=3)
        gmm.fit(X[:start])                          # ONLY the past
        remap = _sorted_by_vol(gmm)
        out[start:stop] = remap[gmm.predict(X[start:stop])]
        start = stop

    labels = pd.Series(out, index=feats.index, name="regime")
    return labels.reindex(pd.Series(returns).index)


def regime_stats(returns: pd.Series, labels: pd.Series,
                 periods_per_year: float = 252) -> pd.DataFrame:
    """Per-regime performance: how do returns behave inside each state?

    The table that justifies the whole exercise — e.g. 'regime 2 (turbulent)
    has negative mean return and 3x the volatility; my trend strategy only
    earns its Sharpe in regimes 0–1' → scale down or stand aside in 2.
    """
    df = pd.DataFrame({"ret": pd.Series(returns), "regime": pd.Series(labels)})
    df = df.dropna()
    rows = []
    for g, sub in df.groupby("regime"):
        mu, sd = sub["ret"].mean(), sub["ret"].std(ddof=1)
        rows.append({
            "regime": int(g),
            "bars": len(sub),
            "frequency": len(sub) / len(df),
            "ann_return": mu * periods_per_year,
            "ann_vol": sd * np.sqrt(periods_per_year),
            "sharpe": (mu / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        })
    return pd.DataFrame(rows).set_index("regime")
