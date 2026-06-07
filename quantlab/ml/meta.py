"""
quantlab.ml.meta
================

**Meta-labeling** (López de Prado, AFML Ch. 3): a second model that decides
*whether to act* on a primary signal — not which way to bet. The primary model
(any strategy, indicator rule, or human discretion) provides the **side**
(long/short); the meta-model provides the **conviction** P(this bet wins).

Why it works: separating *side* from *size* lets a simple, high-recall primary
generate candidates while the meta-model raises **precision** — filtering out the
bad signals and (via the probability) sizing the good ones. It reliably improves
risk-adjusted returns without touching the primary's logic.

The pipeline:
  1. Primary emits a per-bar ``side`` in {-1, 0, +1}.
  2. :func:`triple_barrier_meta` labels each active bar 1 if holding that side
     for the next ``horizon`` bars would have won (profit barrier / positive
     time-out) and 0 otherwise — the meta-target.
  3. Train a classifier (walk-forward, out-of-sample) on features at those bars
     to predict P(win).
  4. :func:`apply_meta` gates (or sizes) the primary side by that probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_meta(close: pd.Series, side: pd.Series, horizon: int = 10,
                        pt: float = 1.0, sl: float = 1.0,
                        vol_window: int = 20) -> pd.DataFrame:
    """Meta-labels for a primary ``side`` via side-adjusted triple barriers.

    For each bar where ``side != 0``, race a profit barrier (``pt`` × recent vol)
    and a stop barrier (``sl`` × recent vol) over the next ``horizon`` bars,
    measured on the **side-adjusted** return ``side·(close/close_t − 1)``. Label
    1 if the profit barrier is hit first (or the position is profitable at the
    time-out), else 0.

    Returns a DataFrame aligned to ``close`` with columns ``meta_label``,
    ``ret`` (side-adjusted realized return at the touch), and ``bars_held``;
    non-event bars and bars without a full observable horizon are NaN.
    """
    c = close.to_numpy(dtype="float64")
    s = pd.Series(side).reindex(close.index).fillna(0.0).to_numpy()
    n = len(c)
    vol = pd.Series(close).pct_change().rolling(vol_window).std().to_numpy()

    label = np.full(n, np.nan)
    realized = np.full(n, np.nan)
    bars = np.full(n, np.nan)

    for i in range(n):
        if s[i] == 0 or np.isnan(vol[i]):
            continue
        if i + horizon > n - 1:        # need the full horizon to be observable
            continue
        up, dn = pt * vol[i], sl * vol[i]
        end = i + horizon
        outcome, j = None, end
        for k in range(i + 1, end + 1):
            r = s[i] * (c[k] / c[i] - 1.0)     # side-adjusted path return
            if r >= up:
                outcome, j = 1, k
                break
            if r <= -dn:
                outcome, j = 0, k
                break
        if outcome is None:                     # time-out: win iff profitable
            r_end = s[i] * (c[end] / c[i] - 1.0)
            outcome, j = (1 if r_end > 0 else 0), end
        label[i] = outcome
        realized[i] = s[i] * (c[j] / c[i] - 1.0)
        bars[i] = j - i

    return pd.DataFrame({"meta_label": label, "ret": realized, "bars_held": bars},
                        index=close.index)


def meta_dataset(features: pd.DataFrame, meta: pd.DataFrame):
    """Restrict features to event rows (where a meta-label exists) and return
    ``(X, y)`` ready for a classifier. Rows with missing features are dropped."""
    y = meta["meta_label"].dropna()
    X = features.reindex(y.index)
    data = X.join(y.rename("__meta__")).replace([np.inf, -np.inf], np.nan).dropna()
    return data.drop(columns="__meta__"), data["__meta__"]


def apply_meta(side: pd.Series, meta_proba: pd.Series, threshold: float = 0.5,
               size: pd.Series | None = None) -> pd.Series:
    """Combine the primary ``side`` with meta probabilities into a final position.

    * Filter mode (default): take the bet only when ``meta_proba > threshold``,
      i.e. ``final = side · 1[proba > threshold]``.
    * Size mode: pass a ``size`` series (e.g. from
      :func:`quantlab.backtest.sizing.bet_size`) to scale continuously —
      ``final = side · size`` where a probability exists, else flat.

    Bars without a meta probability (non-events / un-predicted) are flat.
    """
    side = pd.Series(side).astype("float64")
    proba = pd.Series(meta_proba).reindex(side.index)
    if size is not None:
        gate = pd.Series(size).reindex(side.index).fillna(0.0)
    else:
        gate = (proba > threshold).astype("float64")
    gate = gate.where(proba.notna(), 0.0)
    return side * gate
