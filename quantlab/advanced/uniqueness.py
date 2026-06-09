"""
quantlab.advanced.uniqueness
============================

**Sample uniqueness weights** (López de Prado, AFML Ch. 4) — the fix for a bias
almost every ML-for-trading pipeline silently carries.

Triple-barrier labels span *intervals*: the label decided at bar t describes the
path over t+1..t+h. Consecutive labels therefore **overlap** — twenty labels
opened during the same week mostly describe the *same* market move. A model
trained on them with equal weight effectively sees one event twenty times: it
overfits that episode, and every IID assumption behind cross-validation and
bagging quietly breaks.

The remedy: weight each sample by how much *unique* information it carries.

* **Concurrency** c_t = number of labels whose interval contains bar t.
* A label's **average uniqueness** = mean over its lifespan of 1/c_t.
  A label alone in its window scores 1.0; one of k fully-overlapping labels
  scores 1/k.

Feed the result into ``model.fit(X, y, sample_weight=...)`` (every sklearn /
XGBoost / LightGBM estimator accepts it). Same data, same model — but the
optimizer now sees each market episode roughly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_concurrency(index: pd.Index, event_starts: np.ndarray,
                      event_ends: np.ndarray) -> pd.Series:
    """c_t: how many label intervals are 'alive' at each bar.

    ``event_starts`` / ``event_ends`` are integer positions into ``index``
    (inclusive on both ends). Vectorized via a difference array: +1 at each
    start, −1 after each end, cumulative-sum — O(n + events), no loops over
    bars.
    """
    n = len(index)
    diff = np.zeros(n + 1)
    for s, e in zip(event_starts, event_ends):
        s = int(max(0, s))
        e = int(min(n - 1, e))
        if e < s:
            continue
        diff[s] += 1.0
        diff[e + 1] -= 1.0
    return pd.Series(np.cumsum(diff[:-1]), index=index, name="concurrency")


def average_uniqueness(index: pd.Index, event_starts: np.ndarray,
                       event_ends: np.ndarray) -> pd.Series:
    """Average uniqueness of each label: mean of 1/c_t over its own lifespan.

    Returns a Series indexed by the event start bars, in (0, 1]. Use directly as
    ``sample_weight`` (optionally rescaled so weights average 1, which keeps
    learning rates comparable)."""
    conc = label_concurrency(index, event_starts, event_ends).to_numpy()
    out = np.empty(len(event_starts))
    for i, (s, e) in enumerate(zip(event_starts, event_ends)):
        s = int(max(0, s))
        e = int(min(len(conc) - 1, e))
        c = conc[s:e + 1]
        c = c[c > 0]
        out[i] = float(np.mean(1.0 / c)) if len(c) else 0.0
    starts_idx = pd.Index([index[int(s)] for s in event_starts])
    return pd.Series(out, index=starts_idx, name="uniqueness")


def uniqueness_from_labels(labels: pd.DataFrame, horizon: int) -> pd.Series:
    """Convenience for QuantLab's triple-barrier output.

    ``labels`` is the frame from ``triple_barrier_labels`` /
    ``triple_barrier_meta`` (rows with a non-NaN label are events; the
    ``touch_idx``/``bars_held`` column says how long each ran — falling back to
    ``horizon`` if absent). Returns per-event uniqueness aligned to the event
    bars; reindex onto your training X and fillna(0) for non-events.
    """
    lab_col = "label" if "label" in labels.columns else "meta_label"
    mask = labels[lab_col].notna()
    positions = np.flatnonzero(mask.to_numpy())
    span_col = "touch_idx" if "touch_idx" in labels.columns else "bars_held"
    if span_col in labels.columns:
        spans = labels.loc[mask, span_col].fillna(horizon).to_numpy()
    else:
        spans = np.full(len(positions), horizon, dtype="float64")
    ends = positions + spans.astype(int)
    return average_uniqueness(labels.index, positions, ends)
