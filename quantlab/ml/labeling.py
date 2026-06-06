"""
quantlab.ml.labeling
====================

Labeling: turning "what happened next" into a supervised target — correctly.

Naive labeling ("did price rise over the next 5 days?") ignores *path*: a trade
that drops 8% then closes +0.1% is labeled the same as a smooth +0.1% grind, yet
you'd have been stopped out of the first. **Triple-barrier labeling** (Lopez de
Prado) fixes this: from each bar, race three barriers — an upper (profit-take),
a lower (stop-loss), and a vertical (time limit) — and label by whichever is
touched first. The barriers are sized in volatility units so they adapt to
regime.

This is the single biggest upgrade most retail ML-for-trading pipelines are
missing, and it's the foundation for honest Module 11/12 models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(close: pd.Series,
                          horizon: int = 10,
                          upper: float = 2.0,
                          lower: float = 2.0,
                          vol_window: int = 20) -> pd.DataFrame:
    """Label each bar by the first of three barriers it touches.

    Parameters
    ----------
    close:
        Close price series.
    horizon:
        Vertical barrier: max bars to hold before giving up (time-out).
    upper, lower:
        Profit-take / stop-loss distances in units of rolling daily volatility.
        e.g. ``upper=2`` means "take profit at +2 sigma of recent returns".
    vol_window:
        Lookback for the volatility estimate that sizes the barriers.

    Returns
    -------
    DataFrame with columns:
      * ``label``     : +1 (upper hit first), -1 (lower hit first), 0 (timed out)
      * ``ret``       : the realized return at the touched barrier
      * ``touch_idx`` : integer offset (1..horizon) of the touch
    The last ``horizon`` rows are NaN (their outcome isn't fully known yet).
    """
    c = close.to_numpy(dtype="float64")
    n = c.shape[0]
    # Per-bar return volatility, used to size the barriers (adapts to regime).
    ret = pd.Series(c).pct_change()
    vol = ret.rolling(vol_window).std().to_numpy()

    label = np.full(n, np.nan)
    realized = np.full(n, np.nan)
    touch = np.full(n, np.nan)

    for i in range(n):
        sig = vol[i]
        if np.isnan(sig) or i + 1 >= n:
            continue
        up_lvl = c[i] * (1.0 + upper * sig)
        dn_lvl = c[i] * (1.0 - lower * sig)
        end = min(i + horizon, n - 1)
        if end <= i:
            continue
        outcome = 0
        for j in range(i + 1, end + 1):
            if c[j] >= up_lvl:
                outcome = 1
                break
            if c[j] <= dn_lvl:
                outcome = -1
                break
        else:
            j = end  # vertical barrier (time-out)
        # Only emit a label if the full horizon could be observed.
        if i + horizon <= n - 1:
            label[i] = outcome
            realized[i] = c[j] / c[i] - 1.0
            touch[i] = j - i

    return pd.DataFrame(
        {"label": label, "ret": realized, "touch_idx": touch},
        index=close.index,
    )


def binary_labels(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Simple up/down label over a fixed horizon (the naive baseline).

    Useful as a contrast to triple-barrier labels in the lesson — same model,
    very different (and usually worse) results."""
    fwd = close.pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype("float").where(fwd.notna())
