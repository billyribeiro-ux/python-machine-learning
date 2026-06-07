"""
Tests for meta-labeling and position sizing. Where possible we assert against
KNOWN ground truth (oracle/anti-oracle sides, constant-vol assets) rather than
"did it improve Sharpe", which is never guaranteed.
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import (
    bet_size,
    drawdown_throttle,
    estimate_payoff_ratio,
    kelly_fraction,
    kelly_size,
    vol_target_scalar,
)
from quantlab.ml import apply_meta, meta_dataset, triple_barrier_meta


# --------------------------------------------------------------------------- #
# Sizing
# --------------------------------------------------------------------------- #
def test_bet_size_linear_known_points():
    p = pd.Series([0.0, 0.5, 0.75, 1.0])
    s = bet_size(p, method="linear", cap=1.0)
    # 2(p-0.5) clipped at 0; p is ε-clamped off {0,1} for the normal method's
    # safety, so 1.0 -> ~1.0.
    assert s.iloc[0] == 0.0 and s.iloc[1] == 0.0
    assert s.iloc[2] == pytest.approx(0.5, abs=1e-4)
    assert s.iloc[3] == pytest.approx(1.0, abs=1e-4)


def test_bet_size_normal_monotone_and_bounded():
    p = pd.Series(np.linspace(0.01, 0.99, 50))
    s = bet_size(p, method="normal", cap=1.0)
    assert s.is_monotonic_increasing
    assert (s >= 0).all() and (s <= 1).all()
    assert bet_size(pd.Series([0.5]), "normal").iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_kelly_fraction_and_size():
    # p=0.6, payoff 1:1 -> f* = 0.6 - 0.4/1 = 0.2
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 1.0) < 0                # no edge -> negative
    ks = kelly_size(pd.Series([0.4, 0.6]), payoff_ratio=1.0, fraction=0.5, cap=1.0)
    assert ks.tolist() == [0.0, pytest.approx(0.1)]    # half of 0.2; floored at 0


def test_estimate_payoff_ratio():
    r = pd.Series([0.02, -0.01, 0.02, -0.01])          # avg win 0.02, avg loss 0.01
    assert estimate_payoff_ratio(r) == pytest.approx(2.0)


def test_vol_target_scalar_is_causal_and_constant_on_constant_vol():
    # Alternating +/- constant magnitude -> constant rolling std -> constant scalar.
    ar = pd.Series([0.01, -0.01] * 100)
    sc = vol_target_scalar(ar, target_ann_vol=0.15, lookback=20)
    assert sc.iloc[:20].eq(0.0).all()                  # warmup (lagged) -> 0
    tail = sc.iloc[30:]
    assert tail.std() < 1e-9                            # constant after warmup


def test_drawdown_throttle_cuts_then_restores_and_is_causal():
    # A position of all-1 on an asset that drops then recovers.
    idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    pos = pd.Series(1.0, index=idx)
    ar = pd.Series([0.0, -0.1, -0.1, -0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0], index=idx)
    out = drawdown_throttle(pos, ar, max_dd=0.15, throttle=0.5)
    mult = out / pos
    assert set(np.unique(mult)) <= {0.5, 1.0}
    assert (mult == 0.5).any()                         # throttled during the drop
    # Causal: early multipliers don't change when later bars are appended.
    out_short = drawdown_throttle(pos.iloc[:6], ar.iloc[:6], max_dd=0.15, throttle=0.5)
    assert np.allclose((out_short / pos.iloc[:6]).to_numpy(),
                       mult.iloc[:6].to_numpy())


# --------------------------------------------------------------------------- #
# Meta-labeling (ground truth via oracle / anti-oracle sides)
# --------------------------------------------------------------------------- #
@pytest.fixture
def gbm_close():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=600, tz="UTC")
    idx.name = "timestamp"
    px = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=600)))
    return pd.Series(px, index=idx, name="close")


def test_meta_label_oracle_is_all_wins(gbm_close):
    """If the side is the sign of the forward return (an oracle), and the barriers
    are effectively disabled, the time-out outcome must always be a win -> 1."""
    h = 10
    fwd = gbm_close.pct_change(h).shift(-h)
    side = np.sign(fwd).fillna(0.0)
    meta = triple_barrier_meta(gbm_close, side, horizon=h, pt=100, sl=100)  # barriers off
    lab = meta["meta_label"].dropna()
    assert len(lab) > 0
    assert lab.mean() > 0.98                            # essentially all wins


def test_meta_label_anti_oracle_is_all_losses(gbm_close):
    h = 10
    fwd = gbm_close.pct_change(h).shift(-h)
    side = -np.sign(fwd).fillna(0.0)                    # deliberately wrong side
    meta = triple_barrier_meta(gbm_close, side, horizon=h, pt=100, sl=100)
    lab = meta["meta_label"].dropna()
    assert lab.mean() < 0.02                            # essentially all losses


def test_meta_labels_only_on_events_and_observable(gbm_close):
    side = pd.Series(0.0, index=gbm_close.index)
    side.iloc[100:110] = 1.0                            # a few active bars
    meta = triple_barrier_meta(gbm_close, side, horizon=10)
    # Labels only where side != 0 ...
    assert meta["meta_label"].notna().sum() <= 10
    assert meta["meta_label"].iloc[:100].isna().all()
    # ... and never in the final `horizon` bars (outcome not yet observable).
    assert meta["meta_label"].iloc[-10:].isna().all()


def test_meta_dataset_alignment(gbm_close):
    from quantlab.data.base import normalize_ohlcv
    df = pd.DataFrame({"open": gbm_close, "high": gbm_close * 1.01,
                       "low": gbm_close * 0.99, "close": gbm_close,
                       "volume": 1e6}, index=gbm_close.index)
    from quantlab.ml import make_features
    side = pd.Series(1.0, index=gbm_close.index)
    meta = triple_barrier_meta(gbm_close, side, horizon=5)
    X, y = meta_dataset(make_features(df), meta)
    assert len(X) == len(y) and not X.isna().any().any()
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_apply_meta_filter_and_size():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    side = pd.Series([1, 1, -1, 1, -1], index=idx, dtype="float64")
    proba = pd.Series([np.nan, 0.6, 0.6, 0.4, 0.7], index=idx)
    # Filter at 0.5: bar0 no proba->flat; 0.6>0.5 keep; 0.4<0.5 drop.
    filt = apply_meta(side, proba, threshold=0.5)
    assert filt.tolist() == [0.0, 1.0, -1.0, 0.0, -1.0]
    # Size mode multiplies side by a provided size series.
    size = pd.Series([0.0, 0.2, 0.2, 0.0, 0.5], index=idx)
    sized = apply_meta(side, proba, size=size)
    assert sized.tolist() == [0.0, 0.2, -0.2, 0.0, -0.5]
