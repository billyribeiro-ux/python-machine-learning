"""
Tests for the overfitting capstone: CSCV-based PBO and Combinatorial Purged CV.

The PBO tests use controlled synthetic matrices with KNOWN ground truth — the
only honest way to test an overfitting estimator:
  * pure-noise configs  -> selection is luck      -> PBO ≈ 0.5
  * one dominant config -> winner stays best OOS  -> PBO ≈ 0.0
"""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.research import (
    CombinatorialPurgedCV,
    cpcv_sharpe_distribution,
    cscv_pbo,
    pbo_for_template,
    returns_matrix,
    sample_param_sets,
)
from quantlab.research.templates import TEMPLATES


# ----- PBO ground-truth behavior ------------------------------------------
def test_pbo_noise_is_about_half():
    rng = np.random.default_rng(0)
    M = rng.normal(0.0, 0.01, size=(1500, 25))      # all configs pure noise
    res = cscv_pbo(M, s_blocks=10, metric="sharpe")
    assert 0.30 <= res.pbo <= 0.70                  # ~0.5: selection is chance
    assert res.n_splits == math.comb(10, 5)
    assert len(res.logits) == res.n_splits


def test_pbo_dominant_config_is_low():
    rng = np.random.default_rng(1)
    M = rng.normal(0.0, 0.01, size=(1500, 25))
    M[:, 0] += 0.0015                               # config 0 has real edge
    res = cscv_pbo(M, s_blocks=10, metric="sharpe")
    assert res.pbo <= 0.2                           # winner generalizes
    assert "Low overfitting" in res.verdict


def test_pbo_in_unit_interval_and_fields():
    rng = np.random.default_rng(2)
    M = rng.normal(0, 0.01, size=(800, 10))
    res = cscv_pbo(M, s_blocks=8)
    assert 0.0 <= res.pbo <= 1.0
    assert 0.0 <= res.prob_oos_loss <= 1.0
    assert res.n_configs == 10


def test_cscv_validates_inputs():
    M = np.random.default_rng(0).normal(0, 1, size=(100, 5))
    with pytest.raises(ValueError):
        cscv_pbo(M, s_blocks=7)                      # odd
    with pytest.raises(ValueError):
        cscv_pbo(M[:, :1], s_blocks=8)              # < 2 configs


# ----- building the performance matrix from a template --------------------
def test_sample_param_sets_respects_validity():
    build, space, invalid = TEMPLATES["trend_pullback"]
    sets = sample_param_sets(space, 20, invalid=invalid, seed=0)
    assert len(sets) == 20
    assert all(not invalid(p) for p in sets)        # fast < slow for all


def test_returns_matrix_shape(ohlcv):
    build, space, invalid = TEMPLATES["sma_cross"]
    sets = sample_param_sets(space, 8, invalid=invalid, seed=0)
    M = returns_matrix(ohlcv, build, sets)
    assert M.shape == (len(ohlcv), 8)


def test_pbo_for_template_runs(ohlcv):
    build, space, invalid = TEMPLATES["sma_cross"]
    res = pbo_for_template(ohlcv, build, space, invalid=invalid,
                           n_configs=12, s_blocks=8, seed=0)
    assert 0.0 <= res.pbo <= 1.0
    assert res.n_configs == 12


# ----- Combinatorial Purged CV --------------------------------------------
def test_cpcv_counts():
    cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2)
    assert cv.get_n_splits() == math.comb(10, 2)        # 45
    assert cv.n_paths() == 2 * math.comb(10, 2) // 10    # 9


def test_cpcv_splits_are_disjoint_and_purged():
    cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2, embargo=0.02,
                               label_horizon=5)
    n = 1000
    splits = list(cv.split(n))
    assert len(splits) == math.comb(8, 2)
    bounds = np.linspace(0, n, 9).astype(int)
    for train, test in splits:
        # train and test never overlap
        assert len(np.intersect1d(train, test)) == 0
        # purge: no training index in the 5 bars immediately before a test block
        test_set = set(test.tolist())
        for g in range(8):
            start = bounds[g]
            if start in test_set:  # this group is a test block
                purge_zone = set(range(max(0, start - 5), start))
                assert purge_zone.isdisjoint(set(train.tolist()))


def test_cpcv_every_sample_is_tested():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    n = 600
    tested = np.zeros(n, dtype=bool)
    for _, test in cv.split(n):
        tested[test] = True
    assert tested.all()        # combinatorial coverage: every bar is OOS somewhere


def test_cpcv_sharpe_distribution_length(close):
    r = close.pct_change()
    dist = cpcv_sharpe_distribution(r, n_groups=8, n_test_groups=2)
    assert len(dist) == math.comb(8, 2)
    assert np.all(np.isfinite(dist))
