"""
Tests for the frontier toolkit (Module 18): fractional differentiation, pairs /
Kalman, regime detection, HRP, and sample uniqueness. Wherever possible we test
against KNOWN ground truth — simulated processes whose right answer we control —
plus the course-wide causality invariant (corrupting the future must not change
the past).
"""

import numpy as np
import pandas as pd
import pytest

from quantlab.advanced import (
    adf_tstat,
    average_uniqueness,
    causal_regimes,
    engle_granger,
    fit_regimes,
    fracdiff,
    fracdiff_weights,
    half_life,
    hrp_weights,
    inverse_variance,
    kalman_hedge_ratio,
    label_concurrency,
    min_fracdiff_order,
    pairs_backtest,
    portfolio_backtest,
    regime_stats,
    rolling_hedge_ratio,
)


def _bidx(n, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    idx.name = "timestamp"
    return idx


# --------------------------------------------------------------------------- #
# Fractional differentiation
# --------------------------------------------------------------------------- #
def test_fracdiff_weights_anchor_cases():
    # d=0 -> identity operator; d=1 -> plain first difference.
    assert fracdiff_weights(0.0).tolist() == [1.0]
    w1 = fracdiff_weights(1.0)
    assert np.allclose(w1, [1.0, -1.0])


def test_fracdiff_d1_equals_diff():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.standard_normal(300)) + 100, index=_bidx(300))
    fd = fracdiff(x, 1.0)
    ref = x.diff()
    m = fd.notna() & ref.notna()
    assert np.allclose(fd[m], ref[m], atol=1e-12)


def test_fracdiff_d0_is_identity():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.standard_normal(100) + 50, index=_bidx(100))
    assert np.allclose(fracdiff(x, 0.0), x, atol=1e-12)


def test_fracdiff_is_causal():
    """Fixed-width window => truncating the future cannot change the past."""
    rng = np.random.default_rng(2)
    x = pd.Series(np.cumsum(rng.standard_normal(400)) + 100, index=_bidx(400))
    full = fracdiff(x, 0.4)
    trunc = fracdiff(x.iloc[:300], 0.4)
    m = full.iloc[:300].notna() & trunc.notna()
    assert np.allclose(full.iloc[:300][m], trunc[m], atol=1e-12)


def test_adf_separates_random_walk_from_white_noise():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(1000)
    walk = np.cumsum(noise)
    t_noise = adf_tstat(noise)
    t_walk = adf_tstat(walk)
    assert t_noise < -5.0          # stationary: strongly rejects unit root
    assert t_walk > -3.0           # random walk: cannot reject unit root


def test_min_fracdiff_order_on_random_walk():
    rng = np.random.default_rng(4)
    walk = pd.Series(np.cumsum(rng.standard_normal(1500)) + 500, index=_bidx(1500))
    d_star, table = min_fracdiff_order(walk, d_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Some 0 < d* <= 1 achieves stationarity, and stationarity (ADF) improves
    # monotonically-ish with d while memory (corr with levels) decays.
    assert 0.0 < d_star <= 1.0
    assert table.loc[0.0, "adf_t"] > table.loc[1.0, "adf_t"]
    assert table.loc[0.2, "memory_corr"] > table.loc[1.0, "memory_corr"]


# --------------------------------------------------------------------------- #
# Pairs: hedge ratios, cointegration, half-life, strategy
# --------------------------------------------------------------------------- #
def _cointegrated_pair(n=1500, beta=2.0, theta=0.15, noise=0.5, seed=5):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n) * 0.5) + 100.0
    # OU spread around 0: s_{t} = s_{t-1} - theta*s_{t-1} + eps
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] * (1.0 - theta) + rng.normal(0, noise)
    y = beta * x + s
    idx = _bidx(n)
    return pd.Series(y, index=idx, name="y"), pd.Series(x, index=idx, name="x"), s


def test_kalman_tracks_a_drifting_beta():
    rng = np.random.default_rng(6)
    n = 1200
    x = np.cumsum(rng.standard_normal(n) * 0.5) + 100.0
    true_beta = np.linspace(1.0, 2.0, n)            # beta drifts upward
    y = true_beta * x + rng.normal(0, 0.5, n)
    idx = _bidx(n)
    beta = kalman_hedge_ratio(pd.Series(y, index=idx), pd.Series(x, index=idx),
                              delta=1e-4)
    # After burn-in the filter should track the drifting truth closely.
    err = np.abs(beta.to_numpy()[200:] - true_beta[200:])
    assert err.mean() < 0.08
    assert abs(beta.iloc[-1] - 2.0) < 0.1


def test_kalman_is_causal():
    """The filter runs strictly forward: corrupting the future can't change
    past estimates."""
    y, x, _ = _cointegrated_pair(seed=7)
    full = kalman_hedge_ratio(y, x)
    y2 = y.copy()
    y2.iloc[1000:] = y2.iloc[1000:] * 3.0           # vandalize the future
    corrupted = kalman_hedge_ratio(y2, x)
    assert np.allclose(full.iloc[:1000], corrupted.iloc[:1000], atol=1e-12)


def test_engle_granger_detects_cointegration():
    y, x, _ = _cointegrated_pair(seed=8)
    res = engle_granger(y, x)
    assert res["cointegrated_5pct"]
    assert abs(res["beta"] - 2.0) < 0.1
    # Two INDEPENDENT random walks must not look cointegrated.
    rng = np.random.default_rng(9)
    a = pd.Series(np.cumsum(rng.standard_normal(1500)), index=_bidx(1500))
    b = pd.Series(np.cumsum(rng.standard_normal(1500)), index=_bidx(1500))
    assert not engle_granger(a, b)["cointegrated_5pct"]


def test_half_life_recovers_ou_speed():
    _, _, s = _cointegrated_pair(theta=0.15, seed=10)
    hl = half_life(pd.Series(s))
    true_hl = np.log(2) / 0.15                       # ~4.6 bars
    assert 0.5 * true_hl < hl < 2.0 * true_hl
    # A random walk has no mean reversion: half-life should be far larger.
    rng = np.random.default_rng(11)
    walk = pd.Series(np.cumsum(rng.standard_normal(1500)))
    assert half_life(walk) > 10 * true_hl


def test_pairs_backtest_profits_on_a_real_ou_spread():
    """With a LONG hedge window (the module's documented guidance), the
    strategy must capture the genuine OU edge across several seeds. A short
    window trades hedge-estimation noise instead — asserted below as the
    counter-case, because that failure mode is the module's core lesson."""
    sharpes = []
    for seed in (12, 40, 41):
        y, x, _ = _cointegrated_pair(seed=seed)
        res = pairs_backtest(y, x, hedge="rolling", window=500, z_window=20,
                             entry=2.0, exit=0.5, fee_bps=1.0)
        assert set(np.unique(res["positions"])) <= {-1.0, 0.0, 1.0}
        assert res["stats"]["max_drawdown"] <= 0.0
        sharpes.append(res["stats"]["sharpe"])
    assert min(sharpes) > 0.5                       # robust across seeds
    assert np.mean(sharpes) > 1.0


def test_pairs_short_hedge_window_trades_noise():
    """The documented failure mode: a 60-bar hedge window injects beta noise
    (~price level x beta error) that swamps the true spread, destroying the
    edge that window=500 captures on the SAME data."""
    long_s, short_s = [], []
    for seed in (12, 40, 41):
        y, x, _ = _cointegrated_pair(seed=seed)
        long_s.append(pairs_backtest(y, x, hedge="rolling", window=500,
                                     z_window=20)["stats"]["sharpe"])
        short_s.append(pairs_backtest(y, x, hedge="rolling", window=60,
                                      z_window=20)["stats"]["sharpe"])
    assert np.mean(long_s) > np.mean(short_s) + 0.5


def test_pairs_backtest_is_causal():
    y, x, _ = _cointegrated_pair(seed=13)
    full = pairs_backtest(y, x, hedge="kalman")
    y2 = y.copy()
    y2.iloc[1200:] = y2.iloc[1200:] + 50.0
    corrupted = pairs_backtest(y2, x, hedge="kalman")
    assert np.allclose(full["positions"].iloc[:1200],
                       corrupted["positions"].iloc[:1200], atol=1e-12)


# --------------------------------------------------------------------------- #
# Regime detection
# --------------------------------------------------------------------------- #
def _two_regime_returns(n_calm=600, n_wild=600, seed=14):
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0005, 0.005, n_calm)
    wild = rng.normal(-0.0005, 0.03, n_wild)
    r = np.concatenate([calm, wild])
    return pd.Series(r, index=_bidx(len(r)))


def test_fit_regimes_separates_calm_from_wild():
    r = _two_regime_returns()
    labels, _ = fit_regimes(r, n_regimes=2)
    calm_label = labels.iloc[100:550].dropna()
    wild_label = labels.iloc[700:1150].dropna()
    # Vol-sorted relabeling: regime 0 = calm, regime 1 = turbulent.
    assert (calm_label == 0).mean() > 0.8
    assert (wild_label == 1).mean() > 0.8


def test_causal_regimes_warmup_and_causality():
    r = _two_regime_returns(seed=15)
    labels = causal_regimes(r, n_regimes=2, min_train=252, refit_every=63)
    # No model existed before min_train (+ vol warmup): labels there are NaN.
    assert labels.iloc[:252].isna().all()
    assert labels.iloc[300:].notna().any()
    # Causality: corrupting the future does not change earlier labels.
    r2 = r.copy()
    r2.iloc[900:] = r2.iloc[900:] * 5.0
    labels2 = causal_regimes(r2, n_regimes=2, min_train=252, refit_every=63)
    a, b = labels.iloc[:850], labels2.iloc[:850]
    m = a.notna() & b.notna()
    assert (a[m] == b[m]).all()


def test_regime_stats_table():
    r = _two_regime_returns(seed=16)
    labels, _ = fit_regimes(r, n_regimes=2)
    table = regime_stats(r, labels)
    assert abs(table["frequency"].sum() - 1.0) < 1e-9
    # The turbulent regime must show the higher annualized vol.
    assert table.loc[1, "ann_vol"] > table.loc[0, "ann_vol"]


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def _panel(seed=17, n=750):
    """Four assets: A,B near-clones (corr ~0.95); C,D independent. Equal vol."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.01, n)
    a = base + rng.normal(0, 0.003, n)
    b = base + rng.normal(0, 0.003, n)
    c = rng.normal(0, 0.0105, n)
    d = rng.normal(0, 0.0105, n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=_bidx(n))


def test_inverse_variance_exact():
    rng = np.random.default_rng(18)
    rets = pd.DataFrame({"lo": rng.normal(0, 0.01, 2000),
                         "hi": rng.normal(0, 0.02, 2000)})
    w = inverse_variance(rets)
    assert abs(w.sum() - 1.0) < 1e-9
    # Variance ratio ~1:4 -> weights ~4/5 : 1/5.
    assert abs(w["lo"] - 0.8) < 0.03


def test_hrp_weights_sane_and_diversifying():
    rets = _panel()
    w = hrp_weights(rets)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()
    # The near-clone pair shares one risk bucket: together they should get
    # LESS than the two independent assets, and each clone less than each
    # independent asset. (This is HRP's whole point vs equal weight.)
    assert w["A"] + w["B"] < w["C"] + w["D"]
    assert w["A"] < w["C"] and w["B"] < w["D"]


def test_portfolio_backtest_is_walk_forward():
    rets = _panel(seed=19)
    res = portfolio_backtest(rets, weight_fn=hrp_weights,
                             lookback=252, rebalance_every=63)
    assert len(res["returns"]) == len(rets)
    # Nothing invested before the first lookback window completes.
    assert (res["returns"].iloc[:252] == 0).all()
    assert "sharpe" in res["stats"]
    # Causality: changing the future doesn't change past portfolio returns.
    rets2 = rets.copy()
    rets2.iloc[600:] = rets2.iloc[600:] * 3.0
    res2 = portfolio_backtest(rets2, weight_fn=hrp_weights,
                              lookback=252, rebalance_every=63)
    assert np.allclose(res["returns"].iloc[:600], res2["returns"].iloc[:600],
                       atol=1e-12)


# --------------------------------------------------------------------------- #
# Sample uniqueness
# --------------------------------------------------------------------------- #
def test_uniqueness_non_overlapping_is_one():
    idx = _bidx(100)
    starts = np.array([0, 20, 40, 60])
    ends = np.array([9, 29, 49, 69])
    u = average_uniqueness(idx, starts, ends)
    assert np.allclose(u, 1.0)


def test_uniqueness_fully_overlapping_is_one_over_k():
    idx = _bidx(50)
    starts = np.array([10, 10, 10, 10])              # 4 identical events
    ends = np.array([19, 19, 19, 19])
    u = average_uniqueness(idx, starts, ends)
    assert np.allclose(u, 0.25)


def test_concurrency_counts():
    idx = _bidx(30)
    conc = label_concurrency(idx, np.array([0, 5]), np.array([9, 14]))
    assert conc.iloc[0] == 1          # only event 1 alive
    assert conc.iloc[7] == 2          # both alive
    assert conc.iloc[12] == 1         # only event 2 alive
    assert conc.iloc[20] == 0         # none alive


def test_uniqueness_from_triple_barrier_labels(close):
    from quantlab.advanced import uniqueness_from_labels
    from quantlab.ml import triple_barrier_labels

    lab = triple_barrier_labels(close, horizon=10, upper=2.0, lower=2.0)
    u = uniqueness_from_labels(lab, horizon=10)
    assert len(u) == int(lab["label"].notna().sum())
    assert ((u > 0) & (u <= 1.0)).all()
    # Daily consecutive labels overlap heavily -> typical uniqueness well below 1.
    assert u.mean() < 0.5
