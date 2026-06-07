"""
End-to-end scanner tests — deterministic and offline (CI-safe).

The live harness in ``code/m14_scanners_e2e.py`` proves the scanners work on real
Yahoo data; this module proves the *invariants* hold for EVERY scanner, on
synthetic seeded data, without a network — so CI guards them on every push.

Key design choice: setups are **auto-discovered** from
``quantlab.scanners.scanner``, so any new scanner you add is tested here
automatically with no edits.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from quantlab.data.base import DataProvider, OHLCVRequest
from quantlab.scanners import latest_snapshot, scan
from quantlab.scanners import scanner as scanner_mod


# --------------------------------------------------------------------------- #
# A multi-symbol fake provider: distinct seeded GBM path per symbol so the
# universe has variety (different trends, vols, and price levels).
# --------------------------------------------------------------------------- #
def _make_symbol(seed: int, n: int = 320, drift: float = 0.0004,
                 vol: float = 0.012, start_px: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=int(n), tz="UTC")
    idx.name = "timestamp"
    ret = rng.normal(drift, vol, size=n)
    close = start_px * np.exp(np.cumsum(ret))
    # Overnight gaps so the gap_up scanner has something realistic to find.
    gap = rng.normal(0.0, 0.006, size=n)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1.0 + gap[1:])
    spread = np.abs(rng.normal(0, 0.004, size=n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(2_000_000, 9_000_000, size=n).astype("float64")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class MultiFakeProvider(DataProvider):
    name = "multifake"

    def __init__(self):
        # A mix of strong uptrends, mild trends, and downtrends + price levels.
        # "SPY" is included so the relative_strength benchmark is available.
        specs = {
            "SPY": (0, 0.0005, 0.010, 400),   # the benchmark
            "AAA": (1, 0.0010, 0.010, 50),    # strong uptrend (leads SPY)
            "BBB": (2, 0.0008, 0.013, 200),   # uptrend, pricier
            "CCC": (3, 0.0003, 0.011, 120),   # mild trend
            "DDD": (4, 0.0000, 0.014, 80),    # flat/choppy
            "EEE": (5, -0.0006, 0.012, 300),  # downtrend
            "FFF": (6, 0.0012, 0.016, 30),    # strong, volatile (leads SPY)
            "GGG": (7, 0.0005, 0.009, 150),   # steady up
            "HHH": (8, -0.0003, 0.013, 90),   # mild down
        }
        self._frames = {
            s: _make_symbol(seed=seed, drift=drift, vol=vol, start_px=px)
            for s, (seed, drift, vol, px) in specs.items()
        }

    def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        return self._frames[req.symbol].copy()


UNIVERSE = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


_NON_SETUP = {"scan", "latest_snapshot", "orb_scan", "intraday_orb_features"}


def discover_setups(probe_snapshot):
    """Every snapshot setup that actually works on the STANDARD snapshot.

    We probe by calling each candidate on a real snapshot and keeping only those
    that return a boolean mask aligned to it. This auto-includes new snapshot
    scanners and auto-excludes intraday ones (e.g. opening_range_breakout, which
    needs an ORB snapshot) — exactly mirroring the live harness.
    """
    out = {}
    for name, obj in inspect.getmembers(scanner_mod, inspect.isfunction):
        if name.startswith("_") or name in _NON_SETUP:
            continue
        if obj.__module__ != scanner_mod.__name__:
            continue
        params = list(inspect.signature(obj).parameters.values())
        if not params or params[0].name != "snap":
            continue
        try:
            mask = obj(probe_snapshot)
        except Exception:
            continue
        if (isinstance(mask, pd.Series) and mask.dtype == bool
                and mask.index.equals(probe_snapshot.index)):
            out[name] = obj
    return out


_PROBE_SNAP = latest_snapshot(MultiFakeProvider(), UNIVERSE, start="2021-01-01")
SETUPS = discover_setups(_PROBE_SNAP)


@pytest.fixture(scope="module")
def provider():
    return MultiFakeProvider()


@pytest.fixture(scope="module")
def snapshot(provider):
    return latest_snapshot(provider, UNIVERSE, start="2021-01-01")


def test_at_least_the_known_scanners_exist():
    assert {"swing_pullback", "momentum_breakout", "relative_strength",
            "gap_up", "new_high_breakout"} <= set(SETUPS)


def test_orb_is_excluded_from_snapshot_setups():
    # opening_range_breakout needs an intraday ORB snapshot, not the standard one.
    assert "opening_range_breakout" not in SETUPS


def test_snapshot_is_one_row_per_symbol(snapshot):
    assert len(snapshot) == len(UNIVERSE)
    assert set(snapshot.index) == set(UNIVERSE)
    for col in ("sma20", "sma50", "sma200", "rsi14", "z20", "atr14",
                "mom20", "mom60", "dollar_vol"):
        assert col in snapshot.columns


@pytest.mark.parametrize("name", sorted(SETUPS))
def test_setup_returns_bool_mask(name, snapshot):
    mask = SETUPS[name](snapshot)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert mask.index.equals(snapshot.index)


@pytest.mark.parametrize("name", sorted(SETUPS))
def test_scan_results_are_valid(name, provider, snapshot):
    """The core end-to-end invariant set, for EVERY scanner."""
    setup = SETUPS[name]
    hits = scan(provider, UNIVERSE, setup=setup, rank_by="mom20", start="2021-01-01")

    # Subset of the universe, unique symbols.
    assert set(hits.index).issubset(set(UNIVERSE))
    assert not hits.index.has_duplicates

    # Every hit must re-pass the setup mask computed on the snapshot.
    expected = set(snapshot[setup(snapshot)].index)
    assert set(hits.index) == expected

    # No NaNs in the displayed result columns.
    if len(hits):
        assert not hits.isna().any().any()

    # Liquidity floor respected (all built-in setups gate on dollar_vol > 5e7).
    if len(hits):
        assert (snapshot.loc[hits.index, "dollar_vol"] > 5e7).all()


@pytest.mark.parametrize("name", sorted(SETUPS))
def test_scan_ranking_is_sorted(name, provider):
    hits = scan(provider, UNIVERSE, setup=SETUPS[name], rank_by="mom20",
                ascending=False, start="2021-01-01")
    if len(hits) > 1:
        vals = hits["mom20"].to_numpy()
        assert np.all(np.diff(vals) <= 1e-12)  # descending


@pytest.mark.parametrize("name", sorted(SETUPS))
def test_top_cap_is_respected(name, provider):
    hits = scan(provider, UNIVERSE, setup=SETUPS[name], rank_by="mom20",
                start="2021-01-01", top=2)
    assert len(hits) <= 2


def test_swing_pullback_economic_rules(provider, snapshot):
    """Spot-check the setup's actual economic conditions on its hits."""
    from quantlab.scanners import swing_pullback

    hits = scan(provider, UNIVERSE, setup=swing_pullback, start="2021-01-01")
    for sym in hits.index:
        r = snapshot.loc[sym]
        assert r["close"] > r["sma200"]
        assert r["sma20"] > r["sma50"]
        assert 40 <= r["rsi14"] <= 55


def test_momentum_breakout_economic_rules(provider, snapshot):
    from quantlab.scanners import momentum_breakout

    hits = scan(provider, UNIVERSE, setup=momentum_breakout, start="2021-01-01")
    for sym in hits.index:
        r = snapshot.loc[sym]
        assert r["sma20"] > r["sma50"]
        assert r["z20"] > 1.0
        assert r["mom60"] > 0.10


def test_relative_strength_needs_benchmark_and_rules(provider, snapshot):
    from quantlab.scanners import relative_strength

    # The benchmark RS columns must be present in the snapshot.
    assert "rs_pct_from_high" in snapshot.columns
    hits = scan(provider, UNIVERSE, setup=relative_strength, rank_by="rs_mom60",
                start="2021-01-01")
    for sym in hits.index:
        r = snapshot.loc[sym]
        assert r["rs_pct_from_high"] >= -0.02
        assert r["rs_mom60"] > 0
        assert r["sma20"] > r["sma50"]


def test_relative_strength_empty_without_benchmark(provider):
    from quantlab.scanners import relative_strength

    # With benchmark=None there are no RS columns -> the setup yields no hits,
    # gracefully (no crash). This is the "missing benchmark" safety path.
    snap = latest_snapshot(provider, UNIVERSE, start="2021-01-01", benchmark=None)
    assert "rs_pct_from_high" not in snap.columns
    mask = relative_strength(snap)
    assert mask.dtype == bool and not mask.any()


def test_gap_up_economic_rules(provider, snapshot):
    from quantlab.scanners import gap_up

    hits = scan(provider, UNIVERSE, setup=gap_up, rank_by="gap", start="2021-01-01")
    for sym in hits.index:
        r = snapshot.loc[sym]
        assert r["gap"] > 0.02
        assert r["close"] > r["sma50"]


def test_scanner_depends_only_on_interface():
    """The scanner must accept ANY DataProvider — proving no Yahoo coupling."""
    prov = MultiFakeProvider()
    snap = latest_snapshot(prov, ["AAA", "EEE"], start="2021-01-01")
    assert len(snap) == 2


# --------------------------------------------------------------------------- #
# Intraday opening-range breakout — its own pipeline, synthetic session data.
# --------------------------------------------------------------------------- #
def _make_intraday(seed: int, sessions: int = 3, bars_per_day: int = 78,
                   breakout: bool = False) -> pd.DataFrame:
    """Synthetic 5-minute bars across several sessions. If ``breakout`` is True,
    price drifts up AFTER the opening range so it breaks the opening-range high."""
    rng = np.random.default_rng(seed)
    frames = []
    base = pd.Timestamp("2024-03-01 14:30", tz="UTC")   # ~09:30 US/Eastern
    px = 100.0
    for d in range(sessions):
        idx = pd.date_range(base + pd.Timedelta(days=d), periods=bars_per_day,
                            freq="5min", tz="UTC")
        rets = rng.normal(0.0, 0.001, size=bars_per_day)
        if breakout:
            rets[bars_per_day // 3:] += 0.0015     # strong drift up after open
        else:
            rets[bars_per_day // 3:] -= 0.0010     # drift down -> no breakout
        close = px * np.exp(np.cumsum(rets)); px = float(close[-1])
        open_ = np.empty(bars_per_day); open_[0] = close[0]; open_[1:] = close[:-1]
        spread = np.abs(rng.normal(0, 0.0004, size=bars_per_day)) * close
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        vol = rng.integers(10_000, 60_000, size=bars_per_day).astype("float64")
        frames.append(pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
            index=idx))
    df = pd.concat(frames)
    df.index.name = "timestamp"
    return df


class IntradayFakeProvider(DataProvider):
    name = "intraday_fake"

    def __init__(self):
        self._frames = {
            "UP1": _make_intraday(11, breakout=True),
            "UP2": _make_intraday(12, breakout=True),
            "FLAT": _make_intraday(13, breakout=False),
            "DOWN": _make_intraday(14, breakout=False),
        }

    def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        return self._frames[req.symbol].copy()


def test_orb_features_and_scan():
    from quantlab.scanners import orb_scan, intraday_orb_features

    prov = IntradayFakeProvider()
    syms = ["UP1", "UP2", "FLAT", "DOWN"]
    panel = prov.get_ohlcv_multi(syms, timeframe="5m")
    snap = intraday_orb_features(panel, open_bars=6)

    # One row per symbol, with the ORB columns.
    assert set(snap.index) == set(syms)
    for col in ("or_high", "or_low", "last", "orb_up", "session_dollar_vol"):
        assert col in snap.columns

    hits = orb_scan(prov, syms, timeframe="5m", open_bars=6)
    # Every hit must genuinely have broken its opening range.
    assert bool(hits["orb_up"].all())
    # The two designed breakout names must be flagged; the down name must not.
    assert {"UP1", "UP2"} <= set(hits.index)
    assert "DOWN" not in set(hits.index)
    # Ranking by session dollar volume is descending.
    if len(hits) > 1:
        v = hits["session_dollar_vol"].to_numpy()
        assert np.all(np.diff(v) <= 1e-6)


def test_orb_handles_too_few_bars():
    """A session shorter than the opening range must not crash; it just can't
    break out."""
    from quantlab.scanners import intraday_orb_features

    prov = IntradayFakeProvider()
    panel = prov.get_ohlcv_multi(["UP1"], timeframe="5m")
    snap = intraday_orb_features(panel, open_bars=10_000)   # absurd opening range
    assert bool(~snap["orb_up"].iloc[0])
