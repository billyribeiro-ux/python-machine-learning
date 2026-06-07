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
    idx = pd.bdate_range("2021-01-01", periods=n, tz="UTC")
    idx.name = "timestamp"
    ret = rng.normal(drift, vol, size=n)
    close = start_px * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.004, size=n)) * close
    open_ = np.empty(n); open_[0] = close[0]; open_[1:] = close[:-1]
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
        specs = {
            "AAA": (1, 0.0010, 0.010, 50),    # strong uptrend
            "BBB": (2, 0.0008, 0.013, 200),   # uptrend, pricier
            "CCC": (3, 0.0003, 0.011, 120),   # mild trend
            "DDD": (4, 0.0000, 0.014, 80),    # flat/choppy
            "EEE": (5, -0.0006, 0.012, 300),  # downtrend
            "FFF": (6, 0.0012, 0.016, 30),    # strong, volatile
            "GGG": (7, 0.0005, 0.009, 150),   # steady up
            "HHH": (8, -0.0003, 0.013, 90),   # mild down
        }
        self._frames = {
            s: _make_symbol(seed=seed, drift=drift, vol=vol, start_px=px)
            for s, (seed, drift, vol, px) in specs.items()
        }

    def _fetch_ohlcv(self, req: OHLCVRequest) -> pd.DataFrame:
        return self._frames[req.symbol].copy()


UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def discover_setups():
    """Every public (snapshot -> bool mask) callable defined in the scanner module."""
    out = {}
    for name, obj in inspect.getmembers(scanner_mod, inspect.isfunction):
        if name.startswith("_") or name in ("scan", "latest_snapshot"):
            continue
        if obj.__module__ != scanner_mod.__name__:
            continue
        # A setup takes exactly the snapshot (+ optional kwargs with defaults).
        params = list(inspect.signature(obj).parameters.values())
        if params and params[0].name == "snap":
            out[name] = obj
    return out


SETUPS = discover_setups()


@pytest.fixture(scope="module")
def provider():
    return MultiFakeProvider()


@pytest.fixture(scope="module")
def snapshot(provider):
    return latest_snapshot(provider, UNIVERSE, start="2021-01-01")


def test_at_least_the_known_scanners_exist():
    assert {"swing_pullback", "momentum_breakout"} <= set(SETUPS)


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


def test_scanner_depends_only_on_interface():
    """The scanner must accept ANY DataProvider — proving no Yahoo coupling."""
    prov = MultiFakeProvider()
    snap = latest_snapshot(prov, ["AAA", "EEE"], start="2021-01-01")
    assert len(snap) == 2
