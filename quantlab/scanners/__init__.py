"""Multi-symbol scanners for swing, momentum, relative-strength, gap, breakout,
and intraday opening-range setups."""

from .scanner import (
    BENCHMARK_DEFAULT,
    gap_up,
    intraday_orb_features,
    latest_snapshot,
    momentum_breakout,
    new_high_breakout,
    opening_range_breakout,
    orb_scan,
    relative_strength,
    scan,
    swing_pullback,
)

__all__ = [
    "scan",
    "latest_snapshot",
    # snapshot setups
    "swing_pullback",
    "momentum_breakout",
    "relative_strength",
    "gap_up",
    "new_high_breakout",
    # intraday opening-range breakout
    "orb_scan",
    "opening_range_breakout",
    "intraday_orb_features",
    "BENCHMARK_DEFAULT",
]
