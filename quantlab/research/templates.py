"""
quantlab.research.templates
===========================

Parameterized **strategy templates**: functions ``build(params) ->
StrategyConfig`` plus a declared search ``space``. These turn the configurable
backtester into something an optimizer can sweep generically — the bridge
between Module 10 (build any strategy) and Module 13 (optimize anything).

A template owns three things:
  * ``SPACE`` — the search space as ``name -> ("int"|"float"|"categorical", ...)``
  * ``build(params)`` — assembles a runnable :class:`StrategyConfig`
  * ``invalid(params)`` — rejects nonsensical combinations (pruned, not scored)

Adding a new template is how you give the Optimize page a new strategy family.
"""

from __future__ import annotations

from quantlab.backtest.strategy import StrategyConfig


# --------------------------------------------------------------------------- #
# Trend + momentum + pullback (a generalized, optimizable combo)
# --------------------------------------------------------------------------- #
TREND_PULLBACK_SPACE = {
    "fast": ("int", 5, 50),
    "slow": ("int", 60, 250),
    "rsi_w": ("int", 5, 30),
    "rsi_floor": ("float", 30.0, 55.0),
    "z_w": ("int", 10, 40),
    "z_entry": ("float", -1.0, 1.5),
}


def trend_pullback(params: dict) -> StrategyConfig:
    """Long while: fast SMA > slow SMA (trend), RSI in a healthy band
    (momentum intact, not overbought), and the price z-score has dipped
    (buy the pullback). Column names are built from the chosen windows."""
    fast, slow = int(params["fast"]), int(params["slow"])
    rw, zw = int(params["rsi_w"]), int(params["z_w"])
    rf, ze = float(params["rsi_floor"]), float(params["z_entry"])
    rule = (f"(sma_{fast} > sma_{slow}) & (rsi_{rw} > {rf:.2f}) "
            f"& (rsi_{rw} < 78) & (zscore_{zw} <= {ze:.3f})")
    return StrategyConfig(
        name="trend_pullback",
        indicators=[("sma", {"window": fast}), ("sma", {"window": slow}),
                    ("rsi", {"window": rw}), ("zscore", {"window": zw})],
        rule=rule,
        fee_bps=float(params.get("fee_bps", 1.0)),
    )


def trend_pullback_invalid(params: dict) -> bool:
    # fast must be strictly faster than slow (also keeps the two SMA columns distinct).
    return int(params["fast"]) >= int(params["slow"])


# --------------------------------------------------------------------------- #
# SMA crossover (a small, classic space — good for teaching overfitting)
# --------------------------------------------------------------------------- #
SMA_CROSS_SPACE = {
    "fast": ("int", 5, 80),
    "slow": ("int", 50, 300),
}


def sma_cross(params: dict) -> StrategyConfig:
    fast, slow = int(params["fast"]), int(params["slow"])
    return StrategyConfig(
        name="sma_cross",
        indicators=[("sma", {"window": fast}), ("sma", {"window": slow})],
        rule=f"sma_{fast} > sma_{slow}",
        fee_bps=float(params.get("fee_bps", 1.0)),
    )


def sma_cross_invalid(params: dict) -> bool:
    return int(params["fast"]) >= int(params["slow"])


# Registry of templates so UIs can offer them by name.
TEMPLATES = {
    "trend_pullback": (trend_pullback, TREND_PULLBACK_SPACE, trend_pullback_invalid),
    "sma_cross": (sma_cross, SMA_CROSS_SPACE, sma_cross_invalid),
}
