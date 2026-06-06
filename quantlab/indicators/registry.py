"""
quantlab.indicators.registry
============================

A unified **indicator registry** + a generic ``combine`` function. This is the
Module 7 payoff: instead of fifty bespoke call sites, every indicator is
addressable **by name with settings**, so a config file, a parameter sweep
(Module 9), or an Optuna trial (Module 13) can request indicators generically::

    feats = combine(ohlcv, [
        ("sma",   {"window": 50}),
        ("rsi",   {"window": 14}),
        ("zscore",{"window": 20}),
        ("atr",   {"window": 14}),
    ])

Each registered indicator is a function ``f(ohlcv: DataFrame, **params) ->
DataFrame`` returning one or more **named** columns (the name encodes the
settings, e.g. ``rsi_14``), aligned to the input index with honest NaN warmup.

Registering your own is one decorator — that's what makes this "a better
indicator library": it is *open for extension*. TA-Lib, pandas-ta, and Numba
indicators all plug into the same registry, so downstream code never cares which
engine computed a column.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .numpy_indicators import atr, ewma, rolling_std, rolling_zscore, rsi, sma

# name -> function(ohlcv, **params) -> DataFrame
IndicatorFn = Callable[..., pd.DataFrame]
REGISTRY: dict[str, IndicatorFn] = {}


def register(name: str) -> Callable[[IndicatorFn], IndicatorFn]:
    """Decorator to register an indicator under ``name`` (case-insensitive)."""

    def deco(fn: IndicatorFn) -> IndicatorFn:
        REGISTRY[name.lower()] = fn
        return fn

    return deco


def available() -> list[str]:
    """List registered indicator names — discovery for configs and UIs."""
    return sorted(REGISTRY)


def make(name: str, **params) -> pd.DataFrame:  # convenience, single indicator
    """Compute one indicator by name on a frame passed via ``params['ohlcv']``.
    Mostly you'll use :func:`combine`; this exists for quick one-offs."""
    ohlcv = params.pop("ohlcv")
    return compute_one(ohlcv, name, params)


def compute_one(ohlcv: pd.DataFrame, name: str, params: dict) -> pd.DataFrame:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown indicator {name!r}. Available: {available()}")
    return REGISTRY[key](ohlcv, **params)


def combine(ohlcv: pd.DataFrame, specs: list[tuple[str, dict]],
            join_input: bool = False) -> pd.DataFrame:
    """Compute many indicators with custom settings and join into one frame.

    Parameters
    ----------
    ohlcv:
        Normalized OHLCV frame.
    specs:
        List of ``(name, params)`` — the "multiple indicators with custom
        settings" the whole course is about.
    join_input:
        If True, prepend the original OHLCV columns to the output.
    """
    parts = [compute_one(ohlcv, name, params) for name, params in specs]
    out = pd.concat(parts, axis=1)
    if join_input:
        out = ohlcv.join(out)
    return out


# --------------------------------------------------------------------------- #
# Built-in registrations: wrap our from-scratch NumPy indicators. Each returns
# a single, settings-named column so combined frames are self-documenting.
# --------------------------------------------------------------------------- #
def _col(values: np.ndarray, index: pd.Index, name: str) -> pd.DataFrame:
    return pd.DataFrame({name: values}, index=index)


@register("sma")
def _sma(ohlcv: pd.DataFrame, window: int = 20, source: str = "close") -> pd.DataFrame:
    return _col(sma(ohlcv[source].to_numpy(), window), ohlcv.index, f"sma_{window}")


@register("ewma")
def _ewma(ohlcv: pd.DataFrame, span: int = 20, source: str = "close") -> pd.DataFrame:
    return _col(ewma(ohlcv[source].to_numpy(), span), ohlcv.index, f"ewma_{span}")


@register("std")
def _std(ohlcv: pd.DataFrame, window: int = 20, source: str = "close") -> pd.DataFrame:
    return _col(rolling_std(ohlcv[source].to_numpy(), window), ohlcv.index, f"std_{window}")


@register("zscore")
def _zscore(ohlcv: pd.DataFrame, window: int = 20, source: str = "close") -> pd.DataFrame:
    return _col(rolling_zscore(ohlcv[source].to_numpy(), window), ohlcv.index, f"zscore_{window}")


@register("rsi")
def _rsi(ohlcv: pd.DataFrame, window: int = 14, source: str = "close") -> pd.DataFrame:
    return _col(rsi(ohlcv[source].to_numpy(), window), ohlcv.index, f"rsi_{window}")


@register("atr")
def _atr(ohlcv: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    a = atr(ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy(),
            ohlcv["close"].to_numpy(), window)
    return _col(a, ohlcv.index, f"atr_{window}")


@register("ret")
def _ret(ohlcv: pd.DataFrame, periods: int = 1, source: str = "close") -> pd.DataFrame:
    return _col(ohlcv[source].pct_change(periods).to_numpy(), ohlcv.index, f"ret_{periods}")
