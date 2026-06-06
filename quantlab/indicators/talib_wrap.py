"""
quantlab.indicators.talib_wrap
==============================

Adapters that make **TA-Lib** obey the QuantLab contract.

TA-Lib is a C library exposing 150+ canonical indicators. It is fast and
authoritative, but it speaks raw NumPy arrays and has sharp edges: every
function has a warmup region of NaNs, inputs must be float64 and the right
length, and the *abstract* API names its inputs/outputs in ways you must respect.

These adapters take our normalized OHLCV ``DataFrame`` (lowercase columns,
tz-aware index) and return aligned indicator columns with honest NaN warmup, so
a TA-Lib indicator drops into the same pipeline as our from-scratch ones.

TA-Lib is an *optional* dependency (it needs the C library — see Module 0). We
import it lazily and raise a clear, actionable error if it's missing, rather
than failing at import time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_talib():
    try:
        import talib  # noqa: F401

        return talib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "TA-Lib is not installed. It needs the C library first:\n"
            "  conda install -c conda-forge ta-lib    (easiest, any OS)\n"
            "  macOS:  brew install ta-lib  &&  pip install TA-Lib\n"
            "  Linux:  install libta-lib, then pip install TA-Lib\n"
            "See Module 0 for details."
        ) from exc


def talib_func(name: str, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
    """Call any TA-Lib function by name on a normalized OHLCV frame.

    Uses TA-Lib's *abstract* API, which introspects which inputs a function
    needs (close only? high/low/close? volume?) and what it outputs (one column
    for RSI, three for MACD/BBANDS). This is what lets us write ONE generic
    wrapper instead of 150 hand-written ones — the foundation of settings-driven,
    config-file indicator pipelines.

    Parameters
    ----------
    name:
        TA-Lib function name, e.g. ``"RSI"``, ``"MACD"``, ``"BBANDS"``, ``"ATR"``.
    df:
        Normalized OHLCV frame (lowercase open/high/low/close/volume).
    **params:
        Function parameters, e.g. ``timeperiod=14`` or ``fastperiod=12``.

    Returns
    -------
    A DataFrame aligned to ``df.index`` with one column per TA-Lib output,
    prefixed by the function name (e.g. ``MACD_macd``, ``MACD_macdsignal``).
    """
    talib = _require_talib()
    from talib import abstract

    func = abstract.Function(name)

    # TA-Lib's abstract API maps its required input names (open/high/low/close/
    # volume) to our columns automatically because we already use those names.
    inputs = {
        col: df[col].to_numpy(dtype="float64")
        for col in ("open", "high", "low", "close", "volume")
        if col in df.columns
    }
    func.set_parameters(**params)
    out = func(inputs)

    # Normalize the output to a DataFrame with stable, prefixed column names.
    output_names = func.output_names
    if isinstance(out, list):
        cols = {f"{name}_{nm}": np.asarray(o, dtype="float64")
                for nm, o in zip(output_names, out)}
    else:
        cols = {f"{name}_{output_names[0]}": np.asarray(out, dtype="float64")}
    return pd.DataFrame(cols, index=df.index)


def add_indicators(df: pd.DataFrame, specs: list[tuple[str, dict]]) -> pd.DataFrame:
    """Apply a list of ``(talib_name, params)`` specs and join the results.

    This is the "combine multiple indicators with custom settings" idea in
    TA-Lib form: pass a config list, get back the original frame plus every
    requested indicator column.

    Example::

        feats = add_indicators(ohlcv, [
            ("RSI",    {"timeperiod": 14}),
            ("MACD",   {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}),
            ("ATR",    {"timeperiod": 14}),
            ("BBANDS", {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2}),
        ])
    """
    out = df.copy()
    for name, params in specs:
        cols = talib_func(name, df, **params)
        out = out.join(cols)
    return out


def list_functions() -> dict[str, list[str]]:
    """Return TA-Lib's functions grouped by category (overlap, momentum, ...).

    Handy for discovery: ``list_functions()["Momentum Indicators"]`` shows
    every momentum study available.
    """
    talib = _require_talib()
    return {group: list(funcs) for group, funcs in talib.get_function_groups().items()}
