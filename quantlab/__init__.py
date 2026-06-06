"""
QuantLab
========

The reusable Python spine of the QuantLab course: a provider-agnostic data
layer, a from-scratch indicator library, a caching market-data warehouse, and
(in later modules) a backtesting + stats engine, an ML feature/labeling
pipeline, and a Streamlit dashboard.

Quick start::

    from quantlab.data import get_provider
    from quantlab.indicators import sma, rsi

    px = get_provider("yahoo").get_ohlcv("SPY", start="2020-01-01")
    px["sma50"] = sma(px["close"], 50)
    px["rsi14"] = rsi(px["close"], 14)

Design philosophy, in one sentence: *depend on interfaces, vectorize the math,
never peek at the future, and test everything.*
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
