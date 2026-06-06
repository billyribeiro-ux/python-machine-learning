"""
Tests for the options analytics (Module 15): pricing, parity, Greeks, and IV.

These are closed-form formulas, so we can assert exact mathematical identities —
the strongest kind of test.
"""

import numpy as np
import pytest

from quantlab.options import bs_price, greeks, implied_vol


def test_put_call_parity():
    # C - P == S - K*exp(-rT) must hold exactly under Black-Scholes.
    S, K, T, r, sigma = 100, 95, 0.75, 0.03, 0.3
    c = bs_price(S, K, T, r, sigma, "call")
    p = bs_price(S, K, T, r, sigma, "put")
    assert np.isclose(c - p, S - K * np.exp(-r * T), atol=1e-8)


def test_call_price_monotonic_in_vol():
    # A call is worth more when volatility is higher (vega > 0).
    base = bs_price(100, 100, 0.5, 0.02, 0.20, "call")
    higher = bs_price(100, 100, 0.5, 0.02, 0.40, "call")
    assert higher > base


def test_implied_vol_round_trips():
    # Price at a known vol, then recover that vol from the price.
    for sigma in (0.10, 0.25, 0.60):
        price = bs_price(100, 100, 0.5, 0.02, sigma, "call")
        iv = float(implied_vol(price, 100, 100, 0.5, 0.02, "call"))
        assert np.isclose(iv, sigma, atol=1e-3)


def test_call_delta_in_unit_interval():
    g = greeks(100, 100, 0.5, 0.02, 0.25, "call")
    assert 0.0 <= float(g["delta"]) <= 1.0
    assert float(g["gamma"]) >= 0.0
    assert float(g["vega"]) >= 0.0


def test_vectorized_over_strikes():
    # The whole chain at once: pass an array of strikes.
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    prices = bs_price(100, strikes, 0.5, 0.02, 0.25, "call")
    assert prices.shape == strikes.shape
    # Call value decreases as strike rises (deeper OTM is cheaper).
    assert np.all(np.diff(prices) < 0)
