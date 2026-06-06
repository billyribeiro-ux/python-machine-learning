"""Module 15 — options: pricing, greeks, implied volatility, and a chain scan.

Run:  python code/m15_options.py
"""
import numpy as np
import pandas as pd
from quantlab.data import get_provider
from quantlab.options import bs_price, greeks, implied_vol

# --- 1) Pricing & greeks from scratch ---
S, K, T, r, sigma = 100, 100, 0.5, 0.02, 0.25
call = bs_price(S, K, T, r, sigma, "call")
g = greeks(S, K, T, r, sigma, "call")
print(f"ATM call: price={call:.3f}  delta={g['delta']:.3f}  "
      f"gamma={g['gamma']:.4f}  vega={g['vega']/100:.3f}/%vol  "
      f"theta={g['theta']/365:.3f}/day")

# --- 2) Implied vol round-trips back to the input ---
iv = implied_vol(call, S, K, T, r, "call")
print(f"Implied vol recovered from price: {float(iv):.4f} (input was {sigma})")

# --- 3) A real chain through the provider, with our IV/greeks layered on ---
provider = get_provider("yahoo")
try:
    chain = provider.get_options_chain("SPY")
    spot = provider.get_ohlcv("SPY", start="2024-01-01")["close"].iloc[-1]
    expiry = pd.Timestamp(chain["expiry"].iloc[0])
    T = max((expiry - pd.Timestamp.utcnow().tz_localize(None)).days, 1) / 365
    calls = chain[chain["option_type"] == "call"].copy()
    mid = (calls["bid"] + calls["ask"]) / 2
    calls["iv_ours"] = implied_vol(mid.to_numpy(), spot, calls["strike"].to_numpy(),
                                   T, 0.04, "call")
    near = calls.reindex(calls["strike"].sub(spot).abs().sort_values().index).head(5)
    print(f"\nSPY ~{spot:.0f}, expiry {expiry.date()}: near-the-money calls")
    print(near[["strike", "bid", "ask", "impliedvolatility", "iv_ours"]]
          .round(4).to_string(index=False))
except Exception as e:
    print("\n(Live options fetch skipped:", e, ")")
