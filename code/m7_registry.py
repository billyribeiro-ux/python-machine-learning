"""Module 7 — the unified indicator registry: combine indicators by name+settings.

Run:  python code/m7_registry.py
"""
from quantlab.data import get_provider
from quantlab.indicators import available, combine, register
import pandas as pd

df = get_provider("yahoo").get_ohlcv("MSFT", start="2020-01-01")

print("Registered indicators:", available())

# Declare a feature set as DATA (name, settings) — not code. This is what lets a
# config file, a sweep, or Optuna request indicators generically.
specs = [
    ("sma", {"window": 20}),
    ("sma", {"window": 50}),
    ("rsi", {"window": 14}),
    ("zscore", {"window": 20}),
    ("atr", {"window": 14}),
    ("ret", {"periods": 5}),
]
feats = combine(df, specs, join_input=False)
print("\nCombined feature columns:", list(feats.columns))
print(feats.tail(3).round(3))


# Extend the library with ONE decorator — open for extension.
@register("hl_range")
def _hl_range(ohlcv, window: int = 14):
    """Normalized high-low range, smoothed — a quick volatility proxy."""
    rng = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"]
    return pd.DataFrame({f"hl_range_{window}": rng.rolling(window).mean()},
                        index=ohlcv.index)


print("\nAfter registering a custom indicator:", "hl_range" in available())
custom = combine(df, [("hl_range", {"window": 10})])
print(custom.tail(3).round(4))
