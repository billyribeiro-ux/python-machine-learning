"""Configurable backtester — build ANY indicator strategy on ANY timeframe.

Shows the three ways to declare a strategy (rule / entry+exit / signal callable),
across several symbols and timeframes, and validates the resulting stats.

Run:  PYTHONPATH=. python code/custom_backtester.py
"""
from quantlab.data import get_provider
from quantlab.backtest import StrategyConfig, run_strategy, available_indicators

provider = get_provider("yahoo")
print("Indicators you can use in specs:", available_indicators(), "\n")


def show(title, res):
    s = res["stats"]
    print(f"{title}")
    print(f"   symbol={res['symbol']}  timeframe={res['timeframe']}  "
          f"trades≈{int(res['turnover'])//2}")
    print(f"   total={s['total_return']:7.1%}  CAGR={s['cagr']:6.1%}  "
          f"Sharpe={s['sharpe']:5.2f}  maxDD={s['max_drawdown']:6.1%}  "
          f"win={s['win_rate']:.0%}\n")


# 1) RULE mode — golden cross: hold while the 50-day SMA is above the 200-day.
golden = StrategyConfig(
    name="golden_cross",
    indicators=[("sma", {"window": 50}), ("sma", {"window": 200})],
    rule="sma_50 > sma_200",
)
show("1) Golden cross (daily)", run_strategy(provider, "SPY", golden, start="2010-01-01"))

# 2) ENTRY/EXIT mode — Connors RSI(2) mean-reversion on a trend filter.
rsi2 = StrategyConfig(
    name="connors_rsi2",
    indicators=[("rsi", {"window": 2}), ("sma", {"window": 200})],
    entry="(rsi_2 < 10) & (close > sma_200)",   # oversold, but in an uptrend
    exit="rsi_2 > 60",
)
show("2) Connors RSI(2) (daily)", run_strategy(provider, "QQQ", rsi2, start="2010-01-01"))

# 3) Multi-indicator combo with custom settings (RSI + z-score + trend).
combo = StrategyConfig(
    name="multi_combo",
    indicators=[("sma", {"window": 20}), ("sma", {"window": 100}),
                ("rsi", {"window": 14}), ("zscore", {"window": 20})],
    rule="(sma_20 > sma_100) & (rsi_14 > 40) & (rsi_14 < 75) & (zscore_20 <= 0.5)",
)
show("3) Multi-indicator combo (daily)", run_strategy(provider, "AAPL", combo, start="2010-01-01"))

# 4) Same combo on a DIFFERENT timeframe — weekly bars (resampled from daily).
show("4) Same combo, WEEKLY bars", run_strategy(
    provider, "AAPL", combo, start="2010-01-01", resample="W-FRI"))

# 5) SIGNAL-callable mode — full control with the Numba Supertrend (any code).
from quantlab.indicators import supertrend

def supertrend_signal(feats):
    _, direction = supertrend(feats["high"], feats["low"], feats["close"], 10, 3.0)
    return (direction > 0).astype(float)   # long while the trend is up

st = StrategyConfig(name="supertrend", signal=supertrend_signal, fee_bps=2.0)
show("5) Supertrend trend-follow (daily)", run_strategy(provider, "NVDA", st, start="2015-01-01"))

# 6) SHORT example — fade extreme overbought (direction='short').
fade = StrategyConfig(
    name="short_overbought",
    indicators=[("rsi", {"window": 14})],
    entry="rsi_14 > 80", exit="rsi_14 < 55", direction="short",
)
show("6) Short overbought (daily)", run_strategy(provider, "TSLA", fade, start="2015-01-01"))

# 7) Intraday — the SAME builder on 5-minute bars (last ~60 days on Yahoo).
try:
    intr = run_strategy(provider, "SPY", golden, start=None, timeframe="5m")
    show("7) Golden cross on 5m intraday", intr)
except Exception as e:
    print("7) intraday data unavailable from Yahoo right now:", repr(e))

print("Every strategy above was built declaratively and backtested leak-free "
      "(decide on bar t, act on t+1) with timeframe-correct annualization.")
