"""
quantlab.backtest.strategy
==========================

A **configurable backtester**: declare any indicators (with custom settings),
any entry/exit rules, any symbol, and any timeframe — then get full stats. This
is the "build whatever strategy I want and test it" layer on top of the
leak-free engine.

The design keeps three promises from the rest of the course:

* **Indicators are data.** You pass ``(name, settings)`` specs resolved by the
  Module 7 registry, so you can mix SMA/EWMA/RSI/z-score/ATR/returns (and any
  custom indicator you register) without writing indicator code.
* **No look-ahead.** Whatever signal you build, the engine applies the one-bar
  execution delay, so a strategy can't peek.
* **Timeframe-agnostic & correctly annualized.** Pick ``"1d"``, ``"1h"``,
  ``"5m"``, ``"1wk"`` … and the stats are annualized with the right factor.

Three ways to express a strategy, from simplest to most powerful:

1. **rule** — a single boolean expression; hold the position while it's True::

       StrategyConfig(indicators=[("sma", {"window": 50}), ("sma", {"window": 200})],
                      rule="sma_50 > sma_200")            # classic golden-cross hold

2. **entry / exit** — separate boolean expressions; enter on ``entry``, stay in
   until ``exit`` (a proper state machine)::

       StrategyConfig(indicators=[("rsi", {"window": 2})],
                      entry="rsi_2 < 10", exit="rsi_2 > 60")   # Connors RSI(2)

3. **signal** — a Python callable ``f(features) -> position Series`` for total
   control (use any library you like inside it).

Expressions are evaluated against a frame containing the OHLCV columns
(``open/high/low/close/volume``) **plus** every indicator column you declared
(named with its settings, e.g. ``rsi_14``, ``sma_200``, ``zscore_20``). Use
vectorized ``&`` / ``|`` / ``~`` (not Python ``and``/``or``) to combine
conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quantlab.indicators.registry import available, combine
from quantlab.utils.returns import annualization_factor
from .engine import backtest_signal

# OHLCV aggregation for optional resampling to a coarser timeframe.
RESAMPLE_AGG = {"open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"}

# Periods-per-year for common resample rule prefixes (for correct annualization).
_RESAMPLE_PPY = {"W": 52, "M": 12, "Q": 4, "Y": 1, "D": 252, "B": 252, "H": 252 * 6.5}


@dataclass
class StrategyConfig:
    """A fully-declared, runnable strategy.

    Provide exactly one of ``signal``, ``entry`` (optionally with ``exit``), or
    ``rule``. Everything is a value you can log, hash, or store — so a strategy
    is reproducible and sweepable.
    """

    indicators: list[tuple[str, dict]] = field(default_factory=list)
    rule: str | Callable | None = None          # hold while True
    entry: str | Callable | None = None         # enter when True
    exit: str | Callable | None = None          # exit when True
    signal: Callable | None = None              # features -> position Series
    direction: str = "long"                     # "long" or "short"
    fee_bps: float = 1.0
    name: str = "custom"


def available_indicators() -> list[str]:
    """Names you can use in ``indicators`` specs (from the Module 7 registry)."""
    return available()


# --------------------------------------------------------------------------- #
# Expression evaluation
# --------------------------------------------------------------------------- #
def _eval_cond(cond, feats: pd.DataFrame) -> pd.Series:
    """Evaluate a condition (string expression or callable) to a boolean Series.

    String expressions are evaluated with the feature columns as variables and
    ``np`` available, but with builtins disabled — safe enough for a local
    research tool while staying expressive.
    """
    if callable(cond):
        result = cond(feats)
    else:
        namespace = {col: feats[col] for col in feats.columns}
        namespace["np"] = np
        try:
            result = eval(cond, {"__builtins__": {}}, namespace)  # noqa: S307 (local tool)
        except NameError as exc:
            raise ValueError(
                f"Unknown name in expression {cond!r}: {exc}. "
                f"Available columns: {list(feats.columns)}. "
                "Did you declare the indicator in `indicators`?"
            ) from exc
        except SyntaxError as exc:
            raise ValueError(f"Bad expression {cond!r}: {exc}. "
                             "Use & | ~ (not and/or/not) to combine conditions.") from exc
    if not isinstance(result, pd.Series):
        raise ValueError(
            f"Condition {cond!r} did not produce a Series (got {type(result)}). "
            "Make sure it references column(s)."
        )
    return result.reindex(feats.index).fillna(False).astype(bool)


def build_signal(ohlcv: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """Turn a StrategyConfig into a position Series aligned to ``ohlcv``.

    Position values are in {-1, 0, +1}. The engine later shifts by one bar, so
    this signal may use everything up to and including bar *t* without leaking.
    """
    feats = (combine(ohlcv, config.indicators, join_input=True)
             if config.indicators else ohlcv.copy())

    sign = -1.0 if config.direction == "short" else 1.0

    # 1) Full control: a callable returning the position directly.
    if config.signal is not None:
        pos = config.signal(feats)
        pos = pd.Series(pos, index=feats.index) if not isinstance(pos, pd.Series) else pos
        return pos.reindex(feats.index).fillna(0.0).astype("float64")

    # 2) entry / exit state machine.
    if config.entry is not None:
        entries = _eval_cond(config.entry, feats)
        if config.exit is not None:
            exits = _eval_cond(config.exit, feats)
            # Build a state series: 1 at entry, 0 at exit (exit takes precedence
            # on a same-bar conflict), forward-filled to hold the position.
            ev = pd.Series(np.nan, index=feats.index)
            ev[entries] = 1.0
            ev[exits] = 0.0
            base = ev.ffill().fillna(0.0)
        else:
            base = entries.astype("float64")   # no exit -> hold while entry True
        return (base * sign).astype("float64")

    # 3) hold-while-rule.
    if config.rule is not None:
        base = _eval_cond(config.rule, feats).astype("float64")
        return (base * sign).astype("float64")

    raise ValueError(
        "StrategyConfig needs one of: `signal`, `entry` (+ optional `exit`), or `rule`."
    )


def _resample(ohlcv: pd.DataFrame, rule: str) -> pd.DataFrame:
    return ohlcv.resample(rule).agg(RESAMPLE_AGG).dropna()


def _infer_ppy(timeframe: str, resample: str | None) -> float:
    if resample:
        return _RESAMPLE_PPY.get(resample[0].upper(), 252)
    return annualization_factor(timeframe)


def run_strategy(provider, symbol: str, config: StrategyConfig,
                 start=None, end=None, timeframe: str = "1d",
                 resample: str | None = None,
                 periods_per_year: float | None = None) -> dict:
    """Fetch data, build the signal, backtest it, and return results + stats.

    Parameters
    ----------
    provider:
        Any :class:`DataProvider` (Yahoo by default, or your own).
    symbol:
        The instrument to test.
    config:
        The strategy declaration.
    start, end, timeframe:
        Passed to the provider. ``timeframe`` can be any value the provider
        supports ("1d", "1h", "5m", "1wk", ...).
    resample:
        Optional pandas resample rule (e.g. "W-FRI", "M") applied to the bars
        before computing indicators — for timeframes the provider doesn't serve
        natively (e.g. weekly from daily).
    periods_per_year:
        Override the annualization factor; otherwise inferred from the timeframe
        (or the resample rule).

    Returns
    -------
    dict with ``returns``, ``positions``, ``turnover``, ``stats``, the built
    ``signal``, and the ``config`` (as a dict) — fully self-describing.
    """
    ohlcv = provider.get_ohlcv(symbol, start, end, timeframe)
    if resample:
        ohlcv = _resample(ohlcv, resample)

    sig = build_signal(ohlcv, config)
    ppy = periods_per_year if periods_per_year is not None else _infer_ppy(timeframe, resample)

    res = backtest_signal(ohlcv["close"], sig, fee_bps=config.fee_bps,
                          periods_per_year=ppy)
    res["signal"] = sig
    res["symbol"] = symbol
    res["timeframe"] = resample or timeframe
    res["config"] = asdict_safe(config)
    return res


def run_strategy_multi(provider, symbols, config: StrategyConfig,
                       start=None, end=None, timeframe: str = "1d",
                       resample: str | None = None,
                       periods_per_year: float | None = None) -> dict:
    """Run the same strategy across many symbols and combine into an
    equal-weight portfolio. Returns per-symbol stats plus the portfolio result.
    """
    per_symbol = {}
    rets = []
    for sym in symbols:
        try:
            r = run_strategy(provider, sym, config, start, end, timeframe,
                             resample, periods_per_year)
        except Exception as exc:  # one bad symbol shouldn't sink the portfolio
            per_symbol[sym] = {"error": repr(exc)}
            continue
        per_symbol[sym] = r["stats"]
        rets.append(r["returns"].rename(sym))

    if not rets:
        return {"per_symbol": per_symbol, "portfolio": None}

    # Equal-weight average of per-symbol returns (rebalanced each bar).
    port = pd.concat(rets, axis=1).fillna(0.0).mean(axis=1)
    from .stats import compute_stats

    ppy = periods_per_year if periods_per_year is not None else _infer_ppy(timeframe, resample)
    return {
        "per_symbol": per_symbol,
        "portfolio_returns": port,
        "portfolio": compute_stats(port, periods_per_year=ppy),
    }


def asdict_safe(config: StrategyConfig) -> dict:
    """asdict, but drop callables (which aren't serializable) for clean logging."""
    d = asdict(config)
    for k in ("rule", "entry", "exit", "signal"):
        if callable(d.get(k)):
            d[k] = f"<callable {getattr(config, k).__name__}>"
    return d
