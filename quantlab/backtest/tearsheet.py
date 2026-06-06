"""
quantlab.backtest.tearsheet
===========================

Bridge to **QuantStats** for professional performance reports. Our
``quantlab.backtest.stats`` computes the headline metrics from scratch (so you
understand them); QuantStats turns a returns series into a full HTML tearsheet
with rolling Sharpe, drawdown underwater plots, monthly heatmaps, and benchmark
comparison — the artifact you'd actually share.

QuantStats is optional; imported lazily.
"""

from __future__ import annotations

import pandas as pd


def _qs_ready(s: pd.Series | None) -> pd.Series | None:
    """Coerce a returns series into the shape QuantStats expects.

    QuantStats predates tz-aware / non-nanosecond datetime indexes and raises on
    them. Our data layer (correctly) uses tz-aware UTC, often at millisecond
    precision via Arrow. We strip the timezone and force nanosecond precision
    here — purely a presentation-layer concession, the analytics are unchanged.
    """
    if s is None:
        return None
    s = s.copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx.as_unit("ns")
    return s


def html_report(returns: pd.Series, benchmark: pd.Series | None = None,
                output: str = "tearsheet.html", title: str = "QuantLab Strategy"):
    """Write a full QuantStats HTML tearsheet for a returns series.

    Parameters
    ----------
    returns:
        Strategy per-period simple returns (from our backtest engine's
        ``result["returns"]``).
    benchmark:
        Optional benchmark returns (e.g. SPY) to compare against.
    output:
        Path for the generated HTML file.
    """
    try:
        import quantstats as qs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("quantstats not installed: pip install quantstats") from exc

    qs.reports.html(_qs_ready(returns), benchmark=_qs_ready(benchmark),
                    output=output, title=title)
    return output


def metrics(returns: pd.Series) -> pd.Series:
    """Return QuantStats' full metric set as a Series (no file written).

    Handy for comparing strategies in a notebook side by side. Falls back to our
    own stats bundle if QuantStats isn't installed.
    """
    try:
        import quantstats as qs
    except ImportError:
        from .stats import compute_stats

        return pd.Series(compute_stats(returns))
    return qs.reports.metrics(_qs_ready(returns), display=False)
