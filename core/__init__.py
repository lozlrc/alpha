"""Shared library for the alpha backtest suite (offline, no live market).

Typical use inside a strategy script::

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core import generate_market, long_short_backtest, print_summary

"""
from __future__ import annotations

from . import backtest, data, metrics, plotting
from .backtest import (
    BacktestResult,
    backtest_weights,
    cross_sectional_weights,
    long_short_backtest,
    zscore,
)
from .data import MarketData, generate_market
from .metrics import (
    equity_curve,
    format_leaderboard,
    print_summary,
    sharpe,
    summary,
)
from .plotting import plot_equity

__all__ = [
    "backtest", "data", "metrics", "plotting",
    "BacktestResult", "backtest_weights", "cross_sectional_weights",
    "long_short_backtest", "zscore",
    "MarketData", "generate_market",
    "equity_curve", "format_leaderboard", "print_summary", "sharpe", "summary",
    "plot_equity",
]
