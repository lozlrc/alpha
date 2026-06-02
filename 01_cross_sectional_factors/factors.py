"""Cross-sectional equity factor signals.

Each function takes a `core.MarketData` bundle and returns a signal
DataFrame (dates x tickers) where a *higher* value = more bullish (go long).
All signals use only point-in-time information (no lookahead); the backtest
engine adds a further 1-day execution lag.
"""
from __future__ import annotations

import core


def momentum(md, lookback: int = 252, skip: int = 21):
    """12-1 momentum: trailing 12-month return, skipping the most recent month.

    The skip avoids contaminating the medium-term trend with short-term
    reversal (last month's losers tend to bounce)."""
    p = md.prices
    return p.shift(skip) / p.shift(lookback) - 1.0


def value(md):
    """Book-to-price. High B/P = 'cheap' = expected to outperform."""
    return md.book_to_price()


def quality(md):
    """Profitability (ROE). More profitable firms earn a premium."""
    return md.fundamentals["profitability"]


def low_volatility(md, window: int = 63):
    """Negative trailing realized vol. Low-vol names earn better risk-adjusted
    returns (the low-volatility anomaly), so we go long low vol / short high."""
    return -md.returns.rolling(window).std()


def short_term_reversal(md, window: int = 5):
    """Negative of the last `window`-day return: recent losers tend to bounce.
    (Included as a bonus short-horizon factor.)"""
    return -(md.prices / md.prices.shift(window) - 1.0)


def combined(md):
    """Equal-weight blend of the four classic factors, each cross-sectionally
    z-scored so they're on a common scale. Blending weak, decorrelated signals
    usually beats any single one."""
    parts = [core.zscore(momentum(md)), core.zscore(value(md)),
             core.zscore(quality(md)), core.zscore(low_volatility(md))]
    out = parts[0].copy() * 0.0
    for p in parts:
        out = out.add(p.fillna(0.0), fill_value=0.0)
    return out
