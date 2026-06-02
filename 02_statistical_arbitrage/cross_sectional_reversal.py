"""Short-term cross-sectional reversal on a purpose-built mean-reverting panel.

Why a custom panel?
-------------------
``core.generate_market`` deliberately embeds MOMENTUM (a persistent AR(1)
expected-return state, positive autocorrelation), so a short-horizon reversal
signal loses money there. To show reversal as a real effect we generate our
own equity-like panel with SHORT-TERM NEGATIVE autocorrelation baked in.

Embedded structure (see ``generate_reversal_panel``)
----------------------------------------------------
Each asset's daily return is

    r_t = beta * market_t  -  kappa * (r_{t-1} + r_{t-2})  +  sigma * z_t

The ``-kappa * (past returns)`` term is overreaction/liquidity-provision:
yesterday's (and the prior day's) idiosyncratic move is partly given back, so
recent losers tend to bounce and recent winners pull back. A shared market
factor is included so dollar-neutral (cross-sectional) trading is what
extracts the effect.

Signal & backtest
------------------
Signal = NEGATIVE of the trailing ``window``-day return (1-5d). Higher signal
= bigger recent loser = expected to bounce -> go long. Traded with
``core.long_short_backtest`` (dollar-neutral deciles, 1-day execution lag,
cost on turnover) so it is directly comparable to the cross-sectional book.

Real-world gotcha: short-term reversal is a high-turnover signal that lives or
dies on transaction costs and capacity -- it is largely a paid liquidity-
provision premium, so realistic spreads/impact and crowding erode it fast
(and it inverts in momentum-dominated regimes, as the control panel shows).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core


def generate_reversal_panel(n_assets: int = 100, n_days: int = 1750, seed: int = 31,
                            kappa: float = 0.02, idio_vol: float = 0.018,
                            market_vol: float = 0.010):
    """Equity-like panel with 1-2 day NEGATIVE return autocorrelation.

    ``kappa`` is the overreaction/reversal strength (higher = stronger bounce).
    Returns a ``core.MarketData`` so it plugs straight into the engine.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)
    tickers = [f"REV{i:03d}" for i in range(n_assets)]
    n = n_assets

    beta = np.clip(rng.normal(1.0, 0.2, n), 0.3, 1.8)
    sigma = idio_vol * rng.uniform(0.7, 1.4, n)

    ret = np.zeros((n_days, n))
    market = np.empty(n_days)
    prev1 = np.zeros(n)
    prev2 = np.zeros(n)
    for t in range(n_days):
        m_t = market_vol * rng.standard_normal()
        idio = sigma * rng.standard_normal(n)
        # overreaction: partial reversal of the last two days' idiosyncratic move
        r = beta * m_t - kappa * (prev1 + prev2) + idio
        ret[t] = r
        market[t] = m_t
        prev2 = prev1
        prev1 = r - beta * m_t  # reverse only the idiosyncratic part

    returns = pd.DataFrame(ret, index=dates, columns=tickers)
    prices = 100.0 * (1.0 + returns).cumprod()
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    return core.data.MarketData(
        prices=prices, returns=returns, volume=volume,
        market=pd.Series(market, index=dates, name="market"),
        sectors=pd.Series("Equity", index=tickers, name="sector"),
        betas=pd.Series(beta, index=tickers, name="beta"),
        fundamentals={},
        meta={"seed": seed, "kappa": kappa, "panel": "reversal"},
    )


def reversal_signal(md, window: int = 3):
    """Negative trailing ``window``-day return: recent losers expected to bounce."""
    return -(md.prices / md.prices.shift(window) - 1.0)
