"""Lead-lag trading: laggards partially track a liquid leader's prior-day move.

Embedded structure (see ``generate_lead_lag``)
----------------------------------------------
A single liquid LEADER follows its own random walk of daily returns. Each
LAGGARD's return reacts to the leader with a one-day delay (slow information
diffusion / illiquidity):

    leader_t  = mu + sigma_L * z^L_t
    laggard_i,t = a_i * leader_{t-1}  +  b_i * leader_t  +  sigma_i * z^i_t

The dominant term is ``a_i * leader_{t-1}`` (yesterday's leader move), so the
leader's most recent return PREDICTS today's laggard returns. ``b_i`` is a
small same-day component so the relationship is noisy (realistic).

Signal & backtest
-----------------
For every laggard the signal is simply the leader's most recent daily return
(known at the close of t, applied to t+1 by the engine's lag). We trade it
dollar-neutral across laggards with ``core.long_short_backtest`` (so when the
leader was up we are net-long the laggards that load most on it, and short the
rest) -- net of cost, lookahead-safe.

Real-world gotcha: lead-lag edges come from real frictions (latency, illiquid
followers) and therefore sit exactly where costs/impact bite hardest; they are
also heavily arbitraged, so the predictive horizon shrinks toward zero and the
laggard's "delay" can vanish once enough capital chases it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core


def generate_lead_lag(n_laggards: int = 80, n_days: int = 1750, seed: int = 43,
                      lead_coef: float = 0.28, same_day: float = 0.15,
                      leader_vol: float = 0.013, idio_vol: float = 0.018):
    """Leader + ``n_laggards`` whose returns track ``leader_{t-1}``.

    ``lead_coef`` is the mean loading on yesterday's leader return (the
    tradable, predictable part). Returns ``(core.MarketData, leader_returns)``
    where the MarketData holds ONLY the laggards (the leader is exogenous and
    its prior return is the signal).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)
    tickers = [f"LAG{i:03d}" for i in range(n_laggards)]
    n = n_laggards

    leader = leader_vol * rng.standard_normal(n_days)  # leader daily returns
    leader[0] += 0.0
    a = lead_coef * rng.uniform(0.5, 1.5, n)           # per-laggard lead loading
    b = same_day * rng.uniform(0.0, 1.0, n)            # weak same-day component
    sigma = idio_vol * rng.uniform(0.7, 1.4, n)

    lead_prev = np.concatenate([[0.0], leader[:-1]])   # leader_{t-1}
    ret = (a[None, :] * lead_prev[:, None]
           + b[None, :] * leader[:, None]
           + sigma[None, :] * rng.standard_normal((n_days, n)))

    returns = pd.DataFrame(ret, index=dates, columns=tickers)
    prices = 100.0 * (1.0 + returns).cumprod()
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    md = core.data.MarketData(
        prices=prices, returns=returns, volume=volume,
        market=pd.Series(leader, index=dates, name="leader"),
        sectors=pd.Series("Follower", index=tickers, name="sector"),
        betas=pd.Series(a, index=tickers, name="lead_loading"),
        fundamentals={},
        meta={"seed": seed, "lead_coef": lead_coef, "panel": "lead_lag"},
    )
    return md, pd.Series(leader, index=dates, name="leader_ret")


def lead_lag_signal(md, leader_returns: pd.Series, beta_window: int = 60):
    """Signal per laggard = (trailing lead-loading) x leader's most recent return.

    A flat broadcast of the leader return would be constant across laggards and
    so produce no cross-sectional ranking (and zero dollar-neutral weights). We
    instead estimate each laggard's loading on YESTERDAY's leader return over a
    trailing window (lookahead-safe), then multiply by the most recent leader
    return. The result disperses across names -- big positive when the leader
    just rose AND the laggard loads heavily on it -- so the L/S book goes long
    the strongest followers and short the weakest after an up move (and flips
    after a down move). All inputs are known at the close of t.
    """
    lead_prev = leader_returns.shift(1)                      # leader_{t-1}, the predictor
    lag_ret = md.returns                                     # laggard returns
    # trailing-window OLS slope of each laggard's return on the prior leader return
    cov = lag_ret.rolling(beta_window).cov(lead_prev)
    var = lead_prev.rolling(beta_window).var()
    loading = cov.div(var, axis=0)                           # dates x laggards, trailing beta
    # most recent leader return (known at close of t) scales the cross-section
    sig = loading.mul(leader_returns, axis=0)
    return sig
