"""Classic two-leg pairs trading on synthetic cointegrated pairs.

Embedded structure (see ``generate_pairs``)
-------------------------------------------
Each pair is built so the two legs are genuinely cointegrated:

    common_t = common_{t-1} + eta_t           (shared random-walk factor)
    spread_t = rho * spread_{t-1} + xi_t       (stationary AR(1) / OU spread)

    logP_A,t = common_t                  + 0.5 * spread_t + idio_A,t
    logP_B,t = beta_true * common_t      - 0.5 * spread_t + idio_B,t

The random-walk ``common`` factor dominates each leg's variance (so neither
leg is mean-reverting on its own), but the linear combination
``logP_A - beta_true*logP_B`` cancels the common factor and leaves the
*stationary* spread. That spread is what we trade.

Strategy (fully lookahead-safe)
-------------------------------
* Hedge ratio ``beta_hat`` is estimated by OLS on a TRAILING window of log
  prices (re-estimated daily, expanding the position only on info up to t).
* The spread ``logP_A - beta_hat*logP_B`` is z-scored on a trailing window.
* Enter long-spread (long A, short beta_hat*B) when z < -entry; enter
  short-spread when z > +entry; flat when |z| < exit. Positions are held
  until the exit band is crossed (state machine), then LAGGED one day before
  earning returns, and charged ``cost_bps`` on turnover.

Real-world gotcha: cointegration is not structural -- relationships drift or
break (an index reconstitution, M&A, a regime change), at which point the
"spread" trends instead of reverting and the book bleeds. Trailing estimation
and a stop on |z| (here a hard re-entry band) only partly mitigate this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_pairs(n_pairs: int = 60, n_days: int = 1750, seed: int = 11,
                   rho: float = 0.94, spread_vol: float = 0.018,
                   common_vol: float = 0.014, idio_vol: float = 0.004):
    """Generate ``n_pairs`` cointegrated (A, B) log-price pairs.

    ``rho`` is the AR(1) persistence of the tradable spread (closer to 1 =
    slower mean reversion). Returns a dict of pair-id -> dict with DataFrames
    ``A`` and ``B`` (dates x 1) of *prices*.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)
    pairs = {}
    for k in range(n_pairs):
        # The two legs share the common factor with the SAME loading so it
        # cancels in (logA - logB); the cointegrating hedge ratio is ~1. (A
        # different loading would leave a non-stationary common term in the
        # spread, i.e. break cointegration -- exactly the bug to avoid.)
        beta_true = 1.0
        # shared random-walk common factor (non-stationary; dominates variance)
        common = np.cumsum(common_vol * rng.standard_normal(n_days))
        # stationary AR(1) spread (this is the mean-reverting, tradable part)
        spread = np.empty(n_days)
        spread[0] = rng.standard_normal() * spread_vol / np.sqrt(1 - rho ** 2)
        eps = spread_vol * rng.standard_normal(n_days)
        for t in range(1, n_days):
            spread[t] = rho * spread[t - 1] + eps[t]

        base_a = np.log(rng.uniform(30, 120))
        base_b = np.log(rng.uniform(30, 120))
        # STATIONARY (white) idiosyncratic noise -- a random walk here would
        # also break cointegration since the two walks never cancel.
        idio_a = idio_vol * rng.standard_normal(n_days)
        idio_b = idio_vol * rng.standard_normal(n_days)

        logA = base_a + common + 0.5 * spread + idio_a
        logB = base_b + beta_true * common - 0.5 * spread + idio_b
        pairs[f"P{k:02d}"] = {
            "A": pd.Series(np.exp(logA), index=dates, name=f"P{k:02d}_A"),
            "B": pd.Series(np.exp(logB), index=dates, name=f"P{k:02d}_B"),
            "beta_true": beta_true,
        }
    return pairs


def _rolling_hedge_ratio(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Trailing-window OLS slope of log_a on log_b (with intercept).

    beta_t uses only observations in (t-window, t]; computed via rolling
    covariance / variance so there is no lookahead.
    """
    cov = log_a.rolling(window).cov(log_b)
    var = log_b.rolling(window).var()
    return cov / var


def _positions_from_z(z: pd.Series, entry: float, exit: float) -> pd.Series:
    """State-machine: +1 = long spread (z too low), -1 = short spread, 0 = flat.

    Position decided at the close of t from z_t; the caller lags it before
    applying returns.
    """
    pos = np.zeros(len(z))
    state = 0
    zv = z.to_numpy()
    for t in range(len(zv)):
        zt = zv[t]
        if np.isnan(zt):
            state = 0
        elif state == 0:
            if zt <= -entry:
                state = 1
            elif zt >= entry:
                state = -1
        elif state == 1 and zt >= -exit:      # long spread closes back near 0
            state = 0
        elif state == -1 and zt <= exit:       # short spread closes back near 0
            state = 0
        pos[t] = state
    return pd.Series(pos, index=z.index)


def pair_pnl(price_a: pd.Series, price_b: pd.Series, lookback: int = 250,
             z_window: int = 30, entry: float = 2.0, exit: float = 0.5,
             cost_bps: float = 2.0, lag: int = 1) -> pd.Series:
    """Net-of-cost daily return stream for one pair.

    The spread is traded dollar-neutral: +1 unit of A vs -beta_hat units of B
    (gross exposure normalized to 1 so per-pair P&L is comparable). Returns
    are simple per-leg returns weighted by the (lagged) target weights, less
    turnover * cost.

    ``lookback`` (hedge-ratio window) is deliberately long: a short window
    under-identifies the cointegrating beta (the common factor barely moves
    over a few weeks, so OLS attenuates the slope toward 0). ~1y of data lets
    OLS recover the true hedge ratio.
    """
    log_a, log_b = np.log(price_a), np.log(price_b)
    beta = _rolling_hedge_ratio(log_a, log_b, lookback)

    spread = log_a - beta * log_b
    mu = spread.rolling(z_window).mean()
    sd = spread.rolling(z_window).std()
    z = (spread - mu) / sd

    sig = _positions_from_z(z, entry, exit)  # +1 long spread / -1 short / 0

    # dollar-neutral target weights, gross exposure = 1 (|wA| + |wB| = 1)
    denom = (1.0 + beta.abs()).replace(0, np.nan)
    w_a = sig / denom
    w_b = -sig * beta / denom
    w = pd.concat([w_a.rename("A"), w_b.rename("B")], axis=1).fillna(0.0)

    ret = pd.concat([price_a.pct_change(fill_method=None).rename("A"),
                     price_b.pct_change(fill_method=None).rename("B")], axis=1)

    held = w.shift(lag).fillna(0.0)
    gross = (held * ret).sum(axis=1)
    trades = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = trades * (cost_bps / 1e4)
    return (gross - cost).rename("pnl")


def pairs_portfolio(pairs: dict, lookback: int = 250, z_window: int = 30,
                    entry: float = 2.0, exit: float = 0.5, cost_bps: float = 2.0,
                    lag: int = 1) -> pd.Series:
    """Aggregate equal-weight portfolio return across all pairs (net of cost)."""
    legs = []
    for pid, leg in pairs.items():
        legs.append(pair_pnl(leg["A"], leg["B"], lookback=lookback,
                             z_window=z_window, entry=entry, exit=exit,
                             cost_bps=cost_bps, lag=lag).rename(pid))
    panel = pd.concat(legs, axis=1)
    # equal-weight across pairs each day (a pair contributes 0 when flat)
    return panel.mean(axis=1).rename("pairs_trading")
