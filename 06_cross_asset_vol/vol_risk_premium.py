"""Variance Risk Premium (VRP) -- systematically SHORT variance.

Embedded structure (the variance risk premium): each day we observe an IMPLIED
volatility (what the market charges to insure against moves) and the SUBSEQUENT
REALIZED volatility (what actually happens). On AVERAGE implied > realized --
options/variance swaps are, on average, overpriced because investors pay up for
crash insurance. Selling that insurance earns a steady premium.

Crucial realism -- a FAT LEFT TAIL. Realized vol has occasional upward spikes
(``realized >> implied``) on which the short-variance seller takes a sharp loss.
So the P&L is "picking up nickels in front of a steamroller": steady positive
carry punctuated by violent drawdowns. Sharpe alone *flatters* this trade, which
is why we also report max drawdown and the single worst day.

Strategy: each day sell a variance-swap proxy on a fixed vega notional, earning
``scale * (implied_var - realized_var)`` net of a transaction cost. This is a
single P&L stream, so we build a net-return Series and use ``core.metrics`` /
``core.plot_equity`` directly (per the house convention for vol strategies).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class VRPData:
    implied_vol: pd.Series      # daily implied (annualized) vol, set at t for [t, t+1]
    realized_vol: pd.Series     # daily realized (annualized) vol over (t, t+1]
    meta: dict = field(default_factory=dict)


def generate_vrp_process(
    n_days: int = 2520,
    seed: int = 23,
    base_vol: float = 0.16,         # long-run vol level (~16% VIX-like)
    vrp_mean: float = 0.018,        # avg implied-minus-realized vol gap (the premium)
    iv_persist: float = 0.94,       # AR(1) persistence of implied vol
    iv_innov: float = 0.012,        # implied-vol innovation scale
    rv_noise: float = 0.035,        # day-to-day realized-vol noise (gap is often negative)
    spike_prob: float = 0.02,       # daily probability of a realized-vol spike
    spike_size: float = 0.10,       # mean extra realized vol on a spike day
) -> VRPData:
    """Simulate implied and *subsequent* realized vol with implied > realized on
    average (the premium) but a fat left tail (realized spikes).

    Implied vol follows a persistent, mean-reverting positive process. Realized
    vol equals implied minus the premium plus noise, EXCEPT on rare spike days
    when a large positive jump pushes realized far above implied (the steamroller).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)

    iv = np.empty(n_days)
    rv = np.empty(n_days)
    logiv = np.log(base_vol)
    mu_log = np.log(base_vol)

    for t in range(n_days):
        # persistent, mean-reverting implied vol in log space (stays positive)
        logiv = mu_log + iv_persist * (logiv - mu_log) + iv_innov * rng.standard_normal()
        iv_t = float(np.exp(logiv))

        # baseline: realized sits BELOW implied by ~vrp_mean (the premium earned),
        # but with substantial day-to-day noise so the gap is frequently negative
        # -- the seller does NOT win every day.
        rv_t = iv_t - vrp_mean + rv_noise * rng.standard_normal()

        # FAT LEFT TAIL for the seller: occasional realized-vol spike up
        if rng.random() < spike_prob:
            rv_t += abs(spike_size * rng.standard_normal()) + spike_size

        iv[t] = iv_t
        rv[t] = max(rv_t, 0.01)   # vol floor

    return VRPData(
        implied_vol=pd.Series(iv, index=dates, name="implied_vol"),
        realized_vol=pd.Series(rv, index=dates, name="realized_vol"),
        meta={"seed": seed, "vrp_mean": vrp_mean, "spike_prob": spike_prob,
              "n_days": n_days},
    )


def short_variance_returns(vd: VRPData, scale: float = 0.75, cost_bps: float = 2.0) -> pd.Series:
    """Daily net return of a SHORT-variance position.

    P&L proxy = scale * (implied_var - realized_var), i.e. you collected the
    implied variance and pay out the realized variance (a variance swap struck
    at the implied level). ``scale`` sets the vega notional / capital, calibrated
    so daily vol of the strategy is reasonable. A flat per-day cost models the
    bid/ask + roll of continuously selling the swap.

    No lookahead: implied is observed at t; realized accrues over (t, t+1]; the
    P&L is therefore attributed to t+1 (we shift by one day).
    """
    implied_var = vd.implied_vol ** 2
    realized_var = vd.realized_vol ** 2
    pnl = scale * (implied_var - realized_var)
    net = pnl - cost_bps / 1e4
    # the position is set at t and pays off over the next day -> attribute to t+1
    return net.shift(1).dropna().rename("short_variance")
