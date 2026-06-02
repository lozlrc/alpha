"""Synthetic web-traffic / app-downloads / card-spend NOWCAST (offline).

Alt-data's other classic use: *nowcasting a fundamental* before it is reported.
A vendor's daily web-traffic, app-download, or credit-card-spend panel for a
company is a real-time proxy for revenue/earnings, so it forecasts the next
earnings event -- and the price drift that accrues as the market converges to
that soon-to-be-reported fundamental.

Embedded structure
------------------
The market publishes earnings quarterly (``md.fundamentals['earnings']`` steps
every ~63 business days). For each stock we anchor the nowcast to the *next
earnings event*: the tradable thing web traffic proxies is the cumulative price
move from ``t`` up to that next report date -- the convergence-to-fundamental
drift the data lets you front-run. We build it causally::

    next_rep(t)        = first report date strictly after t
    target(t)          = price[next_rep] / price[t] - 1        # future, unknown to traders
    web_traffic(t)     = b * zscore_xs(target(t)) + noise(t)   # what the data vendor sees

``web_traffic(t)`` is dated at ``t`` and only references that *future* window,
so it is causal (it never uses a past or same-day return). ``b`` sets how
informative the traffic is and the noise std buries most of it -- a modest
direction hit-rate (~55%), as real nowcasting data delivers. The signal is held
within a quarter (you keep watching the traffic until the company files).

No lookahead
------------
The engine holds ``weights.shift(lag)`` then earns ``returns``; with ``lag=1`` a
position from ``web_traffic(t)`` earns ``r_{t+1}`` onward -- exactly the
pre-announcement drift the traffic foresaw. Nothing peeks at the realized P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core


def _report_positions(md) -> np.ndarray:
    """Integer row positions where the published earnings numerator changes
    (i.e. the quarterly report/announcement dates)."""
    earn = md.fundamentals["earnings"]
    changed = (earn.diff().abs() > 1e-12).any(axis=1)
    return np.where(changed.values)[0]


def make_web_traffic(md, b: float = 0.20, noise: float = 3.0,
                     seed: int = 23) -> pd.DataFrame:
    """Daily per-stock web-traffic signal nowcasting the next earnings event.

    Higher = stronger expected beat / pre-announcement drift (go long). Held
    flat within a quarter until the next filing; days with no upcoming report
    are NaN (excluded from trading).
    """
    rng = np.random.default_rng(seed)
    dates = md.returns.index
    cols = md.returns.columns
    prices = md.prices

    report_pos = _report_positions(md)
    all_pos = np.arange(len(dates))
    # index (into report_pos) of the first report strictly after each day t.
    nxt = np.searchsorted(report_pos, all_pos, side="right")

    # target = cumulative return from t up to that next report date.
    target = pd.DataFrame(np.nan, index=dates, columns=cols)
    for t in all_pos:
        j = nxt[t]
        if j < len(report_pos):
            rp = report_pos[j]
            if rp > t:
                target.iloc[t] = (prices.iloc[rp] / prices.iloc[t] - 1.0).values

    target_z = core.zscore(target, axis=1)  # cross-sectional standardize per day
    eps = pd.DataFrame(rng.standard_normal(target_z.shape), index=dates, columns=cols)
    traffic = b * target_z + noise * eps
    # mask rows with no upcoming earnings event to forecast.
    return traffic.where(target_z.notna())


def web_traffic_signal(md, **kwargs) -> pd.DataFrame:
    """Public alias used by run.py (cross-sectionally z-scored)."""
    return core.zscore(make_web_traffic(md, **kwargs), axis=1)
