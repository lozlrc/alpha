"""Index reconstitution (rebalance) effect.

We simulate a periodic index reconstitution every ~`period` trading days
(e.g. quarterly). At each reconstitution a handful of names are flagged as
ADDITIONS and a handful as DELETIONS. The change is ANNOUNCED `announce_lead`
trading days before it becomes EFFECTIVE (the real S&P/Russell pattern).

Embedded effect (index-fund demand)
------------------------------------
Between announcement and the effective date, passive funds must buy the
additions and sell the deletions. We embed a steady price PRESSURE over that
window: additions drift UP, deletions drift DOWN. After the effective date the
pressure unwinds, so a fraction of the move REVERTS over the following weeks.

Strategy (lookahead-safe)
-------------------------
From the day AFTER the announcement until the effective date: long additions /
short deletions, then exit. We set equal-weight target weights on the
[announce, effective-1] rows; the engine's +1 lag means we are actually
positioned from announce+1 through the effective date, capturing the demand
push but exiting before the post-effective reversion.

Real-world gotcha: this trade is crowded and well known -- arbitrageurs
front-run the index funds, so the drift increasingly happens *before* the
official announcement and the post-effective reversion eats latecomers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_rebalance_panel(md, seed: int = 23, period: int = 63,
                          announce_lead: int = 10, n_add: int = 6, n_del: int = 6,
                          push_per_day: float = 0.0022, revert_frac: float = 0.5,
                          revert_days: int = 20):
    """Overlay reconstitution events + price pressure on the base returns.

    Returns
    -------
    returns : DataFrame (dates x tickers) base returns + embedded pressure.
    events  : list of dicts with announce/effective rows and add/del tickers.
    signal  : DataFrame (dates x tickers) in {+1 add, -1 del, 0} marking the
              tradable window [announce_row, effective_row-1] (point-in-time).
    """
    rng = np.random.default_rng(seed)
    base = md.returns.fillna(0.0)
    dates = base.index
    tickers = list(base.columns)
    n_days, n = len(dates), len(tickers)
    col = {t: i for i, t in enumerate(tickers)}

    overlay = np.zeros((n_days, n))
    signal = pd.DataFrame(0.0, index=dates, columns=tickers)
    events = []

    # effective dates on the schedule; announcement is announce_lead days earlier
    eff_days = np.arange(period, n_days - revert_days, period)
    for eff in eff_days:
        ann = eff - announce_lead
        if ann < 1:
            continue
        chosen = rng.choice(n, size=n_add + n_del, replace=False)
        adds = chosen[:n_add]
        dels = chosen[n_add:]

        for j in adds:
            overlay[ann:eff, j] += push_per_day            # drift up into eff
            # partial reversion after the effective date
            tot = push_per_day * (eff - ann)
            overlay[eff:eff + revert_days, j] -= revert_frac * tot / revert_days
        for j in dels:
            overlay[ann:eff, j] -= push_per_day            # drift down into eff
            tot = push_per_day * (eff - ann)
            overlay[eff:eff + revert_days, j] += revert_frac * tot / revert_days

        signal.iloc[ann:eff, adds] = 1.0
        signal.iloc[ann:eff, dels] = -1.0
        events.append({
            "announce_row": int(ann), "effective_row": int(eff),
            "additions": [tickers[j] for j in adds],
            "deletions": [tickers[j] for j in dels],
        })

    returns = base + pd.DataFrame(overlay, index=dates, columns=tickers)
    return returns, events, signal


def rebalance_weights(signal: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    """Equal-weight, dollar-neutral long-add / short-del weights per date.

    `signal` is +1 for additions / -1 for deletions during their tradable
    window. Each leg is equal-weighted and the book scaled to unit gross.
    """
    longs = signal > 0
    shorts = signal < 0
    n_long = longs.sum(axis=1)
    n_short = shorts.sum(axis=1)

    long_w = longs.div(n_long.replace(0, np.nan), axis=0).fillna(0.0)
    short_w = shorts.div(n_short.replace(0, np.nan), axis=0).fillna(0.0)
    w = 0.5 * gross_leverage * (long_w - short_w)
    # only trade when at least one name on each side exists
    valid = (n_long > 0) & (n_short > 0)
    return w.where(valid, 0.0)
