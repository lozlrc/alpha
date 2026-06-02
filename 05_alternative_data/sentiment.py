"""Synthetic news/social SENTIMENT alt-data signal (offline, no live feed).

The whole point of alt-data is *timing*: a signal you can observe at the close
of day ``t`` that genuinely previews the near future. We build that causally.

Embedded structure
-------------------
Pick a preview horizon ``k`` (days). The "true" thing sentiment is a noisy
proxy for is an *exponentially-decay-weighted* cumulative forward return over
``t+1 .. t+k`` -- it previews the next day most strongly and each successive day
less. This is what makes the edge *decay*: most of the information is about the
immediate future, so acting even a few days late throws away the bulk of it::

    w_j            = 0.5 ** ((j-1) / half_life)                # weight on r_{t+j}
    fwd_k(t)       = sum_{j=1..k} w_j * r_{t+j}                # future, unknown to traders
    sentiment(t)   = a * zscore_xs(fwd_k(t)) + noise(t)        # what the crowd "knows" at t

So ``sentiment(t)`` is information *dated* at ``t`` (it does not use ``r_t`` or
anything before, only the future), and it is deliberately a *leading* indicator
of returns starting ``t+1``. The cross-sectional z-score keeps every day on a
common scale; ``a`` sets signal strength and the noise std sets how much real
edge is buried. With ``a`` small relative to noise the per-name information
coefficient is low (~0.05), which is exactly how real sentiment data behaves --
the high *portfolio* Sharpe at zero delay comes from diversifying that tiny edge
across the whole cross-section, and it collapses as you trade later.

No lookahead
------------
The backtest engine holds ``weights.shift(lag)`` and earns ``returns``. With
``lag=1`` a position decided from ``sentiment(t)`` earns ``r_{t+1}`` -- the very
first day of the window sentiment previews. The signal never peeks at the
same-day or a *past* return; it is a forward-looking-but-causal "nowcast" of
sentiment that real desks buy from data vendors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core


def make_sentiment(md, horizon: int = 10, half_life: float = 2.0, a: float = 0.15,
                   noise: float = 5.0, seed: int = 11) -> pd.DataFrame:
    """Build a daily per-stock sentiment score that leads forward returns.

    Parameters
    ----------
    md        : core.MarketData bundle (we only use ``md.returns``).
    horizon   : k, number of forward days the sentiment previews.
    half_life : days for the forward-day weight to halve (smaller => the edge is
                concentrated in the immediate future => decays faster).
    a         : signal strength multiplier on the (standardized) forward move.
    noise     : std of the idiosyncratic noise added on top (>> a => low IC).
    seed      : RNG seed (independent of the market seed).

    Returns
    -------
    DataFrame (dates x tickers); higher = more bullish. The last ``horizon``
    rows are NaN (no full forward window exists -> excluded from trading).
    """
    rng = np.random.default_rng(seed)
    r = md.returns

    # decay-weighted cumulative FORWARD return over t+1 .. t+horizon, dated at t.
    # shift(-j) brings r_{t+j} onto row t; this is the "future" we proxy, with
    # the nearest days weighted most (so most of the edge is in r_{t+1}).
    w = 0.5 ** (np.arange(horizon) / half_life)
    fwd = sum(w[j - 1] * r.shift(-j) for j in range(1, horizon + 1))
    fwd_z = core.zscore(fwd, axis=1)  # cross-sectional standardize per day

    # the observable sentiment: a noisy, scaled view of that forward move.
    eps = pd.DataFrame(rng.standard_normal(fwd_z.shape),
                       index=fwd_z.index, columns=fwd_z.columns)
    sentiment = a * fwd_z + noise * eps

    # rows without a complete forward window are not observable -> drop them.
    sentiment.iloc[-horizon:] = np.nan
    return sentiment


def sentiment_signal(md, **kwargs) -> pd.DataFrame:
    """Public alias used by run.py / alpha_decay.py (z-scored for stacking)."""
    return core.zscore(make_sentiment(md, **kwargs), axis=1)
