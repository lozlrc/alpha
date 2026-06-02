"""Merger arbitrage (risk arbitrage).

We simulate a stream of cash M&A deals on top of the synthetic universe. For
each deal:

  * Before announcement the target trades on its normal (market-driven) path.
  * At announcement the price JUMPS to just below the cash offer price, leaving
    a small positive SPREAD (the arb's expected return if the deal closes).
  * With probability `p_complete` the deal CLOSES: the price converges from the
    post-announcement level to the offer price over the deal horizon (spread ->
    0), and the arb pockets the spread.
  * With probability `1 - p_complete` the deal BREAKS: partway through, the price
    CRASHES back toward (often below) the un-disturbed standalone level -- a
    large loss that wipes out many small wins.

Strategy (lookahead-safe)
-------------------------
Go long the target the day AFTER announcement and hold to resolution
(completion or break). We size every deal equally and aggregate MANY of them
into one daily portfolio return stream. The payoff is the classic
"picking up nickels in front of a steamroller": steady small gains punctuated
by occasional large losses -- so we report max drawdown / worst day, not just
Sharpe.

Embedded edge: positive expected spread (deals close more often than not).
Real-world gotcha: deal-break tail risk. The losses are fat-tailed and
correlated (financing/regulatory regimes), so Sharpe flatters the strategy and
a cluster of breaks can be ruinous despite a high hit rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_deals(md, seed: int = 37, n_deals: int = 360,
                   p_complete: float = 0.90, offer_premium: float = 0.30,
                   entry_spread: float = 0.05, horizon_lo: int = 30,
                   horizon_hi: int = 90, break_drop: float = 0.20):
    """Build a price panel for `n_deals` targets and a per-deal long signal.

    A separate synthetic ticker is created per deal (its pre-announcement path is
    borrowed from a random base-universe name so it looks like a real stock).

    Returns
    -------
    prices : DataFrame (dates x deal_id) target prices with the embedded jump +
             convergence / break dynamics.
    weights: DataFrame (dates x deal_id) equal-weight long positions held over
             each deal's live window (announcement row .. resolution row).
    deals  : list of per-deal dicts (for reporting completion stats).
    """
    rng = np.random.default_rng(seed)
    base_prices = md.prices
    dates = base_prices.index
    n_days = len(dates)
    base_cols = list(base_prices.columns)

    deal_ids = [f"DEAL{i:03d}" for i in range(n_deals)]
    prices = pd.DataFrame(np.nan, index=dates, columns=deal_ids)
    weights = pd.DataFrame(0.0, index=dates, columns=deal_ids)
    deals = []

    for i, did in enumerate(deal_ids):
        horizon = int(rng.integers(horizon_lo, horizon_hi + 1))
        # announcement uniformly in a range that leaves room for resolution
        ann = int(rng.integers(20, n_days - horizon - 2))
        src = base_cols[rng.integers(0, len(base_cols))]
        path = base_prices[src].values.astype(float).copy()

        pre = path[ann]                                   # standalone level
        offer = pre * (1.0 + offer_premium)               # cash offer price
        post = offer * (1.0 - entry_spread)               # jump to below offer

        res = ann + horizon                               # resolution row
        completes = rng.random() < p_complete

        new_path = path.copy()
        new_path[ann] = post                              # the announcement jump
        if completes:
            # linear-ish convergence post -> offer by resolution
            steps = np.linspace(post, offer, res - ann + 1)
            new_path[ann:res + 1] = steps
            # after close, position is gone; leave price flat at offer
            new_path[res + 1:] = offer
        else:
            # drift mildly until a random break day, then crash toward standalone
            brk = ann + int(rng.integers(max(3, horizon // 4), horizon))
            steps = np.linspace(post, post * 1.005, brk - ann + 1)
            new_path[ann:brk + 1] = steps
            crashed = pre * (1.0 - break_drop)            # often below standalone
            new_path[brk + 1] = crashed
            new_path[brk + 1:] = crashed
            res = brk + 1                                 # resolve at the crash

        prices[did] = new_path
        weights.iloc[ann:res, i] = 1.0                    # long over live window
        deals.append({"deal_id": did, "announce_row": ann, "resolve_row": int(res),
                      "completes": bool(completes), "entry_spread": entry_spread})

    return prices, weights, deals


def equalize_weights(weights: pd.DataFrame, gross_leverage: float = 1.0,
                     min_names: int = 12) -> pd.DataFrame:
    """Equal-weight the deals live each day, with a diversification cap.

    Capital is spread across the currently-active deals, but never across fewer
    than `min_names` notional slots -- so when only a couple of deals are open we
    stay deliberately under-invested rather than concentrating the whole book in
    one name. This is what keeps a single deal break from being catastrophic
    (real arb desks cap single-deal exposure for exactly this reason)."""
    live = weights.sum(axis=1)
    denom = live.clip(lower=min_names)
    return weights.div(denom.replace(0, np.nan), axis=0).fillna(0.0) * gross_leverage
