"""Order Flow Imbalance (OFI) strategy on the synthetic LOB tape.

Idea
----
Order flow leads price (see ``orderbook_sim``: mid_{t+1}-mid_t = impact*OFI_t +
noise).  So:

    OFI_t  >  +threshold   ->  go long  for one bar
    OFI_t  <  -threshold   ->  go short for one bar
    otherwise              ->  flat

LOOKAHEAD DISCIPLINE
--------------------
The position decided from OFI_t is *shifted forward one bar* and earns the move
from t to t+1, exactly like the daily engine's ``lag=1``.

COSTS ARE CENTRAL
-----------------
Crossing the book is not free.  Every time the held position *changes* we pay
the half-spread on the traded notional (a marketable order lifts the ask / hits
the bid), plus an optional flat fee.  Because the per-bar edge from a tiny
``impact`` is comparable to a tick, the half-spread eats most of the gross alpha
-- the honest microstructure lesson.  ``net_returns`` therefore reports a
spread-multiplier sweep so you can watch net Sharpe collapse as the book widens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ofi_signal(ob: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    """+1 / -1 / 0 position from OFI vs a symmetric threshold (uses info up to t)."""
    ofi = ob["ofi"]
    pos = pd.Series(0.0, index=ob.index)
    pos[ofi > threshold] = 1.0
    pos[ofi < -threshold] = -1.0
    return pos


def backtest_signal(
    ob: pd.DataFrame,
    position: pd.Series,
    spread_mult: float = 1.0,
    fee_per_turn: float = 0.0,
    lag: int = 1,
) -> dict:
    """Lookahead-safe per-bar P&L for a {-1,0,+1} position on the LOB tape.

    Returns a dict with gross/net return Series and turnover.  The position is
    lagged by `lag` bars; per-bar gross return = held_position * mid_return.
    Costs = (turnover) * (half_spread/mid * spread_mult) + turnover*fee_per_turn,
    where turnover is the absolute change in held position (a full flip = 2 units
    = two book crossings).
    """
    mid = ob["mid"]
    mid_ret = mid.pct_change().fillna(0.0)

    held = position.shift(lag).fillna(0.0)
    gross = held * mid_ret

    # half-spread as a fraction of mid, scaled by the stress multiplier
    half_spread_frac = (0.5 * ob["spread"] / mid) * spread_mult
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * half_spread_frac + turnover * fee_per_turn
    net = gross - cost

    return {
        "gross": gross,
        "net": net,
        "turnover": turnover,
        "position": held,
    }


def spread_sweep(
    ob: pd.DataFrame,
    threshold: float = 1.0,
    multipliers=(0.5, 1.0, 2.0, 4.0),
    fee_per_turn: float = 0.0,
) -> pd.DataFrame:
    """Net mean per-bar return as the spread widens -- shows net degradation.

    Returns a small DataFrame indexed by spread multiplier with the net mean
    per-bar return (in basis points) and the fraction of bars that are still
    profitable after costs.
    """
    pos = ofi_signal(ob, threshold)
    rows = []
    for m in multipliers:
        bt = backtest_signal(ob, pos, spread_mult=m, fee_per_turn=fee_per_turn)
        net = bt["net"]
        rows.append(
            {
                "spread_mult": m,
                "net_mean_bp": float(net.mean() * 1e4),
                "net_hit_rate": float((net[net != 0] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("spread_mult")


if __name__ == "__main__":
    from orderbook_sim import generate_orderbook

    ob = generate_orderbook(n_bars=100_000, seed=11)
    pos = ofi_signal(ob, threshold=1.0)
    bt = backtest_signal(ob, pos, spread_mult=1.0, fee_per_turn=0.0)
    g = bt["gross"]
    n = bt["net"]
    print(f"active bars: {(pos != 0).mean() * 100:.1f}%  "
          f"avg turnover/bar: {bt['turnover'].mean():.3f}")
    print(f"gross mean/bar: {g.mean() * 1e4:+.3f} bp   "
          f"net mean/bar: {n.mean() * 1e4:+.3f} bp")
    print("\nspread sweep (net mean bp, net hit rate):")
    print(spread_sweep(ob).round(4).to_string())
