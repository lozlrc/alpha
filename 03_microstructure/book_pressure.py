"""Top-of-book pressure strategy on the synthetic LOB tape.

Idea
----
The resting size imbalance at the top of book,

    book_pressure_t = (bid_size_t - ask_size_t) / (bid_size_t + ask_size_t)  in (-1, 1),

leads short-term direction: a much thicker bid than ask means buyers are queued
and the next mid tends to tick up.  (In ``orderbook_sim`` the book imbalance is
wired to lead the *next* bar's signed flow, which in turn moves the mid -- so
the predictive link is real, but one step weaker than raw OFI.)

    pressure_t  >  +threshold   ->  long  one bar
    pressure_t  <  -threshold   ->  short one bar
    otherwise                   ->  flat

Same discipline as the OFI strategy: decide at t, earn t->t+1 (lag=1), and pay
the half-spread on every book crossing.  The signal is computed directly from
quoted sizes here (we do *not* peek at the latent ``book_imbalance`` column),
so it carries the realistic lognormal size noise.
"""
from __future__ import annotations

import pandas as pd

# reuse the identical lookahead-safe backtester so cost accounting matches
from order_flow_imbalance import backtest_signal, spread_sweep  # noqa: F401


def book_pressure_feature(ob: pd.DataFrame) -> pd.Series:
    """(bid_size - ask_size) / (bid_size + ask_size) from quoted top-of-book sizes."""
    denom = (ob["bid_size"] + ob["ask_size"]).replace(0, pd.NA)
    return ((ob["bid_size"] - ob["ask_size"]) / denom).astype(float).fillna(0.0)


def book_pressure_signal(ob: pd.DataFrame, threshold: float = 0.20) -> pd.Series:
    """+1 / -1 / 0 position from book pressure vs a symmetric threshold."""
    bp = book_pressure_feature(ob)
    pos = pd.Series(0.0, index=ob.index)
    pos[bp > threshold] = 1.0
    pos[bp < -threshold] = -1.0
    return pos


if __name__ == "__main__":
    from orderbook_sim import generate_orderbook

    ob = generate_orderbook(n_bars=100_000, seed=11)
    pos = book_pressure_signal(ob, threshold=0.20)
    bt = backtest_signal(ob, pos, spread_mult=1.0)
    g, n = bt["gross"], bt["net"]
    print(f"active bars: {(pos != 0).mean() * 100:.1f}%  "
          f"avg turnover/bar: {bt['turnover'].mean():.3f}")
    print(f"gross mean/bar: {g.mean() * 1e4:+.3f} bp   "
          f"net mean/bar: {n.mean() * 1e4:+.3f} bp")
