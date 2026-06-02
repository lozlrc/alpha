"""Alpha decay: the central alt-data lesson.

A sentiment edge lives in a short forward window (here ``horizon`` days). The
faster you act, the more of that window you capture; wait a few days -- because
the data is now public, widely sold, or you are slow -- and the edge evaporates.

We measure this by trading the *same* sentiment signal with increasing
execution delay. The backtest engine already enforces no-lookahead via its
``lag`` parameter (weights are held ``shift(lag)`` then earn ``returns``):

    delay d  ->  lag = 1 + d   ->  a position from sentiment(t) earns r_{t+1+d}

delay 0 captures r_{t+1} (the first, freshest day of the previewed window);
larger delays push past the ``horizon``-day window where there is no signal
left, so Sharpe falls toward zero (or noise).
"""
from __future__ import annotations

import pandas as pd

import core
from sentiment import make_sentiment


def decay_curve(md, delays=(0, 1, 2, 3, 5, 10), horizon: int = 10,
                half_life: float = 2.0, a: float = 0.15, noise: float = 5.0,
                cost_bps: float = 1.0, quantiles: int = 5,
                seed: int = 11) -> pd.DataFrame:
    """Net Sharpe of the sentiment L/S strategy vs execution delay (days).

    Returns a DataFrame indexed by delay with columns [sharpe, ann_return,
    avg_turnover]. Uses the SAME signal throughout; only the trading lag moves.
    """
    sig = make_sentiment(md, horizon=horizon, half_life=half_life, a=a,
                         noise=noise, seed=seed)
    r = md.returns

    rows = []
    for d in delays:
        res = core.long_short_backtest(
            sig, r, quantiles=quantiles, cost_bps=cost_bps, lag=1 + d,
            name=f"sentiment_delay{d}")
        s = res.summary()
        rows.append({"delay": d, "sharpe": s["sharpe"],
                     "ann_return": s["ann_return"],
                     "avg_turnover": s.get("avg_turnover", float("nan"))})
    return pd.DataFrame(rows).set_index("delay")


def plot_decay(curve: pd.DataFrame, path: str,
               title: str = "Alpha decay — sentiment Sharpe vs execution delay") -> str:
    """Save a Sharpe-vs-delay line plot (matplotlib Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve.index, curve["sharpe"], marker="o", lw=1.8, color="#c0392b")
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Execution delay (trading days after signal)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    for d, row in curve.iterrows():
        ax.annotate(f"{row['sharpe']:.2f}", (d, row["sharpe"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8,
                    ha="center")
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
