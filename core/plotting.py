"""Matplotlib helpers that render to PNG (Agg backend -- no display needed)."""
from __future__ import annotations

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics  # noqa: E402


def plot_equity(results, path: str, title: str = "Equity curves", log: bool = True,
                dump_returns: bool = True) -> str:
    """Plot cumulative equity curves and save to `path`.

    `results` may be a dict {name: returns_series} or a list of BacktestResult.
    If `dump_returns` is True, the underlying per-strategy return streams are
    also written to `<path-stem>_returns.csv` (used by the portfolio layer to
    combine strategies across families -- no need to touch each family's code).
    """
    if isinstance(results, dict):
        items = list(results.items())
    else:
        items = [(r.name, r.returns) for r in results]

    if dump_returns:
        try:
            df = pd.DataFrame({name: pd.Series(ret) for name, ret in items})
            df.to_csv(os.path.splitext(path)[0] + "_returns.csv")
        except Exception:
            pass  # plotting must never fail because of the optional dump

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, ret in items:
        eq = metrics.equity_curve(ret)
        ax.plot(eq.index, eq.values, label=name, lw=1.3)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best", fontsize=8)
    if log:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
