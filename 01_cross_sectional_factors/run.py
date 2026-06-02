"""Backtest cross-sectional equity factors (momentum, value, quality, low-vol).

Run:  ../.venv/bin/python run.py [--seed N] [--cost-bps X]

Builds dollar-neutral long/short decile portfolios for each factor, plus a
sector-neutral momentum variant and an equal-weight multifactor blend, then
prints a Sharpe-ranked leaderboard and saves an equity-curve plot.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
import factors  # noqa: E402  (local module)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--quantiles", type=int, default=5)
    args = ap.parse_args()

    md = core.generate_market(seed=args.seed)
    r = md.returns

    signals = {
        "momentum_12_1": factors.momentum(md),
        "value_BP": factors.value(md),
        "quality_ROE": factors.quality(md),
        "low_vol": factors.low_volatility(md),
        "reversal_5d": factors.short_term_reversal(md),
        "multifactor": factors.combined(md),
    }

    results = []
    for name, sig in signals.items():
        res = core.long_short_backtest(sig, r, quantiles=args.quantiles,
                                       cost_bps=args.cost_bps, name=name)
        results.append(res)

    # sector-neutral momentum: demean the signal within each sector first
    results.append(core.long_short_backtest(
        factors.momentum(md), r, quantiles=args.quantiles, cost_bps=args.cost_bps,
        neutralize_groups=md.sectors, name="momentum_sector_neutral"))

    print(f"\n=== Cross-sectional factors (seed={args.seed}, cost={args.cost_bps}bps, "
          f"q={args.quantiles}) ===")
    board = core.format_leaderboard([res.summary() for res in results])
    print(board.round(3).to_string())

    plot_path = os.path.join(HERE, "equity.png")
    core.plot_equity(results, plot_path, title="Cross-sectional factors — L/S equity")
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
