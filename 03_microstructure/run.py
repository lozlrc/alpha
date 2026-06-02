"""Backtest intraday microstructure strategies on a synthetic limit-order book.

Run:  ../.venv/bin/python run.py [--seed N] [--bars N] [--ofi-thr X] [--bp-thr X]

Generates a reduced-form LOB tape with embedded price impact (order flow leads
the next mid), then backtests two fast signals -- Order Flow Imbalance (OFI) and
top-of-book pressure -- with strict lookahead discipline (decide at bar t, earn
the move t->t+1) and the half-spread paid on every book crossing.

The headline lesson is the GROSS-vs-NET gap: gross alpha is large, but crossing
the spread every time you trade eats most of it.  We therefore print both a
GROSS and a NET row per strategy in the leaderboard, plot both equity curves,
and dump a spread-widening sweep showing net P&L collapse as the book widens.

Bars are treated as 1-minute bars, so the annualizer is
periods_per_year = 252 * 390 = 98,280.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402

import book_pressure as bp  # noqa: E402  (local modules)
import order_flow_imbalance as ofi  # noqa: E402
from orderbook_sim import generate_orderbook  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERIODS_PER_YEAR = 252 * 390  # 1-minute bars -> 98,280 bars/year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--bars", type=int, default=200_000)
    ap.add_argument("--ofi-thr", type=float, default=1.0)
    ap.add_argument("--bp-thr", type=float, default=0.20)
    ap.add_argument("--fee-per-turn", type=float, default=0.0,
                    help="flat fee charged per unit of turnover (fraction of notional)")
    args = ap.parse_args()

    ob = generate_orderbook(n_bars=args.bars, seed=args.seed)

    # sanity: how strong is the planted edge in this draw?
    dmid = ob["mid"].diff().shift(-1)
    corr_ofi = ob["ofi"].iloc[:-1].corr(dmid.iloc[:-1])
    corr_bp = bp.book_pressure_feature(ob).iloc[:-1].corr(dmid.iloc[:-1])

    # ---- build {-1,0,+1} positions, then lookahead-safe gross/net P&L ----
    strategies = {
        "OFI": ofi.ofi_signal(ob, threshold=args.ofi_thr),
        "BookPressure": bp.book_pressure_signal(ob, threshold=args.bp_thr),
    }

    curves: dict[str, "object"] = {}
    summaries = []
    for name, pos in strategies.items():
        res = ofi.backtest_signal(ob, pos, spread_mult=1.0, fee_per_turn=args.fee_per_turn)
        # GROSS and NET rows so the cost drag is explicit in the leaderboard
        summaries.append(core.metrics.summary(
            res["gross"], f"{name} (gross)", periods_per_year=PERIODS_PER_YEAR,
            turnover=res["turnover"]))
        summaries.append(core.metrics.summary(
            res["net"], f"{name} (net)", periods_per_year=PERIODS_PER_YEAR,
            turnover=res["turnover"]))
        curves[f"{name} (gross)"] = res["gross"]
        curves[f"{name} (net)"] = res["net"]

    print(f"\n=== Microstructure (seed={args.seed}, bars={args.bars:,}, "
          f"ofi_thr={args.ofi_thr}, bp_thr={args.bp_thr}) ===")
    print(f"mean spread = {ob['spread'].mean():.4f} "
          f"({ob['spread'].mean() / ob.attrs['tick']:.1f} ticks)   "
          f"planted corr: OFI->next mid {corr_ofi:+.3f}, "
          f"BookPressure->next mid {corr_bp:+.3f}")

    board = core.format_leaderboard(summaries)
    print()
    print(board.round(3).to_string())

    # ---- spread-widening sweep for OFI: watch net collapse as book widens ----
    print("\nOFI net P&L vs spread multiplier (the cost-domination lesson):")
    sweep = ofi.spread_sweep(ob, threshold=args.ofi_thr,
                             multipliers=(0.5, 1.0, 2.0, 4.0),
                             fee_per_turn=args.fee_per_turn)
    print(sweep.round(4).to_string())

    # ---- outputs (equity curves are near-linear at this scale -> log=False) ----
    plot_path = os.path.join(HERE, "equity.png")
    core.plot_equity(curves, plot_path,
                     title="Microstructure — gross vs net equity (1-min bars)",
                     log=False)
    csv_path = os.path.join(HERE, "leaderboard.csv")
    board.round(4).to_csv(csv_path)
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
