"""Backtest three event-driven strategies on synthetic, event-overlaid data.

  1. PEAD            -- post-earnings-announcement drift (dollar-neutral L/S).
  2. index_rebalance -- long index additions / short deletions into the
                        effective date (dollar-neutral L/S).
  3. merger_arb      -- long M&A targets to capture the deal spread; fat-tailed
                        payoff (steady gains, occasional crashes on deal breaks).

All three are lookahead-safe: positions are placed on the announcement-date row
and the engine's +1 execution lag means we are only ever invested the session
AFTER the event is observable. Everything is net of transaction costs.

Run:  ../.venv/bin/python run.py [--seed N] [--cost-bps X]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
import index_rebalance  # noqa: E402  (local module)
import merger_arb  # noqa: E402  (local module)
import pead  # noqa: E402  (local module)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    args = ap.parse_args()

    md = core.generate_market(seed=args.seed)
    results = []          # list of (BacktestResult-like) summary dicts
    curves = {}           # name -> net return series for the equity plot

    # ----- 1) PEAD -----------------------------------------------------------
    ret_pead, surprise = pead.build_earnings_panel(md, seed=args.seed + 4)
    w_pead = pead.pead_weights(surprise, hold_days=40)
    res_pead = core.backtest_weights(ret_pead, w_pead, cost_bps=args.cost_bps,
                                     lag=1, name="PEAD")
    results.append(res_pead.summary())
    curves["PEAD"] = res_pead.returns

    # ----- 2) Index rebalance ------------------------------------------------
    ret_idx, idx_events, idx_signal = index_rebalance.build_rebalance_panel(
        md, seed=args.seed + 16)
    w_idx = index_rebalance.rebalance_weights(idx_signal)
    res_idx = core.backtest_weights(ret_idx, w_idx, cost_bps=args.cost_bps,
                                    lag=1, name="index_rebalance")
    results.append(res_idx.summary())
    curves["index_rebalance"] = res_idx.returns

    # ----- 3) Merger arbitrage ----------------------------------------------
    deal_prices, deal_w_raw, deals = merger_arb.simulate_deals(md, seed=args.seed + 30)
    deal_returns = deal_prices.pct_change(fill_method=None)
    w_arb = merger_arb.equalize_weights(deal_w_raw)
    res_arb = core.backtest_weights(deal_returns, w_arb, cost_bps=args.cost_bps,
                                    lag=1, name="merger_arb")
    results.append(res_arb.summary())
    curves["merger_arb"] = res_arb.returns

    # ----- leaderboard -------------------------------------------------------
    print(f"\n=== Event-driven strategies (seed={args.seed}, cost={args.cost_bps}bps) ===")
    board = core.format_leaderboard(results)
    print(board.round(3).to_string())

    # merger-arb tail diagnostics (the whole point of the strategy)
    n_complete = sum(d["completes"] for d in deals)
    worst_day = res_arb.returns.min()
    print(f"\nmerger_arb tail: {n_complete}/{len(deals)} deals completed "
          f"({100 * n_complete / len(deals):.0f}%); "
          f"worst day {100 * worst_day:.2f}%, "
          f"max drawdown {100 * core.metrics.max_drawdown(res_arb.returns):.2f}%")
    print(f"index_rebalance: {len(idx_events)} reconstitution events simulated")

    plot_path = os.path.join(HERE, "equity.png")
    core.plot_equity(curves, plot_path, title="Event-driven strategies — net equity")
    csv_path = os.path.join(HERE, "leaderboard.csv")
    board.round(4).to_csv(csv_path)
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
