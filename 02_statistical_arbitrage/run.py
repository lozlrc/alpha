"""Run all four statistical-arbitrage strategies and rank them.

Strategies
----------
1. pairs_trading          -- rolling-OLS hedge ratio, z-scored spread, entry/exit.
2. cointegration          -- target vs basket, EG/ADF-confirmed, trade the residual.
3. cross_sectional_reversal -- 1-5d reversal on a mean-reverting panel (and a
                             control showing it FAILS on the momentum panel).
4. lead_lag               -- followers track a liquid leader's prior-day return.

Run:  ../.venv/bin/python run.py [--seed N] [--cost-bps X]

Prints a Sharpe-ranked leaderboard and writes equity.png + leaderboard.csv.
All data is synthetic and offline; every P&L stream is net of cost and
lookahead-safe (>=1 day execution lag).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402

import pairs_trading as pt  # noqa: E402
import cointegration as ci  # noqa: E402
import cross_sectional_reversal as xr  # noqa: E402
import lead_lag as ll  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    args = ap.parse_args()
    c = args.cost_bps

    curves = {}          # name -> net return Series (for plotting)
    summaries = []       # list of summary dicts (for leaderboard)

    # ---- 1) pairs trading -------------------------------------------------
    pairs = pt.generate_pairs(n_pairs=60, seed=args.seed + 11)
    pairs_ret = pt.pairs_portfolio(pairs, lookback=250, z_window=30,
                                   entry=2.0, exit=0.5, cost_bps=c)
    # crude per-day turnover proxy for reporting (fraction of pairs trading)
    curves["pairs_trading"] = pairs_ret
    summaries.append(core.metrics.summary(pairs_ret, "pairs_trading"))

    # ---- 2) cointegration (EG/ADF confirmed) ------------------------------
    target, basket, w_true = ci.generate_basket(k=4, seed=args.seed + 23)
    gate = ci.confirm_cointegration(target, basket, insample=0.5)
    print(f"\n[cointegration] EG/ADF gate on in-sample slice "
          f"(n={gate['n_insample']}): ADF p={gate['adf_pvalue']:.4f}, "
          f"coint p={gate['coint_pvalue']:.4f} -> "
          f"{'COINTEGRATED' if gate['is_cointegrated'] else 'NOT cointegrated'}")
    coint_ret = ci.cointegration_pnl(target, basket, gate["beta"],
                                     beta_window=250, z_window=45,
                                     entry=1.5, exit=0.4, cost_bps=c)
    curves["cointegration"] = coint_ret
    summaries.append(core.metrics.summary(coint_ret, "cointegration"))

    # ---- 3) cross-sectional reversal --------------------------------------
    rev_md = xr.generate_reversal_panel(seed=args.seed + 31)
    rev_res = core.long_short_backtest(xr.reversal_signal(rev_md, window=3),
                                       rev_md.returns, quantiles=5, cost_bps=c,
                                       name="xs_reversal")
    curves["xs_reversal"] = rev_res.returns
    summaries.append(rev_res.summary())

    # control: SAME reversal signal on the MOMENTUM panel -> should be negative
    mom_md = core.generate_market(seed=args.seed)
    rev_on_mom = core.long_short_backtest(xr.reversal_signal(mom_md, window=3),
                                          mom_md.returns, quantiles=5, cost_bps=c,
                                          name="xs_reversal_on_momentum(control)")
    s_ctrl = rev_on_mom.summary()
    print(f"[reversal] control on momentum panel: Sharpe "
          f"{s_ctrl['sharpe']:.2f} (expected < 0 -- reversal fails where "
          f"momentum is embedded)")

    # ---- 4) lead-lag ------------------------------------------------------
    ll_md, leader_ret = ll.generate_lead_lag(seed=args.seed + 43)
    ll_res = core.long_short_backtest(ll.lead_lag_signal(ll_md, leader_ret, beta_window=120),
                                      ll_md.returns, quantiles=5, cost_bps=c,
                                      name="lead_lag")
    curves["lead_lag"] = ll_res.returns
    summaries.append(ll_res.summary())

    # ---- leaderboard + outputs -------------------------------------------
    print(f"\n=== Statistical arbitrage (seed={args.seed}, cost={c}bps) ===")
    board = core.format_leaderboard(summaries)
    print(board.round(3).to_string())

    plot_path = os.path.join(HERE, "equity.png")
    core.plot_equity(curves, plot_path,
                     title="Statistical arbitrage — net-of-cost equity")
    csv_path = os.path.join(HERE, "leaderboard.csv")
    board.round(4).to_csv(csv_path)
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
