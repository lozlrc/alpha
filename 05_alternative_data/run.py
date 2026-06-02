"""Alternative-data strategies on synthetic equities (offline, no live feed).

Run:  ../.venv/bin/python run.py [--seed N] [--cost-bps X]

Three alt-data ideas, each built CAUSALLY (a signal observable at day t that
previews the near future, never peeking at the same-day/past return):

  1. sentiment       -- noisy leading indicator of the next few days' return.
  2. web_traffic     -- nowcast of the next earnings surprise's sign.
  3. alpha decay     -- the same sentiment signal traded with growing execution
                        delay; Sharpe falls as the (short) edge goes stale.

Prints a Sharpe-ranked leaderboard and saves equity.png, decay.png,
leaderboard.csv in this folder.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
import sentiment as sent  # noqa: E402  (local module)
import web_traffic as wt  # noqa: E402  (local module)
import alpha_decay as ad  # noqa: E402  (local module)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--quantiles", type=int, default=5)
    args = ap.parse_args()

    md = core.generate_market(seed=args.seed)
    r = md.returns

    # ---- 1) sentiment: noisy leading indicator of the next few days ----
    sentiment = sent.make_sentiment(md, horizon=10, half_life=2.0, a=0.15,
                                    noise=5.0, seed=11)
    res_sent = core.long_short_backtest(
        sentiment, r, quantiles=args.quantiles, cost_bps=args.cost_bps,
        name="sentiment_LS")

    # ---- 3) web traffic: nowcast the next earnings event / drift ----
    traffic = wt.make_web_traffic(md, b=0.20, noise=3.0, seed=23)
    res_wt = core.long_short_backtest(
        traffic, r, quantiles=args.quantiles, cost_bps=args.cost_bps,
        name="web_traffic_LS")

    results = [res_sent, res_wt]

    print(f"\n=== Alternative-data strategies (seed={args.seed}, "
          f"cost={args.cost_bps}bps, q={args.quantiles}) ===")

    # ---- 2) alpha decay: same sentiment, increasing execution delay ----
    delays = (0, 1, 2, 3, 5, 10)
    curve = ad.decay_curve(md, delays=delays, horizon=10, half_life=2.0, a=0.15,
                           noise=5.0, cost_bps=args.cost_bps,
                           quantiles=args.quantiles, seed=11)
    print("\nAlpha decay — net Sharpe vs execution delay (days):")
    print(curve.round(3).to_string())

    # leaderboard across the headline strategies + each delayed sentiment book
    board_rows = [res_sent.summary(), res_wt.summary()]
    for d in delays:
        res_d = core.long_short_backtest(
            sentiment, r, quantiles=args.quantiles, cost_bps=args.cost_bps,
            lag=1 + d, name=f"sentiment_delay{d}")
        board_rows.append(res_d.summary())
        if d in (5, 10):  # stale books on the equity chart (delay0 == sentiment_LS)
            results.append(res_d)

    board = core.format_leaderboard(board_rows)
    print("\nLeaderboard (Sharpe-ranked):")
    print(board.round(3).to_string())

    # ---- plots + csv ----
    eq_path = os.path.join(HERE, "equity.png")
    core.plot_equity(results, eq_path,
                     title="Alt-data strategies — L/S equity (net of costs)")
    decay_path = os.path.join(HERE, "decay.png")
    ad.plot_decay(curve, decay_path)
    csv_path = os.path.join(HERE, "leaderboard.csv")
    board.round(4).to_csv(csv_path)

    print(f"\nSaved: {eq_path}")
    print(f"Saved: {decay_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
