"""Backtest the cross-asset volatility / carry strategy family.

Run:  ../.venv/bin/python run.py [--cost-bps X] [--quantiles N]

Three strategies, all offline on self-generated synthetic data:

  1. carry            -- cross-sectional long high-carry / short low-carry (FX /
                         commodity carry premium), with carry-unwind crashes.
  2. short_variance   -- systematically sell a variance-swap proxy, earning the
                         variance risk premium with a fat left tail (vol spikes).
  3. dispersion       -- short index vol / long single-name vol, harvesting the
                         implied-vs-realized correlation gap (short correlation).

Prints a Sharpe-ranked leaderboard (with a max_drawdown column), saves
equity.png and leaderboard.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402

import carry as carry_mod  # noqa: E402  (local modules)
import dispersion as disp_mod  # noqa: E402
import vol_risk_premium as vrp_mod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--quantiles", type=int, default=4)
    args = ap.parse_args()

    summaries = []
    curves = {}

    # ---- 1) cross-sectional carry (long/short, dollar-neutral) ----
    cd = carry_mod.generate_carry_market()
    carry_res = core.long_short_backtest(
        carry_mod.carry_signal(cd), cd.returns, quantiles=args.quantiles,
        cost_bps=args.cost_bps, name="carry_long_short")
    summaries.append(carry_res.summary())
    curves[carry_res.name] = carry_res.returns

    # ---- 2) short variance (variance risk premium) ----
    vd = vrp_mod.generate_vrp_process()
    vrp_ret = vrp_mod.short_variance_returns(vd, cost_bps=args.cost_bps)
    summaries.append(core.metrics.summary(vrp_ret, "short_variance"))
    curves["short_variance"] = vrp_ret

    # ---- 3) dispersion (short index vol / long single-name vol) ----
    dd = disp_mod.generate_dispersion_process()
    disp_ret = disp_mod.dispersion_returns(dd, cost_bps=args.cost_bps)
    summaries.append(core.metrics.summary(disp_ret, "dispersion"))
    curves["dispersion"] = disp_ret

    # ---- leaderboard ----
    print(f"\n=== Cross-asset vol / carry (cost={args.cost_bps}bps, q={args.quantiles}) ===")
    board = core.format_leaderboard(summaries)
    print(board.round(3).to_string())

    # tail emphasis for the short-variance trade -- Sharpe alone flatters it
    worst_day = float(vrp_ret.min())
    worst_date = vrp_ret.idxmin().date()
    print(f"\nVRP tail check  -> worst day: {worst_day * 100:+.2f}% on {worst_date} "
          f"(max drawdown {core.metrics.max_drawdown(vrp_ret) * 100:.1f}%)")
    print(f"Carry tail check-> max drawdown {core.metrics.max_drawdown(carry_res.returns) * 100:.1f}% "
          f"(carry-unwind crashes)")

    plot_path = os.path.join(HERE, "equity.png")
    core.plot_equity(curves, plot_path,
                     title="Cross-asset vol / carry — net equity", log=True)
    csv_path = os.path.join(HERE, "leaderboard.csv")
    board.round(4).to_csv(csv_path)
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
