"""ML-driven cross-sectional strategy: honest OOS vs the overfitting trap.

Run:  ../.venv/bin/python run.py [--seed N] [--cost-bps X] [--splits K]

Pipeline
--------
1. Engineer point-in-time features + a forward H-day return label (features.py).
2. Honest signal: PURGED + EMBARGOED walk-forward CV produces out-of-sample
   predictions for the whole period (cv.py); ranked cross-sectionally and
   backtested net of costs.
3. Leaky signal: the SAME gradient-boosted model fit in-sample with no purge,
   predicting its own training rows -- demonstrates the overfitting trap with an
   absurdly high (fake) Sharpe.
4. Baseline: a simple equal-weight z-score blend of the raw features (the kind
   of hand-built linear factor combo ML is supposed to beat).
5. Leaderboard + equity plot comparing all three; feature importances printed.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

import core  # noqa: E402
import cv as cvmod  # noqa: E402  (local module)
import features as feat  # noqa: E402
import model as mdl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# Classic 4-factor blend (cf. 01_cross_sectional_factors.factors.combined):
# 12-1 momentum, value (book/price), quality (profitability), low-vol.
_BASELINE_FACTORS = {
    "mom_252_21": 1.0,
    "book_to_price": 1.0,
    "profitability": 1.0,
    "vol_63": -1.0,  # low vol = bullish
}


def zscore_baseline(panel: pd.DataFrame) -> pd.Series:
    """Simple linear baseline: equal-weight blend of the four classic
    (already z-scored) factors -- the same hand-built combo as the reference
    factor strategy. No fitting; this is what the ML model has to beat."""
    out = None
    for col, sign in _BASELINE_FACTORS.items():
        contrib = sign * panel[col]
        out = contrib if out is None else out + contrib
    return out.rename("pred")


def backtest_signal(preds: pd.Series, returns: pd.DataFrame, name: str,
                    cost_bps: float, quantiles: int) -> core.BacktestResult:
    """(date,ticker) predictions -> wide signal -> dollar-neutral L/S backtest."""
    sig = mdl.predictions_to_wide(preds)
    sig = sig.reindex(index=returns.index, columns=returns.columns)
    return core.long_short_backtest(sig, returns, quantiles=quantiles,
                                    cost_bps=cost_bps, name=name)


def print_importances(panel: pd.DataFrame, seed: int) -> None:
    """Permutation importance on a held-out tail slice of an honestly-fit model."""
    udates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    cut = udates[int(len(udates) * 0.8)]
    rd = pd.Series(panel.index.get_level_values("date"), index=panel.index)
    # Purge the H-day seam between train and the held-out importance slice.
    seam = cut - pd.Timedelta(days=int(feat.H * 1.6))
    train = panel[rd < seam]
    holdout = panel[rd >= cut]
    model = mdl.fit_model(train, random_state=seed)
    imp = permutation_importance(
        model, holdout[feat.FEATURE_NAMES].to_numpy(), holdout["fwd_ret"].to_numpy(),
        n_repeats=5, random_state=seed, scoring="r2",
    )
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nFeature importances (permutation, held-out R^2 drop):")
    for j in order:
        print(f"    {feat.FEATURE_NAMES[j]:<16} {imp.importances_mean[j]:+.5f} "
              f"+/- {imp.importances_std[j]:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--splits", type=int, default=8)
    args = ap.parse_args()

    print(f"=== ML-driven cross-sectional (seed={args.seed}, cost={args.cost_bps}bps, "
          f"q={args.quantiles}, H={feat.H}d, splits={args.splits}) ===")

    md = core.generate_market(seed=args.seed)
    returns = md.returns
    panel = feat.build_panel(md, horizon=feat.H)
    print(f"Panel: {len(panel):,} (date,ticker) rows, {len(feat.FEATURE_NAMES)} features, "
          f"label = forward {feat.H}d return\n")

    # 1) Honest out-of-sample: purged + embargoed walk-forward.
    print("Honest walk-forward (purged + embargoed) OOS folds:")
    oos = cvmod.walk_forward_oos(panel, n_splits=args.splits, horizon=feat.H,
                                 random_state=args.seed)

    # 2) Leaky in-sample fit predicting its own training rows (the trap).
    leaky = cvmod.fit_leaky_insample(panel, random_state=args.seed)

    # 3) Naive linear z-score blend baseline.
    base = zscore_baseline(panel)

    res_oos = backtest_signal(oos, returns, "ml_honest_oos", args.cost_bps, args.quantiles)
    res_leak = backtest_signal(leaky, returns, "ml_leaky_insample", args.cost_bps, args.quantiles)
    res_base = backtest_signal(base, returns, "zscore_baseline", args.cost_bps, args.quantiles)

    print_importances(panel, args.seed)

    results = [res_oos, res_leak, res_base]
    board = core.format_leaderboard([r.summary() for r in results])
    print("\n=== Leaderboard (Sharpe-ranked) ===")
    print(board.round(3).to_string())

    s_oos = res_oos.summary()["sharpe"]
    s_leak = res_leak.summary()["sharpe"]
    s_base = res_base.summary()["sharpe"]
    print("\n--- The overfitting trap ---")
    print(f"  Honest OOS Sharpe   {s_oos:5.2f}   (purged + embargoed walk-forward)")
    print(f"  Leaky  IS  Sharpe   {s_leak:5.2f}   (in-sample fit, NO purge -> fake alpha)")
    print(f"  Baseline   Sharpe   {s_base:5.2f}   (naive z-score factor blend)")
    print(f"  Leak inflation: {s_leak - s_oos:+.2f} Sharpe of pure look-ahead.")

    plot_path = os.path.join(HERE, "equity.png")
    core.plot_equity(results, plot_path,
                     title="ML-driven L/S: honest OOS vs leaky vs baseline")
    csv_path = os.path.join(HERE, "leaderboard.csv")
    board.round(4).to_csv(csv_path)
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
