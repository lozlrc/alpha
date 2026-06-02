#!/usr/bin/env python
"""Run every strategy family's backtest and print one master leaderboard.

    .venv/bin/python run_all.py

This is the top-level driver for the alpha suite. It runs each NN_*/run.py as
an isolated subprocess (everything is OFFLINE and uses synthetic data -- there
is no live-market or broker connection anywhere in this project), then collects
each family's leaderboard.csv into a single Sharpe-ranked table.

IMPORTANT caveats when reading the combined table:
  * Sharpes are most meaningful WITHIN a family. The microstructure family is
    annualized from intraday bars, so its Sharpes are huge by construction
    (sqrt(N) with ~100k bets/yr) and are NOT comparable to the daily families.
  * All numbers are on SYNTHETIC data with deliberately planted structure, so
    they demonstrate that each strategy *recovers its signal* -- they are not a
    claim about live-market profitability. Real data is far less generous.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable  # the venv python you launched this with


def main():
    # The portfolio family CONSUMES every other family's equity_returns.csv, so it
    # must run AFTER them regardless of its folder number (e.g. 09 feeds into 08).
    folders = sorted((d for d in glob.glob(os.path.join(HERE, "[0-9][0-9]_*"))
                      if os.path.isdir(d)),
                     key=lambda d: (os.path.basename(d).endswith("_portfolio"),
                                    os.path.basename(d)))
    rows, n_pass, n_fail = [], 0, 0

    print("=" * 72)
    print("ALPHA SUITE  --  running all families (offline, synthetic data only)")
    print("=" * 72)

    for folder in folders:
        name = os.path.basename(folder)
        runpy = os.path.join(folder, "run.py")
        if not os.path.exists(runpy):
            print(f"[skip] {name}: no run.py")
            continue
        t0 = time.time()
        try:
            proc = subprocess.run([PY, "run.py"], cwd=folder, capture_output=True,
                                  text=True, timeout=900)
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {name:30s} (timeout)")
            n_fail += 1
            continue
        dt = time.time() - t0
        ok = proc.returncode == 0
        print(f"[{'PASS' if ok else 'FAIL'}] {name:30s} ({dt:5.1f}s)")
        if not ok:
            n_fail += 1
            print("  --- stderr tail ---")
            print("\n".join("  " + ln for ln in proc.stderr.strip().splitlines()[-15:]))
            continue
        n_pass += 1
        csv = os.path.join(folder, "leaderboard.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df = df.rename(columns={df.columns[0]: "strategy"})
            df.insert(0, "family", name)
            rows.append(df)

    if rows:
        master = pd.concat(rows, ignore_index=True)
        if "sharpe" in master.columns:
            master = master.sort_values("sharpe", ascending=False)
        cols = [c for c in ["family", "strategy", "sharpe", "ann_return", "ann_vol",
                            "max_drawdown", "hit_rate"] if c in master.columns]
        print("\n" + "=" * 72)
        print("MASTER LEADERBOARD  (sorted by Sharpe -- compare WITHIN family; see header)")
        print("=" * 72)
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(master[cols].round(3).to_string(index=False))
        out = os.path.join(HERE, "leaderboard_all.csv")
        master.to_csv(out, index=False)
        print(f"\nFamilies: {n_pass} passed, {n_fail} failed."
              f"  Combined leaderboard -> {out}")
    else:
        print(f"\nNo leaderboards collected. {n_pass} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
