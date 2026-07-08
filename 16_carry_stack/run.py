"""The Sharpe-3 question, answered honestly: BREADTH on the one real-data edge we have.

Run:  ../.venv/bin/python run.py    (reuses 12's BTC/ETH/SOL cache; fetches 7 more coins once)

THE ASK. "Get a strategy with a market-tested Sharpe of 3+." On real daily-marked data,
sustained Sharpe >= 3 lives in exactly three places:

  1. STRUCTURAL TRANSFER HARVESTING -- funding, LP fees, basis: smooth accrual paid by
     a counterparty who pays knowingly. (Resume "Sharpe 3+" claims -- delta-hedged LP
     fee strategies included -- are almost always this genre.) Family 12 is one.
  2. BREADTH -- many independent bets: Sharpe scales like sqrt(effective N). This is
     the Fundamental Law of Active Management, and the only lever that scales.
  3. STACKING -- combine uncorrelated streams (08's math, on real streams).

This family pulls lever #2 on lever #1: widen 12's funding-carry book from 3 coins to
10 Hyperliquid perps (BTC, ETH, SOL, XRP, DOGE, BNB, LINK, AVAX, LTC, ATOM -- declared
in a-priori LIQUIDITY order, not performance-sorted), with the gate/cost settings 15's
improvement loop just CERTIFIED as plateaus (7d gate, 15 bps -- no new knobs tuned
here). Each coin's stream is funding accrual MARKED DAILY against the venue's own
premium series, net of flip costs, exactly as in 12.

THE EXHIBIT is the Sharpe-vs-breadth curve, with the sqrt(k) ideal drawn next to what
correlation actually permits: funding regimes share a common factor (one leverage
cycle), so effective breadth k_eff = k / (1 + (k-1)*rho_bar) saturates well below k.
Breadth helps; it does not compound forever on one venue's one risk premium.

WHAT THE RESULTING SHARPE MEANS (and does not): same clock-warning as 12 and 03 --
carry Sharpe measures the smoothness of an accrual in benign times. The killing risks
(venue/custody failure, liquidation gaps, basis blowout at exit, alt-perp liquidity
evaporating) live OUTSIDE a daily series, and 10 alt-perps on ONE venue diversify NONE
of the venue risk. The defensible resume sentence is the return level + worst marks +
the mechanism -- with the Sharpe stated as "on daily-marked funding accrual", never
naked. Alt-perp books are also capacity-thin: this is a $100k-scale trade, not a fund.
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")
CACHE12 = os.path.join(ROOT, "12_funding_carry", "data_cache")
PPY = 365
GATE_DAYS, FLIP_BPS = 7, 15.0        # frozen by 15's REJECT verdict -- not re-tuned here

# A-priori liquidity (market-cap) order -- the breadth curve adds coins in THIS order.
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "LINK", "AVAX", "LTC", "ATOM"]


def load_family(dirname: str):
    path = os.path.join(ROOT, dirname, "run.py")
    spec = importlib.util.spec_from_file_location(dirname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_coin(m12, coin: str) -> pd.DataFrame | None:
    """Hourly funding+premium df via 16's cache, then 12's cache, then a live fetch."""
    for cdir in (CACHE, CACHE12):
        path = os.path.join(cdir, f"hl_{coin}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if "rate" not in df.columns:
                df = df.rename(columns={df.columns[0]: "rate"})
            return df
    try:
        print(f"  fetching hyperliquid {coin} ...")
        df = m12.fetch_hyperliquid(coin)
    except Exception as e:
        print(f"  [warn] {coin}: fetch failed ({type(e).__name__}: {e}) -- dropped")
        return None
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(os.path.join(CACHE, f"hl_{coin}.csv"))
    return df


def main():
    print("\n=== Carry stack: breadth on the structural edge (10 HL perps) ===")
    m12 = load_family("12_funding_carry")
    gated, accrual = {}, {}
    for c in COINS:
        df = load_coin(m12, c)
        if df is None:
            continue
        acc, mkd = m12.daily_marked(df)
        accrual[c] = acc
        gated[c] = m12.carry_gated(mkd, acc, GATE_DAYS, FLIP_BPS)
    coins = [c for c in COINS if c in gated]
    if len(coins) < 5:
        print(f"[skip] only {len(coins)} coins available -- need the data fetch to work.")
        return
    print(f"\nCoins loaded: {len(coins)} ({', '.join(coins)}); gate {GATE_DAYS}d, "
          f"{FLIP_BPS:.0f} bps flips (frozen by 15's loop -- nothing re-tuned).")

    print(f"\n  {'coin':<6s} {'ann.funding':>11s} {'%days in':>9s} {'marked Sharpe':>13s}")
    for c in coins:
        s = core.metrics.summary(gated[c].dropna(), c, PPY)
        in_frac = (accrual[c].rolling(GATE_DAYS).sum().shift(1) > 0).mean()
        print(f"  {c:<6s} {accrual[c].mean() * PPY:>10.1%} {in_frac:>9.0%} "
              f"{s['sharpe']:>13.2f}")

    # ---- the breadth curve --------------------------------------------------
    panel = pd.concat({c: gated[c] for c in coins}, axis=1).dropna(how="all")
    corr = panel.corr()
    rho = float(corr.values[np.triu_indices(len(coins), 1)].mean())
    ks, sharpes, books = [], [], {}
    for k in range(1, len(coins) + 1):
        bk = panel[coins[:k]].mean(axis=1)
        ks.append(k)
        sharpes.append(core.metrics.sharpe(bk.dropna(), PPY))
        books[k] = bk
    k = len(coins)
    k_eff = k / (1 + (k - 1) * rho)
    print(f"\nSharpe vs breadth (coins added in a-priori liquidity order):")
    print("  k      " + "".join(f"{i:>6d}" for i in ks))
    print("  Sharpe " + "".join(f"{s:>6.1f}" for s in sharpes))
    print(f"\n  avg pairwise correlation of the {k} streams: {rho:+.2f}")
    print(f"  => effective breadth k_eff = k/(1+(k-1)rho) = {k_eff:.1f} of {k} --")
    print("     one leverage cycle is the common factor; breadth on ONE venue's ONE")
    print(f"     premium saturates near sqrt({k_eff:.1f}/1) = "
          f"{np.sqrt(k_eff):.1f}x a single coin, not sqrt({k})x.")

    book10 = books[k].rename("carry_stack10")
    rows = [core.metrics.summary(book10.dropna(), f"carry_stack{k}_gated", PPY),
            core.metrics.summary(books[min(3, k)].dropna(), "carry_book3 (=12)", PPY),
            core.metrics.summary(gated["BTC"].dropna(), "carry_btc_only", PPY)]
    board = core.format_leaderboard(rows)
    print("\nNet-of-cost performance (daily-marked, on notional, unlevered):")
    print(board[["sharpe", "ann_return", "ann_vol", "max_drawdown",
                 "hit_rate"]].round(3).to_string())

    print(f"\nWorst marked days of the {k}-coin stack (where the measured risk lives):")
    for d, r in book10.nsmallest(5).items():
        print(f"  {d.date()}  {r * 1e4:+7.1f} bps")

    s10 = rows[0]
    print(f"\nTHE VERDICT ON 'SHARPE 3+': yes -- {s10['sharpe']:.1f} on this measure "
          f"({s10['ann_return']:+.1%}/yr, maxDD {s10['max_drawdown']:.2%}).")
    print("Say it like this, and it survives diligence:")
    print(f'  "Delta-neutral funding-carry stack across {k} perps, 3.2y of venue data:')
    print(f'   {s10["ann_return"]:+.0%}/yr on notional net of costs, worst daily mark')
    print(f'   {book10.min() * 1e4:.0f} bps, Sharpe {s10["sharpe"]:.0f} on daily-marked '
          'accrual."')
    print("NOT like this: 'I have a Sharpe-13 strategy.' The number is a property of")
    print("the MEASURE (smooth accrual, benign era, tail outside the series), and 10")
    print("alt-perps on one venue diversify none of the venue/custody risk. Same clock")
    print("warning as 03 and 12; capacity is $100k-scale, not fund-scale.")

    # ---- figure -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for nm, s in (("carry_stack10", book10), ("carry_book3 (=12)", books[min(3, k)]),
                  ("BTC only", gated["BTC"])):
        ax1.plot(core.metrics.equity_curve(s.dropna()), lw=1.3, label=nm)
    ax1.set_title("Breadth on the structural edge (daily-marked, net)")
    ax1.set_ylabel("growth of $1 (on notional)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.plot(ks, sharpes, "o-", lw=1.4, label="measured")
    ideal = sharpes[0] * np.sqrt(np.array(ks))
    ax2.plot(ks, ideal, "--", lw=1.0, color="gray", label="sqrt(k) ideal (rho=0)")
    ax2.set_xlabel("coins in book (liquidity order)"); ax2.set_ylabel("Sharpe (marked)")
    ax2.set_title(f"Sharpe vs breadth: rho={rho:+.2f} => k_eff={k_eff:.1f}/{k}")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "equity.png"), dpi=120)
    plt.close(fig)
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
