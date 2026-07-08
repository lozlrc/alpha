"""Overnight news reaction on REAL open/close data -- can you trade the gap?

Run:  ../.venv/bin/python run.py

THE IDEA UNDER TEST (user strategy): news breaks while the market is closed; at the
open, buy the names whose overnight news was good and short the ones whose news was
bad. An LLM/RAG reads the headlines and picks the sides.

WHY THIS FILE CONTAINS NO LLM AND NO HEADLINES. Backtesting an LLM on HISTORICAL news
is structurally contaminated: every modern model was trained on text written AFTER
those nights -- it "knows" Nvidia won and Peloton lost, so any backtest where an LLM
scores 2015-2024 headlines is the look-ahead trap of 07/10 in its most seductive form
(the leak hides inside the weights, where no purge can reach). The honest split is:

  1. THIS FILE -- backtest the MECHANICAL core on real prices: the overnight gap
     (open_t/close_{t-1}-1) is the market's own summary of the night's news. If
     "buy good news at the open" works, gap-signed portfolios must earn something
     open->close. This is testable without any NLP, and it prices the SLOT the LLM
     would have to beat: the LLM only adds value over the gap itself if it can tell
     WHICH gaps under/over-react -- after ~20 bps/day of round-trip costs.
  2. live_harness.py -- the LLM/RAG version runs FORWARD-ONLY as a pre-open paper
     trader (score overnight headlines, log picks, settle against realized returns
     later). Zero look-ahead by construction; evidence accumulates in weeks.

DATA: real ADJUSTED open+close for the same ~62 mega-cap universe as 10_real_data,
2010-2024 (one-time yfinance fetch, cached to data_cache/, offline afterwards).

WHAT TO EXPECT (so results read honestly): US large caps are the most efficient slice
on earth at the open auction; the literature finding is that big gaps mostly REVERT
intraday (market-maker overshoot) and that the equity premium accrues OVERNIGHT, not
intraday (Asness et al.). Costs are brutal for a daily open->close flip book.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")
START, END = "2010-01-01", "2024-12-31"
COST_BPS = 5.0                      # per unit one-way turnover
DECILE = 0.10                       # long top / short bottom fraction by gap
BIG_GAP = 0.02                      # |overnight| > 2% = "a news night", not noise
PPY = 252

UNIVERSE = [
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "ADBE", "AMAT",
    "GOOG", "DIS", "VZ", "CMCSA", "JPM", "BAC", "WFC", "GS", "AXP", "C", "MS",
    "USB", "BLK", "JNJ", "PFE", "MRK", "UNH", "ABT", "AMGN", "GILD", "MDT",
    "PG", "KO", "PEP", "WMT", "MCD", "COST", "CL", "HD", "NKE", "LOW", "SBUX",
    "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "BA", "HON", "UPS", "GE", "MMM",
    "LMT", "DE", "DUK", "SO", "NEE", "APD", "SHW", "FCX", "NEM",
]


# ---------------------------------------------------------------- data
def load_open_close() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Adjusted open & close panels, cached; None (clean skip) if unavailable."""
    os.makedirs(CACHE, exist_ok=True)
    p_open, p_close = os.path.join(CACHE, "open.csv"), os.path.join(CACHE, "close.csv")
    if os.path.exists(p_open) and os.path.exists(p_close):
        return (pd.read_csv(p_open, index_col=0, parse_dates=True),
                pd.read_csv(p_close, index_col=0, parse_dates=True))
    try:
        import yfinance as yf
        print("  one-time fetch of adjusted OHLC (cached afterwards) ...")
        raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True,
                          progress=False, group_by="column")
        opens, closes = raw["Open"], raw["Close"]
    except Exception as e:                              # offline / yfinance missing
        print(f"[skip] could not load real data: {type(e).__name__}: {e}")
        print("       Run once WITH network (and yfinance installed) to populate "
              "data_cache/. Skipping cleanly.")
        return None
    keep = [t for t in UNIVERSE if t in closes.columns and closes[t].notna().sum() > 3000]
    opens, closes = opens[keep].dropna(how="all"), closes[keep].dropna(how="all")
    opens.to_csv(p_open), closes.to_csv(p_close)
    return opens, closes


# ---------------------------------------------------------------- books
def ls_weights(signal: pd.DataFrame, frac: float = DECILE) -> pd.DataFrame:
    """Dollar-neutral long-top / short-bottom decile weights from a daily signal."""
    r = signal.rank(axis=1, pct=True)
    long = (r >= 1 - frac).astype(float)
    short = (r <= frac).astype(float)
    W = long.div(long.sum(axis=1), axis=0).fillna(0.0) \
        - short.div(short.sum(axis=1), axis=0).fillna(0.0)
    return W


def book(returns: pd.DataFrame, W: pd.DataFrame, hold_turnover: float,
         name: str) -> pd.Series:
    """Net daily P&L of weights applied to same-day returns (weights are formed AT
    the open from pre-open info; returns accrue strictly after -- see docstring)."""
    gross = (W * returns).sum(axis=1)
    return (gross - hold_turnover * COST_BPS / 1e4).rename(name)


# ---------------------------------------------------------------- main
def main():
    print("\n=== Overnight news reaction (REAL adjusted open/close, 2010-2024) ===")
    loaded = load_open_close()
    if loaded is None:
        return
    opens, closes = loaded
    overnight = opens / closes.shift(1) - 1.0           # known AT the open
    intraday = closes / opens - 1.0                     # earned open -> close
    close2close = closes.pct_change()
    n_names = closes.shape[1]
    print(f"Universe: {n_names} mega-caps; span {closes.index[0].date()} -> "
          f"{closes.index[-1].date()}; costs {COST_BPS:.0f} bps/turnover.\n")

    # ---- exhibit 1: where does the equity premium live? --------------------
    ew_night = overnight.mean(axis=1)
    ew_day = intraday.mean(axis=1)
    print("Exhibit 1 -- the equal-weight universe, split at the bell (GROSS, no costs):")
    for nm, r in (("overnight only (close->open)", ew_night),
                  ("intraday only (open->close)", ew_day),
                  ("close-to-close (sum of both)", close2close.mean(axis=1))):
        s = core.metrics.summary(r.dropna(), nm, PPY)
        print(f"  {nm:<32s} ann {s['ann_return']:+7.1%}  Sharpe {s['sharpe']:+5.2f}")

    # ---- exhibit 2: trade the gap at the open ------------------------------
    # Daily flip book: enter at open, exit at close => one-way turnover ~4/day.
    streams = {
        "gap_continuation_1d": book(intraday, ls_weights(overnight), 4.0,
                                    "gap_continuation_1d"),
        "gap_fade_1d": book(intraday, -ls_weights(overnight), 4.0, "gap_fade_1d"),
    }
    # Big-gap ("real news") conditioning: same books, only nights with |gap|>2%.
    news = overnight.where(overnight.abs() > BIG_GAP)
    sides = np.sign(news)
    live = sides.abs().sum(axis=1)
    cont = (sides * intraday).sum(axis=1) / live.replace(0, np.nan)
    n_turn = (live > 0) * 2.0                            # gross 1, in+out on live days
    streams["biggap_continuation_1d"] = (cont.fillna(0.0)
                                         - n_turn * COST_BPS / 1e4).rename("biggap_cont")
    streams["biggap_fade_1d"] = (-cont.fillna(0.0)
                                 - n_turn * COST_BPS / 1e4).rename("biggap_fade")
    # 5-day hold (overlapping fifths): enter at open, exit close t+4; turnover 4/5.
    hold5 = sum((ls_weights(overnight).shift(k) * close2close).sum(axis=1)
                for k in range(5)) / 5.0
    hold5 = hold5 + (ls_weights(overnight) * (intraday - close2close)).sum(axis=1) / 5.0
    streams["gap_continuation_5d"] = (hold5 - 0.8 * COST_BPS / 1e4).rename("gap_cont_5d")

    rows = [core.metrics.summary(r.dropna(), n, PPY) for n, r in streams.items()]
    board = core.format_leaderboard(rows)
    print("\nExhibit 2 -- gap-signed books, NET of costs (the slot an LLM must beat):")
    print(board[["sharpe", "ann_return", "ann_vol", "max_drawdown",
                 "hit_rate"]].round(3).to_string())

    g = core.metrics.summary((ls_weights(overnight) * intraday).sum(axis=1).dropna(),
                             "g", PPY)
    print(f"\nGross-vs-net, quantified: continuation GROSS Sharpe {g['sharpe']:+.2f} "
          f"({g['ann_return']:+.1%}/yr) => the FADE is {-g['sharpe']:+.2f} GROSS "
          f"({-g['ann_return']:+.1%}/yr):\nthe overnight crowd OVER-reacts and gaps "
          f"revert -- the same fade edge as family 09, on real data. But a daily "
          f"open->close\nflip pays ~{4 * COST_BPS:.0f} bps/day, "
          f"~{4 * COST_BPS * 2.52:.0f}%/yr -- more than the whole gross edge. The "
          "effect is real; the FREQUENCY is unaffordable\nat retail costs in "
          "mega-caps. (An LLM enters exactly this slot: it must pick WHICH gaps "
          "revert.)")

    # ---- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.4), height_ratios=[1, 1])
    for nm, r in (("overnight only", ew_night), ("intraday only", ew_day)):
        ax1.plot(core.metrics.equity_curve(r.dropna()), lw=1.5, label=nm)
    ax1.set_yscale("log"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_title("Exhibit 1: the premium accrues overnight -- you can't buy the "
                  "night after reading its news")
    ax1.set_ylabel("growth of $1 (log)")
    for nm in ("gap_continuation_1d", "gap_fade_1d", "biggap_fade_1d"):
        ax2.plot(core.metrics.equity_curve(streams[nm].dropna()), lw=1.2, label=nm)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_title("Exhibit 2: gap books net of costs")
    ax2.set_ylabel("growth of $1")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "equity.png"), dpi=120)
    plt.close(fig)

    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print("\nHonest caveats: entry AT the open print is optimistic (auction slippage);")
    print("mega-caps are the hardest slice (this is where the crowd is sharpest -- the")
    print("edge-law says look where it ISN'T); and NO LLM appears in this backtest on")
    print("purpose -- an LLM scoring HISTORICAL headlines already knows the outcome")
    print("(leak lives in the weights). The forward-only LLM harness: live_harness.py.")
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
