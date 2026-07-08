"""The improvement loop, done honestly -- walk-forward tuning with a trials ledger.

Run:  ../.venv/bin/python run.py     (needs 11/12/13 data_cache/ populated -- run those once)

THE TEMPTATION. "Loop over my strategies: backtest, tweak, keep what's better, repeat."
Run that loop naively and it ALWAYS reports progress -- because each iteration is one
more draw from the noise distribution, and the loop keeps the best draw. An improvement
loop without multiple-testing discipline is 14's mining mirage applied to your own
strategies: automated snooping. 10_real_data/optimize.py already showed what that's
worth (snooped max -0.08 avg Sharpe; honest OOS -0.18).

WHAT THIS ENGINE DOES INSTEAD. For each strategy family it takes a small a-priori
parameter grid (the trials ledger -- N is COUNTED and paid for), then:

  1. WALK-FORWARD SELECTION -- at each fold start, pick the variant with the best
     Sharpe on data strictly BEFORE the fold; hold it through the fold. The stitched
     "tuned" stream is what a disciplined re-tuning loop would actually have earned.
  2. PAIRED TEST vs THE DEFAULT -- the a-priori setting shipped by the family is the
     null. Daily return differences give a paired t-stat (the probcup shadow-A/B rule,
     offline): ADOPT only if t > max(2, sqrt(2 ln N)) -- the bar rises with every
     variant you let the loop try.
  3. THE OVERFITTING TAX, PRINTED -- the full-sample best variant ("snooped max") is
     shown next to the walk-forward result; the gap between them is what a naive loop
     would have claimed but never earned.
  4. STABILITY READOUT -- the pick history per fold. A real plateau re-picks the same
     region; a noise surface jumps around.

Families looped (real-data only -- tuning on synthetic data tunes the generator):
  11_tactical_allocation  vol_lookback x target_vol      (9 variants, yearly folds)
  12_funding_carry        gate_days on the marked book    (6 variants, quarterly folds)
  13_overnight_news       fade decile width / big-gap     (4 variants, yearly folds)

WHAT TO EXPECT (stated before you scroll): 11 and 12 already showed FLAT robustness
surfaces -- an honest loop should REJECT tuning there (plateaus don't need a tuner);
and no loop can tune a dead strategy (13) alive -- it can only find the least-dead
corner in-sample. If everything says REJECT, that is the loop WORKING: it is the
certificate that the shipped defaults aren't cherry-picked.

The live sibling of this engine is the probcup bot's shadow A/B (candidates score on
REAL settled outcomes, adopt at t<=-2 over n>=60). Offline loops can only reject;
adoption of anything new belongs on forward data. An LLM proposing variants plugs in
fine as a GENERATOR -- but every proposal lands in the same ledger and pays the same
bar, and the generator never sees the fold it is judged on.
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
MIN_OBS = 90                       # min pre-fold observations before selection kicks in


def load_family(dirname: str):
    path = os.path.join(ROOT, dirname, "run.py")
    spec = importlib.util.spec_from_file_location(dirname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- the engine
def walk_forward(streams: dict[str, pd.Series], default: str, ppy: int,
                 fold_freq: str, burn: int) -> tuple[pd.Series, list[tuple]]:
    """Stitch the stream a disciplined re-tuning loop would have held.

    At each fold start: pick the variant with the best Sharpe on ALL data before
    the fold (expanding window), hold it through the fold. Before `burn` folds of
    history exist, hold the default (a loop with no evidence has no business tuning).
    """
    panel = pd.concat(streams, axis=1).dropna(how="all")
    folds = pd.date_range(panel.index[0], panel.index[-1], freq=fold_freq)[1:]
    edges = [panel.index[0], *folds, panel.index[-1] + pd.Timedelta(days=1)]
    tuned, picks = [], []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        past = panel.loc[: lo - pd.Timedelta(days=1)]
        if k < burn or len(past) < MIN_OBS:
            choice = default
        else:
            sh = {n: core.metrics.sharpe(past[n].dropna(), ppy) for n in panel.columns}
            choice = max(sh, key=lambda n: (np.nan_to_num(sh[n], nan=-9e9)))
        seg = panel[choice].loc[lo:hi - pd.Timedelta(days=1)]
        tuned.append(seg)
        picks.append((str(lo.date()), choice))
    return pd.concat(tuned).rename("wf_tuned"), picks


def paired_t(a: pd.Series, b: pd.Series) -> float:
    d = (a - b).dropna()
    if len(d) < MIN_OBS or d.std(ddof=1) == 0:
        return np.nan
    return float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)))


def report(name: str, streams: dict[str, pd.Series], default: str, ppy: int,
           fold_freq: str, burn: int, rows: list, curves: dict) -> None:
    tuned, picks = walk_forward(streams, default, ppy, fold_freq, burn)
    base = streams[default].reindex(tuned.index)
    n_var = len(streams)
    bar = max(2.0, float(np.sqrt(2 * np.log(n_var))))
    sh = {n: core.metrics.sharpe(s.dropna(), ppy) for n, s in streams.items()}
    snoop_key = max(sh, key=lambda n: np.nan_to_num(sh[n], nan=-9e9))
    t = paired_t(tuned, base)
    s_def, s_tun = core.metrics.sharpe(base.dropna(), ppy), core.metrics.sharpe(tuned.dropna(), ppy)
    verdict = "ADOPT" if (not np.isnan(t)) and t > bar else "REJECT (keep default)"

    print(f"\n--- {name}  ({n_var} variants -> adoption bar t > {bar:.2f}) ---")
    print(f"  default   [{default:<22s}] Sharpe {s_def:+6.2f}")
    print(f"  wf-tuned                           Sharpe {s_tun:+6.2f}   paired t {t:+5.2f}"
          f"   -> {verdict}")
    print(f"  snooped max [{snoop_key:<20s}] Sharpe {sh[snoop_key]:+6.2f}   "
          f"<- in-sample claim; the wf gap is the overfitting tax")
    switch = sum(1 for i in range(1, len(picks)) if picks[i][1] != picks[i - 1][1])
    uniq = sorted({p for _, p in picks})
    print(f"  pick history: {switch} switches over {len(picks)} folds; "
          f"visited {len(uniq)}/{n_var} variants "
          f"({'plateau-ish' if len(uniq) <= max(2, n_var // 3) else 'jumpy = noise surface'})")
    rows += [core.metrics.summary(base.dropna(), f"{name}_default", ppy),
             core.metrics.summary(tuned.dropna(), f"{name}_wf_tuned", ppy)]
    curves[name] = (base, tuned)


# ---------------------------------------------------------------- family adapters
def family_11(rows, curves):
    m = load_family("11_tactical_allocation")
    try:
        md = core.data.load_yfinance(list(m.ASSETS) + [m.CASH_ETF], start=m.START,
                                     end=m.END,
                                     cache_dir=os.path.join(ROOT, "11_tactical_allocation",
                                                            "data_cache"))
    except Exception as e:
        print(f"\n--- 11_tactical_allocation --- [skip] no data: {type(e).__name__}: {e}")
        return
    trade = [a for a in m.ASSETS if a in md.prices.columns]
    prices, returns = md.prices[trade], md.returns[trade]
    cash = (md.returns[m.CASH_ETF] if m.CASH_ETF in md.returns.columns
            else pd.Series(0.0, index=returns.index))
    streams = {f"vl{vl}_tv{tv:.2f}": m.rta_returns(prices, returns, cash,
                                                   vol_lookback=vl, target_vol=tv)
               for vl in (40, 60, 90) for tv in (0.08, 0.10, 0.12)}
    report("11_tactical_allocation", streams, "vl60_tv0.10", 252, "YS", 3, rows, curves)


def family_12(rows, curves):
    m = load_family("12_funding_carry")
    loaded = m.load_all()
    if loaded is None:
        print("\n--- 12_funding_carry --- [skip] no data cache")
        return
    acc, mkd = {}, {}
    for c in m.COINS:
        acc[c], mkd[c] = m.daily_marked(loaded[0][c])
    streams = {}
    for gd in (3, 5, 7, 10, 14, 21):
        streams[f"gate{gd}d"] = pd.concat(
            [m.carry_gated(mkd[c], acc[c], gd, m.FLIP_BPS) for c in m.COINS],
            axis=1).mean(axis=1)
    report("12_funding_carry", streams, "gate7d", 365, "QS", 4, rows, curves)


def family_13(rows, curves):
    m = load_family("13_overnight_news")
    loaded = m.load_open_close()
    if loaded is None:
        print("\n--- 13_overnight_news --- [skip] no data cache")
        return
    opens, closes = loaded
    overnight = opens / closes.shift(1) - 1.0
    intraday = closes / opens - 1.0
    streams = {f"fade_d{int(f * 100):02d}": m.book(intraday, -m.ls_weights(overnight, f),
                                                   4.0, "x")
               for f in (0.05, 0.10, 0.20)}
    news = overnight.where(overnight.abs() > m.BIG_GAP)
    sides = np.sign(news)
    live = sides.abs().sum(axis=1)
    cont = (sides * intraday).sum(axis=1) / live.replace(0, np.nan)
    streams["biggap_fade"] = (-cont.fillna(0.0) - (live > 0) * 2.0 * m.COST_BPS / 1e4)
    report("13_overnight_news (fade)", streams, "fade_d10", 252, "YS", 3, rows, curves)


# ---------------------------------------------------------------- main
def main():
    print("\n=== The improvement loop, with a trials ledger (real-data families) ===")
    print("Rule: walk-forward selection vs the a-priori default; adopt only past the")
    print("multiplicity bar. A loop that can't say REJECT is a snooping machine.")
    rows, curves = [], {}
    family_11(rows, curves)
    family_12(rows, curves)
    family_13(rows, curves)
    if not rows:
        print("\n[skip] no family data caches found -- run 11/12/13 once first.")
        return

    board = core.format_leaderboard(rows)
    print("\nDefault vs walk-forward-tuned, side by side:")
    print(board[["sharpe", "ann_return", "ann_vol", "max_drawdown"]].round(3).to_string())

    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (name, (base, tuned)) in zip(axes, curves.items()):
        ax.plot(core.metrics.equity_curve(base.dropna()), lw=1.3, label="default")
        ax.plot(core.metrics.equity_curve(tuned.dropna()), lw=1.1, ls="--", label="wf-tuned")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("The improvement loop vs the a-priori defaults -- overlap = tuning is noise",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "equity.png"), dpi=120)
    plt.close(fig)
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))

    print("\nHow to read a REJECT: it is the loop WORKING -- the certificate that the")
    print("shipped defaults sit on plateaus, not cherry-picks. New strategies join by")
    print("adding an adapter + grid; LLM-proposed variants join the same ledger and")
    print("pay the same bar. ADOPTING anything new belongs on FORWARD data (the live")
    print("sibling of this file is the probcup bot's shadow-A/B loop).")
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
