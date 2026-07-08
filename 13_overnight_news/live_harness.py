"""Forward-only paper trader for the overnight-news idea -- the honest LLM half.

Why forward-only: an LLM scoring HISTORICAL headlines is contaminated (it was trained
on text written after those nights -- the outcome leaks in through the weights, where
no purged CV can reach). The only clean test of "RAG the overnight news, buy good /
short bad at the open" is to run it FORWARD and let evidence accumulate. This harness
does that with zero look-ahead by construction:

  BEFORE THE OPEN (e.g. 9:00-9:25 ET):
      ../.venv/bin/python live_harness.py --score
  pulls each name's headlines published since the previous close (yfinance news),
  scores them, and APPENDS picks to paper_log.csv. Scores are frozen pre-open.

  ANY LATER DAY:
      ../.venv/bin/python live_harness.py --settle
  fills in each logged day's realized open->close and open->close(+5d) returns and
  prints the running scorecard: IC(score, realized), long-short spread of the picks,
  and n. Judge with the same rule as everything else in this suite: no real money
  until the FORWARD IC is positive with n in the hundreds.

Scorers (pluggable, --scorer):
  lexicon  (default)  -- a small embedded finance polarity lexicon. Crude on purpose:
                         free, fast, and a floor the LLM must beat.
  claude              -- shells out to `claude -p` (Claude Code CLI) with a compact
                         prompt per name; requires the CLI installed & authed.
                         This is the RAG half: retrieval = the overnight headlines,
                         generation = a single calibrated score in [-2, +2].

No orders are placed anywhere. It writes a CSV. That is the point.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "paper_log.csv")
FIELDS = ["date", "ticker", "scorer", "score", "n_headlines", "headlines",
          "ret_1d", "ret_5d"]

UNIVERSE = [
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "ADBE", "AMAT",
    "GOOG", "DIS", "VZ", "CMCSA", "JPM", "BAC", "WFC", "GS", "AXP", "C", "MS",
    "USB", "BLK", "JNJ", "PFE", "MRK", "UNH", "ABT", "AMGN", "GILD", "MDT",
    "PG", "KO", "PEP", "WMT", "MCD", "COST", "CL", "HD", "NKE", "LOW", "SBUX",
    "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "BA", "HON", "UPS", "GE", "MMM",
    "LMT", "DE", "DUK", "SO", "NEE", "APD", "SHW", "FCX", "NEM",
]

POS = {"beat", "beats", "tops", "exceeds", "raises", "raised", "upgrade", "upgraded",
       "buyback", "dividend", "record", "surge", "surges", "approval", "approved",
       "wins", "won", "award", "awarded", "expands", "growth", "strong", "outperform",
       "bullish", "breakthrough", "partnership", "contract", "guidance-raise"}
NEG = {"miss", "misses", "cuts", "cut", "downgrade", "downgraded", "recall",
       "lawsuit", "sues", "sued", "probe", "investigation", "fraud", "bankruptcy",
       "layoffs", "warning", "warns", "plunge", "plunges", "halt", "halted", "delay",
       "delayed", "weak", "underperform", "bearish", "fine", "fined", "breach",
       "guidance-cut", "resigns", "default"}


def _since_prev_close() -> float:
    now = dt.datetime.now(dt.timezone.utc)
    prev = now - dt.timedelta(days=3 if now.weekday() == 0 else 1)
    return prev.replace(hour=20, minute=0, second=0).timestamp()   # ~4pm ET


def overnight_headlines(ticker) -> list[str]:
    import yfinance as yf
    cutoff = _since_prev_close()
    out = []
    for a in (yf.Ticker(ticker).news or []):
        c = a.get("content", a)
        ts = a.get("providerPublishTime") or 0
        if not ts:
            pub = str(c.get("pubDate", ""))
            try:
                ts = dt.datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
            except ValueError:
                ts = 0
        title = (c.get("title") or "").strip()
        if ts >= cutoff and title:
            out.append(title)
    return out[:8]


def score_lexicon(_t: str, heads: list[str]) -> float:
    s = 0
    for h in heads:
        w = set(h.lower().replace(",", " ").split())
        s += len(w & POS) - len(w & NEG)
    return max(-2.0, min(2.0, float(s)))


def score_claude(ticker: str, heads: list[str]) -> float:
    prompt = (f"Overnight headlines for {ticker}:\n- " + "\n- ".join(heads) +
              "\nAs a sober sell-side analyst, score the likely effect on TODAY'S "
              "open-to-close return. Reply with ONLY a number in [-2, 2] "
              "(0 = no tradeable news).")
    r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True,
                       timeout=120)
    return max(-2.0, min(2.0, float(r.stdout.strip().split()[-1])))


def cmd_score(scorer: str) -> None:
    fn = score_claude if scorer == "claude" else score_lexicon
    today = dt.date.today().isoformat()
    new = 0
    exists = os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for t in UNIVERSE:
            try:
                heads = overnight_headlines(t)
            except Exception as e:
                print(f"  {t}: news fetch failed ({type(e).__name__}) -- skipped")
                continue
            if not heads:
                continue
            try:
                s = fn(t, heads)
            except Exception as e:
                print(f"  {t}: scorer failed ({type(e).__name__}) -- skipped")
                continue
            w.writerow({"date": today, "ticker": t, "scorer": scorer, "score": s,
                        "n_headlines": len(heads), "headlines": " | ".join(heads)[:400],
                        "ret_1d": "", "ret_5d": ""})
            new += 1
            if abs(s) >= 1:
                print(f"  {today} {t:<5s} score {s:+.1f}  ({heads[0][:60]})")
    print(f"logged {new} scored names -> {LOG} (frozen pre-open; settle later)")


def cmd_settle() -> None:
    import pandas as pd
    import yfinance as yf
    if not os.path.exists(LOG):
        print("no paper_log.csv yet -- run --score on a market morning first")
        return
    log = pd.read_csv(LOG)
    need = log[(log.ret_1d.isna()) | (log.ret_5d.isna())]
    if len(need):
        start = pd.to_datetime(need.date.min()) - pd.Timedelta(days=1)
        px = yf.download(sorted(need.ticker.unique()), start=start.date().isoformat(),
                         auto_adjust=True, progress=False, group_by="column")
        for i, row in need.iterrows():
            try:
                o = px["Open"][row.ticker]
                c = px["Close"][row.ticker]
                d = pd.Timestamp(row.date)
                if d not in o.index:
                    continue
                log.loc[i, "ret_1d"] = c.loc[d] / o.loc[d] - 1.0
                pos = o.index.get_loc(d)
                if pos + 4 < len(c):
                    log.loc[i, "ret_5d"] = c.iloc[pos + 4] / o.loc[d] - 1.0
            except (KeyError, TypeError):
                continue
        log.to_csv(LOG, index=False)
    done = log.dropna(subset=["ret_1d"])
    done = done[done.score != 0]
    if len(done) < 5:
        print(f"settled rows so far: {len(done)} -- keep logging mornings.")
        return
    for h in ("ret_1d", "ret_5d"):
        d = done.dropna(subset=[h])
        if len(d) < 5:
            continue
        ic = d.score.corr(d[h], method="spearman")
        ls = d[d.score > 0][h].mean() - d[d.score < 0][h].mean()
        print(f"{h}: n={len(d):4d}  forward IC {ic:+.3f}  "
              f"long-short spread {ls:+.4%}")
    print("Rule: no real money until forward IC > 0 with n in the hundreds.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", action="store_true", help="score this morning's news")
    ap.add_argument("--settle", action="store_true", help="fill realized returns + report")
    ap.add_argument("--scorer", default="lexicon", choices=["lexicon", "claude"])
    a = ap.parse_args()
    if a.score:
        cmd_score(a.scorer)
    elif a.settle:
        cmd_settle()
    else:
        ap.print_help()
        sys.exit(1)
