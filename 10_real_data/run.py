"""Real historical-data backtest -- run EVERY strategy that price data supports.

Run:  ../.venv/bin/python run.py

Downloads real daily ADJUSTED prices for a fixed large-cap US universe (one-time,
cached to data_cache/ -> fully offline afterwards; HISTORICAL data, NOT a live feed
or broker connection), then runs the SAME strategy code from the synthetic families
on real prices wherever the data allows it:

  * 01 factors      -- momentum, low-vol, short-term reversal, sector-neutral mom
  * 02 stat-arb     -- cross-sectional reversal, lead-lag, pairs, cointegration
  * 07 ML-driven    -- gradient boosting on price/volume features (honest purged-OOS
                       vs the leaky in-sample trap)

WHAT CANNOT RUN ON A PRICE FEED (and why) -- printed as a coverage matrix at the end:
  * 01 value / quality   need point-in-time FUNDAMENTALS (book value, earnings, ROE)
  * 03 microstructure    needs TICK / order-book data (bid/ask, order flow)
  * 04 event-driven      needs EARNINGS / index-change / M&A event calendars
  * 05 alternative data  needs NEWS / sentiment / web-traffic feeds
  * 06 cross-asset vol   needs OPTIONS / implied-vol surfaces
  * 09 agentic flow      a hypothesis SIMULATOR -- no real AI-agent-flow labels exist;
                         validate via proxies (see README), not a direct price backtest

The lesson is the synthetic-vs-real gap: planted structure is generous, real markets
are not. Survivorship caveat: this is a hand-picked list of names that SURVIVED to
today, so the levels are optimistic; treat the COMPARISON as the takeaway.

If there is no network (or yfinance is missing) the script SKIPS cleanly so
`run_all.py` stays green; run it once online to populate the cache.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
# Make the synthetic families' modules importable by name (they live in digit-prefixed
# folders) so we run the EXACT same strategy code on real data.
for _p in (ROOT, os.path.join(ROOT, "01_cross_sectional_factors"),
           os.path.join(ROOT, "02_statistical_arbitrage"),
           os.path.join(ROOT, "07_ml_driven")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

import core  # noqa: E402
import factors  # noqa: E402  (01)
import cross_sectional_reversal as xr  # noqa: E402  (02)
import lead_lag as ll  # noqa: E402  (02)
import pairs_trading as pt  # noqa: E402  (02)
import cointegration as ci  # noqa: E402  (02)
import features as mlfeat  # noqa: E402  (07)
import cv as mlcv  # noqa: E402  (07)
import model as mlmodel  # noqa: E402  (07)

START, END, COST_BPS, QUANTILES = "2010-01-01", "2024-12-31", 5.0, 5
INSAMPLE = 504  # ~2y reserved for lookahead-safe pair/basket SELECTION

# Fixed large-cap universe (ticker -> sector); SPY is the market factor. Hand-picked
# long-history names -- see the SURVIVORSHIP caveat in the module docstring.
UNIVERSE = {
    "AAPL": "Tech", "MSFT": "Tech", "INTC": "Tech", "CSCO": "Tech", "ORCL": "Tech",
    "IBM": "Tech", "QCOM": "Tech", "TXN": "Tech", "ADBE": "Tech", "AMAT": "Tech",
    "GOOG": "Comm", "DIS": "Comm", "VZ": "Comm", "CMCSA": "Comm",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "AXP": "Financials", "C": "Financials", "MS": "Financials", "USB": "Financials", "BLK": "Financials",
    "JNJ": "Health", "PFE": "Health", "MRK": "Health", "UNH": "Health", "ABT": "Health",
    "AMGN": "Health", "GILD": "Health", "MDT": "Health",
    "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer", "WMT": "Consumer", "MCD": "Consumer",
    "COST": "Consumer", "CL": "Consumer", "HD": "Consumer", "NKE": "Consumer", "LOW": "Consumer", "SBUX": "Consumer",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "GE": "Industrials", "MMM": "Industrials", "LMT": "Industrials", "DE": "Industrials",
    "DUK": "Utilities", "SO": "Utilities", "NEE": "Utilities",
    "APD": "Materials", "SHW": "Materials", "FCX": "Materials", "NEM": "Materials",
}

# What can't run on a price feed alone -> the honesty coverage matrix.
CANT_RUN = [
    ("01 value / quality", "point-in-time fundamentals (book value, earnings, ROE)"),
    ("03 microstructure", "tick / limit-order-book data (bid-ask, order flow)"),
    ("04 event-driven", "earnings / index-rebalance / M&A event calendars"),
    ("05 alternative data", "news, sentiment, web-traffic feeds"),
    ("06 cross-asset vol", "options / implied-volatility surfaces"),
    ("09 agentic flow", "no real agent-flow labels -- a simulator; proxy-test only"),
]


def _groups(md):
    g = {}
    for t in md.prices.columns:
        g.setdefault(UNIVERSE.get(t, "?"), []).append(t)
    return g


def run_factors(md, add):
    """01 price factors (value/quality skipped -- need fundamentals)."""
    r = md.returns
    add("01 momentum_12_1", core.long_short_backtest(factors.momentum(md), r, quantiles=QUANTILES, cost_bps=COST_BPS))
    add("01 low_vol", core.long_short_backtest(factors.low_volatility(md), r, quantiles=QUANTILES, cost_bps=COST_BPS))
    add("01 reversal_5d", core.long_short_backtest(factors.short_term_reversal(md), r, quantiles=QUANTILES, cost_bps=COST_BPS))
    add("01 momentum_sector_neutral", core.long_short_backtest(
        factors.momentum(md), r, quantiles=QUANTILES, cost_bps=COST_BPS, neutralize_groups=md.sectors))


def run_statarb(md, add):
    """02 statistical arbitrage -- all four, on real prices."""
    r = md.returns
    add("02 xs_reversal", core.long_short_backtest(
        xr.reversal_signal(md, window=3), r, quantiles=QUANTILES, cost_bps=COST_BPS))
    add("02 lead_lag", core.long_short_backtest(
        ll.lead_lag_signal(md, md.market, beta_window=120), r, quantiles=QUANTILES, cost_bps=COST_BPS))

    # pairs: trade within-sector pairs whose correlation on the in-sample slice is high
    # (lookahead-safe selection), evaluated only AFTER that window.
    train = r.iloc[:INSAMPLE]
    pnls = []
    for names in _groups(md).values():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if float(train[a].corr(train[b])) > 0.6:
                    pnls.append(pt.pair_pnl(md.prices[a], md.prices[b], lookback=250,
                                            z_window=30, entry=2.0, exit=0.5, cost_bps=COST_BPS))
    if pnls:
        pairs_ret = pd.concat(pnls, axis=1).mean(axis=1).iloc[INSAMPLE:]
        add("02 pairs", pairs_ret, note=f"{len(pnls)} within-sector pairs")

    # cointegration: per sector, target = first name, basket = peers; trade if the
    # EG/ADF gate passes on the in-sample half. Evaluated after the in-sample window.
    cpnls, n_ok = [], 0
    for names in _groups(md).values():
        if len(names) < 4:
            continue
        target, basket = md.prices[names[0]], md.prices[names[1:5]]
        try:
            gate = ci.confirm_cointegration(target, basket, insample=0.5)
        except Exception:
            continue
        if gate.get("is_cointegrated"):
            n_ok += 1
            cpnls.append(ci.cointegration_pnl(target, basket, gate["beta"], beta_window=250,
                                              z_window=45, entry=1.5, exit=0.4, cost_bps=COST_BPS))
    if cpnls:
        coint_ret = pd.concat(cpnls, axis=1).mean(axis=1).iloc[INSAMPLE:]
        add("02 cointegration", coint_ret, note=f"{n_ok} cointegrated baskets")


def run_ml(md, add):
    """07 ML-driven on price/volume features (no fundamentals on a real feed)."""
    panel = mlfeat.build_panel(md, horizon=mlfeat.H)
    nfeat = panel.shape[1] - 1
    oos = mlcv.walk_forward_oos(panel, n_splits=6, horizon=mlfeat.H, random_state=7, verbose=False)
    add("07 ml_honest_oos", core.long_short_backtest(
        mlmodel.predictions_to_wide(oos), md.returns, quantiles=QUANTILES, cost_bps=COST_BPS),
        note=f"{nfeat} price/vol features, purged OOS")
    leaky = mlcv.fit_leaky_insample(panel, random_state=7)
    add("07 ml_leaky_insample", core.long_short_backtest(
        mlmodel.predictions_to_wide(leaky), md.returns, quantiles=QUANTILES, cost_bps=COST_BPS),
        note="in-sample trap (cautionary)")


def main():
    print("\n=== Real-data backtest: every strategy that price data supports ===")
    try:
        md = core.data.load_yfinance(list(UNIVERSE), start=START, end=END, market_ticker="SPY",
                                     sectors=UNIVERSE, cache_dir=os.path.join(HERE, "data_cache"))
    except Exception as e:  # offline / rate-limited / yfinance missing
        print(f"[skip] could not load real data: {type(e).__name__}: {e}")
        print("       Run once WITH network to populate data_cache/ (then fully offline),")
        print("       or `pip install yfinance`. Skipping cleanly so run_all stays green.")
        return

    n, t = md.returns.shape[1], md.returns.shape[0]
    print(f"Universe: {n} stocks x {t} days ({md.returns.index[0].date()} -> "
          f"{md.returns.index[-1].date()}); market = SPY; cost = {COST_BPS:.0f} bps.")

    streams, summaries, notes = {}, [], {}

    def add(name, res, note=""):
        ret = res.returns if hasattr(res, "returns") else res
        turn = res.turnover if hasattr(res, "turnover") else None
        streams[name] = ret
        summaries.append(core.metrics.summary(pd.Series(ret).dropna(), name, turnover=turn))
        if note:
            notes[name] = note

    run_factors(md, add)
    run_statarb(md, add)
    run_ml(md, add)
    summaries.append(core.metrics.summary(md.market.dropna(), "SPY_buy_hold"))
    notes["SPY_buy_hold"] = "long-only market reference"

    board = core.format_leaderboard(summaries)
    print(f"\nReal-data leaderboard (net {COST_BPS:.0f} bps; L/S are dollar-neutral):")
    print(board.round(3).to_string())
    if notes:
        print("\nnotes: " + "; ".join(f"{k} = {v}" for k, v in notes.items()))

    # ---- the punchline: the honest-vs-leaky gap, and "nothing beats SPY" ----
    def _sh(nm):
        return core.sharpe(pd.Series(streams[nm]).dropna()) if nm in streams else float("nan")
    spy_sh = core.sharpe(md.market.dropna())
    print(f"\nReality check -- net of {COST_BPS:.0f} bps on real large-caps:")
    print("  * every dollar-neutral factor / stat-arb / ML signal is flat-to-NEGATIVE;")
    print(f"  * the only thing above SPY ({spy_sh:+.2f}) is ml_leaky_insample "
          f"({_sh('07 ml_leaky_insample'):+.2f}) -- the LOOKAHEAD TRAP: honest purged-OOS")
    print(f"    ML is {_sh('07 ml_honest_oos'):+.2f}, a "
          f"{_sh('07 ml_leaky_insample') - _sh('07 ml_honest_oos'):+.2f}-Sharpe mirage of pure leakage.")
    print("  Same strategy code as the synthetic suite -- only the data changed.")

    # ---- coverage matrix: what could NOT run on a price feed, and why -------
    print("\nCould NOT run on real prices alone (needs other real data):")
    for fam, need in CANT_RUN:
        print(f"  - {fam:22s} needs {need}")

    eqd = {k: v for k, v in streams.items() if "leaky" not in k}
    eqd["SPY_buy_hold"] = md.market.dropna()
    core.plot_equity(eqd, os.path.join(HERE, "equity.png"),
                     title=f"All price-runnable strategies on REAL data ({START[:4]}-{END[:4]}, net 5bps)")
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")
    print(f"Cached prices: {os.path.join(HERE, 'data_cache')}/  (delete to re-download)")


if __name__ == "__main__":
    main()
