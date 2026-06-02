"""Real historical-data backtest -- the honesty check for the synthetic suite.

Run:  ../.venv/bin/python run.py

Downloads real daily ADJUSTED prices for a fixed large-cap US universe (one-time,
cached to data_cache/ -> fully offline afterwards; this is HISTORICAL data, NOT a
live feed or broker connection), then runs the SAME price-based factor code from
01_cross_sectional_factors on real prices and prints the real Sharpes next to the
synthetic ones. Expect the real numbers to be far lower -- that gap is the whole
point: planted synthetic structure is generous; real markets are not.

Only PRICE-based factors run here (momentum, low-vol, short-term reversal). Value
and quality need point-in-time fundamentals, which a price feed does not provide.

Caveats that the engine cannot fix for you:
  * SURVIVORSHIP BIAS -- this is a hand-picked list of names that SURVIVED to today;
    a true point-in-time index would also include the delisted/merged names, so the
    levels here are optimistic. Treat the levels as illustrative, the COMPARISON as
    the lesson.
  * Costs are a flat 5 bps/turnover -- still no market impact, borrow, or slippage.

If there is no network (or yfinance is not installed) the script SKIPS cleanly so
`run_all.py` stays green; run it once online to populate the cache.
"""
from __future__ import annotations

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
START, END, COST_BPS, QUANTILES = "2010-01-01", "2024-12-31", 5.0, 5

# Fixed large-cap universe (ticker -> sector). Hand-picked long-history names;
# see the SURVIVORSHIP caveat in the module docstring. SPY is the market factor.
UNIVERSE = {
    "AAPL": "Tech", "MSFT": "Tech", "INTC": "Tech", "CSCO": "Tech", "ORCL": "Tech", "IBM": "Tech",
    "GOOG": "Comm", "DIS": "Comm", "VZ": "Comm", "T": "Comm",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials", "AXP": "Financials",
    "JNJ": "Health", "PFE": "Health", "MRK": "Health", "UNH": "Health", "ABT": "Health",
    "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer", "WMT": "Consumer", "MCD": "Consumer",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "DUK": "Utilities", "SO": "Utilities", "NEE": "Utilities",
}

# Same price-only factor code as 01_cross_sectional_factors, applied to real data.
PRICE_FACTORS = {
    "momentum_12_1": lambda f, md: f.momentum(md),
    "low_vol": lambda f, md: f.low_volatility(md),
    "reversal_5d": lambda f, md: f.short_term_reversal(md),
}


def _load_factors():
    """Import 01_cross_sectional_factors/factors.py by path (folder starts with a digit)."""
    path = os.path.join(ROOT, "01_cross_sectional_factors", "factors.py")
    spec = importlib.util.spec_from_file_location("factors01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("\n=== Real-data backtest (price factors on REAL prices) ===")
    factors = _load_factors()

    cache = os.path.join(HERE, "data_cache")
    try:
        md = core.data.load_yfinance(list(UNIVERSE), start=START, end=END,
                                     market_ticker="SPY", sectors=UNIVERSE, cache_dir=cache)
    except Exception as e:  # offline / rate-limited / yfinance missing
        print(f"[skip] could not load real data: {type(e).__name__}: {e}")
        print("       Run once WITH network to populate data_cache/ (then it is fully offline),")
        print("       or `pip install yfinance`. Skipping cleanly so run_all stays green.")
        return

    n_names, n_days = md.returns.shape[1], md.returns.shape[0]
    print(f"Universe: {n_names} stocks x {n_days} trading days "
          f"({md.returns.index[0].date()} -> {md.returns.index[-1].date()}); market = SPY.")

    # ---- real-data leaderboard (same factor code as 01, price-only) --------
    results = [core.long_short_backtest(fn(factors, md), md.returns, quantiles=QUANTILES,
                                        cost_bps=COST_BPS, name=name)
               for name, fn in PRICE_FACTORS.items()]
    results.append(core.long_short_backtest(
        factors.momentum(md), md.returns, quantiles=QUANTILES, cost_bps=COST_BPS,
        neutralize_groups=md.sectors, name="momentum_sector_neutral"))

    summaries = [r.summary() for r in results]
    summaries.append(core.metrics.summary(md.market.dropna(), "SPY_buy_hold"))  # market reference
    board = core.format_leaderboard(summaries)
    print(f"\nReal-data Sharpes (net of {COST_BPS:.0f} bps, dollar-neutral L/S; "
          "SPY_buy_hold is long-only market):")
    print(board.round(3).to_string())

    eqd = {r.name: r.returns for r in results}
    eqd["SPY_buy_hold"] = md.market.dropna()
    core.plot_equity(eqd, os.path.join(HERE, "equity.png"),
                     title="Price factors on REAL data (2010-2024, net 5bps)")
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))

    # ---- the honesty contrast: same factor code, synthetic vs real ---------
    syn = core.generate_market(seed=7)
    print("\nSame factor code -- SYNTHETIC (planted) vs REAL (honest), net Sharpe:")
    print(f"  {'factor':22s} {'synthetic':>10s} {'real':>8s}   gap")
    for name, fn in PRICE_FACTORS.items():
        s_syn = core.sharpe(core.long_short_backtest(fn(factors, syn), syn.returns,
                            quantiles=5, cost_bps=COST_BPS).returns)
        s_real = core.sharpe([r for r in results if r.name == name][0].returns)
        print(f"  {name:22s} {s_syn:10.2f} {s_real:8.2f}   {s_syn - s_real:+.2f}")
    print("\nThe synthetic suite recovers PLANTED structure; real markets are far less\n"
          "generous (and far more competitive). The machinery is identical -- only the\n"
          "data changed. This is why every synthetic Sharpe in this repo is a\n"
          "demonstration of mechanics, never a forward-looking return estimate.")

    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")
    print(f"Cached prices: {cache}/  (delete to re-download; otherwise fully offline)")


if __name__ == "__main__":
    main()
