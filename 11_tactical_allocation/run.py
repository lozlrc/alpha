"""Robust multi-asset tactical allocation on REAL data -- high Sharpe, low drawdown.

Run:  ../.venv/bin/python run.py

The honest lesson of 10_real_data was that you cannot squeeze alpha out of 62 mega-cap
cross-sections net of costs. So this does NOT try to. Instead it harvests the most
robust, best-documented sources of real risk-adjusted return -- DIVERSIFICATION across
asset classes, TREND-following, and RISK MANAGEMENT -- combined into one portfolio.

Universe: 14 liquid ETFs spanning asset classes (US large/small cap, intl & EM equity,
Treasuries across the curve, TIPS, IG & high-yield credit, gold, silver, commodities,
REITs), real ADJUSTED prices, ~2007-2024 (so the 2008 GFC, 2020 COVID crash, and 2022
bear are all in-sample for the drawdown test). Idle cash earns the T-bill yield (BIL).
Breadth matters: more genuinely-diversifying sleeves -> higher Sharpe, lower drawdown.

Strategy ("RTA"), monthly rebalance, 1-day lag, net of costs:
  1. TREND filter  -- an ENSEMBLE of 8/10/12-month moving-average filters; an asset's
                      exposure scales with how many say "uptrend" (0, 1/3, 2/3, 1).
  2. RISK PARITY   -- size each held sleeve by inverse trailing volatility; the weights
                      are normalized by the WHOLE universe, so weak-trend months sit
                      partly in cash (which earns the T-bill yield).
  3. VOL TARGET    -- scale the book toward ~10% annual vol, capped at 1x (NO leverage).

These are STANDARD, a-priori rules from the literature -- not parameters fit to this data
-- so the full-sample result is honest, and a robustness sweep shows it is not a knife
edge. Benchmarks: SPY buy-and-hold and a fixed 60/40 (SPY/IEF).

HISTORICAL data only (cached to data_cache/, fully offline after the first fetch); NOT a
live feed. Skips cleanly if offline so run_all stays green.
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
START, END, COST_BPS = "2007-01-01", "2024-12-31", 10.0
CASH_ETF = "BIL"  # 1-3 month T-bill ETF -- the yield earned on the un-invested sleeve

# Liquid cross-asset ETF universe (ticker -> asset class), all with history from ~2007.
# Breadth matters: more (genuinely diversifying) sleeves -> higher Sharpe, lower drawdown.
ASSETS = {
    "SPY": "US large cap", "IWM": "US small cap", "EFA": "Intl equity", "EEM": "EM equity",
    "IEF": "Treasuries 7-10y", "TLT": "Treasuries 20y+", "SHY": "Treasuries 1-3y",
    "TIP": "TIPS", "LQD": "IG credit", "HYG": "High yield",
    "GLD": "Gold", "SLV": "Silver", "DBC": "Commodities", "VNQ": "REITs",
}
TREND_MONTHS = (8, 10, 12)  # ensemble of moving-average lookbacks


def _month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    last = idx.to_series().groupby([idx.year, idx.month]).last()
    return pd.DatetimeIndex(last.values)


def _trend_score(prices: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the 8/10/12-month MA filters that say 'uptrend' (0..1 per asset)."""
    score = sum((prices > prices.rolling(m * 21, min_periods=m * 10).mean()).astype(float)
                for m in TREND_MONTHS) / len(TREND_MONTHS)
    return score


def tactical_weights(prices, returns, vol_lookback=60, target_vol=0.10) -> pd.DataFrame:
    """Ensemble trend x inverse-vol risk parity x vol target, set monthly. Weights sum
    to <=1; the remainder is cash. NaN between rebalances so the engine's ffill HOLDS
    each month (a zero-init matrix would silently sit in cash 20 of every 21 days)."""
    score = _trend_score(prices)
    inv_vol = 1.0 / returns.rolling(vol_lookback).std()
    W = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    for d in _month_end_dates(prices.index):
        iv = inv_vol.loc[d]
        denom = iv.sum()
        if not denom > 0:
            W.loc[d] = 0.0
            continue
        w = (score.loc[d] * iv / denom).fillna(0.0)        # partial risk parity (more cash if fewer trends)
        if w.sum() <= 0:
            W.loc[d] = 0.0
            continue
        cov = returns.loc[:d].tail(vol_lookback).cov()
        pvol = float(np.sqrt(max(w.values @ cov.values @ w.values, 1e-12)) * np.sqrt(252))
        w = w * min(target_vol / pvol, 1.0)                # vol target, NO leverage
        g = float(w.sum())
        if g > 1.0:
            w = w / g                                      # never exceed fully invested
        W.loc[d] = w
    return W


def fixed_weights(returns, alloc: dict) -> pd.DataFrame:
    W = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for d in _month_end_dates(returns.index):
        W.loc[d] = 0.0
        for k, v in alloc.items():
            if k in W.columns:
                W.loc[d, k] = v
    return W


def rta_returns(prices, returns, cash_ret, name="tactical_RTA", **kw) -> pd.Series:
    """Net RTA return = backtested book + T-bill yield on the un-invested cash sleeve."""
    W = tactical_weights(prices, returns, **kw)
    res = core.backtest_weights(returns, W, cost_bps=COST_BPS, lag=1, name=name)
    cash_frac = (1.0 - res.weights.abs().sum(axis=1)).clip(lower=0.0)
    return (res.returns + cash_frac * cash_ret.reindex(res.returns.index).fillna(0.0)).rename(name)


def main():
    print("\n=== Robust multi-asset tactical allocation (REAL ETF data) ===")
    try:
        md = core.data.load_yfinance(list(ASSETS) + [CASH_ETF], start=START, end=END,
                                     cache_dir=os.path.join(HERE, "data_cache"))
    except Exception as e:  # offline / rate-limited / yfinance missing
        print(f"[skip] could not load real data: {type(e).__name__}: {e}")
        print("       Run once WITH network to populate data_cache/ (then fully offline),")
        print("       or `pip install yfinance`. Skipping cleanly so run_all stays green.")
        return

    trade = [a for a in ASSETS if a in md.prices.columns]
    prices, returns = md.prices[trade], md.returns[trade]
    cash_ret = md.returns[CASH_ETF] if CASH_ETF in md.returns.columns else pd.Series(0.0, index=returns.index)
    print(f"Assets: {len(trade)} ETFs ({', '.join(trade)}); cash sleeve = {CASH_ETF}")
    print(f"Span: {returns.index[0].date()} -> {returns.index[-1].date()}; "
          f"cost {COST_BPS:.0f} bps; monthly rebalance.\n")

    # ---- the strategy + benchmarks ----------------------------------------
    rta = rta_returns(prices, returns, cash_ret)
    spy = returns["SPY"].rename("SPY_buy_hold")
    bench = core.backtest_weights(returns, fixed_weights(returns, {"SPY": 0.6, "IEF": 0.4}),
                                  cost_bps=COST_BPS, lag=1, name="60/40").returns

    rows = [core.metrics.summary(rta.dropna(), "tactical_RTA"),
            core.metrics.summary(spy.dropna(), "SPY_buy_hold"),
            core.metrics.summary(bench.dropna(), "60/40")]
    board = core.format_leaderboard(rows)
    cols = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "calmar", "hit_rate"]
    print("Strategy vs benchmarks (net of cost):")
    print(board[cols].round(3).to_string())

    R, S = rows[0], rows[1]
    print(f"\nTactical RTA:  Sharpe {R['sharpe']:.2f}, max drawdown {R['max_drawdown']*100:.1f}%, "
          f"Calmar {R['calmar']:.2f}, vol {R['ann_vol']*100:.1f}%")
    print(f"SPY buy-hold:  Sharpe {S['sharpe']:.2f}, max drawdown {S['max_drawdown']*100:.1f}%, "
          f"Calmar {S['calmar']:.2f}, vol {S['ann_vol']*100:.1f}%")
    print(f"=> {R['sharpe']-S['sharpe']:+.2f} Sharpe, {(1 - R['max_drawdown']/S['max_drawdown'])*100:.0f}% "
          f"shallower drawdown, and {R['calmar']/S['calmar']:.1f}x the Calmar of just owning stocks --\n"
          "   from diversification + trend + risk control, not a fitted signal.")

    # ---- robustness: standard rules, not a tuned knife-edge ----------------
    print("\nRobustness (Sharpe / max-DD across un-tuned rule settings):")
    print(f"  {'vol_lookback':>12s} {'target_vol':>10s} {'Sharpe':>7s} {'maxDD':>7s}")
    for vl in (40, 60, 90):
        for tv in (0.08, 0.10, 0.12):
            r = rta_returns(prices, returns, cash_ret, vol_lookback=vl, target_vol=tv)
            s = core.metrics.summary(r.dropna(), "x")
            print(f"  {vl:>12d} {tv:>10.2f} {s['sharpe']:>7.2f} {s['max_drawdown']*100:>6.1f}%")

    # ---- figure: equity (log) + drawdown -----------------------------------
    curves = {"tactical_RTA": rta, "60/40": bench, "SPY_buy_hold": spy}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.4), height_ratios=[2, 1])
    for nm, r in curves.items():
        eq = core.metrics.equity_curve(r.dropna())
        ax1.plot(eq.index, eq.values, lw=1.6, label=nm)
    ax1.set_yscale("log"); ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Tactical allocation vs SPY & 60/40 — higher Sharpe, far shallower drawdowns")
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(True, alpha=0.3)
    for nm, r in curves.items():
        dd = core.metrics.drawdown_series(r.dropna())
        ax2.plot(dd.index, dd.values * 100, lw=1.2, label=nm)
    ax2.set_ylabel("drawdown (%)"); ax2.grid(True, alpha=0.3); ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "equity.png"), dpi=120)
    plt.close(fig)

    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print("\nHonest caveats: survivor ETFs (these funds existed the whole window); flat 10 bps,"
          " no market-impact model; it is long-only alternative-beta (diversification + trend),")
    print("not market-neutral alpha -- but it is a real, robust, low-drawdown way to hold risk.")
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
