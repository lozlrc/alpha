"""Optimize Sharpe on the real-data strategies -- the HONEST way (train/test OOS).

Run:  ../.venv/bin/python optimize.py   (needs the data_cache/ from run.py)

Naively grid-searching parameters to maximize Sharpe on the SAME data you then
report is the overfitting trap -- the exact mistake `07`'s leaky-insample demo and
the deflated-Sharpe warning exist to expose. So instead:

  1. Split the history: TRAIN = 2010-2017, TEST = 2018-2024 (held out).
  2. For each strategy, grid-search its parameters to maximize the TRAIN Sharpe.
  3. Report the OUT-OF-SAMPLE Sharpe of that train-optimal setting on TEST.

The gap between the optimized TRAIN Sharpe and the realized TEST Sharpe is the
"optimization decay" -- how much of the tuned edge was just fitting noise. The grid
also includes a REBALANCE-frequency knob, the one genuinely a-priori lever here:
the strategies that lost worst (reversal, lead-lag) died to turnover x cost, and
trading less often is a defensible cost cut, not snooping.

Everything is the same strategy code from 01/02 on the same cached real prices.
"""
from __future__ import annotations

import itertools
import os
import sys
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, "01_cross_sectional_factors"),
           os.path.join(ROOT, "02_statistical_arbitrage")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402
import factors  # noqa: E402
import cross_sectional_reversal as xr  # noqa: E402
import lead_lag as ll  # noqa: E402
import pairs_trading as pt  # noqa: E402

# Reuse the exact universe + helpers from run.py -- load it BY PATH to avoid the
# name clash with 01/02's own run.py modules already on sys.path.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("rd_run", os.path.join(HERE, "run.py"))
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)

COST, TRAIN_END, TEST_START, END = 5.0, "2017-12-31", "2018-01-01", "2024-12-31"


def _resample(sig: pd.DataFrame, every: int) -> pd.DataFrame:
    """Hold the signal for `every` days (rebalance less often -> less turnover)."""
    if every <= 1:
        return sig
    keep = pd.Series((np.arange(len(sig)) % every) == 0, index=sig.index)
    return sig.where(keep, axis=0).ffill()


def _ls(md, sig, q, every):
    return core.long_short_backtest(_resample(sig, every), md.returns,
                                    quantiles=q, cost_bps=COST).returns


def _sh(ret, lo, hi):
    return core.sharpe(ret.loc[lo:hi])


# ---- strategy return-series builders (full period, lookahead-safe) ----------
def r_momentum(md, lookback, skip, q, every):
    return _ls(md, factors.momentum(md, lookback=lookback, skip=skip), q, every)


def r_low_vol(md, window, q, every):
    return _ls(md, factors.low_volatility(md, window=window), q, every)


def r_reversal(md, window, q, every):
    return _ls(md, factors.short_term_reversal(md, window=window), q, every)


def r_xs_reversal(md, window, q, every):
    return _ls(md, xr.reversal_signal(md, window=window), q, every)


def r_lead_lag(md, beta_window, q, every):
    return _ls(md, ll.lead_lag_signal(md, md.market, beta_window=beta_window), q, every)


def r_multifactor(md, q, every):
    z = lambda s: core.zscore(s, axis=1)
    sig = z(factors.momentum(md)).add(z(factors.low_volatility(md)), fill_value=0.0)
    return _ls(md, sig, q, every)


def _pairs_list(md, corr):
    train = md.returns.loc[:TRAIN_END]
    out = []
    for names in rd._groups(md).values():
        for a, b in combinations(names, 2):
            if float(train[a].corr(train[b])) > corr:
                out.append((a, b))
    return out


def r_pairs(md, entry, exit_, corr):
    pairs = _pairs_list(md, corr)
    if not pairs:
        return pd.Series(dtype=float)
    pnl = [pt.pair_pnl(md.prices[a], md.prices[b], lookback=250, z_window=30,
                       entry=entry, exit=exit_, cost_bps=COST) for a, b in pairs]
    return pd.concat(pnl, axis=1).mean(axis=1)


STRATS = {
    "momentum":    (r_momentum,    dict(lookback=[126, 189, 252], skip=[0, 21], q=[3, 5], every=[1, 5, 21])),
    "low_vol":     (r_low_vol,     dict(window=[21, 63, 126], q=[3, 5], every=[1, 5, 21])),
    "reversal_5d": (r_reversal,    dict(window=[2, 3, 5, 10], q=[3, 5], every=[1, 3, 5])),
    "xs_reversal": (r_xs_reversal, dict(window=[2, 3, 5, 10], q=[3, 5], every=[1, 3, 5])),
    "lead_lag":    (r_lead_lag,    dict(beta_window=[60, 120, 250], q=[3, 5], every=[1, 5, 21])),
    "multifactor": (r_multifactor, dict(q=[3, 5], every=[1, 5, 21])),
    "pairs":       (r_pairs,       dict(entry=[1.5, 2.0, 2.5], exit_=[0.0, 0.5], corr=[0.6])),
}
DEFAULTS = {
    "momentum": dict(lookback=252, skip=21, q=5, every=1),
    "low_vol": dict(window=63, q=5, every=1),
    "reversal_5d": dict(window=5, q=5, every=1),
    "xs_reversal": dict(window=3, q=5, every=1),
    "lead_lag": dict(beta_window=120, q=5, every=1),
    "multifactor": dict(q=5, every=1),
    "pairs": dict(entry=2.0, exit_=0.5, corr=0.6),
}


def main():
    print("\n=== Optimize Sharpe on real data -- TRAIN 2010-2017, report OOS 2018-2024 ===")
    try:
        md = core.data.load_yfinance(list(rd.UNIVERSE), start="2010-01-01", end=END,
                                     market_ticker="SPY", sectors=rd.UNIVERSE,
                                     cache_dir=os.path.join(HERE, "data_cache"))
    except Exception as e:
        print(f"[skip] could not load real data: {type(e).__name__}: {e}")
        print("       Run `python run.py` once online first to populate data_cache/.")
        return
    print(f"{md.returns.shape[1]} stocks; cost {COST:.0f} bps; "
          f"train -> {TRAIN_END}, test {TEST_START} ->.\n")

    hdr = (f"{'strategy':13s} {'best-on-train params':34s} {'snoop':>6s} "
           f"{'train':>6s} {'OOS':>6s} {'defOOS':>7s}")
    print(hdr)
    print("-" * len(hdr))
    snoop_l, oos_l = [], []
    for name, (fn, grid) in STRATS.items():
        keys = list(grid)
        best = None       # best on TRAIN (honest)
        full_best = None  # best on FULL sample (the SNOOPED number)
        for combo in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            ret = fn(md, **params)
            if ret.dropna().empty:
                continue
            full = _sh(ret, "2010-01-01", END)
            if not np.isnan(full) and (full_best is None or full > full_best):
                full_best = full
            tr = _sh(ret, "2010-01-01", TRAIN_END)
            if np.isnan(tr):
                continue
            if best is None or tr > best[0]:
                best = (tr, params, ret)
        if best is None:
            continue
        tr_sh, params, ret = best
        oos = _sh(ret, TEST_START, END)
        d_oos = _sh(fn(md, **DEFAULTS[name]), TEST_START, END)
        snoop_l.append(full_best)
        oos_l.append(oos)
        ptxt = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"{name:13s} {ptxt:34s} {full_best:6.2f} {tr_sh:6.2f} {oos:6.2f} {d_oos:7.2f}")

    print("-" * len(hdr))
    print(f"SPY buy-and-hold OOS (2018-2024): {_sh(md.market, TEST_START, END):+.2f}")
    print(f"\nMean across strategies:  SNOOPED (tune on ALL data) {np.mean(snoop_l):+.2f}   "
          f"vs   HONEST out-of-sample {np.mean(oos_l):+.2f}")
    print("\nThat spread IS the trap. 'snoop' = the best Sharpe if you grid-search the whole\n"
          "history and report the max (the tempting mistake -- exactly what 'optimize Sharpe'\n"
          "naively means); 'OOS' = what that tuning actually delivers on unseen data. The\n"
          "snooped numbers look like alpha; the honest ones don't. On real large-caps net of\n"
          "5 bps these factors have no edge -- optimization only recovers turnover cost\n"
          "(rebalance less often), it cannot manufacture a signal. Nothing beats SPY.")


if __name__ == "__main__":
    main()
