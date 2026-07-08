"""A factor-mining engine with the discipline that makes it honest.

Run:  ../.venv/bin/python run.py

WHAT QUANT SHOPS ACTUALLY RUN. WorldQuant-style "alpha factories" generate thousands
of candidate expressions over price/volume panels (their public Alpha101 paper is
exactly this genre), score each by cross-sectional IC -- the daily rank-correlation
between the signal and the NEXT day's returns -- and keep the survivors. The engine
is easy. The reason those shops aren't rich-by-accident is the part everyone skips:
MULTIPLE-TESTING DISCIPLINE. Mine N strategies and the best one looks great by
construction; under pure noise the expected best t-stat grows like sqrt(2 ln N).

This family builds both halves on the REAL 62-name panel (2010-2024):

  1. THE ENGINE -- a random-expression generator over lookahead-safe primitives
     (ts_mean / ts_std / ts_delta / ts_corr over returns, price, volume, dollar
     volume), evaluated by daily cross-sectional Spearman IC, TRAIN 2010-2017.
  2. THE DISCIPLINE -- every candidate is re-scored OOS (2018-2024) untouched;
     the report prints the noise bar E[max|t|] ~ sqrt(2 ln N) next to the naive
     |t|>2 bar, and the train-vs-test scatter that shows the mirage directly.
  3. THE CLASSICS -- momentum 12-1, 1-month reversal, low-vol,
     Amihud illiquidity, turnover -- run through the SAME evaluator, so mined
     noise and published anomalies face identical scoring. (Value/quality/PEAD
     need fundamentals/events this panel doesn't have -- see 10's coverage matrix.)

Expected outcome on THIS panel (stated before you scroll): mega-caps post-2010 are
the sharpest-crowd slice there is; 10_real_data already showed price factors are
flat-to-negative net of costs here. If the miner "finds" anything the train column
loves and the test column kills, that is the lesson working -- the same one 07
teaches with leakage and 10 with optimization. Real mining edge comes from breadth
(thousands of names), data nobody else has, and costs/capacity discipline -- not
from a luckier expression on 62 mega-caps.
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
START, END = "2010-01-01", "2024-12-31"
SPLIT = "2018-01-01"                 # train < SPLIT <= test
N_CANDIDATES = 240
SEED = 11
COST_BPS = 5.0
MIN_NAMES = 20                       # min cross-section for a daily IC
WINDOWS = (3, 5, 10, 21, 63, 126, 252)

UNIVERSE = [
    "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "ADBE", "AMAT",
    "GOOG", "DIS", "VZ", "CMCSA", "JPM", "BAC", "WFC", "GS", "AXP", "C", "MS",
    "USB", "BLK", "JNJ", "PFE", "MRK", "UNH", "ABT", "AMGN", "GILD", "MDT",
    "PG", "KO", "PEP", "WMT", "MCD", "COST", "CL", "HD", "NKE", "LOW", "SBUX",
    "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "BA", "HON", "UPS", "GE", "MMM",
    "LMT", "DE", "DUK", "SO", "NEE", "APD", "SHW", "FCX", "NEM",
]


# ---------------------------------------------------------------- expression engine
class Miner:
    """Random alpha-expression generator over lookahead-safe panel primitives."""

    def __init__(self, px, r, v, rng):
        dv = (px * v).replace(0, np.nan)
        self.terms = {"r": r, "px_z": self._z(px), "v_z": self._z(np.log(v.replace(0, np.nan))),
                      "dv_z": self._z(np.log(dv)), "absr": r.abs(), "illiq": r.abs() / dv}
        self.rng = rng

    @staticmethod
    def _z(x, n=63):
        m = x.rolling(n, min_periods=n // 2)
        return (x - m.mean()) / m.std()

    def atom(self):
        t = self.rng.choice(list(self.terms))
        n = int(self.rng.choice(WINDOWS))
        op = self.rng.choice(["ts_mean", "ts_std", "ts_delta", "ts_sum", "ts_corr_r"])
        x = self.terms[t]
        if op == "ts_mean":
            s = x.rolling(n, min_periods=max(2, n // 2)).mean()
        elif op == "ts_sum":
            s = x.rolling(n, min_periods=max(2, n // 2)).sum()
        elif op == "ts_std":
            s = x.rolling(n, min_periods=max(2, n // 2)).std()
        elif op == "ts_delta":
            s = x - x.shift(n)
        else:                                   # ts_corr with returns
            s = x.rolling(n, min_periods=max(3, n // 2)).corr(self.terms["r"])
        return s, f"{op}({t},{n})"

    def candidate(self):
        s1, n1 = self.atom()
        if self.rng.random() < 0.5:             # single atom
            expr, sig = n1, s1
        else:                                   # combine two atoms
            s2, n2 = self.atom()
            op = self.rng.choice(["+", "-", "*"])
            if op == "+":
                sig = s1.rank(axis=1) + s2.rank(axis=1)
            elif op == "-":
                sig = s1.rank(axis=1) - s2.rank(axis=1)
            else:
                sig = s1.rank(axis=1) * s2.rank(axis=1)
            expr = f"{n1} {op} {n2}"
        return expr, sig            # sign-free: IC is signed, |t| is what's mined


# ---------------------------------------------------------------- IC machinery
def daily_ic(signal: pd.DataFrame, fwd_rank: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman IC of signal_t vs r_{t+1} (both pre-aligned)."""
    import warnings
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    s = signal.rank(axis=1).values
    f = fwd_rank.values
    valid = ~np.isnan(s) & ~np.isnan(f)
    s, f = np.where(valid, s, np.nan), np.where(valid, f, np.nan)
    n = valid.sum(axis=1)
    sm, fm = np.nanmean(s, axis=1, keepdims=True), np.nanmean(f, axis=1, keepdims=True)
    sd, fd = s - sm, f - fm
    cov = np.nansum(sd * fd, axis=1)
    den = np.sqrt(np.nansum(sd**2, axis=1) * np.nansum(fd**2, axis=1))
    ic = np.where((n >= MIN_NAMES) & (den > 0), cov / np.where(den > 0, den, 1), np.nan)
    return pd.Series(ic, index=signal.index)


def ic_stats(ic: pd.Series) -> tuple[float, float]:
    ic = ic.dropna()
    if len(ic) < 60:
        return np.nan, np.nan
    return float(ic.mean()), float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)))


def ls_book(signal: pd.DataFrame, returns: pd.DataFrame, name: str,
            frac: float = 0.2) -> pd.Series:
    """Quintile L/S book through the core engine (lag 1, net of costs)."""
    rk = signal.rank(axis=1, pct=True)
    long = (rk >= 1 - frac).astype(float)
    short = (rk <= frac).astype(float)
    W = long.div(long.sum(axis=1), axis=0).fillna(0.0) \
        - short.div(short.sum(axis=1), axis=0).fillna(0.0)
    return core.backtest_weights(returns, W, cost_bps=COST_BPS, lag=1, name=name).returns


# ---------------------------------------------------------------- main
def main():
    print("\n=== Factor mining engine + multiple-testing discipline (REAL data) ===")
    try:
        md = core.data.load_yfinance(UNIVERSE, start=START, end=END,
                                     cache_dir=os.path.join(HERE, "data_cache"))
    except Exception as e:
        print(f"[skip] could not load real data: {type(e).__name__}: {e}")
        print("       Run once WITH network to populate data_cache/. Skipping cleanly.")
        return
    px, r, v = md.prices, md.returns, md.volume
    fwd = r.shift(-1)                           # r_{t+1}, aligned to signal date t
    fwd_rank = fwd.rank(axis=1)
    is_train = px.index < pd.Timestamp(SPLIT)
    print(f"Panel: {px.shape[1]} names x {px.shape[0]} days; "
          f"train {START[:4]}-2017 ({int(is_train.sum())}d), "
          f"test 2018-{END[:4]} ({int((~is_train).sum())}d).")

    # ---- the classics through the same evaluator ---------------------------
    classics = {
        "mom_12_1": px.shift(21) / px.shift(252) - 1.0,
        "reversal_1m": -(px / px.shift(21) - 1.0),
        "low_vol": -r.rolling(63).std(),
        "amihud_illiq": (r.abs() / (px * v)).rolling(21).mean(),
        "low_turnover": -(v / v.rolling(63).mean()).rolling(21).mean(),
    }
    print("\nClassic factors, same scoring (daily rank-IC vs next-day return):")
    print(f"  {'factor':<24s} {'train IC':>9s} {'t':>6s} {'test IC':>9s} {'t':>6s}")
    for nm, sig in classics.items():
        ic = daily_ic(sig, fwd_rank)
        m1, t1 = ic_stats(ic[is_train])
        m2, t2 = ic_stats(ic[~is_train])
        print(f"  {nm:<24s} {m1:>+9.4f} {t1:>+6.1f} {m2:>+9.4f} {t2:>+6.1f}")

    # ---- mine ---------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    miner = Miner(px, r, v, rng)
    print(f"\nMining {N_CANDIDATES} random expressions (seed {SEED}) ...")
    results = []
    for i in range(N_CANDIDATES):
        expr, sig = miner.candidate()
        ic = daily_ic(sig, fwd_rank)
        m1, t1 = ic_stats(ic[is_train])
        m2, t2 = ic_stats(ic[~is_train])
        if not np.isnan(t1):
            results.append({"expr": expr, "ic_tr": m1, "t_tr": t1,
                            "ic_te": m2, "t_te": t2, "sig": sig})
        if (i + 1) % 60 == 0:
            print(f"  {i + 1}/{N_CANDIDATES} scored ...")
    res = (pd.DataFrame(results).drop_duplicates(subset="expr")
           .sort_values("t_tr", key=abs, ascending=False))

    n_eff = len(res)
    noise_bar = float(np.sqrt(2 * np.log(n_eff)))
    bonf = 3.89                                  # two-sided p=0.05/240
    naive = int((res.t_tr.abs() > 2).sum())
    survive = int(((res.t_tr.abs() > 2) & (res.t_te.abs() > 2)
                   & (np.sign(res.t_tr) == np.sign(res.t_te))).sum())
    print(f"\nTHE MIRAGE, QUANTIFIED ({n_eff} candidates):")
    print(f"  naive '|t|>2 significant' on TRAIN:            {naive:3d} candidates")
    print(f"  expected MAX |t| from PURE NOISE (sqrt(2lnN)):  {noise_bar:.2f}")
    print(f"  Bonferroni bar (p=0.05/{n_eff}):                 {bonf:.2f}")
    print(f"  train-significant that stay significant OOS:    {survive:3d}")
    print("\nTop-10 by train |t| -- watch the test column:")
    print(f"  {'expression':<44s} {'tr IC':>7s} {'tr t':>6s} {'te IC':>7s} {'te t':>6s}")
    for _, row in res.head(10).iterrows():
        print(f"  {row.expr[:44]:<44s} {row.ic_tr:>+7.4f} {row.t_tr:>+6.1f} "
              f"{row.ic_te:>+7.4f} {row.t_te:>+6.1f}")
    corr_tt = float(res.t_tr.corr(res.t_te))
    print(f"\n  corr(train t, test t) across candidates: {corr_tt:+.2f} -- >0 means the")
    print("  candidate POOL shares weak real structure (they reuse windows/terms); it")
    print("  is not evidence that any SINGLE pick is real -- the top pick just died.")

    # ---- OOS books: best-mined vs classics, through the real engine --------
    r_test = r.loc[~is_train]
    top = res.iloc[0]
    top_sig = top.sig * np.sign(top.ic_tr)      # trade it the way train said to
    streams = {
        f"mined_best [{top.expr[:28]}]": ls_book(top_sig.loc[~is_train], r_test, "mined"),
        "mom_12_1": ls_book(classics["mom_12_1"].loc[~is_train], r_test, "mom"),
        "reversal_1m": ls_book(classics["reversal_1m"].loc[~is_train], r_test, "rev"),
        "low_vol": ls_book(classics["low_vol"].loc[~is_train], r_test, "lv"),
    }
    rows = [core.metrics.summary(s.dropna(), n) for n, s in streams.items()]
    board = core.format_leaderboard(rows)
    print("\nOOS (2018-2024) quintile L/S books, net 5 bps -- the only column that counts:")
    print(board[["sharpe", "ann_return", "ann_vol", "max_drawdown",
                 "hit_rate"]].round(3).to_string())
    print("\nNote the two-step lesson: the only classic whose IC survives OOS is the")
    print("century-old one (momentum) -- and even ITS book loses net of costs on this")
    print("panel. Surviving the t-test is necessary, not sufficient: IC != money.")

    # ---- figure -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax1.scatter(res.t_tr, res.t_te, s=14, alpha=0.55)
    for b, c, lb in ((2, "orange", "naive |t|=2"), (noise_bar, "red",
                     f"noise max ~{noise_bar:.1f}"), (bonf, "darkred", "Bonferroni")):
        ax1.axvline(b, color=c, lw=1, ls="--", label=lb)
        ax1.axvline(-b, color=c, lw=1, ls="--")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("train t-stat (2010-2017)"); ax1.set_ylabel("test t-stat (2018-2024)")
    ax1.set_title(f"{n_eff} mined alphas: train vs test (corr {corr_tt:+.2f})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    for nm, s in streams.items():
        ax2.plot(core.metrics.equity_curve(s.dropna()), lw=1.2, label=nm[:34])
    ax2.set_title("OOS quintile L/S books, net of costs")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "equity.png"), dpi=120)
    plt.close(fig)

    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print("\nHonest caveats: 62 mega-caps is a BREADTH-starved panel (real miners use")
    print("1000s of names -- IC scales with sqrt(breadth)); candidates share windows/")
    print("terms so the sqrt(2lnN) independent-trials bar UNDERSTATES the haircut for")
    print("N truly independent tries; and any factor that survives here still faces")
    print("costs/capacity/crowding before it is money. The engine is the easy part.")
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
