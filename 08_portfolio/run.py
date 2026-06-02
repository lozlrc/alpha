"""Multi-strategy portfolio: combine the alphas into one risk-managed book.

This is the payoff of the whole suite: the point of generating many roughly
uncorrelated alphas is to COMBINE them. We take one production-quality net
return stream from each daily family (the intraday microstructure family is
excluded -- different clock; the leaky-ML demo and decayed-sentiment variants
are excluded -- not real strategies), then allocate across them.

Allocation is done with TRAILING-WINDOW estimates and a monthly rebalance, so
there is no lookahead: weights at each rebalance use only the prior `WINDOW`
days, and the engine applies them with a 1-day execution lag. Methods:
  * equal weight
  * inverse volatility
  * risk parity (equal risk contribution, via SLSQP)
  * minimum variance (Ledoit-Wolf shrunk covariance)
  * risk parity + 10% annual vol target

Honest caveat: these synthetic streams come from independent data-generating
processes, so their cross-correlations are near zero and diversification looks
close to ideal. Real strategies share common risk factors and correlate far
more in stress, so expect materially less benefit live. The construction
methods and the math, however, are exactly what you'd use on real streams.
Run:  ../.venv/bin/python run.py    (run ../run_all.py first to create inputs)
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
from scipy.optimize import minimize  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WINDOW, STEP, COST_BPS, VOL_TARGET = 126, 21, 2.0, 0.10

# Curated "book": one or a few honest, net strategies per family.
BOOK = [
    ("factors_multifactor",  "01_cross_sectional_factors", "multifactor"),
    ("statarb_pairs",        "02_statistical_arbitrage",   "pairs_trading"),
    ("statarb_leadlag",      "02_statistical_arbitrage",   "lead_lag"),
    ("event_pead",           "04_event_driven",            "PEAD"),
    ("event_index_rebal",    "04_event_driven",            "index_rebalance"),
    ("event_merger_arb",     "04_event_driven",            "merger_arb"),
    ("altdata_sentiment",    "05_alternative_data",        "sentiment_LS"),
    ("altdata_webtraffic",   "05_alternative_data",        "web_traffic_LS"),
    ("carry",                "06_cross_asset_vol",         "carry_long_short"),
    ("vol_short_variance",   "06_cross_asset_vol",         "short_variance"),
    ("dispersion",           "06_cross_asset_vol",         "dispersion"),
    ("ml_gbm",               "07_ml_driven",               "ml_honest_oos"),
    ("agentic_reversal",     "09_agentic_flow",            "crowd_reversal"),
]


def load_streams() -> dict[str, np.ndarray]:
    out = {}
    for label, folder, col in BOOK:
        path = os.path.join(ROOT, folder, "equity_returns.csv")
        if not os.path.exists(path):
            print(f"  [skip] {label}: {os.path.relpath(path)} missing -- run ../run_all.py first")
            continue
        df = pd.read_csv(path, index_col=0)
        if col not in df.columns:
            print(f"  [skip] {label}: column '{col}' not in {folder}")
            continue
        out[label] = df[col].dropna().to_numpy()
    return out


def align(streams: dict[str, np.ndarray]) -> pd.DataFrame:
    """Align independent streams by length (trailing) onto a shared calendar."""
    T = min(len(v) for v in streams.values())
    data = {k: v[-T:] for k, v in streams.items()}
    return pd.DataFrame(data, index=pd.bdate_range("2015-01-02", periods=T))


# ---- weighting schemes (operate on a trailing window: shape (window, K)) ----
def w_equal(win):
    return np.ones(win.shape[1]) / win.shape[1]


def w_invvol(win):
    v = win.std(axis=0, ddof=1)
    w = np.divide(1.0, v, out=np.zeros_like(v), where=v > 0)
    return w / w.sum() if w.sum() > 0 else w_equal(win)


def w_riskparity(win):
    cov = np.cov(win, rowvar=False)
    x0 = w_invvol(win)

    def obj(w):
        rc = w * (cov @ w)               # risk contributions
        return np.sum((rc - rc.mean()) ** 2)

    res = minimize(obj, x0, method="SLSQP", bounds=[(0, 1)] * len(x0),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else x0


def w_minvar(win):
    cov = LedoitWolf().fit(win).covariance_
    K = cov.shape[0]

    def obj(w):
        return w @ cov @ w

    res = minimize(obj, np.ones(K) / K, method="SLSQP", bounds=[(0, 1)] * K,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500})
    return res.x if res.success else np.ones(K) / K


def build_weights(R: pd.DataFrame, wfun, voltarget=None) -> pd.DataFrame:
    arr = R.to_numpy()
    T = len(R)
    W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
    for r in range(WINDOW, T, STEP):
        win = arr[r - WINDOW:r]
        w = wfun(win)
        if voltarget is not None:
            cov = np.cov(win, rowvar=False)
            pv = np.sqrt(max(w @ cov @ w, 1e-12)) * np.sqrt(252)
            w = w * min(voltarget / pv, 3.0)   # cap leverage at 3x
        W.iloc[r] = w
    return W


def corr_heatmap(R: pd.DataFrame, path: str):
    C = R.corr()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(C.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(C)), C.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(C)), C.columns, fontsize=7)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Strategy return correlations")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    print("\n=== Multi-strategy portfolio ===")
    streams = load_streams()
    if len(streams) < 2:
        print("Not enough streams found. Run `../.venv/bin/python ../run_all.py` first.")
        return
    R = align(streams)
    print(f"Loaded {R.shape[1]} strategies x {R.shape[0]} aligned days.")
    avg_corr = R.corr().where(~np.eye(len(R.columns), dtype=bool)).stack().mean()
    print(f"Average pairwise correlation: {avg_corr:.3f}")

    methods = {
        "equal_weight": build_weights(R, w_equal),
        "inverse_vol": build_weights(R, w_invvol),
        "risk_parity": build_weights(R, w_riskparity),
        "min_variance": build_weights(R, w_minvar),
        f"risk_parity_voltgt{int(VOL_TARGET*100)}": build_weights(R, w_riskparity, voltarget=VOL_TARGET),
    }

    results, summaries = {}, []
    for name, W in methods.items():
        res = core.backtest_weights(R, W, cost_bps=COST_BPS, lag=1, name=name)
        ret = res.returns.iloc[WINDOW:]          # drop warmup for fair comparison
        results[name] = ret
        summaries.append(core.metrics.summary(ret, name, turnover=res.turnover.iloc[WINDOW:]))

    # best individual stream (post-warmup) for reference
    single = {c: core.metrics.sharpe(R[c].iloc[WINDOW:]) for c in R.columns}
    best = max(single, key=single.get)
    results[f"best_single[{best}]"] = R[best].iloc[WINDOW:]
    summaries.append(core.metrics.summary(R[best].iloc[WINDOW:], f"best_single[{best}]"))

    board = core.format_leaderboard(summaries)
    print("\n" + board.round(3).to_string())

    # diversification math
    rp = "risk_parity"
    avg_single = np.mean(list(single.values()))
    print(f"\nAvg single-strategy Sharpe : {avg_single:.2f}")
    print(f"Best single-strategy Sharpe: {single[best]:.2f}  ({best})")
    print(f"Risk-parity portfolio Sharpe: {board.loc[rp, 'sharpe']:.2f}  "
          f"(+{board.loc[rp, 'sharpe'] - single[best]:.2f} vs best single)")

    eq_path = os.path.join(HERE, "portfolio_equity.png")
    core.plot_equity(results, eq_path, title="Multi-strategy portfolio vs best single", log=True)
    corr_path = os.path.join(HERE, "correlation.png")
    corr_heatmap(R, corr_path)
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print(f"\nSaved: {eq_path}\nSaved: {corr_path}\nSaved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
