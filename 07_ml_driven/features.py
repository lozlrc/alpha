"""Point-in-time feature engineering for the ML-driven cross-sectional model.

Every feature for date ``t`` is built from information available *up to and
including* the close of ``t`` (rolling windows, lagged fundamentals). The
LABEL, by contrast, is the *forward* H-day return measured strictly in the
future (t -> t+H) and is shifted back to align with the date the prediction
is made. Because consecutive labels share overlapping forward windows, this
is exactly the structure that demands purged+embargoed CV (see ``cv.py``).

The same effects the data generator plants -- momentum, value (book/price),
quality (profitability), low-vol -- are exposed here as features so a tree
model can rediscover them rather than us hand-coding the blend.

`build_panel` returns a tidy long DataFrame indexed by (date, ticker) with
one column per feature plus a ``fwd_ret`` label column, which is what the
model and CV layers consume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core

# Forward-return horizon (trading days) for the prediction label.
H = 21

# Column order is also the model's feature order.
FEATURE_NAMES = [
    "mom_21",        # 1-month price momentum
    "mom_63",        # 3-month momentum
    "mom_252_21",    # 12-1 momentum (skip last month)
    "rev_5",         # short-term reversal (negated 5d return)
    "book_to_price",
    "earnings_yield",
    "profitability",
    "vol_21",        # realized vol, 1 month
    "vol_63",        # realized vol, 3 month
    "vol_trend",     # 5d vs 21d average-volume ratio (liquidity/attention)
    "beta",          # static market beta
]


def _stack(df: pd.DataFrame, name: str) -> pd.Series:
    """Wide (dates x tickers) -> long Series indexed by (date, ticker)."""
    s = df.stack()
    s.index = s.index.set_names(["date", "ticker"])
    return s.rename(name)


def compute_features(md: "core.MarketData") -> dict[str, pd.DataFrame]:
    """All features as wide (dates x tickers) frames, strictly point-in-time."""
    p = md.prices
    r = md.returns
    v = md.volume

    feats: dict[str, pd.DataFrame] = {}
    # --- momentum, multi-horizon ---
    feats["mom_21"] = p / p.shift(21) - 1.0
    feats["mom_63"] = p / p.shift(63) - 1.0
    # 12-1: trailing 12m return skipping the most recent month (reversal noise).
    feats["mom_252_21"] = p.shift(21) / p.shift(252) - 1.0
    # --- short-term reversal: recent losers tend to bounce -> negate ---
    feats["rev_5"] = -(p / p.shift(5) - 1.0)
    # --- value / quality (lagged, as-reported fundamentals) -- ONLY if present.
    #     Real price feeds carry no fundamentals, so these are skipped there and the
    #     model trains on the price/volume features alone (see 10_real_data). ---
    if md.fundamentals:
        feats["book_to_price"] = md.book_to_price()
        feats["earnings_yield"] = md.earnings_yield()
        feats["profitability"] = md.fundamentals["profitability"]
    # --- realized volatility (low-vol anomaly: model can sign it) ---
    feats["vol_21"] = r.rolling(21).std()
    feats["vol_63"] = r.rolling(63).std()
    # --- volume trend: 5d vs 21d mean volume (relative attention) ---
    vol_5 = v.rolling(5).mean()
    vol_21 = v.rolling(21).mean()
    feats["vol_trend"] = vol_5 / vol_21.replace(0, np.nan) - 1.0
    # --- static market beta, broadcast across dates ---
    feats["beta"] = pd.DataFrame(
        np.broadcast_to(md.betas.values, p.shape), index=p.index, columns=p.columns
    )
    return feats


def forward_return(md: "core.MarketData", horizon: int = H) -> pd.DataFrame:
    """LABEL: simple return over the *next* `horizon` days, aligned to date t.

    fwd_ret[t] = P[t+H]/P[t] - 1, i.e. the return you earn *after* deciding at
    t. Built by shifting the future price back by H so it sits on row t; the
    last H rows are NaN (no realized future yet). This forward overlap is the
    source of label leakage that purging/embargo must defend against.
    """
    p = md.prices
    return p.shift(-horizon) / p - 1.0


def build_panel(md: "core.MarketData", horizon: int = H) -> pd.DataFrame:
    """Tidy long panel: index (date, ticker), feature columns + ``fwd_ret``.

    Cross-sectionally z-scores every feature per date so the tree sees
    comparable, outlier-tamed inputs and decisions transfer across regimes.
    Rows with any missing feature or a missing label are dropped.
    """
    feats = compute_features(md)
    names = [n for n in FEATURE_NAMES if n in feats]   # available features, canonical order
    # Cross-sectional z-score (per date) keeps features on a common scale and
    # makes the signal naturally rank-based / dollar-neutral friendly.
    cols = {name: _stack(core.zscore(feats[name], axis=1), name) for name in names}
    label = _stack(forward_return(md, horizon), "fwd_ret")

    panel = pd.concat(list(cols.values()) + [label], axis=1)
    panel = panel.dropna(subset=names + ["fwd_ret"])
    return panel.sort_index()
