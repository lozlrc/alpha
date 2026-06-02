"""Vectorized, lookahead-safe backtest engine.

Convention
----------
A strategy produces *target weights* (or a cross-sectional *signal*) using
information available up to and including time t. The engine shifts weights
forward by `lag` (default 1) before applying them to returns -- i.e. you
decide at the close of day t and earn r_{t+1}. This is the single most
common source of fake alpha, so it is enforced here rather than left to
each strategy.

Transaction costs are charged on turnover (the L1 change in held weights)
at `cost_bps` basis points per unit traded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    returns: pd.Series      # net portfolio returns (after costs)
    gross: pd.Series        # before costs
    turnover: pd.Series     # per-period one-way turnover (L1 weight change)
    weights: pd.DataFrame   # weights actually held (already lagged)
    cost_bps: float
    name: str = "strategy"
    periods_per_year: int = metrics.TRADING_DAYS

    @property
    def equity(self) -> pd.Series:
        return metrics.equity_curve(self.returns)

    def summary(self) -> dict:
        return metrics.summary(self.returns, self.name, self.periods_per_year, self.turnover)

    def print_summary(self) -> dict:
        return metrics.print_summary(self.returns, self.name, self.periods_per_year, self.turnover)


def backtest_weights(returns: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 1.0,
                     lag: int = 1, name: str = "strategy",
                     periods_per_year: int = metrics.TRADING_DAYS) -> BacktestResult:
    """Backtest an explicit weights matrix against an asset-returns matrix.

    Parameters
    ----------
    returns : DataFrame (dates x assets) of simple returns.
    weights : DataFrame (dates x assets) of *target* weights decided using
              info up to that date. Missing dates are forward-filled (you
              hold the position between rebalances); missing assets -> 0.
    cost_bps: round-trip-agnostic per-unit-traded cost in basis points.
    lag     : periods to delay weights before earning returns (>=1 = no lookahead).
    """
    returns = returns.copy()
    weights = weights.reindex(index=returns.index, columns=returns.columns)
    # hold positions between rebalances, then lag to avoid lookahead
    weights = weights.ffill().fillna(0.0)
    held = weights.shift(lag).fillna(0.0)

    gross = (held * returns).sum(axis=1)
    trades = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = trades * (cost_bps / 1e4)
    net = gross - cost
    return BacktestResult(returns=net, gross=gross, turnover=trades, weights=held,
                          cost_bps=cost_bps, name=name, periods_per_year=periods_per_year)


def zscore(df: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
    """Standardize across assets (axis=1) or time (axis=0)."""
    mu = df.mean(axis=axis)
    sd = df.std(axis=axis, ddof=0)
    if axis == 1:
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return (df - mu) / sd.replace(0, np.nan)


def _demean_by_group(sig: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract the per-date group mean from each asset's signal (e.g. sector neutral)."""
    groups = pd.Series(groups)
    out = sig.copy()
    for _, cols in groups.groupby(groups).groups.items():
        cols = [c for c in cols if c in sig.columns]
        if not cols:
            continue
        block = sig[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def cross_sectional_weights(signal: pd.DataFrame, quantiles: int = 5, long_short: bool = True,
                            gross_leverage: float = 1.0,
                            neutralize_groups: pd.Series | None = None) -> pd.DataFrame:
    """Turn a cross-sectional signal into (optionally dollar-neutral) weights.

    Each date: rank assets by signal, go long the top 1/quantiles, short the
    bottom 1/quantiles, equal-weight within each leg. Long-short books are
    scaled to `gross_leverage` total (half long, half short).

    NaN signal = asset not in the universe that day (excluded).
    """
    sig = signal.astype(float).copy()
    if neutralize_groups is not None:
        sig = _demean_by_group(sig, neutralize_groups)

    ranks = sig.rank(axis=1, method="first")
    counts = sig.notna().sum(axis=1)
    frac = ranks.sub(1).div(counts.replace(0, np.nan), axis=0)  # fractional rank in [0,1)

    long_mask = frac >= (1.0 - 1.0 / quantiles)
    short_mask = frac < (1.0 / quantiles)

    n_long = long_mask.sum(axis=1)
    long_w = long_mask.div(n_long.replace(0, np.nan), axis=0).fillna(0.0)

    if long_short:
        n_short = short_mask.sum(axis=1)
        short_w = short_mask.div(n_short.replace(0, np.nan), axis=0).fillna(0.0)
        w = 0.5 * gross_leverage * (long_w - short_w)
    else:
        w = gross_leverage * long_w

    valid = counts >= quantiles * 2  # need enough names to form both legs
    return w.where(valid, 0.0)


def long_short_backtest(signal: pd.DataFrame, returns: pd.DataFrame, quantiles: int = 5,
                        cost_bps: float = 1.0, lag: int = 1,
                        neutralize_groups: pd.Series | None = None,
                        name: str = "strategy",
                        periods_per_year: int = metrics.TRADING_DAYS) -> BacktestResult:
    """Convenience: cross-sectional signal -> dollar-neutral L/S backtest."""
    w = cross_sectional_weights(signal, quantiles=quantiles, long_short=True,
                                neutralize_groups=neutralize_groups)
    return backtest_weights(returns, w, cost_bps=cost_bps, lag=lag, name=name,
                            periods_per_year=periods_per_year)
