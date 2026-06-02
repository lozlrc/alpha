"""Performance metrics for backtested return streams.

Everything here operates on a pandas Series of *simple* periodic returns
(e.g. daily). No lookahead, no live data -- these are pure post-hoc
analytics applied to a strategy's realized P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Cumulative growth of `start` units of capital."""
    returns = pd.Series(returns).fillna(0.0)
    return start * (1.0 + returns).cumprod()


def ann_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualized return."""
    returns = pd.Series(returns).dropna()
    if len(returns) == 0:
        return np.nan
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(returns)) - 1.0


def ann_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    excess = returns - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / sd)


def sortino(returns: pd.Series, periods_per_year: int = TRADING_DAYS, rf: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / dd)


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return np.nan
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return ann_return(returns, periods_per_year) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    nz = returns[returns != 0]
    if len(nz) == 0:
        return np.nan
    return float((nz > 0).mean())


def summary(returns: pd.Series, name: str = "strategy",
            periods_per_year: int = TRADING_DAYS, turnover: pd.Series | None = None) -> dict:
    """One-stop metrics dict for a return stream."""
    returns = pd.Series(returns).dropna()
    out = {
        "name": name,
        "ann_return": ann_return(returns, periods_per_year),
        "ann_vol": ann_vol(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "hit_rate": hit_rate(returns),
        "n_periods": int(len(returns)),
    }
    if turnover is not None:
        t = pd.Series(turnover).dropna()
        out["avg_turnover"] = float(t.mean()) if len(t) else np.nan
    return out


def format_leaderboard(summaries: list[dict]) -> pd.DataFrame:
    """Sortable table from a list of summary() dicts."""
    df = pd.DataFrame(summaries).set_index("name")
    cols = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "calmar", "hit_rate"]
    if "avg_turnover" in df.columns:
        cols.append("avg_turnover")
    cols.append("n_periods")
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("sharpe", ascending=False)


def print_summary(returns: pd.Series, name: str = "strategy",
                  periods_per_year: int = TRADING_DAYS, turnover: pd.Series | None = None) -> dict:
    s = summary(returns, name, periods_per_year, turnover)
    print(f"  {name}")
    print(f"    Sharpe         {s['sharpe']:.2f}")
    print(f"    Ann. return    {s['ann_return'] * 100:6.1f}%")
    print(f"    Ann. vol       {s['ann_vol'] * 100:6.1f}%")
    print(f"    Max drawdown   {s['max_drawdown'] * 100:6.1f}%")
    print(f"    Hit rate       {s['hit_rate'] * 100:6.1f}%")
    if "avg_turnover" in s:
        print(f"    Avg turnover   {s['avg_turnover'] * 100:6.1f}%")
    return s
