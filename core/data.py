"""Synthetic market data with *embedded* structure -- the offline data layer.

Returns are generated from a factor model with deliberately planted effects
so the demo strategies recover real, non-spurious signal:

  * momentum   -- a persistent AR(1) expected-return state per asset
  * value      -- slow mean-reversion of price toward a fundamental anchor
                  (so high book/price predicts higher future return)
  * quality    -- a profitability premium
  * low-vol    -- a (risk-adjusted) premium for lower-volatility names
  * market + sector common factors (so sector-neutralization matters)

Nothing here touches a live market. To run on REAL data instead, replace
`generate_market(...)` with a loader that returns a `MarketData` bundle with
the same fields (see `load_csv` / `load_yfinance` stubs at the bottom).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SECTORS = ["Tech", "Financials", "Energy", "Health",
           "Consumer", "Industrials", "Utilities", "Materials"]


@dataclass
class MarketData:
    prices: pd.DataFrame      # dates x tickers, close prices
    returns: pd.DataFrame     # simple returns
    volume: pd.DataFrame      # share volume
    market: pd.Series         # market factor return
    sectors: pd.Series        # ticker -> sector label
    betas: pd.Series          # ticker -> true market beta
    fundamentals: dict        # name -> DataFrame (dates x tickers), point-in-time
    meta: dict = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def book_to_price(self) -> pd.DataFrame:
        return self.fundamentals["book_value"] / self.prices

    def earnings_yield(self) -> pd.DataFrame:
        return self.fundamentals["earnings"] / self.prices


def _business_dates(n_days: int, start: str = "2015-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n_days)


def _as_reported(daily: pd.DataFrame, period: int = 63, lag: int = 5) -> pd.DataFrame:
    """Sample a daily fundamental quarterly and publish it with a reporting lag,
    so strategies only ever see point-in-time (non-lookahead) fundamentals."""
    idx = np.arange(len(daily))
    report_rows = daily.iloc[idx % period == 0]
    return report_rows.reindex(daily.index).ffill().shift(lag)


def generate_market(n_assets: int = 120, n_days: int = 1750, seed: int = 7,
                    mom_strength: float = 0.0021, value_kappa: float = 0.0014,
                    quality_premium: float = 0.08, lowvol_premium: float = 0.15,
                    market_mu: float = 0.05, market_vol: float = 0.16,
                    sector_vol: float = 0.10) -> MarketData:
    """Generate a multi-asset equity panel with embedded factor structure.

    The `*_strength` / `*_premium` knobs control how strong each planted
    effect is. Defaults are tuned to give realistic (Sharpe ~0.5-2) factor
    premia net of modest costs. Set any to 0 to switch that effect off.
    """
    rng = np.random.default_rng(seed)
    dates = _business_dates(n_days)
    tickers = [f"SYN{i:03d}" for i in range(n_assets)]
    n = n_assets

    # ---- static per-asset parameters ----
    beta = np.clip(rng.normal(1.0, 0.15, n), 0.4, 1.7)
    sigma_ann = rng.uniform(0.15, 0.55, n)
    sigma_d = sigma_ann / np.sqrt(TRADING_DAYS)
    quality = rng.normal(0.0, 1.0, n)                 # latent quality score
    z_quality = (quality - quality.mean()) / quality.std()
    z_sigma = (sigma_ann - sigma_ann.mean()) / sigma_ann.std()
    sector_idx = rng.integers(0, len(SECTORS), n)
    sector_load = rng.uniform(0.6, 1.2, n)

    roe = 0.08 + 0.06 * z_quality                     # profitability tied to quality
    earn_growth = (0.02 + 0.05 * (z_quality - z_quality.min()) /
                   (z_quality.max() - z_quality.min())) / TRADING_DAYS

    price0 = rng.uniform(20, 200, n)
    logP = np.log(price0)
    logF = logP + rng.normal(0.0, 0.10, n)            # mild initial mispricing
    mom_state = np.zeros(n)
    phi = 0.99  # ~100-day persistence -> 12-1 momentum works without killing value

    # daily tilts that are constant through time
    quality_tilt = quality_premium / TRADING_DAYS * z_quality
    lowvol_tilt = lowvol_premium / TRADING_DAYS * (-z_sigma)

    price_path = np.empty((n_days, n))
    logF_path = np.empty((n_days, n))
    ret_path = np.empty((n_days, n))
    market_path = np.empty(n_days)

    sig_mom = np.sqrt(1 - phi ** 2)
    for t in range(n_days):
        m_t = market_mu / TRADING_DAYS + (market_vol / np.sqrt(TRADING_DAYS)) * rng.standard_normal()
        sect_f = (sector_vol / np.sqrt(TRADING_DAYS)) * rng.standard_normal(len(SECTORS))
        mom_state = phi * mom_state + sig_mom * rng.standard_normal(n)

        value_tilt = value_kappa * (logF - logP)      # pull toward fundamental
        mom_tilt = mom_strength * mom_state
        # -0.5*sigma_d**2 cancels volatility drag so the *simple*-return mean
        # equals the intended expected return (otherwise high-vol names get a
        # spurious convexity boost that swamps the low-vol premium).
        expected = (beta * m_t + sector_load * sect_f[sector_idx]
                    + value_tilt + quality_tilt + lowvol_tilt + mom_tilt
                    - 0.5 * sigma_d ** 2)
        r = expected + sigma_d * rng.standard_normal(n)

        logP = logP + r
        logF = logF + earn_growth + rng.normal(0.0, 0.002, n)
        price_path[t] = np.exp(logP)
        logF_path[t] = logF
        ret_path[t] = r
        market_path[t] = m_t

    prices = pd.DataFrame(price_path, index=dates, columns=tickers)
    returns = prices.pct_change(fill_method=None)
    market = pd.Series(market_path, index=dates, name="market")

    # volume: higher on big-move days, lognormal noise
    vol_base = rng.uniform(2e5, 8e6, n)
    abs_z = np.abs(ret_path) / sigma_d
    volume = pd.DataFrame(vol_base * (1.0 + 0.8 * abs_z) * np.exp(rng.normal(0, 0.3, (n_days, n))),
                          index=dates, columns=tickers).round()

    # point-in-time fundamentals
    book_daily = pd.DataFrame(np.exp(logF_path), index=dates, columns=tickers)
    earn_daily = book_daily.mul(roe, axis=1)
    prof_daily = pd.DataFrame(np.broadcast_to(roe, (n_days, n)), index=dates, columns=tickers)
    fundamentals = {
        "book_value": _as_reported(book_daily),
        "earnings": _as_reported(earn_daily),
        "profitability": _as_reported(prof_daily),
    }

    return MarketData(
        prices=prices, returns=returns, volume=volume, market=market,
        sectors=pd.Series([SECTORS[i] for i in sector_idx], index=tickers, name="sector"),
        betas=pd.Series(beta, index=tickers, name="beta"),
        fundamentals=fundamentals,
        meta={"seed": seed, "n_assets": n_assets, "n_days": n_days,
              "knobs": {"mom_strength": mom_strength, "value_kappa": value_kappa,
                        "quality_premium": quality_premium, "lowvol_premium": lowvol_premium}},
    )


# --------------------------------------------------------------------------
# Real-data hooks (optional). The whole suite runs offline without these;
# they exist so you can swap in real prices without touching strategy code.
# --------------------------------------------------------------------------
def load_csv(prices_csv: str, **kwargs) -> MarketData:
    """Build a (prices/returns-only) MarketData from a wide CSV of close prices
    (index = dates, columns = tickers). Fundamentals/sectors left empty; only
    price-based strategies will work unless you populate them."""
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change(fill_method=None)
    empty = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    return MarketData(prices=prices, returns=returns, volume=empty.copy(),
                      market=returns.mean(axis=1).rename("market"),
                      sectors=pd.Series("Unknown", index=prices.columns),
                      betas=pd.Series(1.0, index=prices.columns),
                      fundamentals={}, meta={"source": prices_csv})


def load_yfinance(tickers, start="2015-01-01", end=None) -> MarketData:
    """Download daily closes via yfinance (requires `pip install yfinance` and
    network access). Provided for convenience -- this is historical data only,
    NOT a live/broker connection."""
    import yfinance as yf  # lazy import; optional dependency
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    returns = prices.pct_change(fill_method=None)
    empty = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    return MarketData(prices=prices, returns=returns, volume=empty.copy(),
                      market=returns.mean(axis=1).rename("market"),
                      sectors=pd.Series("Unknown", index=prices.columns),
                      betas=pd.Series(1.0, index=prices.columns),
                      fundamentals={}, meta={"source": "yfinance"})
