"""Cross-sectional CARRY strategy on a synthetic cross-section of assets.

We simulate ~24 "assets" (think FX pairs or commodity futures). Each asset has
a slowly-varying CARRY characteristic -- an interest-rate differential (FX) or
futures roll yield (commodities). The embedded structure is the *carry premium*:
on average, high-carry assets earn a positive excess return proportional to
their carry, on top of a common market factor and idiosyncratic price noise.

Crucial realism -- this is NOT free money. We add occasional CARRY-UNWIND
crashes: rare "risk-off" days on which high-carry assets sell off together
(carry is negatively skewed; "up the stairs, down the elevator"). A naive
Sharpe overstates how nice this trade is, so we report max drawdown too.

Signal: cross-sectionally rank by carry, go long the top quantile / short the
bottom quantile (dollar-neutral) via ``core.long_short_backtest``. The signal
at date t uses only carry known at t; the engine adds a >=1-day execution lag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class CarryData:
    returns: pd.DataFrame   # dates x assets, daily simple excess returns
    carry: pd.DataFrame     # dates x assets, annualized carry (rate diff / roll yield)
    meta: dict = field(default_factory=dict)


def generate_carry_market(
    n_assets: int = 24,
    n_days: int = 2520,
    seed: int = 11,
    carry_premium: float = 0.9,      # how strongly carry maps to expected return
    asset_vol: float = 0.10,         # annualized idiosyncratic price vol
    market_vol: float = 0.07,        # annualized common-factor vol
    unwind_prob: float = 0.012,      # daily probability of a carry-unwind shock
    unwind_beta: float = 0.85,       # how hard high-carry names get hit on unwinds
    unwind_vol: float = 0.05,        # size (daily stdev) of the unwind shock
) -> CarryData:
    """Simulate a cross-section of carry assets with an embedded carry premium
    and occasional carry-unwind crashes.

    Expected daily excess return of asset i on a normal day is
    ``carry_premium * carry_i / TRADING_DAYS`` (so a +5% carry asset earns,
    in expectation, an extra ``carry_premium*5%`` per year). On rare unwind
    days a common negative shock hits every asset in proportion to how
    high-carry it is, manufacturing the negative skew that makes carry risky.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)
    assets = [f"CARRY{i:02d}" for i in range(n_assets)]

    # Persistent carry process per asset: AR(1) around a static cross-sectional
    # spread, so some assets are structurally high-carry (e.g. EM/high-yield) and
    # some low-carry, but the ranking still drifts through time.
    base_carry = rng.uniform(-0.06, 0.10, n_assets)      # static annualized level
    phi = 0.995                                           # very persistent
    sig_c = 0.02 * np.sqrt(1 - phi ** 2)                  # innovation scale
    carry_state = base_carry.copy()

    sigma_d = asset_vol / np.sqrt(TRADING_DAYS)
    mkt_sd = market_vol / np.sqrt(TRADING_DAYS)

    carry_path = np.empty((n_days, n_assets))
    ret_path = np.empty((n_days, n_assets))

    for t in range(n_days):
        carry_state = base_carry + phi * (carry_state - base_carry) + sig_c * rng.standard_normal(n_assets)
        carry_path[t] = carry_state

        # common market factor (broad risk-on/off, affects all assets ~equally)
        m_t = mkt_sd * rng.standard_normal()

        # carry premium: expected return increases with carry level
        expected = carry_premium * carry_state / TRADING_DAYS

        r = expected + m_t + sigma_d * rng.standard_normal(n_assets)

        # occasional CARRY-UNWIND: high-carry assets crash together
        if rng.random() < unwind_prob:
            shock = -abs(unwind_vol * rng.standard_normal())          # always negative
            # demeaned carry so the unwind is a *relative* hit to high-carry names
            rel_carry = carry_state - carry_state.mean()
            r = r + unwind_beta * shock * (1.0 + 5.0 * rel_carry)

        ret_path[t] = r

    returns = pd.DataFrame(ret_path, index=dates, columns=assets)
    carry = pd.DataFrame(carry_path, index=dates, columns=assets)
    return CarryData(
        returns=returns, carry=carry,
        meta={"seed": seed, "n_assets": n_assets, "n_days": n_days,
              "carry_premium": carry_premium, "unwind_prob": unwind_prob},
    )


def carry_signal(cd: CarryData) -> pd.DataFrame:
    """Cross-sectional carry signal: higher carry => go long. Uses only carry
    observable at date t (the backtester adds the execution lag)."""
    return cd.carry
