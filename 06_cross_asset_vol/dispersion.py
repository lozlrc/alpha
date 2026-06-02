"""Dispersion trade -- short index vol / long single-name vol.

Model: an equal-weighted index of N single names. By definition the index
variance is a correlation-weighted aggregation of the single-name variances::

    var_index = sum_i w_i^2 var_i + sum_{i!=j} w_i w_j rho_ij sigma_i sigma_j

With equal weights and an average pairwise correlation ``rho`` this collapses to

    var_index ~= (1/N) * avg_var  +  (1 - 1/N) * rho * avg_var

so index vol is increasing in correlation. Dealers structurally OVER-price index
options relative to the single-name basket (clients buy index puts for crash
protection), which shows up as the *implied* correlation embedded in index vol
being ABOVE the correlation that subsequently REALIZES. That gap is the
dispersion premium.

Embedded structure: implied correlation > realized correlation on average, so a
trade that is SHORT index variance and LONG the single-name variance basket
earns a positive carry. Real-world gotcha: the trade is short correlation, so in
a crash (correlations spike to ~1) it loses sharply -- we plant occasional
correlation-spike days.

Single P&L stream -> build a net-return Series and use ``core.metrics`` /
``core.plot_equity`` (house convention for vol strategies).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class DispersionData:
    avg_var: pd.Series           # cross-sectional average single-name variance
    implied_corr: pd.Series      # correlation priced into index vol at t
    realized_corr: pd.Series     # correlation that subsequently realizes over (t,t+1]
    n_names: int
    meta: dict = field(default_factory=dict)


def _index_var(avg_var: np.ndarray, corr: np.ndarray, n: int) -> np.ndarray:
    """Index variance from average single-name variance and average correlation."""
    return avg_var * (1.0 / n + (1.0 - 1.0 / n) * corr)


def generate_dispersion_process(
    n_names: int = 30,
    n_days: int = 2520,
    seed: int = 41,
    base_corr: float = 0.35,        # long-run average pairwise correlation
    corr_premium: float = 0.045,    # avg (implied - realized) correlation gap
    corr_persist: float = 0.95,     # AR(1) persistence of the correlation state
    corr_innov: float = 0.02,       # correlation-state innovation scale
    rv_corr_noise: float = 0.21,    # day-to-day realized-corr noise (gap often negative)
    avg_vol: float = 0.25,          # typical single-name vol level
    spike_prob: float = 0.016,      # daily prob of a correlation spike (crash)
    spike_corr: float = 0.5,        # extra correlation added on a spike day
) -> DispersionData:
    """Simulate average single-name variance plus implied & realized correlation.

    Implied correlation (priced into index vol at t) sits ABOVE the correlation
    that realizes over the next day by ``corr_premium`` on average -- the
    dispersion premium. On rare spike days realized correlation jumps toward 1
    (the short-correlation blow-up).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)

    avg_var = np.empty(n_days)
    implied_corr = np.empty(n_days)
    realized_corr = np.empty(n_days)

    log_v = np.log(avg_vol)
    mu_log = np.log(avg_vol)
    corr_state = base_corr   # persistent correlation level KNOWN at t

    for t in range(n_days):
        # persistent single-name vol level (log space, positive)
        log_v = mu_log + 0.97 * (log_v - mu_log) + 0.02 * rng.standard_normal()
        vol_t = float(np.exp(log_v))
        avg_var[t] = vol_t ** 2

        # Persistent correlation state observable at t (drives what implied is priced at).
        corr_state = base_corr + corr_persist * (corr_state - base_corr) + corr_innov * rng.standard_normal()
        corr_state = float(np.clip(corr_state, 0.02, 0.9))

        # Implied correlation priced at t: the state plus the dispersion premium.
        # Uses ONLY info available at t (cannot see the next-period spike).
        ic = corr_state + corr_premium + 0.012 * rng.standard_normal()
        implied_corr[t] = float(np.clip(ic, 0.02, 0.99))

        # Realized correlation over (t, t+1]: centered on the state with sizeable
        # day-to-day noise (so the gap is frequently negative -- not free money),
        # plus rare contagion spikes toward 1 -- unforeseeable at t, so they blow
        # through the implied level and the short-correlation trade loses sharply.
        rc = corr_state + rv_corr_noise * rng.standard_normal()
        if rng.random() < spike_prob:
            rc += abs(spike_corr * rng.standard_normal()) + 0.5 * spike_corr
        realized_corr[t] = float(np.clip(rc, 0.02, 0.99))

    return DispersionData(
        avg_var=pd.Series(avg_var, index=dates, name="avg_var"),
        implied_corr=pd.Series(implied_corr, index=dates, name="implied_corr"),
        realized_corr=pd.Series(realized_corr, index=dates, name="realized_corr"),
        n_names=n_names,
        meta={"seed": seed, "corr_premium": corr_premium, "spike_prob": spike_prob,
              "n_days": n_days, "n_names": n_names},
    )


def dispersion_returns(dd: DispersionData, scale: float = 2.2, cost_bps: float = 2.0) -> pd.Series:
    """Daily net return of a dispersion trade: SHORT index variance, LONG the
    single-name variance basket, vega-neutral so only the *correlation* gap drives P&L.

    P&L proxy = scale * avg_var * (1 - 1/N) * (implied_corr - realized_corr).
    Selling index var at the implied correlation and buying back the basket at the
    realized correlation, the single-name legs cancel and the residual is the
    correlation spread times the off-diagonal weight ``(1 - 1/N)``.

    No lookahead: implied corr known at t, realized accrues over (t,t+1]; P&L
    attributed to t+1.
    """
    off_diag = 1.0 - 1.0 / dd.n_names
    corr_gap = dd.implied_corr - dd.realized_corr
    pnl = scale * dd.avg_var * off_diag * corr_gap
    net = pnl - cost_bps / 1e4
    return net.shift(1).dropna().rename("dispersion")
