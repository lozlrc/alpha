"""Multi-asset cointegration: trade a target vs a basket, confirmed by ADF.

Embedded structure (see ``generate_basket``)
--------------------------------------------
We build K "basket" assets, each a random walk driven by a shared common
trend plus idiosyncratic random-walk noise (so every basket leg is
non-stationary / I(1)). The TARGET is a fixed linear combination of the
basket legs plus a *stationary* AR(1) residual:

    target_t = sum_j w_j * basket_j,t  +  resid_t,      resid_t ~ AR(1)

Hence ``target - basket @ w`` is stationary -> target and the basket are
cointegrated, and the residual is the tradable, mean-reverting object.

Method (fully lookahead-safe)
-----------------------------
1. CONFIRM cointegration with an Engle-Granger style test: regress target on
   the basket (OLS, with intercept) over an in-sample window and run an
   Augmented Dickey-Fuller (statsmodels ``adfuller``) test on the residual.
   We also report statsmodels ``coint``'s p-value. This is done ONCE on an
   in-sample slice (a true researcher's "is this even cointegrated?" gate);
   it does not touch out-of-sample prices used for trading P&L.
2. Re-estimate the cointegrating vector on a TRAILING window each day, form
   the residual, z-score it on a trailing window, and trade it with the same
   entry/exit state machine as the pairs book (positions lagged, cost-charged).

Real-world gotcha: the Engle-Granger vector is itself estimated and unstable
out of sample; a basket that tests cointegrated in-sample can decouple later
(structural break), so the confirmed p-value is necessary but far from
sufficient -- live residuals can wander far outside their historical band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def generate_basket(k: int = 4, n_days: int = 1750, seed: int = 23,
                    rho: float = 0.90, resid_vol: float = 0.016,
                    common_vol: float = 0.009, idio_vol: float = 0.005):
    """Generate (target_price, basket_prices_df, true_weights).

    Cointegration is built on the LOG-PRICE scale so every price stays
    strictly positive (an additive random-walk level can wander through zero
    and produce unphysical >100x daily returns -- real prices never do):

        log B_j = base_j + load_j * common_walk + idio_walk_j   (each I(1))
        log TARGET = base + sum_j w_j * log B_j + resid          (resid ~ AR(1))

    so ``log TARGET - w . log B`` is the stationary, tradable residual.
    ``rho`` controls residual persistence; the *_vol knobs are now daily
    log-return scales (e.g. 0.009 ~ 14% annualized).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-02", periods=n_days)
    cols = [f"BKT{j}" for j in range(k)]

    common = np.cumsum(common_vol * rng.standard_normal(n_days))
    log_basket = np.empty((n_days, k))
    loads = rng.uniform(0.6, 1.3, k)
    base = np.log(rng.uniform(40, 120, k))
    for j in range(k):
        idio = np.cumsum(idio_vol * rng.standard_normal(n_days))
        log_basket[:, j] = base[j] + loads[j] * common + idio

    w_true = rng.uniform(0.2, 0.8, k)
    w_true = w_true / w_true.sum()  # convex combo -> well-scaled target

    # stationary AR(1) residual on the log scale (the tradable part)
    resid = np.empty(n_days)
    resid[0] = rng.standard_normal() * resid_vol / np.sqrt(1 - rho ** 2)
    eps = resid_vol * rng.standard_normal(n_days)
    for t in range(1, n_days):
        resid[t] = rho * resid[t - 1] + eps[t]

    log_target = np.log(80.0) + log_basket @ w_true + resid
    basket_df = pd.DataFrame(np.exp(log_basket), index=dates, columns=cols)
    target_s = pd.Series(np.exp(log_target), index=dates, name="TARGET")
    return target_s, basket_df, pd.Series(w_true, index=cols, name="w_true")


def confirm_cointegration(target: pd.Series, basket: pd.DataFrame,
                          insample: float = 0.5) -> dict:
    """Engle-Granger gate on an IN-SAMPLE slice only.

    Returns OLS hedge vector, ADF p-value on the residual, and statsmodels
    ``coint`` p-value. Uses only the first ``insample`` fraction of the data
    so the trading P&L window stays untouched by this research step.
    """
    n_in = int(len(target) * insample)
    # cointegration holds on the log-price scale (prices are geometric)
    y = np.log(target.iloc[:n_in])
    X = np.log(basket.iloc[:n_in])
    Xc = sm.add_constant(X)
    ols = sm.OLS(y.to_numpy(), Xc.to_numpy()).fit()
    resid = y.to_numpy() - Xc.to_numpy() @ ols.params
    adf_p = adfuller(resid, autolag="AIC")[1]
    # statsmodels coint against the first basket leg as a quick scalar check
    coint_p = coint(y.to_numpy(), X.iloc[:, 0].to_numpy())[1]
    return {
        "beta": pd.Series(ols.params[1:], index=basket.columns),
        "const": float(ols.params[0]),
        "adf_pvalue": float(adf_p),
        "coint_pvalue": float(coint_p),
        "is_cointegrated": bool(adf_p < 0.05),
        "n_insample": n_in,
    }


def _rolling_resid_z(target: pd.Series, basket: pd.DataFrame,
                     beta_window: int, z_window: int) -> pd.Series:
    """Residual z-score from a TRAILING-window cointegrating regression.

    At each t (after the warmup) we solve least squares on the prior
    ``beta_window`` rows, predict the residual at t, then z-score the residual
    on a trailing ``z_window``. No future information enters beta or z.
    Regression is on log prices (the scale on which cointegration holds).
    """
    y = np.log(target.to_numpy())
    X = np.log(basket.to_numpy())
    n = len(y)
    resid = np.full(n, np.nan)
    Xc = np.column_stack([np.ones(n), X])  # intercept + basket
    for t in range(beta_window, n):
        sl = slice(t - beta_window, t)      # strictly trailing, excludes t
        b, *_ = np.linalg.lstsq(Xc[sl], y[sl], rcond=None)
        resid[t] = y[t] - Xc[t] @ b
    resid = pd.Series(resid, index=target.index)
    z = (resid - resid.rolling(z_window).mean()) / resid.rolling(z_window).std()
    return z


def cointegration_pnl(target: pd.Series, basket: pd.DataFrame, w_hedge: pd.Series,
                      beta_window: int = 250, z_window: int = 45,
                      entry: float = 1.5, exit: float = 0.4, cost_bps: float = 2.0,
                      lag: int = 1) -> pd.Series:
    """Net-of-cost daily return from trading the mean-reverting residual.

    We hold the long/short residual position via a dollar-neutral book: +1
    unit of target vs -w_hedge units of the basket (gross normalized to 1).
    ``w_hedge`` is the in-sample EG vector used to scale leg sizes; the
    *signal* itself comes from the trailing-window residual z-score, so timing
    is lookahead-safe even though leg sizing uses the confirmed vector.
    """
    z = _rolling_resid_z(target, basket, beta_window, z_window)

    # state machine -> +1 long residual (target cheap), -1 short, 0 flat
    from pairs_trading import _positions_from_z  # reuse the same logic
    sig = _positions_from_z(z, entry, exit)

    # leg weights: target gets +1 share-equivalent, basket gets -w_hedge,
    # all divided by gross to normalize exposure to 1.
    gross_units = 1.0 + float(w_hedge.abs().sum())
    prices = pd.concat([target.rename("TARGET"), basket], axis=1)
    ret = prices.pct_change(fill_method=None)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["TARGET"] = sig / gross_units
    for c in basket.columns:
        w[c] = -sig * w_hedge[c] / gross_units

    held = w.shift(lag).fillna(0.0)
    g = (held * ret).sum(axis=1)
    trades = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = trades * (cost_bps / 1e4)
    return (g - cost).rename("cointegration")
