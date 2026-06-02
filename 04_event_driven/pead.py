"""Post-Earnings-Announcement Drift (PEAD).

We start from the shared synthetic equity panel (`core.generate_market`) for a
realistic factor-driven base, then OVERLAY an earnings calendar on top of the
generated returns:

  * Every stock reports ~quarterly (every ~63 trading days), with a small random
    phase so announcements are spread across the calendar.
  * Each announcement draws a standardized earnings surprise  SUE ~ N(0, 1).
  * After the announcement we ADD a drift to that stock's daily returns in the
    direction of the surprise. The drift starts at a peak and decays linearly to
    zero over ~`drift_days` (~45) trading days -- the classic underreaction.

Strategy (lookahead-safe)
-------------------------
The day AFTER each announcement we take a position sized & signed by the
(public, already-released) surprise, hold it for ~`hold_days` trading days, then
exit. We aggregate every open event-position into one daily DOLLAR-NEUTRAL book
(per date: demean the signed exposures, then scale to unit gross). The backtest
engine adds the +1 day execution lag, so placing a target weight on the
announcement-date row means we are actually invested starting the next session.

Embedded edge: drift in the surprise direction.
Real-world gotcha: PEAD has been heavily arbitraged -- the drift decays faster
and is largely gone within a few days for liquid names, so realized capture is
far smaller than the textbook effect and sensitive to costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_earnings_panel(md, seed: int = 11, period: int = 63,
                         drift_days: int = 45, drift_peak: float = 0.00038):
    """Overlay an earnings calendar + drift on the base market returns.

    Returns
    -------
    returns : DataFrame (dates x tickers)
        Base returns with the embedded post-earnings drift added.
    surprise : DataFrame (dates x tickers)
        SUE published on each announcement date, NaN elsewhere (point-in-time).
    """
    rng = np.random.default_rng(seed)
    base = md.returns.fillna(0.0)
    dates = base.index
    tickers = base.columns
    n_days, n = len(dates), len(tickers)

    surprise = pd.DataFrame(np.nan, index=dates, columns=tickers)
    drift = np.zeros((n_days, n))

    # linearly decaying drift kernel over `drift_days`, normalized so the *sum*
    # of the kernel is 1 -> drift_peak controls cumulative abnormal drift.
    k = np.linspace(1.0, 0.0, drift_days, endpoint=False)
    k = k / k.sum()

    for j in range(n):
        phase = rng.integers(0, period)               # spread reporting dates
        ann_days = np.arange(phase, n_days, period)
        for d in ann_days:
            sue = rng.standard_normal()               # standardized surprise
            surprise.iat[d, j] = sue
            end = min(d + drift_days, n_days)
            seg = end - d
            # cumulative abnormal return ~ drift_peak * sue, spread by kernel
            drift[d:end, j] += drift_peak * sue * k[:seg] * drift_days

    returns = base + pd.DataFrame(drift, index=dates, columns=tickers)
    return returns, surprise


def pead_weights(surprise: pd.DataFrame, hold_days: int = 40,
                 gross_leverage: float = 1.0) -> pd.DataFrame:
    """Aggregate every open earnings event into a daily dollar-neutral book.

    On each announcement we open a position = sign/size of the surprise and keep
    it live for `hold_days` trading days. The signed, time-overlapping exposures
    are summed per date, then made dollar-neutral (demeaned across active names)
    and scaled to `gross_leverage` total gross.
    """
    s = surprise.values
    n_days, n = s.shape
    raw = np.zeros((n_days, n))

    ann_rows, ann_cols = np.where(~np.isnan(s))
    for d, j in zip(ann_rows, ann_cols):
        end = min(d + hold_days, n_days)
        raw[d:end, j] += s[d, j]                       # overlapping events add up

    raw = pd.DataFrame(raw, index=surprise.index, columns=surprise.columns)
    active = raw != 0.0
    n_active = active.sum(axis=1)

    # dollar-neutral: subtract the per-date mean over *active* names only
    grp_mean = raw.where(active).mean(axis=1)
    centered = raw.sub(grp_mean, axis=0).where(active, 0.0)

    gross = centered.abs().sum(axis=1)
    w = centered.div(gross.replace(0, np.nan), axis=0).fillna(0.0) * gross_leverage
    # need at least 2 names to be dollar-neutral
    return w.where(n_active >= 2, 0.0)
