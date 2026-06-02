"""Strategies that trade AGAINST a crowded population of AI agents.

Every strategy here consumes only the OBSERVABLE fields of an ``AgenticMarket``
(returns, public news sentiment, news salience, volume z-score) -- never the
latent truth -- and returns a (dates x assets) weight matrix that is:
  * dollar-neutral among the names it holds each day, and
  * scaled to gross leverage 1 (sum of |weights| = 1) on active days -- the lone
    exception is ``crowding_timed_fade``, which deliberately runs TIME-VARYING
    gross so it can size up and down with the crowd.

The backtest engine (core.backtest_weights) then applies a 1-day execution lag,
so a position decided from information up to day t earns the return on t+1.

Angles on the same mechanism (crowd over-reacts to salient news, then mean-reverts):
  * reaction_frontrun   -- RIDE the herd: enter with the news for the brief pile-in.
  * crowding_reversal   -- FADE the herd: short the over-extension, harvest reversion.
  * salience_fade       -- strip the headline inflation: fade in proportion to loudness.
  * crowding_nowcast    -- observable estimate of how crowded the tape is right now.
  * crowding_timed_fade -- crowding_reversal sized by the nowcast (deployable).
  * systemic_crowding_nowcast -- observable fragility gauge (PC1 variance share).
  * cascade_reversal    -- harvest the de-risking cascade (convex, positive skew).
  * consensus_fade      -- fade only the events the model camps agree on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agents import AgenticMarket


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hold(sig: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Carry each nonzero signal forward for `hold` days (hold the position)."""
    s = sig.replace(0.0, np.nan)
    if hold > 1:
        s = s.ffill(limit=hold - 1)
    return s.fillna(0.0)


def _normalize(raw: pd.DataFrame, neutral: bool = True) -> pd.DataFrame:
    """Per day: optionally demean across the held names (dollar-neutral), then
    scale to gross leverage 1. Days with no position stay flat."""
    vals = raw.fillna(0.0).to_numpy(dtype=float)
    out = np.zeros_like(vals)
    for t in range(vals.shape[0]):
        row = vals[t]
        nz = row != 0.0
        k = int(nz.sum())
        if k == 0:
            continue
        r = row.copy()
        if neutral and k > 1:
            r[nz] = r[nz] - r[nz].mean()      # dollar-neutral among active names
        gross = np.abs(r).sum()
        if gross > 0:
            out[t] = r / gross                # gross leverage = 1
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def reaction_frontrun(mkt: AgenticMarket, hold: int = 1) -> pd.DataFrame:
    """CHASE the herd -- the cautionary baseline. The intuitive trade is to read
    the same news the agents read and pile in behind them. But the agents are
    machines: they have already moved the price by the time you act, so 'riding
    the AI momentum' on the next bar just buys the top and sits through the
    reversion. Included to show the naive trade LOSES -- you cannot outrun the
    crowd you are trying to follow; the edge is in fading it, not chasing it."""
    raw = np.sign(mkt.news_sent)              # chase the headline direction, equal size
    raw = _hold(raw, hold)
    return _normalize(raw)


def crowding_reversal(mkt: AgenticMarket, look: int = 3, hold: int = 8,
                      vol_thresh: float = 0.5) -> pd.DataFrame:
    """FADE the herd (headline strategy). When a name has run hard over the last
    `look` days AND that move came with loud news and a volume spike, it is most
    likely a crowd over-extension -- so fade it and harvest the reversion over the
    next `hold` days. Uses only trailing information."""
    ret = mkt.returns
    runup = ret.rolling(look).sum()                          # recent cumulative move
    sal_recent = mkt.news_salience.rolling(look, min_periods=1).max()
    gate = (sal_recent > 0) & (mkt.volume_z > vol_thresh)    # crowded event nearby?

    raw = -np.sign(runup) * runup.abs() * sal_recent         # fade, size by move x loudness
    raw = raw.where(gate, 0.0)
    raw = _hold(raw, hold)
    return _normalize(raw)


def crowding_nowcast(mkt: AgenticMarket, hold: int = 8, est_window: int = 63) -> pd.Series:
    """Observable, lookahead-safe nowcast of *how crowded the tape is right now* --
    estimated purely from price + public news, never the latent rho.

    Logic: when the agent crowd is dense, salient-news moves over-extend and then
    REVERT, so fading them pays; when agents are independent, there is nothing to
    fade. So for each past news event we measure the realized reversion (did fading
    the move earn over the next `hold` days?) and take a trailing average over
    `est_window` days. The series is shifted by `hold` so only fully REALIZED
    windows enter: the value at day t uses only information available by t. A higher
    nowcast => overshoots are reverting strongly => the crowd is currently dense."""
    cum = mkt.returns.cumsum()
    fwd = cum.shift(-hold) - cum                          # sum of returns over (s, s+hold]
    event = mkt.news_salience.to_numpy() > 0
    rev = (-np.sign(mkt.news_sent) * fwd).where(event, np.nan)   # realized reversion per event
    rev_sum = rev.sum(axis=1, min_count=1)                # daily reversion P&L (NaN if no event)
    rev_cnt = rev.notna().sum(axis=1).astype(float)       # number of events that day
    num = rev_sum.fillna(0.0).shift(hold).rolling(est_window, min_periods=1).sum()
    den = rev_cnt.shift(hold).rolling(est_window, min_periods=1).sum()
    return num / den.replace(0.0, np.nan)                 # avg realized reversion per event


def crowding_timed_fade(mkt: AgenticMarket, look: int = 3, hold: int = 8,
                        vol_thresh: float = 0.5, est_window: int = 63) -> pd.DataFrame:
    """A DEPLOYABLE crowd-fade: the static `crowding_reversal` book, scaled up and
    down through time by the observable `crowding_nowcast`. When crowding is high
    (overshoots have been reverting) it leans in; when the tape is calm (nothing to
    fade) it shrinks toward flat and stops paying costs. This is the strategy for a
    world where agent crowding WAXES AND WANES -- it times the static fade using only
    price/volume, with no lookahead. Unlike the other strategies it deliberately runs
    TIME-VARYING gross exposure, so it is NOT renormalized to constant leverage."""
    base = crowding_reversal(mkt, look=look, hold=hold, vol_thresh=vol_thresh)
    nowcast = crowding_nowcast(mkt, hold=hold, est_window=est_window)
    scale = nowcast.abs().rolling(252, min_periods=20).mean()    # self-scale -> ~unit size
    mult = (nowcast / scale).clip(lower=0.0, upper=2.0).fillna(0.0)
    return base.mul(mult, axis=0)                         # time-varying gross; no renormalize


def salience_fade(mkt: AgenticMarket, hold: int = 8, entry_delay: int = 0) -> pd.DataFrame:
    """Strip the headline inflation. On a (loud) event, fade it in proportion to
    its salience -- the louder the story, the larger the agent over-reaction and
    the larger the expected reversion. This is the assumption-heavy cousin of
    crowd_reversal: instead of waiting to see the price over-extend, it bets the
    over-reaction will happen from the news loudness alone -- so it only captures
    the *predictable* part of the crowd's flow and is noisier as a result."""
    raw = -np.sign(mkt.news_sent) * mkt.news_salience        # fade, sized by loudness
    if entry_delay > 0:
        raw = raw.shift(entry_delay).fillna(0.0)             # delayed entry (lookahead-safe)
    raw = _hold(raw, hold)
    return _normalize(raw)


# --------------------------------------------------------------------------- #
# systemic crowding -- the dark side: fragility, cascades, convexity
# --------------------------------------------------------------------------- #
def systemic_crowding_nowcast(mkt: AgenticMarket, window: int = 63,
                              stride: int = 5) -> pd.Series:
    """Observable FRAGILITY gauge: the share of cross-sectional variance captured by
    the first principal component over a trailing window. When the agent crowd is
    dense, names move together and PC1's share is high (a fragile, cascade-prone
    regime); when agents trade independently it is low. Computed on a trailing window
    that ends *before* each day and stepped every `stride` days for speed -- so it is
    lookahead-safe. Rises with the monoculture knob rho."""
    arr = mkt.returns.to_numpy()
    T, N = arr.shape
    out = np.full(T, np.nan)
    for t in range(window, T, stride):
        win = arr[t - window:t]                              # rows t-window .. t-1 (strictly past)
        C = np.corrcoef(win, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        top = float(np.linalg.eigvalsh(C)[-1])               # largest eigenvalue
        out[t:t + stride] = top / N                          # fraction of total variance in PC1
    return pd.Series(out, index=mkt.returns.index).ffill()


def cascade_reversal(mkt: AgenticMarket, hold: int = 6, z_thresh: float = 2.5,
                     frag_window: int = 63, frag_q: float = 0.5) -> pd.DataFrame:
    """Harvest the de-risking CASCADE. When a dense monoculture unwinds, every name
    dumps at once: a large, sudden COMMON move that gaps the tape and then partly
    snaps back. Detect it observably -- an outsized one-day move in the equal-weight
    index (standardized by *trailing* vol) that lands in a high-crowding (fragile)
    regime -- and fade it: take the opposite common-factor position for `hold` days to
    capture the snap-back. This is the systemic analogue of `crowding_reversal`, and
    it is DIRECTIONAL by construction (a timed bet on the market factor after a crash,
    not a cross-sectional spread), so its payoff is rare and convex (right-skewed)."""
    R = mkt.returns
    idx = R.mean(axis=1)                                     # equal-weight common move
    vol = idx.rolling(63, min_periods=20).std().shift(1)     # trailing vol (excludes today)
    z = idx / vol.replace(0.0, np.nan)                       # standardized common move
    frag = systemic_crowding_nowcast(mkt, window=frag_window)
    frag_hi = frag > frag.rolling(252, min_periods=60).quantile(frag_q)
    trigger = (z.abs() > z_thresh) & frag_hi.fillna(False)   # sudden big move in a fragile regime
    sig = -np.sign(idx) * trigger.astype(float)              # fade the move (snap-back)
    raw = pd.DataFrame(np.repeat(sig.to_numpy()[:, None], R.shape[1], axis=1),
                       index=R.index, columns=R.columns)
    raw = _hold(raw, hold)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0)                # equal-weight directional bet


# --------------------------------------------------------------------------- #
# model heterogeneity -- trade the dispersion BETWEEN foundation-model camps
# --------------------------------------------------------------------------- #
def consensus_fade(mkt: AgenticMarket, hold: int = 8) -> pd.DataFrame:
    """Fade only the events the model camps AGREE on. When several foundation models
    concur on a story their flows reinforce into a big, fade-able over-reaction; when
    they split, the net flow cancels and there is little to fade. `model_agreement`
    (observable -- you can see each public model's take) measures that concurrence per
    event, so we fade in proportion to salience AND agreement: a cross-model-dispersion
    aware cousin of `salience_fade` that concentrates risk on the consensus over-reactions
    and skips the contested ones. With a single model it reduces to `salience_fade`
    (agreement == 1 on every event)."""
    raw = -np.sign(mkt.news_sent) * mkt.news_salience * mkt.model_agreement
    raw = _hold(raw, hold)
    return _normalize(raw)
