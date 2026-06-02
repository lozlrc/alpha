"""Simulate a population of (homogeneous) AI trading agents and the price
footprints their crowded flow leaves behind.

THESIS
------
As more capital is allocated by trading agents built on a *few shared
foundation models* -- prompted alike, fed the same news APIs and transcripts --
their decisions become correlated. Correlated decisions create synchronized,
same-direction order flow, which pushes price TEMPORARILY away from fundamentals
and then reverts once the crowd is done. That temporary, crowd-driven deviation
is tradeable, and -- the key claim -- the edge GROWS with agent homogeneity.

This module is the DATA-GENERATING PROCESS for that thesis. It is fully
SYNTHETIC and OFFLINE: there are no real agents and no live market. A
`monoculture` knob ``rho in [0, 1]`` controls how correlated the agents are:

    rho = 0  -> independent agents; their idiosyncratic over-reactions cancel
               in aggregate (law of large numbers) => little net crowd flow.
    rho = 1  -> one mind; every agent leans the same way => maximal crowding.

Each news event has a TRUE fundamental component ``g`` (a permanent, correct
price move) and a SALIENCE ``m_s`` (how 'loud'/headline-grabbing the story is,
only loosely related to its true importance). The agent crowd over-reacts to
salience and piles in together; that pile-in is a temporary OVERSHOOT that
later decays back to fundamentals.

SYSTEMIC CROWDING (the dark side of the same knob)
--------------------------------------------------
Homogeneity does not only crowd each *stock's* news; it crowds names TOGETHER.
With ``systemic`` / ``fragility`` turned on, a SHARED crowd factor accumulates
(correlated, same-direction positioning) and occasionally UNWINDS in a cascade --
a violent common move that gaps the tape and then partly snaps back. Cross-name
correlation and tail risk then both rise with ``rho``: the very homogeneity that
makes the fade profitable also makes the tape fragile. (See ``strategies``:
``systemic_crowding_nowcast`` gauges it; ``cascade_reversal`` harvests the snap-back.)

MODEL HETEROGENEITY (monoculture is really about model-share concentration)
---------------------------------------------------------------------------
Real agents run on a handful of distinct foundation models. Agents on the SAME model
are tightly correlated; different models read ambiguous news differently. So the crowd
is a share-weighted blend of ``n_models`` camps, and the structural risk is how
CONCENTRATED the ecosystem is across them (the HHI set by ``model_concentration``).
When many balanced models disagree their idiosyncratic biases cancel; when the market
consolidates onto one model they stop cancelling and that model's bias dominates the
tape -- so crowding rises with model concentration even if no individual agent changes.
``model_agreement`` exposes (observably) how aligned the camps are on each event, and
``strategies.consensus_fade`` fades hardest when they concur. n_models=1 (the default)
collapses back to the single-crowd baseline.

OBSERVABLE vs LATENT (this separation is the honesty contract)
--------------------------------------------------------------
A trader -- and every strategy in ``strategies.py`` -- may use ONLY:
  * ``returns`` / ``prices``
  * ``news_sent``      public sentiment on event days (already salience-inflated)
  * ``news_salience``  how loud the story is (observable: outlet count, shares...)
  * ``volume`` / ``volume_z``  spikes when the crowd trades (trailing z-score)
  * ``model_agreement``  cross-model agreement on each event (you can see each public
                         model's take and whether the camps concur)

The following are LATENT truth, exposed only for plotting/diagnostics. Strategies
must NOT read them:
  * ``truth["perm"]``  the true fundamental price component
  * ``truth["temp"]``  the agent overshoot component and its reversion path
  * ``truth["crowd_ret"]``  the latent SHARED crowd-factor return (systemic crowding)
  * ``truth["unwind"]``     days a de-risking cascade fired
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AgenticMarket:
    returns: pd.DataFrame          # observed simple returns (dates x assets)
    prices: pd.DataFrame           # observed prices
    news_sent: pd.DataFrame        # public sentiment, nonzero only on event days
    news_salience: pd.DataFrame    # >=0 'loudness', nonzero only on event days
    volume: pd.DataFrame           # observed volume
    volume_z: pd.DataFrame         # trailing-window volume z-score (lookahead-safe)
    model_agreement: pd.DataFrame  # cross-model directional agreement per event (observable)
    rho: float                     # the monoculture input
    rho_eff: float                 # effective crowding actually applied
    truth: dict = field(default_factory=dict)   # latent components (diagnostics only)
    meta: dict = field(default_factory=dict)


def simulate_agentic_market(
    n_assets: int = 60,
    n_days: int = 1750,
    seed: int = 11,
    *,
    monoculture: float = 0.6,
    rho_path: np.ndarray | None = None,  # per-day crowding path; overrides `monoculture`
    event_rate: float = 1.0 / 15.0,   # per asset per day (~ every 15 trading days)
    sigma_base: float = 0.010,        # idiosyncratic daily vol
    market_vol: float = 0.006,        # common-factor daily vol
    market_mu: float = 0.0002,        # common-factor daily drift (~5%/yr)
    sigma_g: float = 0.030,           # scale of the TRUE fundamental jump
    salience_infl: float = 0.9,       # how much salience inflates PUBLIC sentiment
    agent_overreact: float = 1.0,     # agents over-react to salience (predictable part)
    idio_overshoot: float = 0.030,    # sd of correlated idiosyncratic overshoot (UNpredictable)
    crowd_floor: float = 0.0,         # residual crowding at rho=0 (shared LLM bias)
    wrong_headline_prob: float = 0.15,  # chance the public headline points the wrong way
    revert_mean: float = 0.6,         # avg fraction of an overshoot that reverts
    revert_sd: float = 0.30,          # spread: sometimes the crowd was actually right
    crowd_speed: float = 0.0,         # crowd pile-in speed: 0 = instant (machines); >0 = gradual
    tau_out: float = 6.0,             # reversion time constant (days)
    systemic: float = 0.0,            # strength of the SHARED crowd factor (cross-name crowding)
    fragility: float = 0.0,           # intensity of de-risking cascades (fat tails, off by default)
    cascade_revert: float = 0.6,      # fraction of a cascade that snaps back (fade-able)
    n_models: int = 1,                # number of distinct foundation-model "camps" (1 = monolith)
    model_concentration: float = 0.0,  # 0 = equal model shares; 1 = one model dominates (HHI->1)
    model_bias: float = 0.8,          # how far each camp's published view scatters from the headline
    horizon: int = 40,                # days an event's footprint is tracked
) -> AgenticMarket:
    """Generate one synthetic agentic market. ``monoculture`` (rho) sets the
    crowding; pass ``rho_path`` to make it time-varying (regime-switching), and
    ``crowd_speed`` to set how fast the crowd piles in. Every random draw is
    independent of rho, so sweeping rho/crowd_speed at a fixed ``seed`` isolates
    the pure effect of each knob."""
    rng = np.random.default_rng(seed)
    T, N = n_days, n_assets

    # Effective crowding: a small floor lets even fully-heterogeneous agents share
    # *some* salience-chasing bias; the rest scales linearly with rho. `monoculture`
    # is the constant-crowding input; pass `rho_path` instead to make crowding
    # TIME-VARYING (regime-switching) -- everything else in the model is identical.
    if rho_path is not None:
        rho_arr = np.clip(np.asarray(rho_path, dtype=float), 0.0, 1.0)
        if rho_arr.shape != (T,):
            raise ValueError(f"rho_path must have shape ({T},), got {rho_arr.shape}")
        rho = float(rho_arr.mean())                  # reporting scalar only
    else:
        rho = float(np.clip(monoculture, 0.0, 1.0))
        rho_arr = np.full(T, rho)
    rho_eff_arr = crowd_floor + (1.0 - crowd_floor) * rho_arr
    rho_eff = float(rho_eff_arr.mean())

    # ---- model ecosystem: K foundation-model "camps" with capital shares ----
    # `monoculture` (rho) is per-AGENT homogeneity; this is a SEPARATE axis -- how
    # CONCENTRATED the ecosystem is across distinct foundation models. The crowd's
    # idiosyncratic over-reaction is a share-weighted sum of per-camp biases, so many
    # balanced models CANCEL, while consolidation onto one model (HHI -> 1) lets a
    # single model's bias dominate the tape. n_models=1 reproduces the baseline.
    w_mod = model_shares(n_models, model_concentration)
    hhi = float((w_mod ** 2).sum())

    # ---- event footprint shape (precomputed once) --------------------------
    # `crowd_speed` is THE knob behind "you can't outrun the machines":
    #   crowd_speed == 0 -> instantaneous machine crowd. The reverting overshoot
    #                       peaks at t0 and only decays, so a trader reacting at the
    #                       close of t0 (earning t+1) can only FADE the reversion --
    #                       never ride the pile-in. (default; the headline regime.)
    #   crowd_speed  > 0 -> a slower (human-paced) crowd that piles in over days, so
    #                       the overshoot RAMPS UP after t0 before reverting -- now
    #                       front-running the pile-in can pay.
    # This shape applies to the *reverting* part of the overshoot; the part that
    # sticks is booked at t0 (it behaves like information). See the event loop.
    tau = np.arange(horizon + 1)
    decay = np.exp(-tau / tau_out)
    if crowd_speed <= 0:
        profile = decay                                  # peak at t0, then decay
    else:
        profile = (1.0 - np.exp(-tau / crowd_speed)) * decay
        profile = profile / profile.max()                # normalize peak to 1

    # ---- draw everything up front (sizes independent of rho) ---------------
    mask = rng.random((T, N)) < event_rate
    g_all = rng.normal(0.0, sigma_g, (T, N))            # true fundamental jump
    ms_all = np.abs(rng.normal(0.0, sigma_g, (T, N)))   # salience (loudness >= 0)
    xi = rng.normal(0.0, 1.0, (T, N, n_models))         # per-camp (per-model) idiosyncratic biases
    z_all = (xi * w_mod).sum(axis=2)                     # share-weighted cross-camp residual (cancels when diverse)
    u_all = rng.random((T, N))                          # headline-direction correctness
    revert_all = np.clip(rng.normal(revert_mean, revert_sd, (T, N)), 0.0, 1.0)
    sent_noise = rng.normal(0.0, 0.15 * sigma_g, (T, N))
    eps = rng.normal(0.0, sigma_base, (T, N))           # idiosyncratic returns
    market = rng.normal(market_mu, market_vol, T)       # common factor
    base_vol = np.exp(rng.normal(0.0, 0.30, (T, N)))    # lognormal baseline volume

    # ---- accumulate event footprints ---------------------------------------
    perm = np.zeros((T, N))      # permanent (fundamental) price component
    temp = np.zeros((T, N))      # temporary (agent overshoot) component
    vol_bump = np.zeros((T, N))  # excess volume around events
    sent = np.zeros((T, N))      # public sentiment (observable, event days only)
    salience = np.zeros((T, N))  # observable loudness
    magree = np.zeros((T, N))    # observable cross-model agreement on each event

    for t0, i in np.argwhere(mask):
        g = g_all[t0, i]
        m_s = ms_all[t0, i]
        d_true = 1.0 if g >= 0 else -1.0                # the fundamental's true sign
        # The public headline usually agrees with the fundamental, but sometimes
        # misleads. The crowd chases whatever the HEADLINE says (it trades the
        # narrative, not the truth) -- so the overshoot follows d_news.
        d_news = d_true if u_all[t0, i] > wrong_headline_prob else -d_true

        # cross-model agreement (OBSERVABLE): each camp publishes a view that chases the
        # headline plus its own model bias; agreement is the share-weighted alignment of
        # those views -- 1 when the camps concur, ->0 when they split. (With one model
        # this is always 1: the single camp agrees with itself, so the baseline is
        # unchanged.) The net crowd flow SCALES with it -- divided models cancel.
        leans = d_news + model_bias * xi[t0, i]               # each camp's published lean
        agree = abs(float(np.dot(w_mod, np.sign(leans))))     # |share-weighted sign| in [0,1]
        magree[t0, i] = agree

        # The crowd's OVERSHOOT: over-reaction to the loud part (shared bias) plus a
        # correlated idiosyncratic lean, SCALED by how much the model camps agree. So an
        # independent population (rho->0) -- or a divided ecosystem (agree->0) -- leaves
        # almost no net footprint; a crowded, consensus event leaves a big, fade-able one.
        overshoot = rho_eff_arr[t0] * agree * (d_news * agent_overreact * salience_infl * m_s
                                               + idio_overshoot * z_all[t0, i])

        # Not every overshoot reverts: a random fraction is genuine reversion (the
        # crowd was wrong), the rest sticks as permanent (the crowd was right). The
        # whole overshoot still pops at t0; only the reverting part decays back.
        revert = revert_all[t0, i]
        end = min(T, t0 + horizon + 1)
        span = end - t0
        perm[t0:, i] += g + (1.0 - revert) * overshoot          # fundamental + stuck overshoot
        temp[t0:end, i] += (revert * overshoot) * profile[:span]  # reverting part decays to zero

        vend = min(T, t0 + 7)
        vol_bump[t0:vend, i] += (abs(g) + abs(overshoot)) * np.exp(-np.arange(vend - t0) / 2.0)

        # public sentiment follows the headline and overstates it via salience
        sent[t0, i] = d_news * (abs(g) + salience_infl * m_s) + sent_noise[t0, i]
        salience[t0, i] = m_s

    # ---- systemic crowding: a SHARED crowd factor + fragility cascades ------
    # The per-event overshoots above are INDEPENDENT across names. A monoculture
    # also crowds names TOGETHER: agents pile into the same macro trade, so a
    # shared positioning factor builds up (correlated, same-direction flow) and
    # occasionally UNWINDS violently when the crowd de-risks at once -- a cascade
    # that gaps the tape and then partly snaps back. This is the systemic face of
    # homogeneity: the same rho that makes the fade pay also makes the tape fragile.
    # Off by default (systemic = fragility = 0), so the baseline is unchanged.
    crowd_ret = np.zeros(T)            # daily common-factor return from crowd flow
    unwind = np.zeros(T, dtype=bool)   # days a de-risking cascade fired
    beta_crowd = np.zeros(N)
    if systemic > 0.0 or fragility > 0.0:
        beta_crowd = np.abs(rng.normal(1.0, 0.30, N))     # every name is in the crowd trade
        shared = rng.normal(0.0, 1.0, T)                  # shared crowd innovations
        u_sys = rng.random(T)                             # unwind triggers
        sys_vol = 0.010                                   # daily common crowd vol scale
        phi_sys = 0.97                                    # crowd-position persistence
        ext_scale = 0.05                                  # extension where unwinds get likely
        base_haz = 0.040                                  # peak daily unwind hazard
        collapse = 0.85                                   # how much position snaps out on unwind
        cdecay = np.exp(-np.arange(1, 6) / 2.0)
        cdecay = cdecay / cdecay.sum()                    # snap-back concentrated over ~5 days, sums to 1
        cascade_temp = np.zeros(T)
        P = 0.0
        for t in range(T):
            re = rho_eff_arr[t]
            ext = abs(P) / ext_scale
            haz = fragility * re * base_haz * min(1.0, ext * ext)   # hazard rises with extension^2
            if u_sys[t] < haz and ext > 0.35:
                unwind[t] = True
                shock = -np.sign(P) * abs(P) * 0.95       # violent move against the position
                crowd_ret[t] += shock                     # the de-risk gaps the tape at t
                back = -cascade_revert * shock            # the part that snaps back (overshoot)
                kk = min(T, t + 1 + len(cdecay))
                cascade_temp[t + 1:kk] += back * cdecay[:kk - (t + 1)]
                P *= (1.0 - collapse)
            build = systemic * re * sys_vol * shared[t]   # gradual correlated pile-in
            P = phi_sys * P + build
            crowd_ret[t] += build
        crowd_ret += cascade_temp

    # ---- assemble observed returns -----------------------------------------
    level = perm + temp
    event_ret = np.vstack([level[:1], np.diff(level, axis=0)])   # level_t - level_{t-1}
    r = market[:, None] + eps + event_ret + beta_crowd[None, :] * crowd_ret[:, None]

    dates = pd.bdate_range("2015-01-02", periods=T)
    cols = [f"A{j:02d}" for j in range(N)]
    R = pd.DataFrame(r, index=dates, columns=cols)
    prices = 100.0 * (1.0 + R).cumprod()

    volume = pd.DataFrame(base_vol * (1.0 + vol_bump), index=dates, columns=cols)
    roll = volume.rolling(63, min_periods=10)
    volume_z = ((volume - roll.mean()) / roll.std()).fillna(0.0)   # trailing -> no lookahead

    return AgenticMarket(
        returns=R,
        prices=prices,
        news_sent=pd.DataFrame(sent, index=dates, columns=cols),
        news_salience=pd.DataFrame(salience, index=dates, columns=cols),
        volume=volume,
        volume_z=volume_z,
        model_agreement=pd.DataFrame(magree, index=dates, columns=cols),
        rho=rho,
        rho_eff=rho_eff,
        truth={"perm": pd.DataFrame(perm, index=dates, columns=cols),
               "temp": pd.DataFrame(temp, index=dates, columns=cols),
               "crowd_ret": pd.Series(crowd_ret, index=dates),     # latent common crowd return
               "unwind": pd.Series(unwind, index=dates)},          # latent cascade days
        meta={"n_events": int(mask.sum()), "seed": seed,
              "crowd_speed": crowd_speed, "tau_out": tau_out,
              "systemic": systemic, "fragility": fragility,
              "n_unwinds": int(unwind.sum()),
              "n_models": n_models, "model_concentration": model_concentration, "hhi": hhi,
              "time_varying_rho": rho_path is not None},
    )


def regime_rho_path(n_days: int, seed: int = 0, *, calm: float = 0.10,
                    crowded: float = 0.90, mean_dwell: int = 150) -> np.ndarray:
    """A piecewise-constant, regime-switching crowding path ``rho_t in [0, 1]``.

    Real agent crowding is not constant: it WAXES (a wave of same-model, AI-driven
    flow piling into one loud macro narrative) and WANES (quiet tape, idiosyncratic
    news the models disagree on). This builds a 2-state path that alternates between
    a CALM regime (low rho) and a CROWDED regime (high rho), with geometric dwell
    times averaging ``mean_dwell`` days. Feed it to
    ``simulate_agentic_market(rho_path=...)`` to get a market whose fade-ability
    switches on and off over time -- the setting a deployable, crowding-TIMED
    strategy is built for. The path itself is latent; a strategy must NOWCAST it
    from observable price/volume (see ``strategies.crowding_nowcast``)."""
    rng = np.random.default_rng(seed)
    levels = (float(calm), float(crowded))
    path = np.empty(int(n_days), dtype=float)
    state = int(rng.random() < 0.5)
    t = 0
    while t < n_days:
        dwell = max(5, int(rng.geometric(1.0 / max(mean_dwell, 1))))
        path[t:t + dwell] = levels[state]
        t += dwell
        state ^= 1
    return path


def model_shares(n_models: int, concentration: float = 0.0) -> np.ndarray:
    """Capital shares across ``n_models`` foundation-model camps. ``concentration``
    in [0, 1] interpolates from an EQUAL split (0 -> every model 1/n, HHI = 1/n: a
    diverse ecosystem) to a single DOMINANT model (1 -> one camp ~1, HHI -> 1: a
    monoculture). Returns a length-``n_models`` array summing to 1; its HHI is the
    'model concentration' axis swept in run.py."""
    n = max(int(n_models), 1)
    if n == 1:
        return np.array([1.0])
    c = float(np.clip(concentration, 0.0, 1.0))
    top = 1.0 / n + c * (1.0 - 1.0 / n)              # dominant share: 1/n (c=0) -> 1 (c=1)
    rest = (1.0 - top) / (n - 1)
    w = np.full(n, rest)
    w[0] = top
    return w
