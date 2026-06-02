"""Agent-behavior alpha: trade the crowded flow of a homogeneous AI-agent population.

Run:  ../.venv/bin/python run.py

The premise (see agents.py for the full write-up): when a lot of capital is
allocated by trading agents built on a few shared foundation models, their
decisions correlate, their order flow synchronizes, and price over-shoots
fundamentals on loud news before reverting. We build strategies that trade that
behavior, then run four experiments:

  1. Base book        -- fade vs chase the crowd at a fixed homogeneity.
  2. Monoculture sweep-- the crowd-reversal edge emerges as agents homogenize.
  3. Crowd-speed sweep-- WHY fading (not chasing) wins: the crowd is fast. Slow
                         the pile-in down and chasing starts to win instead.
  4. Regime timing    -- crowding waxes and wanes; a deployable fade must NOWCAST
                         it from observable data (no access to the latent rho).

Everything is SYNTHETIC and OFFLINE -- no live market, no real agents. The results
demonstrate the LOGIC of the thesis (if agents crowd, their flow is fade-able and
the edge scales with crowding); they are not a claim about live P&L.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402

import agents  # noqa: E402
import strategies as strat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RHO_BASE, COST_BPS, NDAYS, SYS_RHO = 0.6, 1.0, 1750, 0.9
SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CSPEED = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
N_MODELS, MCONC = 4, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

STRATS = {
    "crowd_reversal": strat.crowding_reversal,
    "reaction_frontrun": strat.reaction_frontrun,
    "salience_fade": strat.salience_fade,
}


def _sharpe(returns, W, cost=COST_BPS):
    return core.sharpe(core.backtest_weights(returns, W, cost_bps=cost, lag=1).returns)


def run_book(mkt) -> list:
    out = []
    for name, fn in STRATS.items():
        W = fn(mkt)
        out.append(core.backtest_weights(mkt.returns, W, cost_bps=COST_BPS, lag=1, name=name))
    return out


# --------------------------------------------------------------------------- #
# experiment 2: monoculture sweep -- does the edge grow with homogeneity?
# --------------------------------------------------------------------------- #
def monoculture_sweep() -> list:
    print("\nMonoculture sweep -- crowd_reversal Sharpe vs agent homogeneity rho")
    print("(same seed at every rho, so only the crowding changes):")
    sweep = []
    for rho in SWEEP:
        m = agents.simulate_agentic_market(monoculture=rho, seed=11)
        sh = _sharpe(m.returns, strat.crowding_reversal(m))
        sweep.append(sh)
        print(f"  rho={rho:.1f}  ->  Sharpe {sh:5.2f}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(SWEEP, sweep, "o-", lw=2.2, color="#b2182b", zorder=3)
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.fill_between(SWEEP, 0, sweep, color="#b2182b", alpha=0.10)
    ax.set_xlabel("agent homogeneity  ρ   (0 = independent  →  1 = monoculture)")
    ax.set_ylabel("crowd-reversal Sharpe")
    ax.set_title("Agent-behavior alpha grows with monoculture")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "monoculture.png"), dpi=120)
    plt.close(fig)
    return sweep


# --------------------------------------------------------------------------- #
# experiment 3: crowd-speed sweep -- why fading beats chasing
# --------------------------------------------------------------------------- #
def crowd_speed_sweep():
    print("\nCrowd-speed sweep -- chase (ride 1wk) vs fade, as the crowd slows down")
    print("(crowd_speed=0 is the instantaneous machine crowd; larger = human-paced):")
    chase, fade = [], []
    for cs in CSPEED:
        m = agents.simulate_agentic_market(monoculture=RHO_BASE, crowd_speed=cs, seed=11)
        chase.append(_sharpe(m.returns, strat.reaction_frontrun(m, hold=5)))
        fade.append(_sharpe(m.returns, strat.crowding_reversal(m)))
        print(f"  crowd_speed={cs:4.2f}  chase {chase[-1]:6.2f}   fade {fade[-1]:6.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(CSPEED, fade, "o-", lw=2.2, color="#2166ac", label="fade the crowd (crowd-reversal)")
    ax.plot(CSPEED, chase, "s-", lw=2.2, color="#b2182b", label="chase the crowd (ride 1-wk)")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.axvline(0, color="#444", lw=6, alpha=0.12)
    ax.annotate("machine speed\n(today)", xy=(0.12, -3.0), fontsize=8, color="#444")
    ax.set_xlabel("crowd pile-in time  (days)   —   0 = instantaneous machine crowd")
    ax.set_ylabel("Sharpe")
    ax.set_title("Fade or chase? It flips with the crowd's SPEED")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "crowd_speed.png"), dpi=120)
    plt.close(fig)
    return chase, fade


# --------------------------------------------------------------------------- #
# experiment 4: regime timing -- nowcast the crowd, deploy capital efficiently
# --------------------------------------------------------------------------- #
def regime_timing():
    print("\nRegime timing -- crowding waxes/wanes; NOWCAST it from observable data")
    rp = agents.regime_rho_path(NDAYS, seed=3, calm=0.0, crowded=0.9, mean_dwell=200)
    m = agents.simulate_agentic_market(rho_path=rp, n_days=NDAYS, seed=11)

    W_static = strat.crowding_reversal(m)
    W_timed = strat.crowding_timed_fade(m, est_window=63)
    nowcast = strat.crowding_nowcast(m, est_window=63)
    truth = pd.Series(rp, index=m.returns.index)
    corr = float(nowcast.corr(truth))

    # cost sensitivity + capital efficiency
    costs = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    s_sh = [_sharpe(m.returns, W_static, c) for c in costs]
    t_sh = [_sharpe(m.returns, W_timed, c) for c in costs]
    gross_s, gross_t = float(W_static.abs().sum(axis=1).mean()), float(W_timed.abs().sum(axis=1).mean())
    turn_s = float(core.backtest_weights(m.returns, W_static, cost_bps=COST_BPS, lag=1).turnover.mean())
    turn_t = float(core.backtest_weights(m.returns, W_timed, cost_bps=COST_BPS, lag=1).turnover.mean())

    print(f"  nowcast vs latent rho: correlation {corr:.2f}  (crowding is OBSERVABLE)")
    print(f"  net Sharpe  @ 1bp : static {s_sh[0]:.2f}   timed {t_sh[0]:.2f}")
    print(f"  net Sharpe  @10bp : static {s_sh[-1]:.2f}   timed {t_sh[-1]:.2f}  "
          f"(timing wins once trading isn't free)")
    print(f"  avg gross deployed: static {gross_s:.2f}   timed {gross_t:.2f}  "
          f"({100 * (1 - gross_t / gross_s):.0f}% less capital)")
    print(f"  Sharpe / gross    : static {s_sh[0] / gross_s:.2f}   timed {t_sh[0] / gross_t:.2f}  "
          f"(capital efficiency)   turnover {turn_s:.3f} -> {turn_t:.3f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.4))

    # top: the latent regime vs the observable nowcast
    ax1.step(truth.index, truth.values, where="post", color="#444", lw=1.6,
             label="latent crowding ρ (unobservable)")
    ax1.set_ylabel("latent ρ", color="#444")
    ax1.set_ylim(-0.08, 1.05)
    axb = ax1.twinx()
    axb.plot(nowcast.index, nowcast.values, color="#b2182b", lw=1.1, alpha=0.85,
             label="observable nowcast")
    lo, hi = nowcast.quantile(0.02), nowcast.quantile(0.98)   # crop warmup transient
    axb.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    axb.set_ylabel("crowding nowcast", color="#b2182b")
    ax1.set_title(f"The crowd is observable: nowcast tracks the hidden regime  (corr {corr:.2f})")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = axb.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # bottom: net Sharpe vs cost -- timing pays once trading costs money
    ax2.plot(costs, s_sh, "o-", lw=2.2, color="#2166ac", label="static fade (always on)")
    ax2.plot(costs, t_sh, "s-", lw=2.2, color="#b2182b", label="timed fade (nowcast-scaled)")
    ax2.set_xlabel("transaction cost (bps per unit traded)")
    ax2.set_ylabel("net Sharpe")
    ax2.set_title("Timing deploys less capital and wins net as costs rise")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "regime_timing.png"), dpi=120)
    plt.close(fig)
    return corr, s_sh, t_sh


# --------------------------------------------------------------------------- #
# experiment 5: systemic crowding -- the dark side (fragility & convex harvest)
# --------------------------------------------------------------------------- #
def _skew(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std()
    return float(((x - x.mean()) ** 3).mean() / sd ** 3) if len(x) > 2 and sd > 0 else float("nan")


def fragility_experiment():
    print("\nSystemic crowding -- the dark side: correlation & fragility vs rho")
    print("(shared crowd factor + cascades ON; same seed, so only rho changes):")
    pc1s, tails = [], []
    for rho in SWEEP:
        m = agents.simulate_agentic_market(monoculture=rho, systemic=1.0, fragility=1.0, seed=11)
        frag = strat.systemic_crowding_nowcast(m)
        idx = m.returns.mean(axis=1)
        x = idx.dropna().to_numpy()
        kurt = float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3.0)
        csh = _sharpe(m.returns, strat.cascade_reversal(m))
        pc1s.append(float(frag.mean()))
        tails.append(float(idx.quantile(0.01)))
        print(f"  rho={rho:.1f}  PC1-share {pc1s[-1]:.3f}  worst-1%-day {tails[-1]:+.4f}  "
              f"kurtosis {kurt:5.1f}  cascades {m.meta['n_unwinds']:2d}  cascade_rev Sharpe {csh:+.2f}")

    # high-rho market for the convex-payoff panel
    m = agents.simulate_agentic_market(monoculture=SYS_RHO, systemic=1.0, fragility=1.0, seed=11)
    res = core.backtest_weights(m.returns, strat.cascade_reversal(m), cost_bps=COST_BPS, lag=1,
                                name="cascade_reversal")
    eq = (1.0 + res.returns).cumprod()
    unwind = m.truth["unwind"]
    s = res.summary()
    sk = _skew(res.returns[res.returns != 0])
    print(f"\n  cascade_reversal @ rho={SYS_RHO}: Sharpe {s['sharpe']:.2f}, hit-rate "
          f"{s['hit_rate'] * 100:.0f}%, max-DD {s['max_drawdown'] * 100:.0f}%, skew {sk:+.2f}, "
          f"{int(unwind.sum())} cascades")
    print("  -> the convex MIRROR of merger-arb (high hit-rate, NEGATIVE skew): cascade_reversal\n"
          "     is LOW hit-rate, POSITIVE skew -- small bleeds, paid back in the crashes.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 7.6))

    # top: the SAME rho that pays the alpha also raises correlation + tail risk
    ax1.plot(SWEEP, pc1s, "o-", color="#2166ac", lw=2.2, label="cross-name correlation (PC1 share)")
    ax1.set_xlabel("agent homogeneity  ρ")
    ax1.set_ylabel("PC1 variance share", color="#2166ac")
    axb = ax1.twinx()
    axb.plot(SWEEP, [-t for t in tails], "s-", color="#b2182b", lw=2.2,
             label="worst-1% common day (|loss|)")
    axb.set_ylabel("|worst-1% daily common move|", color="#b2182b")
    ax1.set_title("Same knob: crowding ALPHA rises with ρ — and so does systemic FRAGILITY")
    h1, la1 = ax1.get_legend_handles_labels()
    h2, la2 = axb.get_legend_handles_labels()
    ax1.legend(h1 + h2, la1 + la2, loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # bottom: cascade_reversal's convex equity -- flat, then paid in the crashes
    ax2.plot(eq.index, eq.values, color="#1a9850", lw=1.6, zorder=3, label="cascade_reversal equity")
    for d in unwind.index[unwind.to_numpy()]:
        ax2.axvline(d, color="#b2182b", lw=0.8, alpha=0.22)
    ax2.set_ylabel("growth of $1")
    ax2.set_title(f"Convex harvest @ ρ={SYS_RHO}: paid in the crashes "
                  f"(red = cascade days; skew {sk:+.2f})")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fragility.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# experiment 6: model heterogeneity -- monoculture is model-share concentration
# --------------------------------------------------------------------------- #
def model_concentration_experiment():
    print("\nModel-concentration sweep -- the ecosystem consolidating onto fewer models")
    print(f"(rho fixed at {RHO_BASE}, {N_MODELS} model camps; only the share concentration changes):")
    hhis, cr, sf, cf = [], [], [], []
    for c in MCONC:
        m = agents.simulate_agentic_market(monoculture=RHO_BASE, n_models=N_MODELS,
                                           model_concentration=c, seed=11)
        hhis.append(m.meta["hhi"])
        cr.append(_sharpe(m.returns, strat.crowding_reversal(m)))
        sf.append(_sharpe(m.returns, strat.salience_fade(m)))
        cf.append(_sharpe(m.returns, strat.consensus_fade(m)))
        print(f"  conc={c:.1f}  HHI={hhis[-1]:.2f} (~{1.0 / hhis[-1]:.1f} eff. models)  "
              f"crowd_rev {cr[-1]:.2f}  salience_fade {sf[-1]:.2f}  consensus_fade {cf[-1]:.2f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.4))

    # top: crowding rises as the ecosystem consolidates -- with rho held fixed
    ax1.plot(hhis, cr, "o-", color="#b2182b", lw=2.3, zorder=3)
    ax1.fill_between(hhis, min(cr) - 0.2, cr, color="#b2182b", alpha=0.08)
    ax1.set_xlabel("model-share concentration  (HHI)   —   diverse → monoculture")
    ax1.set_ylabel("crowd-reversal Sharpe")
    ax1.set_title(f"Same agents, fewer models → more crowding (ρ fixed at {RHO_BASE})")
    ax1.annotate(f"{N_MODELS} balanced\nmodels", xy=(hhis[0], cr[0]),
                 xytext=(hhis[0] + 0.02, cr[0] + 0.18), fontsize=8, color="#444")
    ax1.annotate("1 dominant\nmodel", xy=(hhis[-1], cr[-1]),
                 xytext=(hhis[-1] - 0.13, cr[-1] - 0.42), fontsize=8, color="#444")
    ax1.grid(True, alpha=0.3)

    # bottom: the cross-model dispersion trade -- fade only what the camps agree on
    ax2.plot(hhis, sf, "o-", color="#2166ac", lw=2.0, label="salience_fade (ignores agreement)")
    ax2.plot(hhis, cf, "s-", color="#1a9850", lw=2.3, label="consensus_fade (agreement-gated)")
    ax2.set_xlabel("model-share concentration  (HHI)")
    ax2.set_ylabel("Sharpe")
    ax2.set_title("Cross-model agreement tells you WHICH events to fade (largest edge when diverse)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "model_concentration.png"), dpi=120)
    plt.close(fig)


def main():
    print("\n=== Agent-behavior alpha (synthetic AI-agent population) ===")
    mkt = agents.simulate_agentic_market(monoculture=RHO_BASE, seed=11)
    n_events = int((mkt.news_salience.to_numpy() > 0).sum())
    print(f"Universe: {mkt.returns.shape[1]} assets x {mkt.returns.shape[0]} days; "
          f"{n_events} agent-driven news events; "
          f"monoculture rho={mkt.rho:.2f} (effective {mkt.rho_eff:.2f}).")

    # ---- experiment 1: base book -------------------------------------------
    results = run_book(mkt)
    board = core.format_leaderboard([r.summary() for r in results])
    print("\n" + board.round(3).to_string())

    eq_path = os.path.join(HERE, "equity.png")
    core.plot_equity(results, eq_path, title=f"Agent-behavior strategies (rho={RHO_BASE})")
    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))

    # ---- experiment 2: monoculture sweep -----------------------------------
    sweep = monoculture_sweep()
    print(f"\nEdge at ρ=0 (independent agents): Sharpe {sweep[0]:.2f}")
    print(f"Edge at ρ=1 (full monoculture):   Sharpe {sweep[-1]:.2f}")
    print("Interpretation: the crowd-reversal alpha is ~absent when agents are\n"
          "independent and emerges as they homogenize -- the agents' BEHAVIOR is\n"
          "the tradeable signal, not any one stock's fundamentals.")

    # ---- experiment 3: crowd-speed sweep -----------------------------------
    chase, fade = crowd_speed_sweep()
    print(f"\nAt machine speed (instant): chase {chase[0]:5.2f}  vs  fade {fade[0]:5.2f}  "
          "-> can't outrun the machines, so fade them.")
    print(f"At human speed (slow):      chase {chase[-1]:5.2f}  vs  fade {fade[-1]:5.2f}  "
          "-> a slow pile-in is ride-able, so chasing wins.")

    # ---- experiment 4: regime timing ---------------------------------------
    regime_timing()

    # ---- experiment 5: systemic crowding -> fragility & convex harvest ------
    fragility_experiment()

    # ---- experiment 6: model heterogeneity -> concentration drives crowding -
    model_concentration_experiment()

    print(f"\nSaved: {eq_path}")
    for p in ("monoculture.png", "crowd_speed.png", "regime_timing.png",
              "fragility.png", "model_concentration.png", "leaderboard.csv"):
        print(f"Saved: {os.path.join(HERE, p)}")


if __name__ == "__main__":
    main()
