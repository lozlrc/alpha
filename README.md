# Alpha — a backtest suite of quant trading strategies

A self-contained research suite that implements **eight families of trading alpha** —
including a forward-looking one that trades the *behavior of other AI trading agents* —
backtests each on synthetic data with deliberately planted structure, and then
**combines them into one risk-managed portfolio**.

> **Read this first — what this is and isn't.**
> Everything here is **offline and backtest-only**. There is **no live-market feed,
> no broker, no order routing** anywhere in this project. All price/volume/fundamental
> data is **synthetic**, generated with known embedded structure so each strategy can
> be shown to *recover a real signal* rather than overfit noise. The Sharpe ratios
> below demonstrate that the *machinery works*; they are **not** a claim about
> live-market profitability. Real data is far less generous. See
> [Honest caveats](#honest-caveats).

---

## Quick start

Built and tested on **CPython 3.12** (the repo's `.venv` was created with `uv`).

```bash
# 1. create the environment
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# 2. run the whole suite (each family in isolation, then the master leaderboard)
./.venv/bin/python run_all.py

# 3. or run a single family
cd 01_cross_sectional_factors && ../.venv/bin/python run.py
```

`run_all.py` runs each `NN_*/run.py` as an isolated subprocess, then collects every
family's `leaderboard.csv` into a single Sharpe-ranked `leaderboard_all.csv`. Each
family also writes its own equity-curve PNG.

---

## Repository layout

```
alpha/
├── core/                          shared engine — imported by every family
│   ├── data.py                    synthetic market generator (+ real-data hooks)
│   ├── backtest.py                no-lookahead backtest engine, L/S weighting
│   ├── metrics.py                 Sharpe, Sortino, drawdown, leaderboard
│   └── plotting.py                equity-curve / heatmap PNGs (Agg, headless)
│
├── 01_cross_sectional_factors/    momentum · value · quality · low-vol · multifactor
├── 02_statistical_arbitrage/      pairs · cointegration · xs-reversal · lead-lag
├── 03_microstructure/             order-flow imbalance · book pressure · LOB sim
├── 04_event_driven/               PEAD · index rebalance · merger arb
├── 05_alternative_data/           sentiment · web-traffic nowcast · alpha-decay study
├── 06_cross_asset_vol/            carry · variance risk premium · dispersion
├── 07_ml_driven/                  gradient boosting + purged/embargoed CV
├── 08_portfolio/                  combines the alphas into one risk-managed book
├── 09_agentic_flow/               trade the crowded flow of homogeneous AI agents
├── 10_real_data/                  honesty check: the price factors on REAL prices (cached)
│
├── run_all.py                     master driver  →  leaderboard_all.csv
├── requirements.txt
└── README.md
```

Every family folder follows the same shape: one module per strategy, a `run.py`
that wires them together, and outputs (`equity.png`, `leaderboard.csv`).

---

## Design principles — why the numbers are trustworthy *within this synthetic world*

The hard part of a backtest isn't getting a high Sharpe — it's not fooling yourself.
The whole suite is built around four disciplines:

1. **No lookahead, enforced by the engine.** `core.backtest_weights` shifts every
   weight forward by ≥1 period before it touches a return, so a signal computed from
   day *t* data can only earn day *t+1* P&L. Fundamentals are point-in-time (quarterly,
   reported with a lag); the ML family uses **purged + embargoed** cross-validation so
   no training row overlaps the test horizon.

2. **Synthetic data with *real* structure.** `core.generate_market()` builds a factor
   model with genuine momentum persistence, value mean-reversion, a quality/low-vol
   premium, sector co-movement, and idiosyncratic noise. Strategies succeed by
   recovering *that* structure — not by curve-fitting. Defaults are tuned so all four
   classic factors stay positive across many random seeds.

3. **Volatility-drag correction.** Expected *simple* returns are held fixed by
   subtracting ½σ² from the log-drift, so a high-vol asset doesn't get an artificial
   return penalty (a subtle bug that silently creates a fake "low-vol anomaly").

4. **Honesty controls built in as teaching contrasts.** The suite deliberately ships
   the *wrong* way next to the right way so the gap is visible:
   - **`07_ml_driven`** runs `ml_leaky_insample` (Sharpe **6.97**) right beside
     `ml_honest_oos` (**2.22**) — the difference *is* the lookahead leakage.
   - **`05_alternative_data`** shows a sentiment signal decaying from Sharpe **2.59**
     (act same-day) to **negative** by a 5-day delay — alpha is perishable.
   - **`09_agentic_flow`** ships `reaction_frontrun` (**−2.0**) beside the fades
     (**+3.7 / +4.7**) — *chasing* a fast agent crowd loses; only *fading* it pays.

---

## The eight families

Headline results from a representative run (`./.venv/bin/python run_all.py`). Sharpes
are annualized and most meaningful **within** a family.

| Family | Representative strategies (Sharpe) | The idea |
|---|---|---|
| **01 Cross-sectional factors** | multifactor **2.04**, sector-neutral mom **1.69**, quality 1.48, momentum 1.31, value 0.65, low-vol 0.30 | Rank stocks by a characteristic, go long the top decile / short the bottom, dollar-neutral. |
| **02 Statistical arbitrage** | pairs **2.27**, lead-lag 1.92, cointegration 1.66, xs-reversal 1.59 | Trade the spread between historically linked names back to its mean. |
| **03 Microstructure** ⚠️ | OFI **16.4**, book-pressure 7.3 *(net, intraday)* | Predict the next tick from order-flow imbalance and book pressure on a simulated LOB. |
| **04 Event-driven** | index-rebal **1.80**, PEAD 1.54, merger-arb 1.09 | Trade predictable flows around earnings, index reconstitution, and M&A. |
| **05 Alternative data** | sentiment **2.59**, web-traffic 1.52 *(+ decay study)* | Nowcast fundamentals from non-price data; demonstrate how fast that edge decays. |
| **06 Cross-asset & vol** | short-variance **2.40**, dispersion 2.33, carry 0.52 | Harvest the variance risk premium, index-vs-single-name dispersion, and carry. |
| **07 ML-driven** | honest-OOS **2.22** *(vs leaky 6.97)* | Gradient-boosted trees on engineered features, validated without leakage. |
| **09 Agent-behavior** ⚡ | salience-fade **4.74**, crowd-reversal 3.68, *frontrun −2.0 (cautionary)* | Fade the crowded, salience-driven over-reaction of homogeneous AI trading agents. *Forward-looking — see below.* |

> ⚠️ **The microstructure family is on a different clock.** It is annualized from
> intraday bars (~100k bets/year), so its Sharpes are huge by construction (√N) and
> are **not comparable** to the daily families. Compare it only to itself.

One worth a second look: in **04**, merger-arb posts a modest Sharpe (1.09) despite a
**98% hit rate** — the textbook negatively-skewed payoff, where many small wins are
punctuated by the occasional large loss when a deal breaks. (Family **09**'s
`cascade_reversal` is its mirror image — a *low* hit rate and *positive* skew; see below.)

---

## Forward-looking: trading the behavior of other AI agents (`09_agentic_flow`)

As more capital is deployed by trading agents built on a *few shared foundation models*
— prompted alike, fed the same news APIs and transcripts — their decisions correlate,
their order flow synchronizes, and price over-reacts to loud headlines before reverting.
That crowded, salience-driven flow is a **new mechanical participant you can model** —
the way this suite already trades index-fund flow in `04`.

`agents.py` simulates that agent population with a **monoculture knob ρ** (0 =
independent agents whose idiosyncratic errors cancel in aggregate; 1 = one mind, maximal
crowding). Three strategies form the core book (two more — a crowding-timed fade and a
cascade harvester — appear in the deeper lenses below):

- **`crowd_reversal`** (Sharpe **3.68**) — fade the *realized* over-extension, harvest the reversion. *(robust; the sweep's headline)*
- **`salience_fade`** (Sharpe **4.74**) — fade in proportion to headline loudness, straight off the news.
- **`reaction_frontrun`** (Sharpe **−2.0**) — *chase* the herd, and lose: the machines already moved the price, so you buy the top and sit through the reversion. **You can't outrun the crowd you're trying to follow.**

**1. The edge *is* the crowding** (`monoculture.png`). Sweep ρ — same seed at every
point, so *only* the crowding changes — and the crowd-reversal edge emerges precisely as
the agents homogenize:

| agent homogeneity ρ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| crowd-reversal Sharpe | −0.2 | 0.9 | 2.3 | 3.7 | 4.9 | 6.0 |

At ρ=0 there is *no* edge (fading independent noise just pays costs); the alpha is
**created by the crowding itself** — the agents' behavior *is* the signal, not any one
stock's fundamentals.

**2. Why *fade*, not *chase*? Because the crowd is fast** (`crowd_speed.png`). The whole
thesis rests on the agents being faster than you — and that, too, is a knob. `crowd_speed`
sets how long the crowd takes to pile in; sweep it (at ρ=0.6) and the right trade flips:

| crowd pile-in time (days) | **0 = machines** | 0.5 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| chase the move (ride 1-wk) | **−4.0** | 4.4 | 5.1 | 6.1 | 6.7 | 6.9 |
| fade the move (crowd-reversal) | **+3.7** | 2.1 | 1.6 | 0.6 | −0.6 | −1.5 |

At machine speed (the left edge — where AI agents actually live) the overshoot peaks the
*same day* the news breaks, so by the time you can act on the next bar you are already on
the reversion: chasing loses, fading wins. Slow the crowd to human pace and it inverts —
a multi-day pile-in is ride-able, so chasing wins and fading bleeds. The more automated
and homogeneous the crowd, the more reliably you sit on the fade side of that line.

**3. Making it deployable: you can't see ρ, but you can *nowcast* it** (`regime_timing.png`).
Real crowding waxes and wanes — it spikes when a wave of same-model flow piles into one
loud narrative and subsides on quiet, idiosyncratic tape. A live strategy never sees the
latent ρ, so `crowding_nowcast` estimates it from observable data alone: for each past
event, how much the move reverted over the next few days, trailing-averaged and
lookahead-safe. Against a regime-switching ρ it tracks the hidden state at **correlation
0.65** using only price and public news. Feeding that into a **`crowding_timed_fade`**
(lean in when crowding is high, shrink toward flat when it's calm) does *not* beat an
always-on fade in a world where crowding never switches off — but the real world *does*
switch off, and there the timed book deploys **~15% less capital for the same Sharpe**
(capital efficiency 4.0 vs 3.5) and **wins outright on net Sharpe once costs are
realistic** (the two cross around ~4 bps), because the static fade keeps churning in calm
periods where there is nothing to fade. The point isn't a bigger headline number — it's
that the crowd is *observable and timeable* without ever seeing the thing that creates it.

**4. The dark side of the same knob: systemic fragility** (`fragility.png`). Homogeneity
doesn't only crowd each *stock's* news — it crowds names *together*. Turn on the simulator's
`systemic` / `fragility` knobs and a **shared crowd factor** builds up (agents pile into the
same macro trade) and occasionally **unwinds in a cascade** — a violent common move that gaps
the tape and then partly snaps back. Sweep ρ with it on and the *cost* of the monoculture
emerges right alongside the alpha:

| agent homogeneity ρ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| cross-name correlation (PC1 share) | 0.22 | 0.22 | 0.26 | 0.30 | 0.35 | 0.40 |
| worst-1% daily common move | −1.4% | −1.5% | −1.7% | −1.9% | −2.4% | −2.9% |
| return kurtosis | −0.1 | 0.1 | 1.9 | 4.5 | 6.6 | 7.0 |
| de-risking cascades (count) | 0 | 1 | 5 | 8 | 17 | 22 |

The same ρ that makes the fade pay also makes the tape fragile: correlation, tail size, and
crash frequency all climb together. **`cascade_reversal`** harvests that fragility — it spots
an outsized common move inside a high-crowding regime (observable: a standardized index jump
plus an elevated PC1 share) and fades it for the snap-back. Its payoff is **convex**: a modest
Sharpe (~0.5 at ρ=0.9) but **positive skew** at a ~45% hit rate — small bleeds, paid back in
the crashes. It is the exact **mirror of merger-arb** (high hit rate, *negative* skew): one
sells insurance, the other buys it. You'd hold `cascade_reversal` not for its Sharpe but for
*when* it pays — in the cascades, precisely when a crowded book hurts most.

**5. Monoculture is really *model-share concentration*** (`model_concentration.png`). The ρ
knob treats "the crowd" as one blob, but real agents run on a handful of distinct foundation
models. Model the crowd as **`n_models` camps** with capital shares: agents on the same model
move together, different models read ambiguous news differently, and their idiosyncratic
biases *cancel* when many models are balanced but *reinforce* when the market consolidates onto
one. Hold per-agent homogeneity fixed (ρ=0.6) and sweep the ecosystem's concentration (HHI):

| model-share concentration (HHI) | 0.25 | 0.37 | 0.52 | 0.73 | 1.00 |
|---|---:|---:|---:|---:|---:|
| effective # of models | ~4 | ~2.7 | ~1.9 | ~1.4 | 1 |
| crowd-reversal Sharpe | 2.3 | 2.5 | 2.7 | 2.9 | 3.3 |

The crowding edge climbs as the ecosystem consolidates **even though no individual agent
changes** — "monoculture" is as much about *the market consolidating onto one model* as about
any single agent being predictable. The dispersion *between* camps is itself tradeable:
**`consensus_fade`** fades only the events the models *agree* on (observable via
`model_agreement`), where the flows reinforce into a big over-reaction, and skips the contested
ones where they cancel. It beats the agreement-blind `salience_fade` by ~0.1–0.3 Sharpe — and,
neatly, its edge is **largest when the ecosystem is diverse** (when one model dominates, every
event is "consensus" and there is nothing left to gate).

> **Caveat (important).** This demonstrates the *logic conditional on the mechanism*:
> *if* agents crowd and over-react to salience, *then* their flow is fade-able and the
> edge scales with homogeneity. Whether real agent flows actually behave this way — and
> how quickly the effect arbitrages away once everyone trades it — is an empirical
> question that needs real data. The simulator is a **hypothesis you can test, not
> evidence**. (How to test it for real: tag news/social posts by salience and by
> likely LLM-generation, proxy "agent flow" with fast post-headline volume + price
> kinks, and check whether high-salience over-reactions revert more as model
> concentration rises — and sharpen as execution gets faster and more automated,
> exactly as the crowd-speed sweep predicts.)

---

## The payoff: `08_portfolio`

The reason to generate many roughly-uncorrelated alphas is to **combine** them. The
portfolio layer pulls one honest, net-of-cost stream from each daily family and
allocates across them using **trailing-window, no-lookahead** weights (monthly
rebalance, 1-day execution lag): equal-weight, inverse-vol, **risk-parity** (equal
risk contribution via SLSQP), **min-variance** (Ledoit-Wolf shrunk covariance), and a
vol-targeted variant.

| | Sharpe | Ann. vol | Max drawdown |
|---|---:|---:|---:|
| **Risk-parity portfolio** (13 streams) | **5.22** | 1.9% | −2.3% |
| Best single stream (`agentic_reversal`) | 3.75 | 8.1% | −5.0% |

Across the **13 streams** — one honest, net-of-cost stream per daily family, now
including the agent-behavior fade — the average pairwise correlation is **0.002**, so
risk-parity weighting lifts Sharpe **+1.48 over the best individual strategy** and cuts
the drawdown by roughly **2×**. The agent-behavior stream is itself the **best single**
input *and* near-uncorrelated with the rest, so it both raises the bar and diversifies
the book. Outputs: `portfolio_equity.png`, `correlation.png`, `leaderboard.csv`.

---

## Honest caveats

- **The diversification looks too good — because the data is too clean.** These
  synthetic streams come from *independent* data-generating processes, so their
  average pairwise correlation is ~0.00 and diversification is near-ideal. **Real
  strategies share common risk factors and correlate sharply in a crisis**, so expect
  materially less benefit live. The portfolio *construction math*, however, is exactly
  what you'd run on real streams.
- **Synthetic ≠ real.** Every synthetic number demonstrates signal recovery on planted
  structure, not a forward-looking return estimate — `10_real_data` runs the same factor
  code on real prices and shows it going flat-to-negative net of costs.
- **Costs are simple.** A flat per-turnover bps charge; no market impact, borrow,
  slippage, or capacity modeling.

---

## Real data — the honesty check (`10_real_data`)

The synthetic generator sits behind one interface, so the whole suite can point at **real
historical prices** without touching strategy code. `10_real_data/run.py` does exactly
that: it downloads daily *adjusted* prices for a fixed large-cap universe (one time, cached
to `data_cache/` → fully offline afterwards; historical data, **not** a live feed), then
runs the **same** price-factor code from `01` on real prices.

The result is the whole point of the repo:

| factor (net 5 bps, 2010–2024) | synthetic (planted) | real |
|---|---:|---:|
| momentum 12-1 | **+1.16** | **−0.22** |
| low-vol | +0.18 | −0.38 |
| short-term reversal | −2.80 | −1.06 |
| *SPY buy-and-hold (reference)* | — | *+0.84* |

Same machinery, only the data changed — the textbook factors that "work" on planted
structure go flat-to-negative on real large-caps net of costs, and **just holding the
market beat every dollar-neutral factor**. (That's honest, not a bug: cross-sectional
factors in a ~34-name mega-cap universe, net of turnover, are a hard place to find alpha.)
This is why every synthetic Sharpe here demonstrates *mechanics*, never a forward-looking
return estimate.

Loaders (each returns the same `MarketData` the synthetic generator does, so every
price-based family runs unchanged):
- `core.data.load_yfinance(tickers, start, end, market_ticker="SPY", cache_dir=…)` — a
  one-time historical download, cached (needs the optional `yfinance`).
- `core.data.load_csv(prices_csv, volume_csv=…)` — your own adjusted-price files.

**What real *prices* alone can't reach:** value/quality (point-in-time fundamentals),
events (`04`), alt-data (`05`), microstructure (`03`), and `09_agentic_flow` (a hypothesis
simulator — validate via the proxies above). And two traps the engine can't fix for you:
**survivorship bias** (a surviving-names list overstates returns) and **point-in-time**
fundamentals (never use restated data).
