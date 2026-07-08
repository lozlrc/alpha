# Alpha — a backtest suite of quant trading strategies

A self-contained research suite that implements **thirteen families of trading alpha** —
eight demonstrated on synthetic data with deliberately planted structure (including a
forward-looking one that trades the *behavior of other AI trading agents*), then **five
on real market data**: large-cap prices, multi-asset ETFs, crypto venue funding, overnight
open/close gaps, and a factor-mining panel — plus a layer that **combines the synthetic
alphas into one risk-managed portfolio**.

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
├── 10_real_data/                  honesty check: 01/02/07 strategies on REAL prices (cached)
├── 11_tactical_allocation/        real high-Sharpe, low-drawdown multi-asset TAA (cached)
├── 12_funding_carry/              real crypto perp funding carry — structural, delta-neutral (cached)
├── 13_overnight_news/             can you trade overnight news at the open? + forward-only LLM harness
├── 14_factor_mining/              factor-mining engine (因子挖掘) + multiple-testing discipline
├── 15_improvement_loop/           walk-forward re-tuning loop with a trials ledger (runs on 11/12/13)
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

The synthetic generator sits behind one interface, so the suite can point at **real
historical prices** without touching strategy code. `10_real_data/run.py` downloads daily
*adjusted* prices for a fixed ~62-name large-cap universe (one time, cached to `data_cache/`
→ fully offline afterwards; historical data, **not** a live feed), then runs the **same
strategy code** from the synthetic families on real prices — *everything the data supports*:
**01** price factors, **all four `02`** stat-arb strategies, and the **`07`** ML model.

The result is the whole point of the repo:

| strategy (net 5 bps, 2010–2024) | synthetic (suite) | real |
|---|---:|---:|
| 01 momentum 12-1 | +1.3 | −0.06 |
| 01 low-vol | +0.3 | −0.19 |
| 01 short-term reversal | −2.8 | −0.82 |
| 02 pairs / cointegration | +2.3 / +1.7 | −0.14 / −0.07 |
| 02 xs-reversal / lead-lag | +1.6 / +1.9 | −0.89 / −1.17 |
| 07 ML honest-OOS | +2.2 | −0.85 |
| 07 ML **leaky in-sample** ⚠️ | +7.0 | **+0.97** |
| *SPY buy-and-hold (reference)* | — | *+0.84* |

Same code, only the data changed — and **every dollar-neutral factor, stat-arb, and ML
signal is flat-to-negative on real large-caps net of costs.** Just holding SPY (0.84) beat
all of them. The *only* thing above SPY is the **leaky in-sample ML (+0.97) — which is the
look-ahead trap**: its honest purged-OOS twin is **−0.85**, a +1.8-Sharpe mirage of pure
leakage that survives the jump to real data exactly as it does in `07`. *That gap is the
point:* planted structure is generous, real markets are competitive, and a high backtest
Sharpe is worthless until you've ruled out leakage and costs.

**"Can't you just optimize the parameters?"** `10_real_data/optimize.py` answers it the
honest way: tune each strategy on a **train** slice (2010–2017), then report the
**out-of-sample** Sharpe on held-out **2018–2024**. Even *snooping* — grid-searching the
whole history and reporting the max — averages only **−0.08**, and the honest OOS average
is **−0.18**; nothing beats SPY (+0.76). The one setting that improves OOS is **rebalancing
less often** (e.g. monthly momentum −0.07 → +0.14) — a real turnover/cost saving, not a
discovered signal. You can't optimize your way out of no-alpha; you only fit noise. (To
actually *find* edge you change the inputs, not the knobs: a broader small/mid-cap universe
where factors live, or genuinely new data — fundamentals, events, alt-data.)

**What price data alone can't reach** (the script prints this as a coverage matrix):

| family | needs real data of type |
|---|---|
| 01 value / quality | point-in-time fundamentals (book, earnings, ROE) |
| 03 microstructure | tick / limit-order-book (bid-ask, order flow) |
| 04 event-driven | earnings / index-rebalance / M&A calendars |
| 05 alternative data | news, sentiment, web-traffic feeds |
| 06 cross-asset vol | options / implied-vol surfaces |
| 09 agentic flow | no real agent-flow labels — a simulator; proxy-test only |

Loaders (each returns the same `MarketData` the synthetic generator does):
`core.data.load_yfinance(tickers, start, end, market_ticker="SPY", cache_dir=…)` (one-time
cached download; needs the optional `yfinance`) and `core.data.load_csv(prices_csv,
volume_csv=…)`. Two traps the engine can't fix for you: **survivorship bias** (a
surviving-names list overstates returns) and **point-in-time** fundamentals (never restated).

---

## Where real risk-adjusted return actually lives (`11_tactical_allocation`)

If naive equity factors net of costs are a dead end, where does *real* high-Sharpe,
low-drawdown return come from? **Diversification, trend-following, and risk management** —
the most robust, best-documented effects in finance. `11_tactical_allocation` builds them
into one portfolio on **14 real ETFs** (US large/small cap, intl & EM equity, Treasuries
across the curve, TIPS, IG & high-yield credit, gold, silver, commodities, REITs;
**2007–2024**, so the GFC, COVID, and 2022 are all in the test). Idle cash earns the T-bill
yield (BIL). Three *a-priori, literature-standard* rules — not parameters fit to the data:

1. **Trend** — hold each asset only while it's above an *ensemble* of 8/10/12-month moving
   averages (Faber / time-series momentum); exposure scales with trend strength.
2. **Risk parity** — size each sleeve by inverse volatility.
3. **Vol target** — scale toward ~10% annual vol, capped at 1× (no leverage); the rest is cash.

| 2007–2024, net 10 bps | Sharpe | ann vol | **max drawdown** | Calmar |
|---|---:|---:|---:|---:|
| **Tactical allocation** | **1.00** | 3.0% | **−6.2%** | **0.48** |
| 60/40 (SPY/IEF) | 0.74 | 11.3% | −31.4% | 0.26 |
| SPY buy-and-hold | 0.60 | 19.8% | −55.2% | 0.19 |

**+0.40 Sharpe over SPY, an 89% shallower drawdown (−6.2% vs −55%), and 2.6× the Calmar** —
and **robust, not a knife-edge**: Sharpe holds **0.97–1.00** and max-DD **−6.1% to −6.2%**
across *every* lookback/vol-target setting in the sweep `run.py` prints. The drawdown panel of
`equity.png` is the story: in 2008 SPY falls 55% while the tactical book barely dips. Two
things that genuinely helped (neither is curve-fitting): **breadth** (9 → 14 diversifying
sleeves lifted Sharpe 0.88 → 1.00 and cut drawdown 9.5% → 6.2%) and the **ensemble trend**
filter. *Long/short trend was tested and was worse* (≈0.5 Sharpe — shorting whipsaws in a
mostly-rising tape), so the book stays long-only-to-cash.

**Honest framing:** this is long-only **alternative beta** (harvesting diversification + a
trend premium), *not* market-neutral alpha, and it's **unlevered** (3% vol) — so it trades raw
return for safety; lever it to taste and the *Sharpe/drawdown shape* is what travels (≈3× →
~9% return, ~−18% DD, same ~1.0 Sharpe — still a third of SPY's drawdown). It's a backtest on
survivor ETFs net of a flat cost, with no impact model. But unlike the equity factors, the
edge is **structural and robust** — which is exactly why it survives where threshold-tuning
didn't. *That* is the difference between finding alpha and fitting noise.

---

## Structural carry on real crypto venues (`12_funding_carry`)

The same law, applied where the counterparty pays *knowingly*. Perpetual futures track
spot via a **funding rate** exchanged between longs and shorts every hour; retail leverage
demand is chronically long, so funding is positive most of the time. **Short 1× perp +
long 1× spot** is delta-neutral and collects that transfer — a structural insurance
premium for warehousing leverage demand, not a forecast. Nobody is out-predicted (the
sibling of `06`'s carry/variance premium — on **real venue data**).

Data: **Hyperliquid** hourly funding *and premium*, May 2023 → Jul 2026 (~3.2 yr, BTC/ETH/
SOL) + **Kraken Futures** hourly funding (~1 yr) — free public endpoints, one-time fetch
cached to `data_cache/`. (Ops reality is part of the trade: Binance & Bybit are US-geo-
blocked and OKX's public history is capped at ~3 months — this is *why* funding desks run
VPSes next to their venues.)

| 2023-05 → 2026-07, net of flip costs | ann. return (on notional) | max DD (daily-marked) |
|---|---:|---:|
| **carry_book_gated** (3 coins, 7-day gate) | **15.3%** | **−0.3%** |
| avg funding received: BTC **14.4%**, ETH 14.5%, SOL 12.5% | *(87 / 87 / 73% of days positive)* | |

Three honesty devices are built in, in the suite's usual style:

- **The smoothing illusion** — `carry_btc_ACCRUAL_ILLUSION` (funding accrual only,
  Sharpe 13.8) ships beside the properly **daily-marked** `carry_btc` (funding − Δbasis
  via the venue's own premium series, Sharpe 11.0). Same trade, same mean — the gap is
  thrown-away vol. Never score carry unmarked.
- **Read the Sharpe like `03`'s** — it's a category artifact of the clock/measure. Basis
  wiggle is bps while the yield is steady, so Sharpe prints huge; the risks that actually
  kill carry books — venue/custody failure, liquidation gaps, basis blowout at exit —
  live **outside** a daily series. The worst daily mark across 3.2 years is **−14 bps**,
  *including the Oct-2025 liquidation cascade* — that smallness is the warning, not the
  comfort. The honest summary is **~15%/yr structural yield + an unmeasured operational
  tail**. Size to the tail, not the vol.
- **A real negative result** — the cross-venue spread (`xvenue_*`, HL vs Kraken funding
  differential, both legs perps) is reported as the accrual-only **upper bound** and it
  *still loses* 1–4%/yr net of 20 bps flips on the 1-yr overlap: the differential
  mean-reverts faster than a trailing-sign rule catches. Venue desync is real; harvesting
  it needs faster hands than a daily loop.

Carry is **regime beta**: the equity curve compounds steadily through 2024 and visibly
flattens through the compressed-funding 2025-26 tape, and the 7-day gate is what keeps
the book out of the negative-funding stretches (SOL gated Sharpe 8.2 vs 6.0 always-on).
Robust across the gate/cost grid (9.7–11.9 Sharpe for 3/7/14-day gates × 10/25 bps).
Unlevered, on-notional; spot custody/borrow and margin drag excluded; HL/Kraken perps
aren't US-retail venues (the US-legal cousin is the CME basis trade).

---

## Overnight news at the open — and why the LLM half runs forward-only (`13_overnight_news`)

The tempting strategy: *RAG the headlines published while the market was closed; at the
open, buy the good-news names and short the bad-news ones.* Before pointing an LLM at it,
two things had to be done honestly:

**1. No LLM appears in the backtest — on purpose.** Backtesting an LLM on *historical*
news is the suite's look-ahead trap in its most seductive form: every modern model was
trained on text written **after** those nights — it already "knows" which earnings beat
and which CEO resigned in disgrace. The leak lives **inside the weights**, where no
purged CV can reach it. Any 2015-2024 "LLM news backtest" you see is contaminated by
construction.

**2. So the backtest tests the *mechanical core* on real adjusted open/close (2010-2024,
same 62 mega-caps):** the overnight gap `open_t/close_{t-1}−1` *is* the market's own
summary of the night's news, and it prices the slot an LLM would have to beat.

| finding (real data, 2010-2024) | number |
|---|---|
| the equity premium accrues **overnight** (Asness's night effect, reproduced) | overnight-only **+9.1%/yr, Sharpe 0.85** vs intraday-only +5.0%, 0.46 |
| "buy the gap at the open" (continuation, gross) | **Sharpe −1.49** — gaps *revert* |
| so the **fade** is the real gross edge | **+1.49 gross, +25%/yr** — family `09`'s salience-fade, on real data |
| …and a daily open→close flip costs | ~20 bps/day ≈ **50%/yr** — more than the whole gross edge |

Every net-of-cost variant (continuation, fade, big-gap-only, 5-day hold) is **negative**
in mega-caps: the over-reaction is real, but the *frequency is unaffordable* at retail
costs, in the sharpest-crowd universe there is. The edge-law reading: an LLM only earns
its keep here if it can say **which** gaps under- vs over-react — that marginal skill,
after ~20 bps/day, is the bar.

**The honest LLM half: `live_harness.py`** — a forward-only pre-open paper trader.
`--score` pulls each name's overnight headlines and logs frozen pre-open scores
(pluggable scorer: a crude embedded finance lexicon as the free floor, or `claude -p`
as the actual RAG); `--settle` later fills in realized open→close returns and prints
the running **forward IC**. Zero look-ahead by construction; the rule printed at the
bottom is the rule: *no real money until the forward IC is positive with n in the
hundreds.*

---

## A factor-mining engine, with the discipline that makes it honest (`14_factor_mining`)

What "alpha factories" (WorldQuant's Alpha101 genre) actually run: generate thousands of
candidate **expressions** over price/volume panels, score each by **daily cross-sectional
rank-IC** against next-day returns, keep survivors. The engine is the easy part —
`14_factor_mining` builds it (random expressions over `ts_mean/ts_std/ts_delta/ts_corr`
of returns, price, volume, dollar-volume, illiquidity; train 2010-2017, test 2018-2024,
real 62-name panel). The part that separates shops from noise-miners is **multiple-testing
discipline**, and the run prints it as a single exhibit:

| the mirage, quantified (205 unique mined candidates) | |
|---|---:|
| "significant" on train by the naive |t|>2 bar | **39** |
| expected **max** |t| from pure noise, √(2 ln N) | **3.26** |
| Bonferroni bar (p=0.05/205) | 3.89 |
| train-significant that stay significant OOS | **4** |
| the #1 pick, `ts_delta(illiq,63)`: train t **−4.2** (beats even Bonferroni) → OOS t | **−0.2**, book Sharpe **−3.0** |

Mine 205 strategies and the best one *looks* like a discovery **by construction** — the
scatter (`equity.png`) shows the whole cloud. The **classics (经典因子)** run through the
*same* evaluator as the mined noise: momentum 12-1 (动量) is the only one whose IC
survives OOS (t +2.4 — a century of literature showing up in 62 names), 1-month reversal
(反转), low-vol (低波), Amihud illiquidity and turnover all die — **and even momentum's
quintile book loses net of costs** (−0.05 Sharpe). Surviving the t-test is necessary,
not sufficient: **IC ≠ money** — after the statistics come costs, capacity, and crowding.

Honest caveats: 62 mega-caps is a breadth-starved panel (real miners use thousands of
names — IC value scales with √breadth); the candidates share windows/terms, so the
independent-trials noise bar *understates* the real haircut; and the pool-level
train/test t correlation (+0.37) means the *family* of illiquidity/return-delta signals
shares weak structure — which is exactly how mining should be read: evidence about
*pools*, never about the lucky top pick.

---

## The improvement loop, with a trials ledger (`15_improvement_loop`)

"Loop over the strategies: backtest, tweak, keep what's better, repeat" — run that
naively and it **always reports progress**, because each iteration is one more draw from
the noise distribution and the loop keeps the best draw. An improvement loop without
multiple-testing discipline is `14`'s mining mirage pointed at your own book.

`15` is the disciplined version, run on the real-data families (11 / 12 / 13):
**walk-forward selection** (each fold, pick the variant with the best Sharpe on data
strictly *before* the fold — what a re-tuning loop would actually have earned), a
**paired t-test against the a-priori default** with an adoption bar that rises with the
number of variants tried (`t > max(2, √(2 ln N))` — every trial is counted and paid for),
the **snooped full-sample max** printed alongside (the gap to walk-forward is the
overfitting tax), and a **pick-stability readout** (plateaus re-pick the same region;
noise surfaces jump).

| family (variants) | default | wf-tuned | paired t (bar) | verdict |
|---|---:|---:|---:|---|
| 11 TAA (9) | 1.00 | 0.98 | −1.04 (2.10) | **REJECT** — plateau, tuner chased wobbles |
| 12 funding carry (6) | 11.58 | 11.72 | +1.67 (2.00) | **REJECT** — "better" number, below the bar |
| 13 overnight fade (4) | −1.26 | −0.75 | +1.27 (2.00) | **REJECT** — can't tune a dead strategy alive |

Three REJECTs is the loop **working**: it is the certificate that the shipped defaults
sit on robust plateaus rather than cherry-picks — the offline sibling of a live
shadow-A/B rule (candidates judged on *forward* data, adopted only past a paired-t bar).
New strategies join by adding an adapter + grid; LLM-proposed variants are welcome as a
*generator*, but every proposal lands in the same ledger, pays the same bar, and never
sees the fold it is judged on.
