# Category-Selection Alpha — Results

**Alpha tested:** *"We can't predict who wins (high variance), but team counting-stats
(corners, shots, fouls…) are more predictable from team strength / form / style. If we
forecast only the categories we're best at, do we win more with less variance?"*

**Verdict (one line):** The hypothesis is **right about *where the edge is* but wrong about
*why*, and wrong about the remedy.** Stats are **not** more predictable than who-wins — on
17,784 club matches they're *comparable* (who-wins is actually slightly more predictable, and
shots-on-target most of all). The reason team-stats are where our competition edge lives is
that **there's no sharp market there, so the crowd is weak** — whereas on who-wins the crowd
≈ an efficient bookmaker, leaving no edge. On the *remedy*: **how you select matters.**
*Noisy per-family selection* (pick what looked best on data) **backfires** — loses OOS,
raises variance. But *structural selection* — skip the families where a sharp public market
makes the crowd beat you (who-wins, totals, player props), keep all the no-market stats —
is the **vindicated form of the user's "skip who-wins" instinct** (conditional on the crowd's
sharpness and a neutral abstention rule). Winning play: **answer all no-market stat families,
treat market families as drop-candidates, and invest modeling effort in the high-edge stats
(esp. SoT — big headroom — fouls, corners).** Independently reproduced + adversarially verified
(two multi-agent reviews) and cross-checked on a large external dataset.

> **The unifying law:  edge ≈ predictability − crowd_competence.**
> Predictability is similar across categories; *crowd competence* is what varies — high where
> a liquid market exists (who-wins, totals), ~zero on niche stats. So edge concentrates in
> the no-market stats not because they're easier to predict, but because no one else is.

---

## 1. Data (real, not simulated)

- **176 settled** Probability-Cup questions pulled from the live API (`pull_data.py`):
  our submitted probability + the competition's **actual Brier score** per question.
- Each question's `family` (category) is joined from the bot's `forecasts.json`; the binary
  **outcome is recovered** from `brier = (prob/100 − outcome)²`. Data integrity verified:
  **0/176 inconsistent**, all probs in range. (Caveat: 5 records with prob=50 are
  outcome-unidentifiable since Brier=0.25 either way — immaterial.)
- 23 families, grouped into **WINNER_GOALS** (who-wins / scorelines), **TEAM_STAT**
  (corners/shots/fouls/cards/offsides), **PLAYER** (props).

### The field is the "CROWD" — and it is *not* dumb
The rules PDF shows RBP is scored **per question, relative to the crowd's probability**
(YOU vs CROWD vs reality), and is **additive** across questions. In the one fully-visible
settled match (Korea 2–1 Czechia) the crowd's probs ranged 33–67% — **moderately
calibrated, not a coin-flip.** The crowd probabilities are **not exposed by the REST API**
(every per-market/per-match route 500s; settled markets return `[]`), so we cannot measure
true per-question edge at scale. We proxy the field two ways, both stated as proxies:

| proxy | definition | realistic for | biased for |
|---|---|---|---|
| **coin-flip** | skill = 0.25 − brier | no-market niche questions | *over*-states winner/goals & rare props |
| **climatology** | edge vs family base rate | removing base-rate freebies | small-n artifacts |

Because the real crowd is sharper than a coin-flip, **our true edge is thinner than the
coin-flip "skill" numbers below** — they are an upper bound.

---

## 2. Predictability by category (real Brier, coin-flip field)

| bucket | n | Brier | skill vs coin-flip | 95% CI | std (variance) |
|---|---|---|---|---|---|
| WINNER_GOALS | 66 | 0.2395 | +0.0105 | [−0.026, +0.046] | **0.152 (worst)** |
| TEAM_STAT | 84 | 0.2345 | +0.0155 | [−0.013, +0.043] | 0.131 |
| PLAYER | 26 | 0.2148 | +0.0352 | [+0.002, +0.069] | 0.088 |

**Standout family:** `fouls` (n=14) — skill **+0.0735, CI [+0.013, +0.121]** (excludes 0) →
a **genuine, significant alpha**. `corners` (n=13, +0.044) and `team_scores` (+0.028) are
positive but underpowered. **`result` (who-wins, n=17): −0.003** — no edge, highest variance.
Value-destroyers (≤ 0): `cards` (−0.062), `cards_compare` (−0.054), `pen_or_red` (−0.034),
`offsides` (−0.008).

✅ **Hypothesis half #1 (who-wins is noise) — confirmed.**

---

## 3. Genuine edge, base-rate freebie removed

`skill = 0.25 − brier` flatters families with lopsided base rates (correctly calling a rare
prop "unlikely" scores free points the crowd also banks). **13% of records (rare props)
supplied 43% of total skill.** Stripping that — looking only at **balanced** questions
(family base rate 0.35–0.65), the true match-specific picture:

| bucket | genuine edge (balanced) | 95% CI |
|---|---|---|
| WINNER_GOALS | **+0.0030** (≈ zero) | [−0.043, +0.047] |
| TEAM_STAT | **+0.0226** (largest) | [−0.009, +0.051] |
| PLAYER | **+0.0051** (freebie was the "edge") | [−0.023, +0.038] |

**Core-claim test** (bootstrap of TEAM_STAT − WINNER_GOALS skill): balanced questions
**+0.0196, P(TEAM_STAT > WINNER_GOALS) = 0.76** — directionally supports the hypothesis but
**not significant at 95%** (n too small). PLAYER's apparent strength was almost entirely the
freebie.

✅ **Hypothesis half #2 (stats carry the edge) — directionally yes, not yet significant.**

---

## 3.5 Large-sample cross-check (17,784 real club matches)

The competition's n=176 cannot *prove* "stats more predictable than who-wins," so I tested it
where it can be powered: football-data.co.uk (7 leagues × 7 seasons), walk-forward online
logistic per category, skill = expanding-base-rate Brier − model Brier (`historical_logit.py`).

| category | skill over base rate | category | skill over base rate |
|---|---|---|---|
| **SOT home>away** | **+0.0248 (+10%)** | CORNERS home>away | +0.0113 (+4.5%) |
| RESULT home-win | +0.0159 (+6.5%) | GOALS over 2.5 | +0.0023 |
| FOULS home>away | +0.0153 (+6.2%) | CORNERS total>9.5 | +0.0013 |
| CARDS over 3.5 | +0.0118 (+4.9%) | CARDS home>away | +0.0020 |
| SOT over 8.5 | +0.0119 (+4.8%) | | |

**Ancillary-stat mean skill +0.0129 vs RESULT +0.0159 → comparable; who-wins is *not* the
unpredictable one.** Shots-on-target comparison is the single most predictable family.

**The market is the difference, not predictability.** On RESULT the devigged Bet365 line
scores Brier **0.2152 (skill +0.030)** — about **2× our model's +0.016** — so a crowd that
knows the line leaves us no edge. No comparable sharp market exists for corners/fouls/SoT, so
our modest model skill there survives as real edge. This is exactly the competition pattern
(§2: edge on fouls/corners, none on result) — explained by crowd sharpness, not by
predictability. *Sub-finding:* "home > away" comparisons predict better than "over X" totals.

**Robustness (`robustness.py`):** the ordering is stable across rolling windows K∈{6,10,20},
across eras (2018–21 vs 2022–25), and across all 7 leagues — SoT, result, fouls, corners are
positive in *every* league. So the finding is not a window/era/league artifact.

(An earlier generative-Poisson version, `historical_backtest.py`, scored FOULS at −36% — a
grid-truncation bug, foul mean ~12 vs grid cap 12; a signal diagnostic, cov(rolling-diff,
outcome)=+0.20, confirmed the signal is real and the logistic rebuild fixed it. Documented for
honesty.)

## 4. The remedy fails: selection loses out-of-sample (the headline)

Leakage-free cross-validation (repeated 50/50, 5-fold, 10-fold, LOO; 400 repeats each),
independently reimplemented and confirmed. Selection rules tried: keep-positive-train-skill,
**drop-only-negative**, top-K, and an a-priori fundamental set {fouls, corners, sot, both_sot}.

**On the metric that decides standing — TOTAL score — `answer everything` beats every rule:**

| rule | OOS total-score win-rate vs answer-all |
|---|---|
| keep-positive | 0.21 |
| drop-only-negative | 0.23 |
| top-3 / top-5 | 0.26 / 0.28 |
| fixed fundamental set | 0.39 |

All **below 0.5** → no rule reliably beats answering everything. The per-question "wins" some
rules show are an artifact of **answering almost nothing** (top-3 answers ~1.5 of 35
questions/fold), forfeiting most of the positive-edge points.

**Variance — the opposite of the hypothesis:** concentrating *lowers raw std only because the
total is tiny*, but **raises the coefficient of variation** (risk per expected point):
answer-all CV **1.22** vs concentrated **1.96–2.18**. Controlled bootstrap: CV rises from
**1.13 (50 questions) → 2.52 (10 questions)**.

❌ **Hypothesis half #3 (concentrate → win more, less variance) — rejected.**

### Why (theory, verified)
RBP total `S = Σ (crowd_brierᵢ − your_brierᵢ)` is a **sum of independent per-question bets**.
With per-question edge mean μ>0, std σ: `E[S]=nμ`, `SD[S]=√n·σ`, so
**CV = σ/(μ√n) ∝ 1/√n** — relative risk *falls* as you answer more. Mean grows linearly,
noise only as √n (classic diversification). **Exception:** a question with μ≤0 lowers E[S]
while adding variance, so genuinely **negative-edge** questions *should* be dropped — the
greedy edge-first CV bottoms near N≈18 then rises as `cards/penalty/offsides/result` are
forced in. **But** at n=176 we cannot reliably identify which families are negative (their
negatives are noise), which is exactly why drop-only-negative also loses OOS.

---

## 5. Calibration corrections also overfit
Per-family bias exists (`cards`: we say YES +0.23 too often; `corners`: we *under*-call by
−0.14). But learning per-family corrections on 176 points and applying OOS makes Brier
**worse**: logit-shift −0.0275, shrink-to-rate −0.0259; only a global shrink-to-0.5 is
~neutral (we're already roughly calibrated, 30–70% band). **Same lesson: too little data to
fit anything category-specific.**

---

## 6. What to actually do

1. **Answer every *no-market* question; reconsider the *market* ones.** Two kinds of
   selection must not be confused:
   - ❌ *Noisy per-family selection* (pick the families that looked best on past data) —
     **fails OOS**, overfits, raises variance. Don't do this.
   - ✅ *Structural selection* (drop families where a sharp public market exists, so the crowd
     ≈ an efficient bookmaker and our RBP there is ≤ 0) — **defensible and potentially large.**
     `field_sensitivity.py`: if the crowd is as sharp as Bet365 on who-wins (it beats our model
     by 0.014 Brier), the 92 market-family questions cost ~129 RBP; **skipping them** (keep the
     84 no-market stat questions) is a big gain — *if* abstaining is neutral (not penalised).
   This is the correct, vindicated form of the user's "skip who-wins" instinct. **Caveat:** the
   WC crowd's sharpness is unmeasured (could be weaker than Bet365 → smaller gain → answer-all
   becomes fine) and the abstention rule is unknown. Net: definitely answer all no-market stats;
   treat who-wins / totals / player-props as drop-candidates pending the abstention rule.
2. **The real alpha is model quality, not category selection** — specifically on the
   fundamentals-driven stat families where the user's thesis genuinely holds and the crowd is
   weakest: **`fouls` (proven), `corners`, shots.** Style/strength priors (pressing &
   physicality → fouls/cards; width & attacking volume → corners; both ~Poisson-stable and
   far more frequent than goals) are the lever. That's where to spend modeling effort.
3. **Fix, don't drop, the negative-edge families** (`cards`, `cards_compare`, `pen_or_red`):
   they're value-destroying today, but dropping them loses OOS — re-examine the model (cards
   are referee-driven → high irreducible variance; consider regressing harder to base rate).
4. **Don't deploy per-category selection or per-family calibration** until n is much larger.
   Re-run this battery as the tournament settles more questions (n grows on match days).

## 6.5 Deployment priorities (`priority.py` — fuses both datasets)

| action | families | why |
|---|---|---|
| **PRIORITISE** | `sot`, `both_sot`, `fouls`, `corners` | no sharp market + historically predictable |
| **#1 HEADROOM** | `sot` | most predictable category (+0.025 large-sample) but our competition skill is ~0 (+0.003) — we're capturing almost none of the available edge |
| **FIX (don't drop)** | `cards` (−0.062), `cards_compare` (−0.054), `pen_or_red` (−0.034) | predictable (`cards` +0.012 hist) yet we're *anti*-predictive → model bug/miscalibration |
| **MONITOR** | `offsides` | no market, but no historical data to confirm a model works |
| **SKIP** | `result`, `total_goals`, `btts`, `ht_winning`, `team_scores`, player props | sharp market → crowd ≈ efficient → our "edge" is illusory (player props = the base-rate freebie) |

*Note `corners` is genuine but venue-sensitive (independent reviewer: +0.011 only with a
home/away venue split, ~0 in a pooled model) → model corners with explicit venue rates.*

## 7. Key open question
**The abstention mechanic is unknown and decisive.** If an unanswered question scores 0
(neutral), dropping negative-edge families would help *if we could identify them*; if it
scores the crowd-average, dropping hurts. Either way the robust action is the same — **answer
everything** — but resolving this (from the official rules / support) would sharpen #3.

---

## Files
- `pull_data.py` — pull & cache real settled results (`data/settled.json`, 176 recs).
- `analyze.py` — per-family / per-bucket Brier, skill, edge, bootstrap CIs → `data/summary.json`.
- `selection_backtest.py` — OOS selection CV (random split) → `data/backtest.json`.
- `genuine_edge.py` — freebie-stripped edge + core-claim difference test.
- `calibration.py` — calibration diagnosis + OOS correction backtest → `data/calibration.json`.
- `historical_logit.py` — **large-sample** walk-forward predictability test, online logistic,
  17,784 club matches → `data/historical_logit.json`. *(Run this; it's the powered test.)*
- `historical_backtest.py` — generative-Poisson version (has the documented foul-grid bug);
  kept for the diagnostic + result-vs-bookmaker comparison.
- `RESEARCH_LOG.md` — cycle-by-cycle journal.
- `data/settled_snapshot176.json` — frozen competition dataset for reproducibility.
- `data/hist/` — cached football-data.co.uk CSVs.

*Independently reproduced and adversarially verified by a 4-agent review (reproduce / OOS-steelman / methodology-critic / theory+literature); all four agreed on the core conclusion.*
