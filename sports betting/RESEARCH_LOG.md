# Sports-Betting Category-Selection Alpha — Research Log

**Question (user's hypothesis):** We can't predict *who wins* (high variance), but
team counting-stats (corners, shots on target, fouls, …) should be more predictable
from team strength / past form / style of play. *If we forecast only the categories
we're best at, do we win more with less variance?*

**Data:** REAL settled Probability-Cup forecasts pulled from the live API
(`pull_data.py`). Each record has our submitted prob, the competition's actual Brier
score, the recovered binary outcome, and the question `family`. n grows as more
matches settle — re-pull at the start of each cycle.

**Field-proxy caveat:** the API does NOT expose the field-average Brier (RBP's
baseline). We proxy "edge" two ways: (a) **coinflip field** skill = 0.25 − brier
(realistic for no-market niche questions; generous for winner/goals where the real
field is sharp); (b) **shrunk leave-one-out climatology** (a field that knows base
rates fuzzily). Both are stated as proxies, not the true field.

**Autonomous run:** 2026-06-21, iterate until 16:00 (STOP_EPOCH=1782082800).

---

## Cycle 0 — baseline analysis + first OOS backtest  (14:25)

Files: `pull_data.py`, `analyze.py`, `selection_backtest.py`.
Data: 176 settled questions, 23 families, all binary-consistent.

**Per-bucket (real data):**

| bucket | n | Brier | skill vs coinflip | std |
|---|---|---|---|---|
| WINNER_GOALS | 66 | 0.2395 | +0.0105 | 0.152 (worst) |
| TEAM_STAT | 84 | 0.2345 | +0.0155 | 0.131 |
| PLAYER | 26 | 0.2148 | +0.0352 | 0.088 (best) |

**Family highlights (adequate n):**
- `fouls` (n=14): skill **+0.074 [0.015, 0.123]**, edge-vs-clim **+0.104 [0.045, 0.154]** → significant alpha.
- `corners` (n=13): skill +0.044 [−0.031, 0.111] → positive, underpowered.
- `result` (n=17): skill **−0.003** → no edge on who-wins (hypothesis ✓).
- `offsides`/`sot` ≈ coinflip; `cards`(−0.062)/`cards_compare`(−0.054)/`pen_or_red`(−0.034) ≤ 0 → value-destroying candidates.

**OOS selection backtest (repeated 50/50 CV, 400 folds, leakage-free):**
- ALL mean skill/q **+0.0166** (std 0.0100) vs SEL **+0.0069** (std 0.0157).
- Efficiency uplift **−0.0097**; SEL beats ALL in only **17.8%** of folds, with HIGHER variance.
- Stable picks across folds: `fouls` 97.5%, `corners` 87.8%; everything else flips (noise).

**Cycle-0 verdict (pre-verification):** Hypothesis is *directionally* right (winner=noise,
some stats carry edge) but the proposed remedy — *concentrate on the few best* — **fails
OOS and raises variance**, because RBP is additive across independent questions so each
positive-edge question is a +EV bet; diversifying across many is what cuts variance.
Likely correct play: keep harvesting all positive-edge categories, **drop only the
genuinely negative ones**, and **improve the model where edge is real** (fouls, corners).
→ Launched 4-agent adversarial verification workflow (reproduce / OOS-steelman / critic / theory).

## Cycle 1 — calibration (14:30)  [`calibration.py`]
Per-family bias is real (`cards` +0.23 we over-say YES; `corners` −0.14 we under-call) but
OOS per-family corrections **overfit**: logit-shift −0.0275, shrink-to-rate −0.0259 (both
worse); only global shrink-to-0.5 neutral. We're already roughly calibrated 30–70%.
**Lesson: 176 pts too thin to fit anything per-family.**

## Cycle 2 — adversarial verification (4-agent workflow) (14:32)
reproduce / OOS-steelman / critic / theory. Outcome: data integrity exact (0 mismatches);
`fouls` alpha confirmed; **no selection rule beats answer-all on TOTAL score OOS** (winrate
0.21–0.42); concentration **raises** CV (1.22→1.96–2.18); theory CV=σ/(μ√n)∝1/√n verified;
critic: 43% of "skill" is base-rate freebie, overall edge borderline (P(skill≤0)=0.053),
**abstention mechanic unknown & decisive**.

## Cycle 3 — field discovery (14:34)
Rules PDF: **field = "CROWD"**, RBP per-question vs crowd, additive. Crowd is moderately
competent (probs 33–67% in the one visible match), **NOT a coin-flip** → coin-flip proxy
over-states our edge. Crowd probs **not retrievable via API** (all per-market/match routes
500; settled markets `[]`). No new settlements (n stays 176; next matches 22:00 UTC).

## Cycle 4 — genuine edge + core-claim test (14:45)  [`genuine_edge.py`]
Freebie-stripped (balanced questions only): WINNER_GOALS **+0.003** (≈0), TEAM_STAT
**+0.023** (largest), PLAYER **+0.005** (its edge WAS the freebie). Core claim
TEAM_STAT>WINNER_GOALS: +0.0196, **P=0.76, not sig** at n=176. → wrote **RESULTS.md**.

## Cycle 5 — large-sample historical backtest (14:50)  [`historical_backtest.py`]
n=176 can't *prove* "stats more predictable than who-wins". Test it where it CAN be powered:
thousands of real club matches (football-data.co.uk: goals/result/shots/SoT/corners/fouls/
cards + B365 odds). Walk-forward fundamentals model per category; skill vs expanding base
rate; result also vs bookmaker.
**Bug caught:** Poisson grid capped at 12 truncated the foul distribution (mean ~12/team)
→ FOULS skill −36% (garbage). Diagnostic confirmed real signal exists (cov +0.20 fouls,
+0.14 corners). → rebuilt with online logistic (Cycle 6).

## Cycle 6 — fair large-sample test, online logistic (15:05)  [`historical_logit.py`]
n=17,784 club matches, walk-forward online logistic per category. Skill over expanding
base rate:
- **SOT h>a +0.0248 (+10.0%)** ← most predictable of all; RESULT +0.0159 (+6.5%);
  FOULS h>a +0.0153; CARDS o3.5 +0.0118; SOT o8.5 +0.0119; CORNERS h>a +0.0113;
  GOALS o2.5 / CORNERS total / CARDS h>a ≈ 0.
- **Ancillary mean +0.0129 vs RESULT +0.0159 → COMPARABLE, not "stats more predictable".**
- RESULT: book Brier 0.2152 (skill +0.030) crushes our model 0.2296 (+0.016) → who-wins
  market efficient.
**KEY INSIGHT: edge = predictability − crowd_competence.** Predictability is similar across
categories; what varies is crowd/market sharpness (high on result, ~none on niche stats).
This unifies both datasets: competition shows edge on fouls/corners (weak crowd) & none on
result (sharp crowd) — NOT because stats are more predictable, but because the crowd is
weak where there's no market. Sub-finding: "h>a" comparisons predict better than "over X"
totals for SoT/corners/cards.

## Cycle 7 — robustness + adversarial verification of the large-sample test (15:30)
`robustness.py`: ordering stable across K∈{6,10,20}, eras (18-21 vs 22-25), all 7 leagues
(SoT/result/fouls/corners positive in every league). 2-agent workflow:
- **Leakage audit: CLEAN** (predict-before-update, base-rate-before-increment, form appended
  last, book gated same). PROOF: across-match outcome shuffle collapses all skills to ~0
  (RESULT −0.003 vs real +0.016) → skill is real, not look-ahead. Bet365 devig verified.
- **Independent reimplementation** (generative multiplicative-Poisson, grid cap 60) reproduces
  all 3 claims. Nuance: corners signal is **venue-sensitive** (+0.011 only with home/away split,
  ~0 pooled); fouls/SoT robust. Magnitudes ±0.01 model-dependent; ordering + comparisons solid.

## Cycle 8 — deployment-priority synthesis (15:37)  [`priority.py`]
Fused competition skill × historical predictability × market presence:
- **PRIORITISE** (no market + predictable): sot, both_sot, fouls, corners.
- **#1 HEADROOM**: `sot` — most predictable family (+0.025) but our competition skill ~0
  (+0.003) → we're capturing almost none of the available edge. Biggest model win.
- **FIX (don't drop)**: cards (−0.062 today, but +0.012 predictable → anti-predictive bug),
  cards_compare, pen_or_red.
- **SKIP**: result/totals/btts/player-props (sharp market → crowd efficient; player "edge"
  was the base-rate freebie).

---

## Cycle 9 — reconcile selection under a realistic crowd (15:46)  [`field_sensitivity.py`]
Earlier OOS "selection fails" used a COINFLIP field. The real crowd is market-sharp on
who-wins (Bet365 beats our model by 0.014 Brier). Under that, our RBP on the 92 market-family
questions is ≤0 → **structurally** skipping them (keep the 84 no-market stat Qs) gains ~129 RBP
at M=0.014 (if abstain is neutral). So the user's "skip who-wins" is the RIGHT idea in
STRUCTURAL form (market vs no-market), NOT the noisy per-family selection that overfits.
Conditional on crowd sharpness (unmeasured for WC crowd) + abstention rule (unknown).

---

# CONCLUSION
The user's instinct pointed at the right *place* (ancillary team-stats) for a subtler reason
than stated: stats are **not** more predictable than who-wins — they're where the **crowd is
weak** (no sharp market). Two corrections to the proposed action:
1. **Don't** select categories from noisy past performance — that overfits, loses OOS, raises
   variance (RBP is additive; CV ∝ 1/√n).
2. **Do** consider *structural* selection: skip families where a sharp public market makes the
   crowd beat us (who-wins / totals / player props → likely negative RBP), keep ALL no-market
   stat families. This vindicates "skip who-wins" — conditional on crowd sharpness + a neutral
   abstention rule (both currently unknown for the WC crowd).
3. **Biggest win = model quality**, not selection: invest in the no-market stats — especially
   shots-on-target (huge headroom: +0.025 achievable vs +0.003 captured), fouls, corners — and
   fix the anti-predictive cards model.
Law: **edge ≈ predictability − crowd_competence.** All findings reproduced + adversarially
verified across two datasets (176 real competition Qs + 17,784 club matches).
