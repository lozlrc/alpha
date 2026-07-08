# sports betting — category-selection alpha

Does forecasting only the categories we're best at (corners/shots/fouls, skipping who-wins)
win more with less variance in the Jump Trading Probability Cup?

**Short answer:** No — and the reasoning matters. → read **[RESULTS.md](RESULTS.md)**.

- Stats are **not** more predictable than who-wins (they're comparable; SoT is most predictable).
- Team-stats are where edge lives only because **there's no sharp market, so the crowd is weak**
  (`edge ≈ predictability − crowd_competence`).
- **Concentrating loses out-of-sample and raises variance.** RBP is additive across independent
  questions → answer everything; invest model effort in the no-market stat families.

## Reproduce (uses the probcup-bot venv)
```
PY=/Users/rainchai/probcup-bot/.venv/bin/python
cd "alpha/sports betting"
$PY pull_data.py            # real competition data -> data/settled.json (176 settled)
$PY analyze.py             # per-category Brier / skill / CIs
$PY genuine_edge.py        # edge stripped of base-rate freebie + core-claim test
$PY selection_backtest.py  # out-of-sample selection vs answer-all
$PY calibration.py         # calibration diagnosis + OOS correction backtest
$PY historical_logit.py    # LARGE-SAMPLE predictability test (17,784 club matches)
$PY robustness.py          # window / era / league robustness
```

- `RESULTS.md` — full writeup + verdict.
- `RESEARCH_LOG.md` — cycle-by-cycle journal.
- `data/` — cached competition data, summaries, and football-data.co.uk CSVs.

Findings independently reproduced and adversarially verified by multi-agent reviews.
