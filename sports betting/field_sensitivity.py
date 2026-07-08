"""Reconciling the two selection results under a REALISTIC (non-coinflip) crowd.

The OOS test rejected selecting by NOISY per-family skill. But there's a different,
STRUCTURAL selection: drop families where a sharp public MARKET exists (so the crowd
is strong and our RBP there is <=0), keep the no-market stat families.

We can't observe the crowd, so we model it and sweep:
  - market families  : crowd beats us by M Brier  -> per-q RBP = -M  (we lose).
    (historical Bet365 beats our RESULT model by ~0.014, so M~0.014 if crowd~market.)
  - no-market families: crowd skill over coinflip = s  -> crowd_brier = 0.25 - s;
    per-q RBP = (0.25 - s) - our_brier  (we win where we beat a weak crowd).

Compare expected TOTAL RBP: ANSWER-ALL vs ANSWER-NO-MARKET-ONLY, using our REAL briers.
This shows WHEN structurally skipping who-wins/markets is +EV (it is, for any M>0 under
abstain=0/neutral) and by how much.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MARKET = {"result", "ht_winning", "ht_tied", "team_scores", "total_goals",
          "team_total_goals", "btts_compound", "half_vs_half_goals", "goals_compare_half",
          "first_goal_2h", "compound_first_and_2h", "penalty", "pen_or_red",
          "player_sot", "player_involve", "player_goalscorer"}
NOMARKET = {"sot", "both_sot", "corners", "fouls", "offsides", "cards", "cards_compare"}


def main():
    recs = json.load(open(os.path.join(DATA, "settled.json")))
    mk = [r for r in recs if r["family"] in MARKET]
    nm = [r for r in recs if r["family"] in NOMARKET]
    print(f"market-family questions: {len(mk)}   no-market: {len(nm)}\n")

    print("Expected TOTAL RBP (×100 Brier units), real our-briers, modelled crowd:")
    print(f"{'M (mkt crowd edge)':>18} {'s (nomkt crowd skill)':>22} "
          f"{'ALL':>8} {'NO-MKT only':>12} {'gain from skipping mkt':>22}")
    for M in (0.0, 0.007, 0.014, 0.025):
        for s in (0.0, 0.005, 0.010):
            rbp_mk = sum(-M for _ in mk)                       # we lose M each on market Qs
            rbp_nm = sum(((0.25 - s) - r["brier"]) for r in nm)
            all_rbp = (rbp_mk + rbp_nm) * 100
            nomkt_rbp = rbp_nm * 100
            print(f"{M:>18.3f} {s:>22.3f} {all_rbp:>8.1f} {nomkt_rbp:>12.1f} "
                  f"{nomkt_rbp-all_rbp:>+22.1f}")
    print("\nReading: 'gain from skipping market families' = -ALL_market_RBP = +M*n_market.")
    print("If the WC crowd is market-sharp (M~0.014, ~Bet365 vs our model on RESULT),")
    print(f"skipping the {len(mk)} market questions saves ~{0.014*len(mk)*100:.0f} RBP — IF abstaining")
    print("is neutral (0) and not penalised. This is STRUCTURAL selection (market vs no-market),")
    print("NOT the noisy per-family selection that failed OOS.")
    print("\nCAVEAT: M for the WC crowd is unmeasured (crowd may be weaker than Bet365 -> M smaller,")
    print("even ~0). And the abstention rule is unknown. So this is the conditional case for")
    print("skipping who-wins, not a proven +EV move.")


if __name__ == "__main__":
    main()
