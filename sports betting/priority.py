"""Actionable synthesis: which competition families to PRIORITISE for modelling.

Fuses three signals per family:
  - COMP skill   : our real competition skill vs coin-flip (data/summary.json, small n).
  - HIST predict : large-sample achievable predictability over base rate
                   (data/historical_logit.json, n~17.8k) -- is the signal real?
  - MARKET       : does a liquid sharp market exist? (crowd is strong there -> low edge).

Priority = predictable (HIST>0) AND no sharp market AND not value-destroying in COMP.
Headroom  = HIST clearly predictable but our COMP skill is low -> model is leaving edge on
            the table (improve the model). Flagged for action.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# competition family -> (historical analog key or None, has_sharp_market)
MAP = {
    "fouls":         ("FOULS_h>a",     False),
    "corners":       ("CORNERS_h>a",   False),
    "sot":           ("SOT_h>a",       False),
    "both_sot":      ("SOT_h>a",       False),
    "cards":         ("CARDS_over3.5", False),
    "cards_compare": ("CARDS_h>a",     False),
    "offsides":      (None,            False),   # no historical data, but no market either
    "pen_or_red":    (None,            False),
    "team_scores":   (None,            True),    # team-to-score ~ has market
    "total_goals":   ("GOALS_over2.5", True),
    "btts_compound": (None,            True),
    "result":        ("RESULT_homewin", True),
    "ht_winning":    ("RESULT_homewin", True),
    "player_sot":    (None,            True),    # player props have markets
    "player_involve": (None,           True),
    "player_goalscorer": (None,        True),
}


def main():
    comp = {o["key"]: o for o in json.load(open(os.path.join(DATA, "summary.json")))["families"]}
    hist = json.load(open(os.path.join(DATA, "historical_logit.json")))

    rows = []
    for fam, (hk, market) in MAP.items():
        if fam not in comp:
            continue
        c = comp[fam]
        comp_skill = c["skill"]          # vs coin-flip
        comp_n = c["n"]
        hist_skill = hist[hk]["skill"] if hk and hk in hist else None
        # priority logic
        predictable = (hist_skill is not None and hist_skill > 0.008) or fam == "offsides"
        if market:
            verdict = "skip (sharp market -> crowd captures it)"
            pr = 0
        elif comp_skill < -0.02:
            verdict = "fix model (negative edge today)"
            pr = 1
        elif predictable:
            # headroom: predictable historically but weak competition skill
            head = (hist_skill is not None and hist_skill > 0.012 and comp_skill < 0.02)
            verdict = "PRIORITISE" + ("  [headroom: model under-capturing]" if head else "")
            pr = 3 if not head else 4
        else:
            verdict = "monitor (no market, unclear signal)"
            pr = 2
        rows.append((pr, fam, comp_n, comp_skill, hist_skill, market, verdict))

    rows.sort(key=lambda r: (-r[0], -(r[3] or 0)))
    print(f"{'family':15} {'compN':>5} {'compSkill':>9} {'histPred':>8} {'mkt':>4}  action")
    print("-" * 78)
    for pr, fam, n, cs, hs, mk, v in rows:
        hsf = f"{hs:+.4f}" if hs is not None else "   n/a"
        print(f"{fam:15} {n:>5} {cs:>+9.4f} {hsf:>8} {'yes' if mk else 'no':>4}  {v}")

    print("\nLegend: compSkill = our real competition skill vs coin-flip (small n, overstates);")
    print("        histPred = large-sample achievable predictability over base rate;")
    print("        mkt = a liquid sharp market exists (crowd strong -> little edge for us).")
    print("\nTop modelling priorities = no market + historically predictable:")
    tops = [r[1] for r in rows if r[0] >= 3]
    print("  " + ", ".join(tops))


if __name__ == "__main__":
    main()
