"""Pull the REAL Probability-Cup settled results and cache them locally.

Joins each settled prediction to its question `family` (via the bot's
forecasts.json, keyed by market_id) and recovers the realized binary outcome
from (probability_submitted, brier_score):  brier = (p/100 - y)^2, y in {0,1}.

Output: data/settled.json  (list of records) + data/open.json (unsettled).
Re-run any time; it overwrites the cache. Read-only against the live API.
"""
import json
import math
import os
import sys

BOT = "/Users/rainchai/probcup-bot"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, BOT)

from probcup import config  # noqa: E402  (loads .env -> API key)
from probcup.api import SportsPredict, items  # noqa: E402


def family_map():
    """market_id -> family, from the bot's forecasts.json."""
    fc = json.load(open(os.path.join(BOT, "state", "forecasts.json")))
    return {mid: rec.get("family", "?") for mid, rec in fc.items()}


def recover_outcome(prob_int, brier):
    """Return (y, our_brier_recomputed). y in {0,1}; None if inconsistent."""
    p = prob_int / 100.0
    b1 = (p - 1.0) ** 2   # if y == 1
    b0 = (p - 0.0) ** 2   # if y == 0
    # pick the outcome whose implied brier matches the reported one
    if abs(b1 - brier) <= abs(b0 - brier):
        return 1, b1
    return 0, b0


def main():
    os.makedirs(DATA, exist_ok=True)
    api = SportsPredict()
    fam = family_map()

    settled_raw = items(api.results())
    preds_raw = items(api.predictions())

    settled = []
    bad = 0
    for r in settled_raw:
        mid = r.get("market_id")
        prob = r.get("probability_submitted", r.get("probability"))
        brier = r.get("brier_score")
        if prob is None or brier is None:
            bad += 1
            continue
        y, b_chk = recover_outcome(prob, brier)
        # sanity: reported brier should match one of the two candidates closely
        consistent = abs(b_chk - brier) < 1e-4
        settled.append({
            "market_id": mid,
            "family": fam.get(mid, "?"),
            "question": r.get("question", ""),
            "prob": prob,
            "brier": brier,
            "outcome": y,
            "consistent": consistent,
        })

    opens = []
    for r in preds_raw:
        if r.get("market_status") == "settled":
            continue
        mid = r.get("market_id")
        opens.append({
            "market_id": mid,
            "family": fam.get(mid, "?"),
            "question": r.get("question", ""),
            "prob": r.get("probability"),
        })

    json.dump(settled, open(os.path.join(DATA, "settled.json"), "w"), indent=1)
    json.dump(opens, open(os.path.join(DATA, "open.json"), "w"), indent=1)

    n_unknown_fam = sum(1 for s in settled if s["family"] == "?")
    n_inconsistent = sum(1 for s in settled if not s["consistent"])
    print(f"settled cached : {len(settled)}  (skipped {bad} missing prob/brier)")
    print(f"  unknown family : {n_unknown_fam}")
    print(f"  brier-inconsistent (non-binary?) : {n_inconsistent}")
    print(f"open cached    : {len(opens)}")
    print(f"-> {DATA}/settled.json , open.json")


if __name__ == "__main__":
    main()
