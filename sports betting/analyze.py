"""Category-selection alpha — core analysis on REAL settled Probability-Cup data.

Question: which question CATEGORIES do we forecast best, and would forecasting
ONLY those win more with less variance?

Two layers:
  L1 PREDICTABILITY (no field assumption) -- per-family / per-bucket Brier, std,
     skill-vs-coinflip, calibration. Tests "stats easier than winner".
  L2 EDGE (field proxy) -- RBP edge = field_Brier - our_Brier. The field-average
     Brier is NOT exposed by the API, so we proxy it with a *leave-one-out
     climatology* field (a competitor who knows each category's base rate but
     nothing match-specific). For no-market niche stats the real field ~ this;
     for liquid markets (result/totals) the real field is SHARPER, so edge there
     is OVER-stated -- flagged explicitly.

Run: ../.venv/bin/python analyze.py     (uses repo .venv for numpy-free stdlib)
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# --- hypothesis buckets ------------------------------------------------------
# WINNER_GOALS: outcome / scoreline questions (user: "can't predict who wins")
# TEAM_STAT   : countable team stats (user: "corners / shots easier")
# PLAYER      : individual player props
BUCKET = {
    "result": "WINNER_GOALS", "ht_winning": "WINNER_GOALS", "ht_tied": "WINNER_GOALS",
    "team_scores": "WINNER_GOALS", "total_goals": "WINNER_GOALS",
    "team_total_goals": "WINNER_GOALS", "btts_compound": "WINNER_GOALS",
    "half_vs_half_goals": "WINNER_GOALS", "goals_compare_half": "WINNER_GOALS",
    "first_goal_2h": "WINNER_GOALS", "compound_first_and_2h": "WINNER_GOALS",
    "penalty": "WINNER_GOALS", "pen_or_red": "WINNER_GOALS",
    "sot": "TEAM_STAT", "both_sot": "TEAM_STAT", "corners": "TEAM_STAT",
    "fouls": "TEAM_STAT", "offsides": "TEAM_STAT", "cards": "TEAM_STAT",
    "cards_compare": "TEAM_STAT",
    "player_sot": "PLAYER", "player_involve": "PLAYER",
    "player_goalscorer": "PLAYER",
}
UNIFORM_BRIER = 0.25  # always-50% on a binary question


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def bootstrap_ci(xs, fn=mean, B=2000, lo=2.5, hi=97.5):
    """Deterministic bootstrap (LCG, no Math.random) for a CI on fn(xs)."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    n = len(xs)
    seed = 1234567
    stats = []
    for _ in range(B):
        samp = []
        for _ in range(n):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            samp.append(xs[seed % n])
        stats.append(fn(samp))
    stats.sort()
    return (stats[int(lo / 100 * B)], stats[int(hi / 100 * B)])


SHRINK_K = 4.0  # pseudo-counts of a 0.5-anchored prior for the climatology field


def climatology_edge(records):
    """Field proxies, per record:
      skill  = 0.25 - our_brier               (field = always 50%)
      edge   = field_brier - our_brier  where field = leave-one-out base rate
               of the family, SHRUNK toward 0.5 with SHRINK_K pseudo-counts.
    Shrinkage kills the small-n LOO pathology (n=2 -> loo rate = the other point)
    and models a field that knows base rates only fuzzily. For symmetric no-market
    questions the shrunk field stays ~0.5 (==coinflip, realistic); for skewed
    families it moves toward the true rate (removes the base-rate freebie)."""
    byfam = defaultdict(list)
    for r in records:
        byfam[r["family"]].append(r)
    for fam, rs in byfam.items():
        tot = sum(x["outcome"] for x in rs)
        n = len(rs)
        for r in rs:
            others_sum = tot - r["outcome"]
            others_n = n - 1
            loo_rate = (others_sum + SHRINK_K * 0.5) / (others_n + SHRINK_K)
            r["field_brier"] = (loo_rate - r["outcome"]) ** 2
            r["edge"] = r["field_brier"] - r["brier"]
            r["skill"] = UNIFORM_BRIER - r["brier"]


def table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))))


def fmt(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, float) and not math.isnan(x) else "  -"


def summarize(records, keyfn, label):
    groups = defaultdict(list)
    for r in records:
        groups[keyfn(r)].append(r)
    out = []
    for k, rs in groups.items():
        briers = [r["brier"] for r in rs]
        skills = [r["skill"] for r in rs]
        edges = [r["edge"] for r in rs]
        rate = mean([r["outcome"] for r in rs])
        mb, sb = mean(briers), std(briers)
        msk = mean(skills)
        me, se = mean(edges), std(edges)
        sharpe = me / se if se > 1e-9 else float("nan")
        skill_ci = bootstrap_ci(skills)
        edge_ci = bootstrap_ci(edges)
        # "sig" = both proxies' 95% CIs are strictly > 0  (edge is real & robust)
        sig = (skill_ci[0] > 0 and edge_ci[0] > 0)
        out.append({
            "key": k, "n": len(rs), "brier": mb, "brier_std": sb,
            "skill": msk, "skill_ci": skill_ci, "base_rate": rate,
            "edge": me, "edge_std": se, "edge_sharpe": sharpe,
            "edge_ci": edge_ci, "sig": sig,
        })
    out.sort(key=lambda d: d["skill"], reverse=True)
    print(f"\n================  {label}  ================")
    rows = [[
        o["key"], o["n"], fmt(o["brier"]), fmt(o["brier_std"]),
        fmt(o["skill"]), f"[{fmt(o['skill_ci'][0],3)},{fmt(o['skill_ci'][1],3)}]",
        fmt(o["edge"]), f"[{fmt(o['edge_ci'][0],3)},{fmt(o['edge_ci'][1],3)}]",
        fmt(o["base_rate"], 2), "YES" if o["sig"] else "",
    ] for o in out]
    table(rows, ["category", "n", "brier", "bstd", "skillVs.5", "skill95CI",
                 "edgeClim", "edge95CI", "baseRt", "sig>0"])
    return out


def main():
    records = json.load(open(os.path.join(DATA, "settled.json")))
    climatology_edge(records)

    print(f"REAL settled questions analysed: {len(records)}")
    print(f"overall Brier            : {fmt(mean([r['brier'] for r in records]))}")
    print(f"overall skill vs coinflip: {fmt(UNIFORM_BRIER - mean([r['brier'] for r in records]))}")
    print(f"overall edge vs climatology: {fmt(mean([r['edge'] for r in records]))}")

    bucket_sum = summarize(records, lambda r: BUCKET.get(r["family"], "OTHER"),
                           "BY BUCKET (hypothesis test)")
    fam_sum = summarize(records, lambda r: r["family"], "BY FAMILY (detailed)")

    # dump machine-readable for the writeup / verification agents
    def clean(o):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}
    json.dump({"buckets": [clean(o) for o in bucket_sum],
               "families": [clean(o) for o in fam_sum],
               "n_total": len(records),
               "overall_brier": mean([r["brier"] for r in records])},
              open(os.path.join(DATA, "summary.json"), "w"), indent=1)
    print(f"\n-> wrote {DATA}/summary.json")


if __name__ == "__main__":
    main()
