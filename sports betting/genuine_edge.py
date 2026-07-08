"""Genuine forecasting edge, stripped of the base-rate freebie + the user's
core claim tested directly.

The critic flagged that skill = 0.25 - brier is base-rate-confounded: correctly
betting that a rare prop won't happen scores "skill" without real forecasting.
43% of our summed skill came from 13% of records (extreme-base-rate props).

So here we:
  1. Recompute skill on BALANCED questions only (family base rate in [0.35,0.65]),
     where there's no easy base-rate lean and the crowd can't trivially win -- the
     cleanest estimate of genuine match-specific edge. Per bucket, with CI.
  2. DIRECT TEST of the hypothesis "stats more predictable than who-wins":
     bootstrap the DIFFERENCE  skill(TEAM_STAT) - skill(WINNER_GOALS).
  3. Freebie decomposition: share of total skill from extreme-base-rate records.
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

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
    "player_sot": "PLAYER", "player_involve": "PLAYER", "player_goalscorer": "PLAYER",
}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def boot_ci(xs, B=4000, lo=2.5, hi=97.5):
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    n, seed, out = len(xs), 987654321, []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            s += xs[seed % n]
        out.append(s / n)
    out.sort()
    return (out[int(lo / 100 * B)], out[int(hi / 100 * B)])


def boot_diff_ci(a, b, B=4000):
    """bootstrap CI for mean(a) - mean(b) (independent resamples)."""
    na, nb, seed, out = len(a), len(b), 555, []
    for _ in range(B):
        sa = 0.0
        for _ in range(na):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            sa += a[seed % na]
        sb = 0.0
        for _ in range(nb):
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            sb += b[seed % nb]
        out.append(sa / na - sb / nb)
    out.sort()
    frac_gt0 = sum(1 for x in out if x > 0) / B
    return (out[int(0.025 * B)], out[int(0.975 * B)], frac_gt0)


def main():
    recs = json.load(open(os.path.join(DATA, "settled.json")))
    for r in recs:
        r["skill"] = 0.25 - r["brier"]
        r["bucket"] = BUCKET.get(r["family"], "OTHER")

    # family base rates -> tag balanced vs lopsided
    byfam = defaultdict(list)
    for r in recs:
        byfam[r["family"]].append(r["outcome"])
    rate = {f: mean(v) for f, v in byfam.items()}
    for r in recs:
        r["balanced"] = 0.35 <= rate[r["family"]] <= 0.65

    print("=== 1. GENUINE EDGE on BALANCED questions (base rate 0.35-0.65) ===")
    print("    (strips the base-rate freebie; closest to true match-specific skill)")
    print(f"{'bucket':14} {'n_bal':>5} {'skill':>8} {'95% CI':>20} {'| all_n':>7} {'all_skill':>9}")
    for bk in ["WINNER_GOALS", "TEAM_STAT", "PLAYER"]:
        allb = [r["skill"] for r in recs if r["bucket"] == bk]
        bal = [r["skill"] for r in recs if r["bucket"] == bk and r["balanced"]]
        ci = boot_ci(bal)
        print(f"{bk:14} {len(bal):>5} {mean(bal):>+8.4f} "
              f"[{ci[0]:>+7.4f},{ci[1]:>+7.4f}] | {len(allb):>5} {mean(allb):>+9.4f}")
    allbal = [r["skill"] for r in recs if r["balanced"]]
    ci = boot_ci(allbal)
    print(f"{'ALL':14} {len(allbal):>5} {mean(allbal):>+8.4f} [{ci[0]:>+7.4f},{ci[1]:>+7.4f}]")

    print("\n=== 2. CORE CLAIM: is TEAM_STAT more predictable than WINNER_GOALS? ===")
    for tag, filt in [("all questions", lambda r: True),
                      ("balanced only", lambda r: r["balanced"])]:
        ts = [r["skill"] for r in recs if r["bucket"] == "TEAM_STAT" and filt(r)]
        wg = [r["skill"] for r in recs if r["bucket"] == "WINNER_GOALS" and filt(r)]
        lo, hi, fg = boot_diff_ci(ts, wg)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not sig (CI crosses 0)"
        print(f"  [{tag:14}] TEAM_STAT({len(ts)}) {mean(ts):+.4f} - "
              f"WINNER_GOALS({len(wg)}) {mean(wg):+.4f} = {mean(ts)-mean(wg):+.4f}"
              f"  95%CI[{lo:+.4f},{hi:+.4f}]  P(>0)={fg:.2f}  -> {sig}")

    print("\n=== 3. BASE-RATE FREEBIE decomposition ===")
    extreme = [r for r in recs if abs(rate[r["family"]] - 0.5) > 0.3]
    balanced = [r for r in recs if abs(rate[r["family"]] - 0.5) <= 0.15]
    tot_skill = sum(r["skill"] for r in recs)
    ex_skill = sum(r["skill"] for r in extreme)
    print(f"  extreme-base-rate (|rate-0.5|>0.3): {len(extreme)} recs "
          f"({len(extreme)/len(recs):.0%}), supply {ex_skill:+.3f} of {tot_skill:+.3f} "
          f"total skill = {ex_skill/tot_skill:.0%}")
    print(f"  balanced (|rate-0.5|<=0.15)        : {len(balanced)} recs, "
          f"mean skill {mean([r['skill'] for r in balanced]):+.4f}")
    print(f"  -> our genuine edge on competitive questions is the balanced number.")


if __name__ == "__main__":
    main()
