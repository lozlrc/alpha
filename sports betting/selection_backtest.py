"""Out-of-sample test of the CATEGORY-SELECTION rule.

Does "forecast only the categories we're best at" generalise, or is it just
overfitting 176 noisy points?

Method: repeated random 50/50 split (deterministic LCG, no Math.random).
  - On TRAIN: rank families by mean skill; SELECT families with mean skill > 0
    and train-n >= MIN_TRAIN_N.
  - On TEST: score every question by `skill` (= 0.25 - brier, i.e. edge vs a
    coin-flip field -- the realistic field for no-market niche questions, and a
    GENEROUS field for winner/goals; so dropping winner under this field is a
    strong result).
  - Compare ALL (answer everything) vs SELECTED (answer only chosen families):
      * mean skill per question (efficiency)
      * total skill (sum)              -- "do we win more?"
      * coverage (fraction kept)
  - Aggregate mean/std across folds and the win-rate (folds SELECTED beats ALL).

Also reports a fixed full-sample SELECTED set for deployment.
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MIN_TRAIN_N = 3
N_FOLDS = 400


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def lcg(seed):
    s = seed
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s / 0x7FFFFFFF


def fold(records, rng):
    train, test = [], []
    for r in records:
        (train if next(rng) < 0.5 else test).append(r)
    return train, test


def select_families(train, min_n=MIN_TRAIN_N):
    byfam = defaultdict(list)
    for r in train:
        byfam[r["family"]].append(r["skill"])
    return {f for f, sk in byfam.items() if len(sk) >= min_n and mean(sk) > 0}


def main():
    records = json.load(open(os.path.join(DATA, "settled.json")))
    for r in records:
        r["skill"] = 0.25 - r["brier"]

    rng = lcg(20260621)
    diffs_mean, diffs_total = [], []
    sel_means, all_means, covs, wins = [], [], [], 0
    fam_pick_count = defaultdict(int)

    for _ in range(N_FOLDS):
        train, test = fold(records, rng)
        if not train or not test:
            continue
        sel = select_families(train)
        for f in sel:
            fam_pick_count[f] += 1
        test_sel = [r for r in test if r["family"] in sel]
        if not test_sel:
            continue
        all_mean = mean([r["skill"] for r in test])
        sel_mean = mean([r["skill"] for r in test_sel])
        all_means.append(all_mean)
        sel_means.append(sel_mean)
        diffs_mean.append(sel_mean - all_mean)
        # total under "answer this fold's worth of questions"
        diffs_total.append(sum(r["skill"] for r in test_sel) - sum(r["skill"] for r in test))
        covs.append(len(test_sel) / len(test))
        if sel_mean > all_mean:
            wins += 1

    nf = len(diffs_mean)
    print(f"folds evaluated            : {nf}")
    print(f"mean coverage (frac kept)  : {mean(covs):.2%}")
    print()
    print(f"ALL  mean skill/question   : {mean(all_means):+.4f}  (std {std(all_means):.4f})")
    print(f"SEL  mean skill/question   : {mean(sel_means):+.4f}  (std {std(sel_means):.4f})")
    print(f"  -> efficiency uplift     : {mean(diffs_mean):+.4f}  (std {std(diffs_mean):.4f})")
    print(f"  -> SEL beats ALL in      : {wins}/{nf} folds = {wins/nf:.1%}")
    print()
    # risk-adjusted: per-question skill Sharpe (mean/std of the per-question skill)
    def qsharpe(rs):
        s = [r["skill"] for r in rs]
        return mean(s) / std(s) if std(s) > 1e-9 else float("nan")
    sel_full = select_families(records, min_n=8)  # deployable set, stricter n
    kept = [r for r in records if r["family"] in sel_full]
    print(f"FULL-SAMPLE deployable set (train-n>=8, mean skill>0):")
    print(f"  families: {sorted(sel_full)}")
    print(f"  ALL : n={len(records)}  mean skill {mean([r['skill'] for r in records]):+.4f}"
          f"  qSharpe {qsharpe(records):.3f}")
    print(f"  SEL : n={len(kept)}  mean skill {mean([r['skill'] for r in kept]):+.4f}"
          f"  qSharpe {qsharpe(kept):.3f}")
    print()
    print("family selection frequency across folds (stability):")
    for f, c in sorted(fam_pick_count.items(), key=lambda kv: -kv[1]):
        print(f"  {c/nf:5.1%}  {f}")

    json.dump({
        "folds": nf, "coverage": mean(covs),
        "all_mean_skill": mean(all_means), "sel_mean_skill": mean(sel_means),
        "uplift": mean(diffs_mean), "uplift_std": std(diffs_mean),
        "winrate": wins / nf,
        "deployable_set": sorted(sel_full),
        "sel_qsharpe": qsharpe(kept), "all_qsharpe": qsharpe(records),
        "pick_freq": {f: c / nf for f, c in fam_pick_count.items()},
    }, open(os.path.join(DATA, "backtest.json"), "w"), indent=1)
    print(f"\n-> wrote {DATA}/backtest.json")


if __name__ == "__main__":
    main()
