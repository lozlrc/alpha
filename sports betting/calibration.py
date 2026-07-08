"""Per-family calibration diagnosis + OOS correction backtest.

If a family has NEGATIVE edge (we're worse than a coin-flip) it may just be
systematically biased (we say "yes" too often/rarely). Calibration can fix bias
(not noise). We:
  1. Diagnose per-family calibration gap = mean(prob) - base_rate(outcome).
  2. Overall reliability bins.
  3. OOS backtest: learn a per-family correction on TRAIN, apply on TEST, measure
     Brier change. Two corrections, both leakage-free:
       (a) logit-shift  : p' = sigmoid(logit(p) + s*),  s* mins train Brier.
       (b) shrink-to-rate: p' = (1-w)*p + w*r_train,     w* mins train Brier.
     Also a GLOBAL shrink-to-0.5 (overconfidence regulariser) as a baseline.
"""
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
N_FOLDS = 300


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def clamp(p, lo=0.01, hi=0.99):
    return max(lo, min(hi, p))


def logit(p):
    p = clamp(p)
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-x))


def brier(p, y):
    return (p - y) ** 2


def lcg(seed):
    s = seed
    while True:
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        yield s / 0x7FFFFFFF


def best_shift(train):
    """grid-search additive logit shift minimising train Brier."""
    best, bs = 1e9, 0.0
    s = -2.0
    while s <= 2.0001:
        loss = mean([brier(sig(logit(r["p"]) + s), r["outcome"]) for r in train])
        if loss < best:
            best, bs = loss, s
        s += 0.05
    return bs


def best_shrink(train, target_key):
    """grid-search blend weight toward target (base rate or 0.5)."""
    if target_key == "rate":
        tgt = mean([r["outcome"] for r in train])
    else:
        tgt = 0.5
    best, bw = 1e9, 0.0
    w = 0.0
    while w <= 1.0001:
        loss = mean([brier((1 - w) * r["p"] + w * tgt, r["outcome"]) for r in train])
        if loss < best:
            best, bw = loss, w
        w += 0.05
    return bw, tgt


def main():
    recs = json.load(open(os.path.join(DATA, "settled.json")))
    for r in recs:
        r["p"] = r["prob"] / 100.0

    # 1. per-family calibration gap
    byfam = defaultdict(list)
    for r in recs:
        byfam[r["family"]].append(r)
    print("PER-FAMILY CALIBRATION (gap = mean_prob - base_rate; +=we say YES too much)")
    print(f"{'family':22} {'n':>3} {'meanP':>6} {'rate':>6} {'gap':>7} {'brier':>7}")
    for f, rs in sorted(byfam.items(), key=lambda kv: -len(kv[1])):
        if len(rs) < 5:
            continue
        mp = mean([r["p"] for r in rs])
        rate = mean([r["outcome"] for r in rs])
        mb = mean([brier(r["p"], r["outcome"]) for r in rs])
        print(f"{f:22} {len(rs):>3} {mp:>6.2f} {rate:>6.2f} {mp-rate:>+7.2f} {mb:>7.4f}")

    # 2. overall reliability bins
    print("\nOVERALL RELIABILITY (bin by predicted prob)")
    bins = defaultdict(list)
    for r in recs:
        bins[min(9, int(r["p"] * 10))].append(r)
    print(f"{'bin':>9} {'n':>4} {'pred':>6} {'actual':>7}")
    for b in sorted(bins):
        rs = bins[b]
        print(f"{b*10:>3}-{b*10+10:>3}%  {len(rs):>4} {mean([r['p'] for r in rs]):>6.2f}"
              f" {mean([r['outcome'] for r in rs]):>7.2f}")

    # 3. OOS correction backtest
    rng = lcg(20260621)
    res = defaultdict(list)  # method -> list of (test Brier)
    for _ in range(N_FOLDS):
        train, test = [], []
        for r in recs:
            (train if next(rng) < 0.5 else test).append(r)
        if len(train) < 20 or not test:
            continue
        tr_by = defaultdict(list)
        for r in train:
            tr_by[r["family"]].append(r)
        shift = {f: best_shift(rs) for f, rs in tr_by.items() if len(rs) >= 5}
        shr_rate = {f: best_shrink(rs, "rate") for f, rs in tr_by.items() if len(rs) >= 5}
        g_w, _ = best_shrink(train, "half")  # global shrink-to-0.5
        b_raw = b_sh = b_sr = b_g = 0.0
        for r in test:
            y, p = r["outcome"], r["p"]
            b_raw += brier(p, y)
            s = shift.get(r["family"], 0.0)
            b_sh += brier(sig(logit(p) + s), y)
            if r["family"] in shr_rate:
                w, tgt = shr_rate[r["family"]]
                b_sr += brier((1 - w) * p + w * tgt, y)
            else:
                b_sr += brier(p, y)
            b_g += brier((1 - g_w) * p + g_w * 0.5, y)
        n = len(test)
        res["raw"].append(b_raw / n)
        res["logit_shift_perfam"].append(b_sh / n)
        res["shrink_rate_perfam"].append(b_sr / n)
        res["shrink_half_global"].append(b_g / n)

    print(f"\nOOS CORRECTION BACKTEST ({len(res['raw'])} folds, lower Brier = better)")
    base = mean(res["raw"])
    print(f"{'method':22} {'testBrier':>10} {'vs raw':>9} {'std':>7}")
    for m in ["raw", "logit_shift_perfam", "shrink_rate_perfam", "shrink_half_global"]:
        mb = mean(res[m])
        print(f"{m:22} {mb:>10.4f} {base-mb:>+9.4f} {std(res[m]):>7.4f}")

    json.dump({m: {"mean": mean(v), "std": std(v)} for m, v in res.items()},
              open(os.path.join(DATA, "calibration.json"), "w"), indent=1)
    print(f"\n-> wrote {DATA}/calibration.json")


if __name__ == "__main__":
    main()
