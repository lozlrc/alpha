"""Fair large-sample predictability test, v2 — walk-forward ONLINE LOGISTIC.

The generative Poisson (historical_backtest.py) truncated the foul distribution
(grid cap 12, foul mean ~12) and under-extracted signal. Here each question family
gets its own online logistic regression on the relevant rolling-form feature,
trained strictly walk-forward (predict with current weights = past only, then
update). Self-calibrating, no truncation. This measures how much each category is
predictable from team fundamentals when modelled COMPETENTLY.

skill = expanding-base-rate Brier - model Brier.  Result also vs devigged B365.
"""
import json
import math
import os
from collections import defaultdict, deque
from historical_backtest import download, parse, devig_home, ROLL, BURN

HERE = os.path.dirname(os.path.abspath(__file__))


def sig(x):
    return 1 / (1 + math.exp(-max(-30, min(30, x))))


class OnlineLogit:
    """1-feature logistic w/ running standardisation, single-pass SGD."""
    def __init__(self, lr=0.02):
        self.w0 = 0.0; self.w1 = 0.0; self.lr = lr
        self.n = 0; self.mu = 0.0; self.M2 = 0.0  # Welford for feature

    def _z(self, x):
        sd = math.sqrt(self.M2 / self.n) if self.n > 1 and self.M2 > 0 else 1.0
        return (x - self.mu) / sd

    def predict(self, x):
        return sig(self.w0 + self.w1 * self._z(x))

    def update(self, x, y):
        z = self._z(x)
        p = sig(self.w0 + self.w1 * z)
        g = p - y
        self.w0 -= self.lr * g
        self.w1 -= self.lr * g * z
        # update feature stats AFTER (so predict used past-only scaling)
        self.n += 1
        d = x - self.mu
        self.mu += d / self.n
        self.M2 += d * (x - self.mu)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    rows = parse(download())
    print(f"matches: {len(rows)}")
    F = defaultdict(lambda: defaultdict(lambda: deque(maxlen=ROLL)))

    def avg(t, s, d):
        dq = F[t][s]
        return sum(dq) / len(dq) if dq else d

    DEF = {"g": 1.35, "c": 5.0, "st": 4.5, "f": 11.0, "k": 1.8}
    # family -> (feature_fn over rates, outcome_fn)  ; feature is diff or total
    models = {}
    preds = defaultdict(list)
    base_briers = defaultdict(list)
    yes = defaultdict(lambda: [0, 0])
    book = []

    def fam_setup(name):
        if name not in models:
            models[name] = OnlineLogit()
        return models[name]

    HFA = 0.20
    for r in rows:
        h, a = r["h"], r["a"]
        if len(F[h]["g_for"]) >= BURN and len(F[a]["g_for"]) >= BURN:
            def rt(sf, sa, t, o, d):
                return 0.5 * (avg(t, sf, d) + avg(o, sa, d))
            lh_g = rt("g_for", "g_against", h, a, DEF["g"]) + HFA
            la_g = rt("g_for", "g_against", a, h, DEF["g"])
            lh_c = rt("c_for", "c_against", h, a, DEF["c"]); la_c = rt("c_for", "c_against", a, h, DEF["c"])
            lh_s = rt("s_for", "s_against", h, a, DEF["st"]); la_s = rt("s_for", "s_against", a, h, DEF["st"])
            lh_f = rt("f_for", "f_against", h, a, DEF["f"]); la_f = rt("f_for", "f_against", a, h, DEF["f"])
            lh_k = rt("k_for", "k_against", h, a, DEF["k"]); la_k = rt("k_for", "k_against", a, h, DEF["k"])
            feats = [
                ("RESULT_homewin", lh_g - la_g, 1 if r["hg"] > r["ag"] else 0),
                ("GOALS_over2.5", lh_g + la_g, 1 if r["hg"] + r["ag"] > 2.5 else 0),
                ("CORNERS_h>a", lh_c - la_c, 1 if r["hc"] > r["ac"] else 0),
                ("CORNERS_over9.5", lh_c + la_c, 1 if r["hc"] + r["ac"] > 9.5 else 0),
                ("SOT_h>a", lh_s - la_s, 1 if r["hst"] > r["ast"] else 0),
                ("SOT_over8.5", lh_s + la_s, 1 if r["hst"] + r["ast"] > 8.5 else 0),
                ("FOULS_h>a", lh_f - la_f, 1 if r["hf"] > r["af"] else 0),
                ("CARDS_over3.5", lh_k + la_k, 1 if r["hcard"] + r["acard"] > 3.5 else 0),
                ("CARDS_h>a", lh_k - la_k, 1 if r["hcard"] > r["acard"] else 0),
            ]
            for fam, x, y in feats:
                m = fam_setup(fam)
                if m.n >= 50:   # let it warm up before scoring
                    p = m.predict(x)
                    yc, tc = yes[fam]
                    base = yc / tc if tc >= 30 else 0.5
                    preds[fam].append((p, y))
                    base_briers[fam].append((base - y) ** 2)
                yes[fam][0] += y; yes[fam][1] += 1
                m.update(x, y)
            bp = devig_home(r.get("bh"), r.get("bd"), r.get("ba"))
            if bp is not None and models["RESULT_homewin"].n >= 50:
                book.append((bp - (1 if r["hg"] > r["ag"] else 0)) ** 2)

        for t, gf, ga, cf, ca, sf, sa, ff, fa, kf, ka in [
            (h, r["hg"], r["ag"], r["hc"], r["ac"], r["hst"], r["ast"], r["hf"], r["af"], r["hcard"], r["acard"]),
            (a, r["ag"], r["hg"], r["ac"], r["hc"], r["ast"], r["hst"], r["af"], r["hf"], r["acard"], r["hcard"]),
        ]:
            F[t]["g_for"].append(gf); F[t]["g_against"].append(ga)
            F[t]["c_for"].append(cf); F[t]["c_against"].append(ca)
            F[t]["s_for"].append(sf); F[t]["s_against"].append(sa)
            F[t]["f_for"].append(ff); F[t]["f_against"].append(fa)
            F[t]["k_for"].append(kf); F[t]["k_against"].append(ka)

    cat = lambda f: f.split("_")[0]
    print(f"\n{'family':18} {'n':>6} {'baseRt':>6} {'model':>7} {'base':>7} {'skill':>8} {'skill%':>7}")
    print("-" * 64)
    out = {}
    for fam in ["RESULT_homewin", "GOALS_over2.5", "CORNERS_h>a", "CORNERS_over9.5",
                "SOT_h>a", "SOT_over8.5", "FOULS_h>a", "CARDS_over3.5", "CARDS_h>a"]:
        ps = preds[fam]
        mb = mean([(p - y) ** 2 for p, y in ps]); bb = mean(base_briers[fam])
        sk = bb - mb; out[fam] = {"n": len(ps), "model_brier": mb, "base_brier": bb,
                                  "skill": sk, "skill_pct": sk / bb * 100}
        print(f"{fam:18} {len(ps):>6} {mean([y for _,y in ps]):>6.2f} {mb:>7.4f} {bb:>7.4f} "
              f"{sk:>+8.4f} {sk/bb*100:>+6.1f}%")

    bk = mean(book)
    rb = out["RESULT_homewin"]
    print(f"\nRESULT vs sharp field: base {rb['base_brier']:.4f} | model {rb['model_brier']:.4f} "
          f"(skill {rb['skill']:+.4f}) | BOOK {bk:.4f} (skill {rb['base_brier']-bk:+.4f})")
    print(f"  -> book beats our model by {rb['model_brier']-bk:+.4f}: who-wins market efficient")

    anc = [out[f]["skill"] for f in out if cat(f) in ("CORNERS", "SOT", "FOULS")]
    print(f"\nancillary-stat mean skill {mean(anc):+.4f} vs RESULT {rb['skill']:+.4f}")
    json.dump(out, open(os.path.join(HERE, "data", "historical_logit.json"), "w"), indent=1)
    print("-> wrote data/historical_logit.json")


if __name__ == "__main__":
    main()
