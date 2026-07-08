"""INDEPENDENT walk-forward predictability test (reimplementation).

Does NOT reuse historical_logit.py's OnlineLogit nor historical_backtest.py's
main() / pois_grid. Only download() and parse() are reused for loading.

Model: generative multiplicative-Poisson with a league-baseline + team
attack/defence log-strengths estimated from rolling form (a different
parameterisation than historical_backtest's additive 0.5*(for+against)).

For a stat S, team T's expected count vs opponent O:
    lambda_T = league_mean_S * (T_for_strength) * (O_against_strength)
where strengths are rolling-mean ratios to the running league mean
(so they're dimensionless multipliers centred on 1.0). Home gets an
extra multiplicative home-field factor for goals/SoT/corners.

Probabilities:
    P(home > away)  via exact grid convolution, GRID CAP = 60  (no foul truncation)
    P(total > thr)  via Poisson-sum (lambda_h+lambda_a) tail, also cap 60
RESULT home-win uses the goals grid P(hg>ag).

skill = expanding-base-rate Brier - model Brier   (all OOS / walk-forward)
Result also scored vs devigged Bet365.
"""
import json
import math
import os
from collections import defaultdict, deque

from historical_backtest import download, parse, devig_home  # loaders only

HERE = os.path.dirname(os.path.abspath(__file__))
ROLL = 12          # rolling window (own, deliberately != backtest's 10)
BURN = 6           # min prior matches per team before scoring
GRID = 60          # grid cap -- large enough that fouls (max~32) aren't truncated
WARM = 50          # global warm-up matches per family before scoring (parity w/ logit)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# ---- Poisson helpers (own implementation, cap GRID) --------------------------
def _pois_pmf(lam, cap=GRID):
    lam = max(0.02, lam)
    # stable iterative pmf
    p = math.exp(-lam)
    out = [p]
    for k in range(1, cap + 1):
        p *= lam / k
        out.append(p)
    s = sum(out)
    return [x / s for x in out]  # renormalise over the cap (mass beyond cap tiny)


def p_home_gt_away(lh, la):
    ph = _pois_pmf(lh)
    pa = _pois_pmf(la)
    # P(H>A) = sum_a pa[a] * P(H >= a+1) = sum_a pa[a]*(1 - cdf_h[a])
    cdf_h = []
    c = 0.0
    for x in ph:
        c += x
        cdf_h.append(c)
    tot = 0.0
    for a, paj in enumerate(pa):
        tot += paj * (1.0 - cdf_h[a])  # cdf_h[a] = P(H<=a)
    return tot


def p_total_gt(lh, la, thr):
    # total ~ Poisson(lh+la); P(total > thr)
    lam = lh + la
    pmf = _pois_pmf(lam, cap=GRID)
    k0 = int(math.floor(thr)) + 1  # smallest integer strictly greater than thr
    return sum(pmf[k:]) if k0 <= GRID else 0.0
    # (note: slice from k0)


def p_total_gt_fixed(lh, la, thr):
    lam = lh + la
    pmf = _pois_pmf(lam, cap=GRID)
    k0 = int(math.floor(thr)) + 1
    return sum(pmf[k0:]) if k0 <= GRID else 0.0


def main():
    rows = parse(download())
    print(f"matches: {len(rows)}  grid cap={GRID} roll={ROLL}")

    # per-team rolling for/against deques per stat
    F = defaultdict(lambda: defaultdict(lambda: deque(maxlen=ROLL)))
    # running league means per stat (expanding) for normalisation -- past only
    lg_sum = defaultdict(float)
    lg_n = defaultdict(int)

    def team_avg(t, s):
        dq = F[t][s]
        return mean(list(dq)) if dq else None

    def lg_mean(s, default):
        return lg_sum[s] / lg_n[s] if lg_n[s] >= 50 else default

    DEF = {"g": 1.35, "c": 5.0, "st": 4.5, "f": 11.0, "k": 1.85}
    # home-field multiplicative factors (mild, fixed priors -- attacking stats)
    HF = {"g": 1.15, "st": 1.12, "c": 1.10, "f": 1.0, "k": 1.0}

    def strength(t, sfor, sagainst, lgm):
        """rolling for-strength and against-strength as ratios to league mean."""
        af = team_avg(t, sfor)
        ag = team_avg(t, sagainst)
        sf = (af / lgm) if (af is not None and lgm > 0) else 1.0
        sa = (ag / lgm) if (ag is not None and lgm > 0) else 1.0
        # shrink toward 1.0 a touch for stability
        sf = 0.85 * sf + 0.15
        sa = 0.85 * sa + 0.15
        return sf, sa

    def lambdas(stat_key, sfor, sagainst, h, a, default, hf):
        lgm = lg_mean(stat_key, default)
        h_for, h_against = strength(h, sfor, sagainst, lgm)
        a_for, a_against = strength(a, sfor, sagainst, lgm)
        lh = lgm * h_for * a_against * hf
        la = lgm * a_for * h_against
        return max(0.05, lh), max(0.05, la)

    preds = defaultdict(list)
    base_briers = defaultdict(list)
    yes = defaultdict(lambda: [0, 0])
    counts = defaultdict(int)  # matches seen per family (for warm-up)
    book = []

    FAMILIES = ["RESULT_homewin", "GOALS_over2.5", "CORNERS_h>a", "CORNERS_over9.5",
                "SOT_h>a", "SOT_over8.5", "FOULS_h>a", "CARDS_over3.5", "CARDS_h>a"]

    for r in rows:
        h, a = r["h"], r["a"]
        ready = (len(F[h]["g_for"]) >= BURN and len(F[a]["g_for"]) >= BURN)
        if ready:
            lh_g, la_g = lambdas("g", "g_for", "g_against", h, a, DEF["g"], HF["g"])
            lh_c, la_c = lambdas("c", "c_for", "c_against", h, a, DEF["c"], HF["c"])
            lh_s, la_s = lambdas("st", "s_for", "s_against", h, a, DEF["st"], HF["st"])
            lh_f, la_f = lambdas("f", "f_for", "f_against", h, a, DEF["f"], HF["f"])
            lh_k, la_k = lambdas("k", "k_for", "k_against", h, a, DEF["k"], HF["k"])

            q = [
                ("RESULT_homewin", p_home_gt_away(lh_g, la_g), 1 if r["hg"] > r["ag"] else 0),
                ("GOALS_over2.5", p_total_gt_fixed(lh_g, la_g, 2.5), 1 if r["hg"] + r["ag"] > 2.5 else 0),
                ("CORNERS_h>a", p_home_gt_away(lh_c, la_c), 1 if r["hc"] > r["ac"] else 0),
                ("CORNERS_over9.5", p_total_gt_fixed(lh_c, la_c, 9.5), 1 if r["hc"] + r["ac"] > 9.5 else 0),
                ("SOT_h>a", p_home_gt_away(lh_s, la_s), 1 if r["hst"] > r["ast"] else 0),
                ("SOT_over8.5", p_total_gt_fixed(lh_s, la_s, 8.5), 1 if r["hst"] + r["ast"] > 8.5 else 0),
                ("FOULS_h>a", p_home_gt_away(lh_f, la_f), 1 if r["hf"] > r["af"] else 0),
                ("CARDS_over3.5", p_total_gt_fixed(lh_k, la_k, 3.5), 1 if r["hcard"] + r["acard"] > 3.5 else 0),
                ("CARDS_h>a", p_home_gt_away(lh_k, la_k), 1 if r["hcard"] > r["acard"] else 0),
            ]
            for fam, pred, y in q:
                counts[fam] += 1
                yc, tc = yes[fam]
                base = yc / tc if tc >= 30 else 0.5
                if counts[fam] > WARM:
                    preds[fam].append((max(.01, min(.99, pred)), y))
                    base_briers[fam].append((base - y) ** 2)
                yes[fam][0] += y
                yes[fam][1] += 1
            bp = devig_home(r.get("bh"), r.get("bd"), r.get("ba"))
            if bp is not None and counts["RESULT_homewin"] > WARM:
                book.append((bp - (1 if r["hg"] > r["ag"] else 0)) ** 2)

        # update rolling form + running league means AFTER predicting
        for s_key, hv, av in [("g", r["hg"], r["ag"]), ("c", r["hc"], r["ac"]),
                              ("st", r["hst"], r["ast"]), ("f", r["hf"], r["af"]),
                              ("k", r["hcard"], r["acard"])]:
            lg_sum[s_key] += hv + av
            lg_n[s_key] += 2
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
    for fam in FAMILIES:
        ps = preds[fam]
        mb = mean([(p - y) ** 2 for p, y in ps])
        bb = mean(base_briers[fam])
        sk = bb - mb
        out[fam] = {"n": len(ps), "base_rate": mean([y for _, y in ps]),
                    "model_brier": mb, "base_brier": bb, "skill": sk,
                    "skill_pct": sk / bb * 100 if bb else 0}
        print(f"{fam:18} {len(ps):>6} {out[fam]['base_rate']:>6.2f} {mb:>7.4f} {bb:>7.4f} "
              f"{sk:>+8.4f} {sk/bb*100:>+6.1f}%")

    bk = mean(book)
    rb = out["RESULT_homewin"]
    print(f"\nRESULT vs sharp field: base {rb['base_brier']:.4f} | model {rb['model_brier']:.4f} "
          f"(skill {rb['skill']:+.4f}) | BOOK {bk:.4f} (skill {rb['base_brier']-bk:+.4f})")
    print(f"  book beats model by {rb['model_brier']-bk:+.4f} Brier")

    anc_fams = [f for f in FAMILIES if cat(f) in ("CORNERS", "SOT", "FOULS")]
    anc = [out[f]["skill"] for f in anc_fams]
    print(f"\nancillary mean skill {mean(anc):+.4f} vs RESULT {rb['skill']:+.4f}  "
          f"-> ancillary {'MORE' if mean(anc) > rb['skill'] else 'NOT more'} predictable")
    # most predictable family
    most = max(out, key=lambda f: out[f]["skill"])
    print(f"most predictable family: {most}  (skill {out[most]['skill']:+.4f})")

    out["_book_result_brier"] = bk
    json.dump(out, open(os.path.join(HERE, "data", "independent.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
