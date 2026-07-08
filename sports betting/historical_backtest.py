"""LARGE-SAMPLE test of the core hypothesis the competition's n=176 can't power:
   are ancillary team-stats (corners, shots-on-target, fouls, cards) more
   PREDICTABLE from team fundamentals than the match RESULT?

Data: football-data.co.uk (real club matches, many leagues/seasons) with goals,
result, shots, shots-on-target, corners, fouls, cards, AND Bet365 odds.

Method (fully walk-forward, leakage-free):
  - Maintain per-team rolling means (last K matches) of for/against rates for each
    stat. Predict the CURRENT match from PAST data only, then update.
  - Generative fundamentals models (no fitting, parameter-light):
      * counts (corners/SoT/fouls/cards): team rate = 0.5*(own for-rate +
        opp against-rate); compare via Skellam grid (home>away) or Poisson tail (total>T).
      * result: independent-Poisson goal grid from rolling goal rates -> P(home win).
  - Score each binary "question family" by Brier, OOS. Baselines:
      * climatology = EXPANDING base rate (running freq of YES so far) -> also OOS.
      * for result also the BOOKMAKER (devigged B365) -> the sharp field benchmark.
  - skill = base_rate_brier - model_brier. Bigger skill => more predictable.

Hypothesis confirmed if corners/SoT/fouls skill-over-baseline >> result skill,
AND the bookmaker crushes our result model (markets efficient on who-wins) while
no such sharp market exists for the ancillary stats.
"""
import json
import math
import os
import urllib.request
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "data", "hist")
LEAGUES = ["E0", "E1", "D1", "D2", "I1", "SP1", "F1"]
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425"]
ROLL = 10          # rolling window (matches)
BURN = 6           # need this many prior matches for both teams
MAXG = 12          # grid cap


def download():
    os.makedirs(HIST, exist_ok=True)
    files = []
    for s in SEASONS:
        for lg in LEAGUES:
            fp = os.path.join(HIST, f"{s}_{lg}.csv")
            if not os.path.exists(fp) or os.path.getsize(fp) < 1000:
                url = f"https://www.football-data.co.uk/mmz4281/{s}/{lg}.csv"
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    data = urllib.request.urlopen(req, timeout=25).read()
                    open(fp, "wb").write(data)
                except Exception as e:
                    print(f"  skip {s}_{lg}: {str(e)[:50]}")
                    continue
            files.append(fp)
    return files


def parse(files):
    rows = []
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
            "B365H", "B365D", "B365A"]
    for fp in files:
        try:
            txt = open(fp, encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        if not txt:
            continue
        hdr = txt[0].split(",")
        idx = {h: i for i, h in enumerate(hdr)}
        if not all(c in idx for c in need[:13]):
            continue
        for ln in txt[1:]:
            c = ln.split(",")
            if len(c) < len(hdr):
                continue
            try:
                d = c[idx["Date"]]
                # date dd/mm/yy or dd/mm/yyyy -> sortable yyyymmdd
                p = d.split("/")
                yr = p[2] if len(p[2]) == 4 else ("20" + p[2])
                key = yr + p[1].zfill(2) + p[0].zfill(2)
                r = {
                    "k": key, "h": c[idx["HomeTeam"]], "a": c[idx["AwayTeam"]],
                    "hg": int(c[idx["FTHG"]]), "ag": int(c[idx["FTAG"]]),
                    "hst": int(c[idx["HST"]]), "ast": int(c[idx["AST"]]),
                    "hc": int(c[idx["HC"]]), "ac": int(c[idx["AC"]]),
                    "hf": int(c[idx["HF"]]), "af": int(c[idx["AF"]]),
                    "hcard": int(c[idx["HY"]]) + int(c[idx["HR"]]),
                    "acard": int(c[idx["AY"]]) + int(c[idx["AR"]]),
                }
                try:
                    r["bh"], r["bd"], r["ba"] = (float(c[idx["B365H"]]),
                                                 float(c[idx["B365D"]]), float(c[idx["B365A"]]))
                except Exception:
                    r["bh"] = r["bd"] = r["ba"] = None
                rows.append(r)
            except Exception:
                continue
    rows.sort(key=lambda r: r["k"])
    return rows


def pois_grid(lh, la):
    lh, la = max(0.05, lh), max(0.05, la)
    ph = [math.exp(-lh) * lh ** i / math.factorial(i) for i in range(MAXG + 1)]
    pa = [math.exp(-la) * la ** i / math.factorial(i) for i in range(MAXG + 1)]
    return ph, pa


def p_home_gt_away(lh, la):
    ph, pa = pois_grid(lh, la)
    return sum(ph[i] * pa[j] for i in range(len(ph)) for j in range(len(pa)) if i > j)


def p_total_gt(lh, la, thr):
    ph, pa = pois_grid(lh, la)
    return sum(ph[i] * pa[j] for i in range(len(ph)) for j in range(len(pa)) if i + j > thr)


def p_home_win(lh, la):
    ph, pa = pois_grid(lh, la)
    return sum(ph[i] * pa[j] for i in range(len(ph)) for j in range(len(pa)) if i > j)


def devig_home(bh, bd, ba):
    if not bh:
        return None
    ih, idd, ia = 1 / bh, 1 / bd, 1 / ba
    return ih / (ih + idd + ia)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    files = download()
    rows = parse(files)
    print(f"matches loaded: {len(rows)}  ({len(files)} league-seasons)")

    # rolling for/against per team
    F = defaultdict(lambda: defaultdict(lambda: deque(maxlen=ROLL)))  # team -> stat -> deque

    def avg(team, stat, default):
        dq = F[team][stat]
        return mean(list(dq)) if len(dq) >= 1 else default

    # accumulators: family -> list of (pred, outcome); plus baselines
    preds = defaultdict(list)
    yes_count = defaultdict(lambda: [0, 0])  # family -> [yes, total] expanding base rate
    base_briers = defaultdict(list)
    book_briers = []   # result only
    HFA = 0.20         # home goal advantage (rough)

    # league-average priors as defaults before burn-in
    DEF = {"g": 1.35, "c": 5.0, "st": 4.5, "f": 11.0, "card": 1.8}

    for r in rows:
        h, a = r["h"], r["a"]
        ready = (len(F[h]["g_for"]) >= BURN and len(F[a]["g_for"]) >= BURN)
        if ready:
            # rolling rates
            def rate(stat_for, stat_against, team, opp, d):
                return 0.5 * (avg(team, stat_for, d) + avg(opp, stat_against, d))
            lh_g = rate("g_for", "g_against", h, a, DEF["g"]) + HFA
            la_g = rate("g_for", "g_against", a, h, DEF["g"])
            lh_c = rate("c_for", "c_against", h, a, DEF["c"])
            la_c = rate("c_for", "c_against", a, h, DEF["c"])
            lh_s = rate("s_for", "s_against", h, a, DEF["st"])
            la_s = rate("s_for", "s_against", a, h, DEF["st"])
            lh_f = rate("f_for", "f_against", h, a, DEF["f"])
            la_f = rate("f_for", "f_against", a, h, DEF["f"])
            lh_k = rate("k_for", "k_against", h, a, DEF["card"])
            la_k = rate("k_for", "k_against", a, h, DEF["card"])

            # questions: (family, model_pred, outcome)
            q = [
                ("RESULT_homewin", p_home_win(lh_g, la_g), 1 if r["hg"] > r["ag"] else 0),
                ("GOALS_over2.5", p_total_gt(lh_g, la_g, 2.5), 1 if r["hg"] + r["ag"] > 2.5 else 0),
                ("CORNERS_h>a", p_home_gt_away(lh_c, la_c), 1 if r["hc"] > r["ac"] else 0),
                ("CORNERS_over9.5", p_total_gt(lh_c, la_c, 9.5), 1 if r["hc"] + r["ac"] > 9.5 else 0),
                ("SOT_h>a", p_home_gt_away(lh_s, la_s), 1 if r["hst"] > r["ast"] else 0),
                ("SOT_over8.5", p_total_gt(lh_s, la_s, 8.5), 1 if r["hst"] + r["ast"] > 8.5 else 0),
                ("FOULS_h>a", p_home_gt_away(lh_f, la_f), 1 if r["hf"] > r["af"] else 0),
                ("CARDS_over3.5", p_total_gt(lh_k, la_k, 3.5), 1 if r["hcard"] + r["acard"] > 3.5 else 0),
                ("CARDS_h>a", p_home_gt_away(lh_k, la_k), 1 if r["hcard"] > r["acard"] else 0),
            ]
            for fam, pred, y in q:
                # expanding base-rate baseline (OOS: uses only prior matches)
                yc, tc = yes_count[fam]
                base = yc / tc if tc >= 30 else 0.5
                preds[fam].append((max(.01, min(.99, pred)), y))
                base_briers[fam].append((base - y) ** 2)
                yes_count[fam][0] += y
                yes_count[fam][1] += 1
            # bookmaker on result
            bp = devig_home(r.get("bh"), r.get("bd"), r.get("ba"))
            if bp is not None:
                book_briers.append((bp - (1 if r["hg"] > r["ag"] else 0)) ** 2)

        # update rolling AFTER predicting
        for team, gf, ga, cf, ca, sf, sa, ff, fa, kf, ka in [
            (h, r["hg"], r["ag"], r["hc"], r["ac"], r["hst"], r["ast"], r["hf"], r["af"], r["hcard"], r["acard"]),
            (a, r["ag"], r["hg"], r["ac"], r["hc"], r["ast"], r["hst"], r["af"], r["hf"], r["acard"], r["hcard"]),
        ]:
            F[team]["g_for"].append(gf); F[team]["g_against"].append(ga)
            F[team]["c_for"].append(cf); F[team]["c_against"].append(ca)
            F[team]["s_for"].append(sf); F[team]["s_against"].append(sa)
            F[team]["f_for"].append(ff); F[team]["f_against"].append(fa)
            F[team]["k_for"].append(kf); F[team]["k_against"].append(ka)

    print(f"\n{'family':18} {'n':>6} {'baseRt':>6} {'modelBrier':>10} {'baseBrier':>9} "
          f"{'skill':>8} {'skill%':>7}")
    print("-" * 70)
    out = {}
    for fam in preds:
        ps = preds[fam]
        n = len(ps)
        mb = mean([(p - y) ** 2 for p, y in ps])
        bb = mean(base_briers[fam])
        rate = mean([y for _, y in ps])
        skill = bb - mb
        skillpct = skill / bb * 100 if bb else 0
        out[fam] = {"n": n, "base_rate": rate, "model_brier": mb,
                    "base_brier": bb, "skill": skill, "skill_pct": skillpct}
        print(f"{fam:18} {n:>6} {rate:>6.2f} {mb:>10.4f} {bb:>9.4f} {skill:>+8.4f} {skillpct:>+6.1f}%")

    # result vs bookmaker
    if book_briers:
        bk = mean(book_briers)
        rm = out["RESULT_homewin"]["model_brier"]
        bbase = out["RESULT_homewin"]["base_brier"]
        print("\nRESULT — sharp-field benchmark:")
        print(f"  base-rate Brier {bbase:.4f} | our model {rm:.4f} (skill {bbase-rm:+.4f}) "
              f"| BOOKMAKER {bk:.4f} (skill {bbase-bk:+.4f})")
        print(f"  bookmaker beats our model by {rm-bk:+.4f} Brier -> who-wins market is efficient")

    json.dump(out, open(os.path.join(HERE, "data", "historical.json"), "w"), indent=1)
    print(f"\n-> wrote data/historical.json")

    # hypothesis read-out
    anc = [out[f]["skill"] for f in out if f.split("_")[0] in ("CORNERS", "SOT", "FOULS")]
    res = out["RESULT_homewin"]["skill"]
    print(f"\nHYPOTHESIS: mean ancillary-stat skill {mean(anc):+.4f} vs RESULT skill {res:+.4f}"
          f"  -> stats {'MORE' if mean(anc) > res else 'NOT more'} predictable over baseline")


if __name__ == "__main__":
    main()
