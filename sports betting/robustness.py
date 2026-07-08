"""Robustness of the large-sample predictability finding:
   vary rolling window K, the era (early vs late seasons), and per-league.
Confirms the headline (SoT/result/fouls/corners all predictable; ancillary ~ result;
book beats us on result) is not an artifact of one configuration.
"""
import io
import json
import os
import contextlib
import historical_backtest as HB
import historical_logit as HL

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ["SOT_h>a", "RESULT_homewin", "FOULS_h>a", "CORNERS_h>a", "CARDS_over3.5", "GOALS_over2.5"]


def run():
    """run HL.main() silently, return the skills dict it writes."""
    with contextlib.redirect_stdout(io.StringIO()):
        HL.main()
    return json.load(open(os.path.join(HERE, "data", "historical_logit.json")))


def show(tag, d):
    print(f"{tag:22} " + "  ".join(f"{k.split('_')[0][:4]}:{d[k]['skill']:+.4f}" for k in KEY))


def main():
    base_seasons = list(HB.SEASONS)
    base_leagues = list(HB.LEAGUES)

    print("=== window K sensitivity (all data) ===")
    print(f"{'config':22} " + "  ".join(f"{k.split('_')[0][:4]:>11}" for k in KEY))
    for k in (6, 10, 20):
        HL.ROLL = k
        show(f"K={k}", run())
    HL.ROLL = 10

    print("\n=== era stability (K=10) ===")
    HB.SEASONS = ["1819", "1920", "2021"]
    show("early 18-21", run())
    HB.SEASONS = ["2223", "2324", "2425"]
    show("late 22-25", run())
    HB.SEASONS = base_seasons

    print("\n=== per-league (K=10, all seasons) ===")
    for lg in base_leagues:
        HB.LEAGUES = [lg]
        try:
            show(lg, run())
        except Exception as e:
            print(f"{lg}: {str(e)[:40]}")
    HB.LEAGUES = base_leagues
    print("\n(skill = base-rate Brier - model Brier; +ve = predictable over base rate)")


if __name__ == "__main__":
    main()
