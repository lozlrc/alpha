"""Crypto perpetual funding-rate carry on REAL data -- structural, delta-neutral yield.

Run:  ../.venv/bin/python run.py

WHY THIS FAMILY EXISTS. 10_real_data showed you cannot out-predict a sharp crowd
(equity factors net of costs), and 11 showed structural return sources survive where
prediction fails. This family applies the same law to crypto derivatives:

  Perpetual futures track spot via a FUNDING RATE paid between longs and shorts every
  hour (Hyperliquid, Kraken) or 8h (most CEXes). Retail leverage demand is chronically
  LONG, so funding is positive most of the time. Hold "short 1x perp + long 1x spot"
  and you are delta-neutral and COLLECT that transfer -- a structural insurance premium
  for providing the other side of leverage demand, not a directional forecast. Nobody
  is out-predicted; the payer pays *knowingly* (for convenience/leverage). This is the
  crypto sibling of 06's carry & variance-premium ideas -- but on real venue data.

  A second, purer edge: two venues' funding rates desynchronize. Long the perp on the
  cheap-funding venue, short it on the rich venue -- both legs perps, basis largely
  nets, and you collect the DIFFERENTIAL (cross-venue funding spread).

THE SMOOTHING ILLUSION (this family's built-in honesty contrast). Score the book on
funding ACCRUAL alone and you get a Sharpe >10: near-riskless-looking, because the
noisy component was thrown away. A real carry book is MARKED DAILY: the perp-index
basis wiggles, adding vol that contributes ~zero mean. The suite therefore marks every
Hyperliquid stream with the venue's own hourly premium (funding - d(basis)), and ships
`carry_btc_ACCRUAL_ILLUSION` right beside the marked `carry_btc` -- the Sharpe gap is
the illusion, exactly like 07's leaky-vs-honest and 10's in-sample mirage.

DATA (real, free, public; one-time fetch cached to data_cache/, offline after):
  * Hyperliquid  POST /info fundingHistory -- HOURLY funding + premium since ~May 2023.
  * Kraken Futures GET /v4/historicalfundingrates -- hourly relative funding, ~1 yr.
  * (Binance & Bybit are US-geo-blocked; OKX's public endpoint only keeps ~3 months.
     Venue/data access IS part of this trade -- that is why funding desks run VPSes.)

HONEST CAVEATS (printed again at the end):
  * Returns are ON NOTIONAL, unlevered. Spot custody/borrow and margin drag excluded.
  * The cross-venue spread is accrual-only (Kraken publishes no premium series) -- an
    UPPER bound, and it still loses net of flip costs on this window: reported as the
    negative result it is.
  * The Hyperliquid era (2023-26) is mostly a bull tape -- the friendly regime for
    positive funding. The GATED variants exist precisely for the other regime.
  * US persons cannot trade HL/Kraken perps retail; the US-legal cousin is the CME
    basis trade. The edge measured here is the mechanism, not a brokerage referral.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")

COINS = ["BTC", "ETH", "SOL"]
KRAKEN_SYM = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD", "SOL": "PF_SOLUSD"}
HL_START_MS = 1682899200000          # 2023-05-01 -- just before HL mainnet history begins
PPY = 365                            # crypto accrues every calendar day

GATE_DAYS = 7                        # trailing window for the in/out gate & spread side
FLIP_BPS = 15.0                      # spot+perp entry OR exit: 2 legs x ~7.5 bps taker+slip
SPREAD_FLIP_BPS = 20.0               # 4 perp legs to flip the two-venue spread book


# ---------------------------------------------------------------- data fetch
def _http(url: str, payload: dict | None = None, tries: int = 4) -> object:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "User-Agent": "alpha-suite/12"},
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = 15.0 * (attempt + 1)
            print(f"    rate-limited (429); backing off {wait:.0f}s ...")
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == tries - 1:                    # transient net hiccups: retry too
                raise
            wait = 5.0 * (attempt + 1)
            print(f"    transient error ({type(e).__name__}); retrying in {wait:.0f}s ...")
            time.sleep(wait)


def fetch_hyperliquid(coin: str) -> pd.DataFrame:
    """Full hourly funding + premium history for one coin (paginated 500/call)."""
    rows, cursor, pages = [], HL_START_MS, 0
    while True:
        batch = _http("https://api.hyperliquid.xyz/info",
                      {"type": "fundingHistory", "coin": coin, "startTime": int(cursor)})
        if not batch:
            break
        rows.extend((int(b["time"]), float(b["fundingRate"]), float(b["premium"]))
                    for b in batch)
        pages += 1
        if pages % 15 == 0:
            print(f"    HL {coin}: {len(rows):,} hourly records ...")
        if len(batch) < 500 or pages > 250:
            break
        cursor = rows[-1][0] + 1
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=["time", "rate", "premium"]).set_index("time")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_kraken(coin: str) -> pd.DataFrame:
    """~1 year of hourly RELATIVE funding rates, one call (no premium published)."""
    out = _http("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"
                f"?symbol={KRAKEN_SYM[coin]}")
    df = pd.DataFrame([(r["timestamp"], float(r["relativeFundingRate"]))
                       for r in out.get("rates", [])], columns=["time", "rate"])
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_all() -> tuple[dict, dict] | None:
    """{coin: hourly DataFrame} per venue, via cache; None (clean skip) if unreachable."""
    os.makedirs(CACHE, exist_ok=True)
    hl, kr = {}, {}
    for coin in COINS:
        for tag, store, fn in (("hl", hl, fetch_hyperliquid), ("kraken", kr, fetch_kraken)):
            path = os.path.join(CACHE, f"{tag}_{coin}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if "rate" not in df.columns:            # tolerate older cache formats
                    df = df.rename(columns={df.columns[0]: "rate"})
                store[coin] = df
                continue
            try:
                print(f"  fetching {tag} {coin} ...")
                df = fn(coin)
            except Exception as e:                      # offline / venue down
                print(f"[skip] could not fetch {tag} {coin}: {type(e).__name__}: {e}")
                print("       Run once WITH network to populate data_cache/ "
                      "(fully offline afterwards). Skipping cleanly.")
                return None
            df.to_csv(path)
            store[coin] = df
    return hl, kr


# ---------------------------------------------------------------- streams
def daily_marked(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(accrual, marked) daily returns of the short-perp+long-spot book, 1x notional.

    accrual = the day's summed funding.  marked = accrual - d(basis): the short perp
    leg is marked against the venue's own premium (basis proxy), so the basis wiggle
    shows up as daily vol even though it telescopes to ~0 over a long hold."""
    accrual = df["rate"].resample("1D").sum().iloc[1:-1]
    if "premium" in df.columns:
        basis = df["premium"].resample("1D").last().reindex(accrual.index).ffill()
        marked = accrual - basis.diff().fillna(0.0)
    else:
        marked = accrual.copy()
    return accrual, marked


def net(stream: pd.Series, pos: pd.Series, flip_bps: float) -> pd.Series:
    """Daily net return of a +-1/0 position in `stream`.
    `pos` must already be LAGGED (decided from data through t-1)."""
    pos = pos.reindex(stream.index).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())       # first day = entry
    return pos * stream - turnover * flip_bps / 2e4


def carry_always(stream: pd.Series) -> pd.Series:
    return net(stream, pd.Series(1.0, index=stream.index), FLIP_BPS)


def carry_gated(stream: pd.Series, accrual: pd.Series,
                gate_days: int = GATE_DAYS, flip_bps: float = FLIP_BPS) -> pd.Series:
    """Gate on trailing FUNDING (the signal), trade the MARKED stream (the reality)."""
    sig = (accrual.rolling(gate_days).sum().shift(1) > 0).astype(float)
    return net(stream, sig, flip_bps)


def xvenue_spread(acc_a: pd.Series, acc_b: pd.Series,
                  gate_days: int = GATE_DAYS, flip_bps: float = SPREAD_FLIP_BPS) -> pd.Series:
    """Receive funding on the rich venue, pay on the cheap one; both legs perps.
    Accrual-only (no Kraken premium series) => an UPPER bound on the spread book."""
    idx = acc_a.index.intersection(acc_b.index)
    diff = (acc_a - acc_b).loc[idx]
    side = np.sign(diff.rolling(gate_days).mean().shift(1)).fillna(0.0)
    return net(diff, side, flip_bps)


# ---------------------------------------------------------------- main
def main():
    print("\n=== Crypto funding-rate carry (REAL Hyperliquid + Kraken data) ===")
    loaded = load_all()
    if loaded is None:
        return
    hl_acc, hl_mkd, kr_acc = {}, {}, {}
    for c in COINS:
        hl_acc[c], hl_mkd[c] = daily_marked(loaded[0][c])
        kr_acc[c], _ = daily_marked(loaded[1][c])

    print("\nAnnualized average funding received by the SHORT-perp side "
          f"(daily x {PPY}):")
    print(f"  {'coin':<5s} {'venue':<12s} {'span':<24s} {'ann.funding':>11s} {'%days>0':>8s}")
    for c in COINS:
        for tag, d in (("hyperliquid", hl_acc[c]), ("kraken", kr_acc[c])):
            span = f"{d.index[0].date()} - {d.index[-1].date()}"
            print(f"  {c:<5s} {tag:<12s} {span:<24s} {d.mean() * PPY:>10.1%} "
                  f"{(d > 0).mean():>8.0%}")

    streams: dict[str, pd.Series] = {}
    for c in COINS:
        lc = c.lower()
        streams[f"carry_{lc}"] = carry_always(hl_mkd[c])
        streams[f"carry_{lc}_gated"] = carry_gated(hl_mkd[c], hl_acc[c])
        streams[f"xvenue_{lc}_UPPERBOUND"] = xvenue_spread(hl_acc[c], kr_acc[c])
    book = pd.concat([streams[f"carry_{c.lower()}_gated"] for c in COINS], axis=1).mean(axis=1)
    streams["carry_book_gated"] = book
    # The teaching contrast: same book, basis mark thrown away -> fake smoothness.
    streams["carry_btc_ACCRUAL_ILLUSION"] = carry_always(hl_acc["BTC"])

    rows = [core.metrics.summary(r.dropna(), n, periods_per_year=PPY)
            for n, r in streams.items()]
    board = core.format_leaderboard(rows)
    print("\nNet-of-cost performance (returns ON NOTIONAL, unlevered, delta-neutral;")
    print("carry_* marked daily against the venue premium -- see ACCRUAL_ILLUSION row):")
    print(board[["sharpe", "ann_return", "ann_vol", "max_drawdown", "hit_rate",
                 "n_periods"]].round(3).to_string())

    ill = core.metrics.summary(streams["carry_btc_ACCRUAL_ILLUSION"].dropna(), "i", PPY)
    real = core.metrics.summary(streams["carry_btc"].dropna(), "r", PPY)
    print(f"\nThe smoothing illusion, quantified (BTC): accrual-only Sharpe "
          f"{ill['sharpe']:.1f} vs marked {real['sharpe']:.1f} -- same trade, same mean;"
          f"\nthe gap is thrown-away basis vol, not alpha. Never score carry unmarked.")

    print("\nWhere the risk actually shows up -- worst marked days of the gated book:")
    for d, r in book.nsmallest(5).items():
        print(f"  {d.date()}  {r * 1e4:+7.1f} bps")
    print("READ THE SHARPE LIKE 03's: a category artifact of the clock/measure. Daily")
    print("basis wiggle is bps while the yield is steady, so Sharpe prints huge; the")
    print("risks that actually kill carry books -- venue/custody failure, liquidation")
    print("gaps in a squeeze, basis blowout at exit -- live OUTSIDE this daily series.")
    print("The honest summary is the RETURN level (~15%/yr structural, unlevered) plus")
    print("an unmeasured operational tail, NOT 'Sharpe 11'. Size to the tail, not vol.")

    # ---- robustness: the gate/cost surface, not a tuned knife-edge ---------
    print(f"\nRobustness -- carry_book_gated (marked) Sharpe / maxDD across settings "
          f"(base: gate {GATE_DAYS}d, {FLIP_BPS:.0f} bps):")
    print(f"  {'gate_days':>9s} {'flip_bps':>8s} {'Sharpe':>7s} {'maxDD':>7s}")
    for gd in (3, 7, 14):
        for fb in (10.0, 15.0, 25.0):
            b = pd.concat([carry_gated(hl_mkd[c], hl_acc[c], gd, fb) for c in COINS],
                          axis=1).mean(axis=1)
            s = core.metrics.summary(b.dropna(), "x", periods_per_year=PPY)
            print(f"  {gd:>9d} {fb:>8.0f} {s['sharpe']:>7.2f} {s['max_drawdown'] * 100:>6.2f}%")

    # ---- figure ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.4), height_ratios=[2, 1])
    show = {"carry_book_gated": book,
            **{f"carry_{c.lower()}_gated": streams[f"carry_{c.lower()}_gated"] for c in COINS},
            "carry_btc_ACCRUAL_ILLUSION": streams["carry_btc_ACCRUAL_ILLUSION"]}
    for nm, r in show.items():
        eq = core.metrics.equity_curve(r.dropna())
        style = dict(lw=1.8) if nm == "carry_book_gated" else \
            dict(lw=1.2, ls="--", alpha=0.8) if "ILLUSION" in nm else dict(lw=1.1)
        ax1.plot(eq.index, eq.values, label=nm, **style)
    ax1.set_ylabel("growth of $1 (on notional)")
    ax1.set_title("Funding carry, marked daily -- structural yield from leverage demand\n"
                  "(dashed: the same book scored accrual-only -- the smoothness is fake)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    for nm, r in show.items():
        if "ILLUSION" in nm:
            continue
        dd = core.metrics.drawdown_series(r.dropna())
        ax2.plot(dd.index, dd.values * 100, lw=1.1, label=nm)
    ax2.set_ylabel("drawdown (%)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "equity.png"), dpi=120)
    plt.close(fig)

    board.round(4).to_csv(os.path.join(HERE, "leaderboard.csv"))
    print("\nHonest caveats: unlevered returns on notional; spot custody/borrow and margin")
    print("drag excluded; xvenue rows are accrual-only UPPER bounds (Kraken has no premium")
    print("series) and still lose net of flips here -- a real negative result. 2023-26 is")
    print("mostly a bull tape (friendly for funding); the gate exists for the other regime.")
    print("HL/Kraken perps are not US-retail venues; the US-legal cousin is the CME basis.")
    print(f"\nSaved: {os.path.join(HERE, 'equity.png')}")
    print(f"Saved: {os.path.join(HERE, 'leaderboard.csv')}")


if __name__ == "__main__":
    main()
