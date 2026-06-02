"""Reduced-form synthetic limit-order-book / tick generator (offline).

This is NOT a full matching-engine LOB. It is a *reduced-form* generator that
reproduces the handful of microstructure facts the demo strategies rely on,
with the predictive structure planted deliberately so the signals recover
real (non-spurious) alpha:

EMBEDDED STRUCTURE
------------------
1. PRICE IMPACT of order flow.  The headline planted effect:

       mid_{t+1} - mid_t  =  impact * OFI_t  +  microstructure_noise

   where OFI_t is the (size-weighted) order-flow imbalance over bar t. So a
   burst of net buying *causes* the next mid to drift up -- order flow leads
   price.  `impact` is small (sub-tick per unit OFI) so a single bar's edge is
   tiny relative to the spread; that is the whole point of the exercise.

2. PERSISTENT order flow.  Signed flow is autocorrelated (an AR(1) "meta-order"
   state) -- real flow clusters because large parents are sliced into children.
   This is what makes OFI forecastable one bar ahead at all.

3. TOP-OF-BOOK SIZE IMBALANCE leads flow/price.  The resting size imbalance
   (bid_size - ask_size)/(bid_size + ask_size) is tied to the *next* bar's flow
   direction (a thick bid tends to get hit less and lifted more), giving the
   book-pressure strategy something to predict.

4. Realistic spread & discreteness.  Prices live on a tick grid; the quoted
   spread is a small integer number of ticks that widens with volatility.

Everything is driven by ``numpy.random.default_rng(seed)`` so runs are
reproducible.  Returns a tidy ``pandas.DataFrame``; one row = one bar (treated
as a 1-minute bar, so 200k bars ~ a couple of trading years and the annualizer
is periods_per_year = 252 * 390 = 98,280).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_orderbook(
    n_bars: int = 200_000,
    seed: int = 11,
    tick: float = 0.01,
    start_price: float = 100.0,
    impact: float = 0.9,           # mid move (in ticks) per unit OFI
    flow_phi: float = 0.55,        # AR(1) persistence of signed flow
    flow_scale: float = 1.0,       # std of the flow innovation
    book_to_flow: float = 1.6,     # how strongly size-imbalance leads next flow
    noise_ticks: float = 10.0,     # microstructure noise on the mid (ticks)
    base_spread_ticks: float = 1.0,
    vol_spread_ticks: float = 0.5,  # extra spread that scales with volatility
    base_size: float = 500.0,
    size_noise: float = 0.18,       # lognormal noise on resting sizes
) -> pd.DataFrame:
    """Generate a reduced-form LOB tape with embedded price impact.

    Parameters
    ----------
    n_bars : number of bars/events to simulate.
    impact : coefficient linking OFI_t to the mid move into t+1 (in ticks per
             unit of standardized OFI).  This is the planted alpha; keep it
             small so gross is large but net is thin.
    flow_phi : AR(1) coefficient making signed order flow persistent.
    book_to_flow : loading of the resting size-imbalance on the *next* bar's
             flow (gives book-pressure something real to predict).
    noise_ticks : std of i.i.d. microstructure noise added to each mid move.
    base_spread_ticks / vol_spread_ticks : spread (in ticks) = base + vol*|state|.

    Returns
    -------
    DataFrame with columns:
        mid, bid, ask, spread, bid_size, ask_size, ofi,
        signed_flow, book_imbalance
    indexed by a synthetic 1-second DatetimeIndex.
    """
    rng = np.random.default_rng(seed)

    # ---- latent volatility regime (slow) controls spread & flow intensity ----
    # A gently mean-reverting positive process; |vol_state| ~ O(1).
    vol_state = np.empty(n_bars)
    v = 1.0
    for t in range(n_bars):
        v = 0.995 * v + 0.005 * 1.0 + 0.05 * rng.standard_normal()
        vol_state[t] = abs(v)
    vol_state /= vol_state.mean()  # normalize to ~1 on average

    # ---- resting top-of-book sizes & their imbalance ----
    # Sizes are lognormal; we induce a persistent imbalance that will *lead* flow.
    imb_state = np.zeros(n_bars)
    z = 0.0
    for t in range(n_bars):
        z = 0.85 * z + np.sqrt(1 - 0.85 ** 2) * rng.standard_normal()
        imb_state[t] = z
    book_imbalance = np.tanh(0.9 * imb_state)  # in (-1, 1)

    bid_size = base_size * np.exp(size_noise * rng.standard_normal(n_bars)) * (1.0 + 0.5 * book_imbalance)
    ask_size = base_size * np.exp(size_noise * rng.standard_normal(n_bars)) * (1.0 - 0.5 * book_imbalance)
    bid_size = np.clip(bid_size, 1.0, None)
    ask_size = np.clip(ask_size, 1.0, None)

    # ---- signed order flow: persistent + led by lagged book imbalance ----
    # signed_flow_t = phi*flow_{t-1} + book_to_flow*book_imbalance_{t-1} + eps
    signed_flow = np.empty(n_bars)
    f = 0.0
    prev_imb = 0.0
    sig_innov = flow_scale * np.sqrt(1 - flow_phi ** 2)
    for t in range(n_bars):
        f = (flow_phi * f
             + book_to_flow * prev_imb
             + sig_innov * vol_state[t] * rng.standard_normal())
        signed_flow[t] = f
        prev_imb = book_imbalance[t]

    # ---- Order Flow Imbalance (OFI): signed flow weighted by available size ----
    # Standardize so `impact` has a stable interpretation across seeds.
    depth = 0.5 * (bid_size + ask_size)
    ofi_raw = signed_flow * np.log1p(depth) / np.log1p(base_size)
    ofi = (ofi_raw - ofi_raw.mean()) / ofi_raw.std()

    # ---- mid price: planted impact of OFI on the NEXT bar + noise ----
    # delta_mid_t (the move realized from t -> t+1) depends on OFI_t.
    noise = noise_ticks * rng.standard_normal(n_bars)
    delta_ticks = impact * ofi + noise            # in ticks
    # The move from bar t to t+1 is delta_ticks[t]; shift so mid[t] excludes it.
    mid_ticks = np.empty(n_bars)
    mid_ticks[0] = round(start_price / tick)
    mid_ticks[1:] = mid_ticks[0] + np.cumsum(delta_ticks[:-1])
    mid = mid_ticks * tick

    # ---- discrete spread (integer ticks, widens with vol) and quotes ----
    spread_ticks = np.round(base_spread_ticks + vol_spread_ticks * (vol_state - 1.0))
    spread_ticks = np.clip(spread_ticks, 1.0, None)
    half = (spread_ticks * tick) / 2.0
    # snap mid to the tick grid, then place symmetric quotes
    mid_snapped = np.round(mid / tick) * tick
    bid = mid_snapped - half
    ask = mid_snapped + half
    spread = ask - bid

    # one bar == one minute (matches periods_per_year = 252*390 in run.py)
    idx = pd.date_range("2020-01-02 09:30:00", periods=n_bars, freq="min")
    df = pd.DataFrame(
        {
            "mid": mid_snapped,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "ofi": ofi,
            "signed_flow": signed_flow,
            "book_imbalance": book_imbalance,
        },
        index=idx,
    )
    df.attrs["tick"] = tick
    df.attrs["impact"] = impact
    df.attrs["seed"] = seed
    return df


if __name__ == "__main__":
    # quick smoke-test / sanity check of the embedded structure
    ob = generate_orderbook(n_bars=50_000, seed=11)
    dmid = ob["mid"].diff().shift(-1)  # realized move t -> t+1
    corr = np.corrcoef(ob["ofi"].iloc[:-1], dmid.iloc[:-1])[0, 1]
    print(ob.head())
    print(f"\nrows={len(ob)}  mean spread={ob['spread'].mean():.4f}  "
          f"tick={ob.attrs['tick']}")
    print(f"corr(OFI_t, mid move t->t+1) = {corr:+.3f}  (planted impact)")
    print(f"mean |book_imbalance| = {ob['book_imbalance'].abs().mean():.3f}")
