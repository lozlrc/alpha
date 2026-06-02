"""Purged + embargoed walk-forward cross-validation.

WHY THIS EXISTS  (the central lesson of this strategy family)
-------------------------------------------------------------
Our label is a *forward* H-day return: ``fwd_ret[t] = P[t+H]/P[t] - 1``. That
means the label attached to date ``t`` is not known until ``t+H``, and two
training rows whose dates are within H of each other share *overlapping future
windows*. If we naively train on everything before a test block, training rows
in the final H days before the test block peek into returns realized *inside*
the test block -- classic label leakage that inflates out-of-sample metrics.

Two defenses, applied on top of a strictly forward walk:

1. PURGE -- drop any TRAIN row whose label window ``[t, t+H]`` overlaps the
   test block's date span. Those rows literally contain test-period returns.

2. EMBARGO -- additionally drop train rows in a small gap (~H days) *after*
   the test block. Serial correlation / slow-moving features mean rows just
   after the test set are still informationally entangled with it; the
   embargo enforces a clean buffer (Lopez de Prado, *Advances in Financial
   Machine Learning*).

``walk_forward_oos`` slides sequential test blocks across the timeline, trains
an honest model on the purged+embargoed past for each, and stitches the test-
block predictions into a single OUT-OF-SAMPLE prediction panel covering (almost)
the whole period -- the signal ``run.py`` then backtests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, H
from model import fit_model, predict


def _date_blocks(dates: pd.DatetimeIndex, n_splits: int, min_train: int):
    """Sequential, equal-ish test blocks over the unique sorted dates.

    The first ``min_train`` dates are reserved as the initial training base
    (never used as a test block), the remainder is chopped into ``n_splits``
    contiguous test blocks. Yields (test_start_date, test_end_date) pairs.
    """
    udates = pd.DatetimeIndex(sorted(dates.unique()))
    n = len(udates)
    if min_train >= n:
        raise ValueError("min_train too large for the available history")
    test_dates = udates[min_train:]
    blocks = np.array_split(test_dates, n_splits)
    for blk in blocks:
        if len(blk):
            yield blk[0], blk[-1]


def walk_forward_oos(panel: pd.DataFrame, n_splits: int = 8, min_train: int = 400,
                     horizon: int = H, embargo: int | None = None,
                     random_state: int = 0, verbose: bool = True) -> pd.Series:
    """Generate purged+embargoed walk-forward OUT-OF-SAMPLE predictions.

    Parameters
    ----------
    panel    : tidy (date,ticker) panel from ``features.build_panel``.
    n_splits : number of sequential test blocks.
    min_train: trading dates held back as the initial training window.
    horizon  : label horizon H (drives the purge window length).
    embargo  : extra buffer of dates dropped after each test block (default H).

    Returns a (date,ticker) Series of OOS predictions (test blocks only).
    """
    embargo = horizon if embargo is None else embargo
    dates = panel.index.get_level_values("date")
    row_date = pd.Series(dates, index=panel.index)

    oos_parts: list[pd.Series] = []
    for i, (t0, t1) in enumerate(_date_blocks(dates, n_splits, min_train), start=1):
        # Test block = rows whose date falls inside [t0, t1].
        test_mask = (row_date >= t0) & (row_date <= t1)

        # PURGE: a train row dated d carries a label over [d, d+H]. It leaks if
        # that window reaches the test block, i.e. d + H >= t0. Combined with
        # "strictly before the block" this drops the contaminated tail.
        # EMBARGO: also drop the [t1, t1 + embargo] buffer just after the block.
        embargo_end = t1 + pd.Timedelta(days=int(embargo * 1.6))  # cal-day pad ~ embargo bdays
        purge_start = t0 - pd.Timedelta(days=int(horizon * 1.6))
        leak_mask = (row_date >= purge_start) & (row_date <= embargo_end)
        train_mask = (~test_mask) & (~leak_mask) & (row_date < t0 - pd.Timedelta(days=0))
        # Keep only genuine past: train rows must predate the test block start.
        train_mask = train_mask & (row_date < t0)

        train = panel[train_mask]
        test = panel[test_mask]
        if len(train) < 1000 or test.empty:
            if verbose:
                print(f"  fold {i}: skipped (train={len(train)}, test={len(test)})")
            continue

        model = fit_model(train, random_state=random_state)
        oos_parts.append(predict(model, test))
        if verbose:
            print(f"  fold {i}: test {t0.date()}..{t1.date()}  "
                  f"train_rows={len(train):>6}  test_rows={len(test):>6}")

    if not oos_parts:
        raise RuntimeError("no OOS predictions produced -- check split sizing")
    return pd.concat(oos_parts).sort_index().rename("pred")


def fit_leaky_insample(panel: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """The OVERFITTING TRAP: fit on the ENTIRE panel and predict the SAME rows.

    No walk-forward, no purge, no embargo -- the model is scored on data it was
    trained on, and every label window leaks freely. The resulting "signal" will
    post an absurd Sharpe that evaporates out of sample. Used in ``run.py`` as a
    cautionary baseline, never as a real strategy.
    """
    model = fit_model(panel, random_state=random_state)
    return predict(model, panel)
