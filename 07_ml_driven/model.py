"""HistGradientBoostingRegressor wrapper for cross-sectional return forecasting.

The model maps the point-in-time feature vector for a (date, stock) to a
prediction of that stock's forward H-day return. Predictions are used purely
*cross-sectionally* (ranked each day into a long/short book), so we care about
the ordering of predictions on a given date far more than their absolute level.

Kept deliberately modest (shallow-ish trees, capped iterations) so the full
walk-forward CV runs in a couple of minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURE_NAMES


def make_model(random_state: int = 0) -> HistGradientBoostingRegressor:
    """A small, regularized gradient-boosted tree regressor.

    Modest depth + learning rate + L2 regularization guard against the
    boosting model memorizing noise -- though as ``run.py`` demonstrates, even
    a regularized model overfits spectacularly if you evaluate it in-sample.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )


def fit_model(train: pd.DataFrame, random_state: int = 0) -> HistGradientBoostingRegressor:
    """Fit on a tidy panel slice (must contain FEATURE_NAMES + ``fwd_ret``)."""
    X = train[FEATURE_NAMES].to_numpy()
    y = train["fwd_ret"].to_numpy()
    model = make_model(random_state=random_state)
    model.fit(X, y)
    return model


def predict(model: HistGradientBoostingRegressor, panel: pd.DataFrame) -> pd.Series:
    """Predict forward returns for a panel slice; returns a (date,ticker) Series."""
    X = panel[FEATURE_NAMES].to_numpy()
    preds = model.predict(X)
    return pd.Series(preds, index=panel.index, name="pred")


def predictions_to_wide(preds: pd.Series) -> pd.DataFrame:
    """Long (date,ticker) predictions -> wide (dates x tickers) signal frame."""
    wide = preds.unstack("ticker")
    return wide.sort_index()
