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


def _feature_cols(panel: pd.DataFrame) -> list[str]:
    """Feature columns = every column except the forward-return label. Inferred from
    the panel so the model adapts to whatever features exist (all 11 on synthetic
    data; the 8 price/volume ones on a real price feed, which has no fundamentals)."""
    return [c for c in panel.columns if c != "fwd_ret"]


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
    """Fit on a tidy panel slice (feature columns + ``fwd_ret``)."""
    feats = _feature_cols(train)
    X = train[feats].to_numpy()
    y = train["fwd_ret"].to_numpy()
    model = make_model(random_state=random_state)
    model.fit(X, y)
    return model


def predict(model: HistGradientBoostingRegressor, panel: pd.DataFrame) -> pd.Series:
    """Predict forward returns for a panel slice; returns a (date,ticker) Series."""
    X = panel[_feature_cols(panel)].to_numpy()
    preds = model.predict(X)
    return pd.Series(preds, index=panel.index, name="pred")


def predictions_to_wide(preds: pd.Series) -> pd.DataFrame:
    """Long (date,ticker) predictions -> wide (dates x tickers) signal frame."""
    wide = preds.unstack("ticker")
    return wide.sort_index()
