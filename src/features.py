"""Shared feature engineering for training (offline) and inference (Flask API).

The model predicts the next-day log return of the BTC close price from
lagged returns and technical indicators. A 7-day forecast is produced
recursively: each predicted close is appended to the history and the
features are recomputed for the next step.
"""

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 5, 7, 14)
MIN_HISTORY = 31  # need 30 closes for the 30-day MA plus one for returns

FEATURE_COLUMNS = [f"ret_lag_{lag}" for lag in LAGS] + [
    "ma7_ratio",
    "ma30_ratio",
    "volatility_7",
    "volatility_30",
]


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, drop duplicates / non-positive prices, forward-fill gaps."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index).ffill()
    df.index.name = "date"
    df = df[df["close"] > 0]
    return df.reset_index()


def build_features(close: pd.Series) -> pd.DataFrame:
    """Compute model features from a close-price series (index = date)."""
    close = close.astype(float)
    log_ret = np.log(close).diff()
    feats = pd.DataFrame(index=close.index)
    for lag in LAGS:
        feats[f"ret_lag_{lag}"] = log_ret.shift(lag - 1)
    feats["ma7_ratio"] = close / close.rolling(7).mean() - 1.0
    feats["ma30_ratio"] = close / close.rolling(30).mean() - 1.0
    feats["volatility_7"] = log_ret.rolling(7).std()
    feats["volatility_30"] = log_ret.rolling(30).std()
    return feats[FEATURE_COLUMNS]


def build_training_frame(close: pd.Series) -> pd.DataFrame:
    """Features at day t with target = log return from t to t+1."""
    feats = build_features(close)
    feats["target"] = np.log(close).diff().shift(-1)
    return feats.dropna()


def recursive_forecast(model, history, days: int = 7):
    """Forecast `days` future closes from a list/array of past closes."""
    closes = [float(x) for x in history]
    if len(closes) < MIN_HISTORY:
        raise ValueError(f"history must contain at least {MIN_HISTORY} closing prices")
    preds = []
    for _ in range(days):
        feats = build_features(pd.Series(closes)).iloc[[-1]]
        next_ret = float(model.predict(feats.values)[0])
        next_close = closes[-1] * float(np.exp(next_ret))
        closes.append(next_close)
        preds.append(next_close)
    return preds
