"""Train the 7-day BTC close-price forecaster and save model artifacts.

Model: scikit-learn GradientBoostingRegressor on lagged log returns and
technical indicators, forecasting recursively 7 days ahead.

Evaluation: the last 7 days are held out; the model trained on earlier data
forecasts them recursively and MAE / RMSE / MAPE are computed. The result is
compared with a naive "last close" baseline. The model is registered (copied to
the serving path) only if MAPE <= MAPE_THRESHOLD.
"""

import argparse
import json
import logging
import os
import pickle
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features import FEATURE_COLUMNS, MIN_HISTORY, build_training_frame, recursive_forecast

log = logging.getLogger("train_forecast")
HORIZON = 7


def make_model():
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42)


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100),
    }


def train(processed_path: str, model_dir: str, mape_threshold: float) -> dict:
    df = pd.read_csv(processed_path, parse_dates=["date"]).set_index("date")
    close = df["close"]
    if len(close) < MIN_HISTORY + HORIZON + 60:
        raise ValueError(f"Not enough data to train: {len(close)} rows")

    # --- Validation on the latest 7-day window --------------------------------
    train_close, test_close = close.iloc[:-HORIZON], close.iloc[-HORIZON:]
    frame = build_training_frame(train_close)
    val_model = make_model().fit(frame[FEATURE_COLUMNS].values, frame["target"].values)
    preds = recursive_forecast(val_model, train_close.values, HORIZON)
    model_metrics = metrics(test_close.values, preds)
    baseline = metrics(test_close.values, [train_close.iloc[-1]] * HORIZON)
    log.info("Validation metrics: %s | naive baseline: %s", model_metrics, baseline)

    # --- Refit on the full history for serving --------------------------------
    full = build_training_frame(close)
    model = make_model().fit(full[FEATURE_COLUMNS].values, full["target"].values)

    version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    source_file = os.getenv("RAW_DATA_PATH", "data/raw/btc_usd.csv") + ".source"
    source = open(source_file).read().strip() if os.path.exists(source_file) else "unknown"
    passed = model_metrics["mape"] <= mape_threshold
    report = {
        "model_version": version,
        "model_type": "GradientBoostingRegressor (recursive 7-day)",
        "data_source": source,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(full)),
        "last_date": close.index[-1].strftime("%Y-%m-%d"),
        "validation_window": [d.strftime("%Y-%m-%d") for d in test_close.index],
        "validation_actual": [round(float(x), 2) for x in test_close.values],
        "validation_predicted": [round(float(x), 2) for x in preds],
        "metrics": model_metrics,
        "baseline_naive_metrics": baseline,
        "mape_threshold": mape_threshold,
        "passed_threshold": passed,
    }
    artifact = {
        "model": model,
        "history": [float(x) for x in close.values[-120:]],
        "last_date": report["last_date"],
        "version": version,
        "metrics": report,
    }

    os.makedirs(model_dir, exist_ok=True)
    candidate = os.path.join(model_dir, "candidate_model.pkl")
    with open(candidate, "wb") as fh:
        pickle.dump(artifact, fh)
    with open(os.path.join(model_dir, "candidate_metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    if passed:
        # Atomic promotion: write to a temp file then rename, so the API never reads a half-written model.
        for src_name, dst_name in (("candidate_model.pkl", "model.pkl"), ("candidate_metrics.json", "metrics.json")):
            tmp = os.path.join(model_dir, f".{dst_name}.tmp")
            shutil.copyfile(os.path.join(model_dir, src_name), tmp)
            os.replace(tmp, os.path.join(model_dir, dst_name))
        log.info("Model %s REGISTERED (MAPE %.2f%% <= %.2f%%)", version, model_metrics["mape"], mape_threshold)
    else:
        log.warning("Model %s REJECTED (MAPE %.2f%% > %.2f%%)", version, model_metrics["mape"], mape_threshold)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", default=os.getenv("PROCESSED_DATA_PATH", "data/processed/btc_features.csv"))
    p.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "models"))
    p.add_argument("--mape-threshold", type=float, default=float(os.getenv("MAPE_THRESHOLD", "10")))
    a = p.parse_args()
    report = train(a.processed, a.model_dir, a.mape_threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
