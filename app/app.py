"""Flask REST API serving the 7-day BTC-USD close-price forecast.

Endpoints
  GET  /health   status, uptime, loaded model version
  POST /predict  {"forecast_days": 1..30, "history": [optional list of >=31 closes]}
  GET  /metrics  MAE / RMSE / MAPE of the latest validation run
  POST /reload   re-read model weights from MODEL_DIR (called by Airflow);
                 requires header X-API-Key when API_KEY is set
"""

import logging
import os
import pickle
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import MIN_HISTORY, recursive_forecast  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("btc-api")

MAX_FORECAST_DAYS = 30
START_TIME = time.time()
_lock = threading.Lock()
_state = {"artifact": None, "loaded_at": None, "error": None}


def model_path() -> str:
    return os.path.join(os.getenv("MODEL_DIR", "models"), "model.pkl")


def load_model() -> bool:
    path = model_path()
    try:
        with open(path, "rb") as fh:
            artifact = pickle.load(fh)
        with _lock:
            _state.update(artifact=artifact, loaded_at=datetime.now(timezone.utc).isoformat(), error=None)
        log.info("Loaded model %s from %s", artifact["version"], path)
        return True
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _state["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Could not load model from %s: %s", path, exc)
        return False


def create_app() -> Flask:
    app = Flask(__name__)
    load_model()

    def error(status: int, message: str):
        return jsonify({"status": "error", "code": status, "message": message}), status

    @app.errorhandler(404)
    def not_found(_):
        return error(404, "endpoint not found")

    @app.errorhandler(405)
    def bad_method(_):
        return error(405, "method not allowed")

    @app.errorhandler(Exception)
    def unhandled(exc):
        log.exception("Unhandled error: %s", exc)
        return error(500, "internal server error")

    @app.get("/health")
    def health():
        art = _state["artifact"]
        return jsonify(
            {
                "status": "ok" if art else "degraded",
                "uptime_seconds": round(time.time() - START_TIME, 2),
                "model_loaded": art is not None,
                "model_version": art["version"] if art else None,
                "model_stage": os.getenv("MODEL_STAGE", "production"),
                "model_loaded_at": _state["loaded_at"],
                "pod": os.getenv("HOSTNAME"),
            }
        )

    @app.get("/metrics")
    def metrics():
        art = _state["artifact"]
        if art is None:
            return error(503, "model not loaded")
        m = art["metrics"]
        return jsonify(
            {
                "model_version": art["version"],
                "metrics": m["metrics"],
                "baseline_naive_metrics": m.get("baseline_naive_metrics"),
                "validation_window": m.get("validation_window"),
                "trained_at": m.get("trained_at"),
                "data_source": m.get("data_source"),
            }
        )

    @app.post("/predict")
    def predict():
        art = _state["artifact"]
        if art is None:
            return error(503, "model not loaded")
        if request.data and not request.is_json:
            return error(400, "Content-Type must be application/json")
        payload = request.get_json(silent=True)
        if request.data and payload is None:
            return error(400, "request body is not valid JSON")
        payload = payload or {}
        if not isinstance(payload, dict):
            return error(400, "JSON body must be an object")

        days = payload.get("forecast_days", 7)
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_FORECAST_DAYS:
            return error(400, f"forecast_days must be an integer between 1 and {MAX_FORECAST_DAYS}")

        history = payload.get("history")
        if history is not None:
            if not isinstance(history, list) or len(history) < MIN_HISTORY:
                return error(400, f"history must be a list of at least {MIN_HISTORY} closing prices")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x is None or x <= 0 for x in history):
                return error(400, "history values must be positive numbers")
            start = datetime.now(timezone.utc).date() + timedelta(days=1)
            source = "custom_history"
        else:
            history = art["history"]
            start = datetime.strptime(art["last_date"], "%Y-%m-%d").date() + timedelta(days=1)
            source = "stored_history"

        try:
            preds = recursive_forecast(art["model"], history, days)
        except ValueError as exc:
            return error(400, str(exc))

        return jsonify(
            {
                "status": "ok",
                "model_version": art["version"],
                "input": source,
                "forecast_days": days,
                "last_observed_close": round(float(history[-1]), 2),
                "predictions": [
                    {"date": (start + timedelta(days=i)).isoformat(), "predicted_close": round(p, 2)}
                    for i, p in enumerate(preds)
                ],
            }
        )

    @app.post("/reload")
    def reload_model():
        api_key = os.getenv("API_KEY")
        if api_key and request.headers.get("X-API-Key") != api_key:
            return error(401, "invalid or missing X-API-Key")
        if not load_model():
            return error(500, f"reload failed: {_state['error']}")
        return jsonify({"status": "reloaded", "model_version": _state["artifact"]["version"]})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
