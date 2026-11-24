"""Airflow DAG: daily BTC-USD ETL -> feature engineering -> retrain/evaluate -> API reload.

Runs inside the Airflow containers from docker-compose.yml, where the project is
mounted at /opt/project and the shared model volume at /opt/project/models
(the same volume the Flask API reads from).
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator, ShortCircuitOperator

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/project")
sys.path.insert(0, PROJECT_DIR)

RAW = os.path.join(PROJECT_DIR, "data/raw/btc_usd.csv")
PROCESSED = os.path.join(PROJECT_DIR, "data/processed/btc_features.csv")
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(PROJECT_DIR, "models"))
API_URL = os.getenv("BTC_API_URL", "http://btc-api:5000")
MAPE_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "10"))

log = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def fetch_data(**_):
    from src import etl_fetch

    etl_fetch.run(RAW, allow_synthetic=os.getenv("ALLOW_SYNTHETIC", "0") == "1")
    return RAW


def preprocess_features(**_):
    from src import preprocess

    preprocess.run(RAW, PROCESSED)
    return PROCESSED


def train_and_evaluate(ti, **_):
    from src import train_forecast

    report = train_forecast.train(PROCESSED, MODEL_DIR, MAPE_THRESHOLD)
    log.info("Training report: %s", json.dumps(report, indent=2))
    ti.xcom_push(key="metrics", value=report["metrics"])
    ti.xcom_push(key="model_version", value=report["model_version"])
    return report["passed_threshold"]


def model_passed(ti, **_):
    passed = ti.xcom_pull(task_ids="train_and_evaluate")
    log.info("Model passed threshold: %s", passed)
    return bool(passed)


def reload_api(ti, **_):
    import requests

    headers = {"X-API-Key": os.getenv("API_KEY", "")}
    resp = requests.post(f"{API_URL}/reload", headers=headers, timeout=30)
    if resp.status_code != 200:
        raise AirflowFailException(f"Reload failed: {resp.status_code} {resp.text}")
    expected = ti.xcom_pull(task_ids="train_and_evaluate", key="model_version")
    served = resp.json()["model_version"]
    log.info("API now serving %s (expected %s)", served, expected)
    forecast = requests.post(f"{API_URL}/predict", json={"forecast_days": 7}, timeout=30).json()
    log.info("Fresh 7-day forecast: %s", json.dumps(forecast["predictions"]))


with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, retraining and model hot-reload",
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["btc", "mlops", "forecast"],
) as dag:
    t_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t_pre = PythonOperator(task_id="preprocess_features", python_callable=preprocess_features)
    t_train = PythonOperator(task_id="train_and_evaluate", python_callable=train_and_evaluate)
    t_gate = ShortCircuitOperator(task_id="check_threshold", python_callable=model_passed)
    t_reload = PythonOperator(task_id="reload_api_model", python_callable=reload_api)

    t_fetch >> t_pre >> t_train >> t_gate >> t_reload
