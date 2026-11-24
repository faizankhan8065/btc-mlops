# BTC-USD 7-Day Forecast — Production MLOps Pipeline

Muhammad Faizan Khan (26L-9004, PhD CS) — MLOps Assignment 3

Flask + Gunicorn API · multi-stage Docker · GitHub Actions CI/CD · Airflow DAG · Kubeflow (KFP v2) · Kubernetes (Deployment, Service, ConfigMap/Secret, PVC, HPA, Ingress).

## Quick start
```bash
pip install -r requirements-dev.txt
python -m src.etl_fetch && python -m src.preprocess && python -m src.train_forecast   # add --allow-synthetic offline
pytest -v
docker build -f app/Dockerfile -t btc-forecast-api:latest .
docker compose up -d --build        # API :5000, Airflow :8080 (admin/admin)
python kubeflow/btc_kfp_pipeline.py # -> kubeflow/btc_kfp_pipeline.yaml
bash scripts_run_k8s_demo.sh        # Minikube deploy + chaos test + HPA load test, logs in docs/logs/k8s/
```

## API
| Method | Path | Description |
|---|---|---|
| GET | /health | status, uptime, model version |
| POST | /predict | `{"forecast_days": 7, "history": [optional >=31 closes]}` |
| GET | /metrics | MAE / RMSE / MAPE of latest validation |
| POST | /reload | hot-reload model (header `X-API-Key`) |

## CI/CD secrets
Set the repository secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` (a Docker Hub access token).

## Layout
`app/` API + Dockerfile · `src/` ETL, features, training · `dags/` Airflow · `kubeflow/` KFP · `k8s/` manifests · `tests/` pytest · `loadtest/` Locust and bench · `docs/` report and logs
