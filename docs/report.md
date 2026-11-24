---
title: "Assignment 3 — Production MLOps Pipeline for 7-Day Bitcoin Forecasting"
subtitle: "Flask · Docker · GitHub Actions · Airflow · Kubeflow · Kubernetes"
---

<div class="authors">

| | |
|---|---|
| **Student** | Muhammad Faizan Khan — Roll No. **26L-9004** — **PhD (Computer Science)** |
| **Group partner** | _[Name — Roll No.]_ |
| **Course** | MLOps — Assignment 3 (60 marks) |
| **GitHub repository** | _[https://github.com/&lt;user&gt;/btc-mlops]_ |
| **Docker Hub images** | _[https://hub.docker.com/r/&lt;user&gt;/btc-forecast-api]_ |
| **Date** | 24 November 2026 |

</div>

## 1. System Architecture

<div class="arch">
<svg viewBox="0 0 900 300" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial" font-size="13">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#334"/></marker></defs>
<g stroke="#334" fill="none" stroke-width="1.4">
<rect x="10" y="20" width="150" height="50" rx="6" fill="#eef4ff"/><rect x="10" y="110" width="150" height="50" rx="6" fill="#eef4ff"/><rect x="10" y="200" width="150" height="70" rx="6" fill="#f3f3f3"/>
<rect x="220" y="15" width="300" height="160" rx="8" fill="#fff7e8"/>
<rect x="220" y="200" width="300" height="70" rx="8" fill="#f0fff3"/>
<rect x="580" y="15" width="310" height="255" rx="8" fill="#eefaff"/>
</g>
<text x="20" y="42">yfinance / CoinGecko</text><text x="20" y="60" font-size="11">daily BTC-USD OHLCV</text>
<text x="20" y="132">GitHub (push / PR)</text><text x="20" y="150" font-size="11">GitHub Actions CI/CD</text>
<text x="20" y="222">Docker Hub</text><text x="20" y="240" font-size="11">btc-forecast-api:&lt;sha&gt;</text><text x="20" y="256" font-size="11">btc-pipeline:&lt;sha&gt;</text>
<text x="232" y="37" font-weight="bold">Apache Airflow DAG (daily) / Kubeflow KFP</text>
<text x="232" y="62">fetch_data → preprocess_features →</text>
<text x="232" y="82">train_and_evaluate → check_threshold →</text>
<text x="232" y="102">reload_api_model (POST /reload)</text>
<text x="232" y="130" font-size="11">Postgres metadata DB · KFP: ingest → features → train</text>
<text x="232" y="150" font-size="11">Gate: register model only if 7-day MAPE ≤ threshold</text>
<text x="232" y="222" font-weight="bold">Shared model store</text>
<text x="232" y="242" font-size="11">PVC btc-models-pvc → /app/models (model.pkl, metrics.json)</text>
<text x="590" y="37" font-weight="bold">Kubernetes (Minikube) — namespace btc-mlops</text>
<text x="590" y="62">Ingress (nginx) host btc-forecast.local</text>
<text x="590" y="84">Service ClusterIP :80→5000 · NodePort 30080</text>
<text x="590" y="106">Deployment btc-api (3 replicas, Gunicorn)</text>
<text x="600" y="126" font-size="11">liveness/readiness GET /health · 100m/128Mi → 500m/512Mi</text>
<text x="590" y="150">HPA 2–8 pods @ 60% CPU (metrics-server)</text>
<text x="590" y="174">ConfigMap (MODEL_STAGE, LOG_LEVEL, PORT)</text>
<text x="590" y="196">Secret (API_KEY) → env vars</text>
<text x="590" y="222">Endpoints: /health /predict /metrics /reload</text>
<g stroke="#334" stroke-width="1.4" marker-end="url(#a)"><line x1="160" y1="45" x2="218" y2="60"/><line x1="160" y1="135" x2="85" y2="198"/><line x1="160" y1="235" x2="578" y2="250"/><line x1="370" y1="175" x2="370" y2="198"/><line x1="520" y1="235" x2="578" y2="200"/><line x1="520" y1="100" x2="578" y2="100"/></g>
</svg>
</div>

**Model.** A scikit-learn `GradientBoostingRegressor` predicts the next-day log return from lagged returns (1, 2, 3, 5, 7, 14 days), MA7/MA30 price ratios and 7/30-day volatility. The 7-day forecast is produced recursively. Feature code (`src/features.py`) is shared by training and serving to prevent training/serving skew, and a unit test checks there is no look-ahead leakage. Validation holds out the latest 7 days, computes MAE/RMSE/MAPE, and compares them against a naive "last close" baseline. The model is promoted atomically (temp file + rename) only if the MAPE is at or below the threshold.

> **Note on data used for the logs in this report.** The build sandbox had no outbound access to Yahoo Finance or CoinGecko (proxy 403, shown in the logs). The ETL therefore fell back to its explicit **synthetic** mode (`ALLOW_SYNTHETIC=1`, a geometric random walk), and every artifact records `data_source: synthetic`. On a random walk no model can beat the naive baseline, which matches the result (MAPE 9.58% vs 7.28% baseline). With internet access the identical code ingests real BTC-USD data; _[replace these numbers with the real-data run]_.

## 2. Part A — Flask REST API (`app/app.py`)

| Endpoint | Behaviour |
|---|---|
| `GET /health` | status, uptime, model version, stage, pod name (used by K8s probes) |
| `POST /predict` | `forecast_days` (1–30, default 7) and optional `history` (≥31 closes); returns dated predictions |
| `GET /metrics` | MAE / RMSE / MAPE and naive baseline from the latest validation run |
| `POST /reload` | hot-reloads `model.pkl` from the PVC; protected by `X-API-Key` (K8s Secret) |

Validation and error handling: JSON content-type and body checks, type and range checks, JSON 400/401/404/405/500/503 responses, and a global exception handler. Gunicorn (`app/gunicorn.conf.py`, gthread workers) is the production WSGI server. Real output from the running container:

```text
$ docker exec btc-api id
uid=10001(app) gid=10001(app) groups=10001(app)
$ curl -s localhost:5000/health
{"model_loaded":true,"model_loaded_at":"2026-11-24T14:15:37.349404+00:00","model_stage":"production","model_version":"v20261124141407","pod":"d1401987bdea","status":"ok","uptime_seconds":1266.68}
$ curl -s -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"forecast_days": 7}'
{"forecast_days":7,"input":"stored_history","last_observed_close":39472.55,"model_version":"v20261124141407","predictions":[{"date":"2026-11-25","predicted_close":39197.75},
  {"date":"2026-11-26","predicted_close":39531.75},
  {"date":"2026-11-27","predicted_close":39513.82},
  {"date":"2026-11-28","predicted_close":39746.41},
  {"date":"2026-11-29","predicted_close":39737.08},
  {"date":"2026-11-30","predicted_close":39870.84},
  {"date":"2026-12-01","predicted_close":40429.66}],"status":"ok"}
$ curl -s localhost:5000/metrics
{"model_version": "v20261124141407", "metrics": {"mae": 3890.726582988192, "mape": 9.578856674073526, "rmse": 4219.982474890972}, "baseline_naive_metrics": {"mae": 2959.303876941211, "mape": 7.277346430868679, "rmse": 3143.14037335182}, "data_source": "synthetic"}
$ curl ... -d '{"forecast_days": 0}'     -> {"code":400,"message":"forecast_days must be an integer between 1 and 30","status":"error"}
 HTTP 400
$ curl ... -d '{bad json'                -> {"code":400,"message":"request body is not valid JSON","status":"error"}
 HTTP 400
$ curl ... -d '{"history": [1,2,3]}'    -> {"code":400,"message":"history must be a list of at least 31 closing prices","status":"error"}
 HTTP 400
$ curl -X POST /reload (no key)          -> {"code":401,"message":"invalid or missing X-API-Key","status":"error"}
 HTTP 401
$ curl -X POST /reload -H 'X-API-Key: …' -> {"model_version":"v20261124143510","status":"reloaded"}
 HTTP 200
```

**pytest** (14 tests: endpoints, validation, malformed JSON, auth, 404/405, metric maths, leakage check):

```text
tests/test_api.py::test_health PASSED                                    [  7%]
tests/test_api.py::test_predict_default_7_days PASSED                    [ 14%]
tests/test_api.py::test_predict_no_body PASSED                           [ 21%]
tests/test_api.py::test_predict_custom_days_and_history PASSED           [ 28%]
tests/test_api.py::test_predict_invalid_days PASSED                      [ 35%]
tests/test_api.py::test_predict_invalid_history PASSED                   [ 42%]
tests/test_api.py::test_predict_malformed_json PASSED                    [ 50%]
tests/test_api.py::test_metrics PASSED                                   [ 57%]
tests/test_api.py::test_reload_requires_key PASSED                       [ 64%]
tests/test_api.py::test_unknown_endpoint_and_method PASSED               [ 71%]
tests/test_pipeline.py::test_features_have_no_lookahead PASSED           [ 78%]
tests/test_pipeline.py::test_metrics_values PASSED                       [ 85%]
tests/test_pipeline.py::test_training_artifacts PASSED                   [ 92%]
tests/test_pipeline.py::test_recursive_forecast_rejects_short_history PASSED [100%]
============================== 14 passed in 1.38s ==============================
```

## 3. Part B — Containerization & CI/CD

**Multi-stage Dockerfile** (`app/Dockerfile`): the builder stage compiles wheels and the runtime stage installs from those wheels only. It uses `python:3.10-slim`, runs as non-root UID 10001 (verified: `uid=10001(app)`), has a `HEALTHCHECK`, and orders layers so that requirements are cached before code. `.dockerignore` excludes data, tests and docs. A second image (`Dockerfile.pipeline`) runs the ETL and training steps for Airflow and Kubeflow.

```text
#10 [builder 4/4] RUN --mount=type=secret,id=ca,required=false if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; pip wheel --wheel-
#14 [builder 2/4] WORKDIR /build
#7 [builder 1/4] FROM docker.io/library/python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d
#9 [builder 3/4] COPY app/requirements.txt .
#11 [runtime 3/7] COPY --from=builder /wheels /wheels
#12 [runtime 4/7] RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels && groupadd --system --gid 10001 app && useradd --system
#13 [runtime 2/7] WORKDIR /app
#15 [runtime 6/7] COPY --chown=app:app app/app.py app/gunicorn.conf.py ./
#16 [runtime 7/7] COPY --chown=app:app models/ ./models/
#8 [runtime 5/7] COPY --chown=app:app src/__init__.py src/features.py ./src/
#17 naming to docker.io/library/btc-forecast-api:latest done
$ docker images
btc-forecast-api:latest  812MB
btc-pipeline:latest  665MB
python:3.10-slim  186MB
```

**docker-compose.yml** starts Postgres, airflow-init, the Airflow webserver and scheduler, and the Flask API. A shared `models` volume lets the DAG's retrained model be hot-reloaded by the API.

**GitHub Actions** (`.github/workflows/ci_cd.yml`) runs on every push or PR to `main`. Job `lint-test` runs flake8, `black --check` and pytest, compiles the KFP pipeline, and uploads the JUnit report. Job `build-push` runs only on push to main after tests pass. It logs in with `secrets.DOCKER_USERNAME` / `secrets.DOCKER_PASSWORD`, then builds and pushes both images tagged with `${{ github.sha }}` and `latest`, using a GHA layer cache. Local lint results:

```text
$ flake8 app src tests dags kubeflow
flake8: 0 issues
$ black --check app src tests kubeflow dags
All done! ✨ 🍰 ✨
13 files would be left unchanged.
```

_[Screenshot: GitHub Actions run — both jobs green]_ &nbsp; _[Screenshot: Docker Hub tags sha + latest]_

## 4. Part C — Airflow ETL DAG & Kubeflow Pipeline

`dags/btc_etl_pipeline.py` is scheduled `@daily` with `catchup=False`, `max_active_runs=1` and 2 retries. Its tasks are **fetch_data** (yfinance → CoinGecko fallback), **preprocess_features** (cleaning, daily reindex, MA7/MA30, volatility), **train_and_evaluate** (7-day hold-out metrics pushed to XCom), **check_threshold** (`ShortCircuitOperator`, which skips the reload when the model is rejected) and **reload_api_model** (POST `/reload`, then checks that the served version equals the new version and logs a fresh forecast). Real manual execution (`airflow dags test`, Airflow 2.9.3) against the running API container:

```text
$ airflow dags list
dag_id           | fileloc                               | owners | is_paused
=================+=======================================+========+==========
btc_etl_pipeline | /opt/airflow/dags/btc_etl_pipeline.py | mlops  | None     
                                                                             

$ airflow tasks list btc_etl_pipeline --tree
<Task(PythonOperator): fetch_data>
    <Task(PythonOperator): preprocess_features>
        <Task(PythonOperator): train_and_evaluate>
            <Task(ShortCircuitOperator): check_threshold>
                <Task(PythonOperator): reload_api_model>
Marking task as SUCCESS. dag_id=btc_etl_pipeline, task_id=fetch_data
Marking task as SUCCESS. dag_id=btc_etl_pipeline, task_id=preprocess_features
Model v20261124143510 REGISTERED (MAPE 9.58% <= 10.00%)
Marking task as SUCCESS. dag_id=btc_etl_pipeline, task_id=train_and_evaluate
Marking task as SUCCESS. dag_id=btc_etl_pipeline, task_id=check_threshold
API now serving v20261124143510 (expected v20261124143510)
Marking task as SUCCESS. dag_id=btc_etl_pipeline, task_id=reload_api_model
DagRun Finished: dag_id=btc_etl_pipeline, execution_date=2026-11-24 00:00:00+00:00
```

After the run, `/health` on the API reported the new version, `v20261124143510` (the old one was `v20261124141407`). This confirms the end-to-end update with no restart. _[Screenshot: Airflow UI Graph view of btc_etl_pipeline, all tasks green]_

**Kubeflow** (`kubeflow/btc_kfp_pipeline.py`, KFP v2 SDK) defines three `@dsl.container_component`s, `ingest_data → build_features → train_model`, which run the `btc-pipeline` image. Data and model are passed as typed `Dataset`/`Model`/`Metrics` artifacts so KFP tracks lineage. The pipeline compiles to `kubeflow/btc_kfp_pipeline.yaml` (this step also runs in CI) and can be uploaded to the KFP UI with a recurring run.

## 5. Part D — Kubernetes Deployment & Scaling

| Manifest | Key settings |
|---|---|
| `configmap-secret.yaml` | ConfigMap: MODEL_STAGE, LOG_LEVEL, PORT, MODEL_DIR; Secret: API_KEY (via `envFrom`) |
| `pvc.yaml` | 1 Gi RWO PVC mounted at `/app/models`; an initContainer seeds it from the image on first start |
| `deployment.yaml` | 3 replicas, RollingUpdate; liveness `/health` (initialDelay 15 s, period 10 s); readiness `/health`; requests 100m/128Mi, limits 500m/512Mi; runAsNonRoot, drop ALL capabilities |
| `service.yaml` | ClusterIP 80→5000 and NodePort 30080 (80→5000) |
| `hpa.yaml` | autoscaling/v2, min 2 / max 8, CPU 60% (plus memory 80%), fast scale-up, 60 s scale-down window |
| `ingress.yaml` | ingressClassName nginx, host `btc-forecast.local` → btc-api:80 |

`scripts_run_k8s_demo.sh` automates the full Minikube run and saves every output to `docs/logs/k8s/`. It starts Minikube with the metrics-server and ingress addons, builds the image inside Minikube, applies the manifests, runs `kubectl get pods/svc/hpa/ingress/pvc`, sends API calls through the NodePort and the Ingress, performs a pod-deletion chaos test, and runs an in-cluster load generator while recording `kubectl get hpa --watch`.

_[Paste: kubectl get pods,svc,hpa,ingress,pvc — docs/logs/k8s/03_get_all.log]_
_[Screenshot: kubectl get hpa --watch showing replicas 2 → N under load, then scale-down]_

**Load-test baseline (single container, 2 Gunicorn workers × 2 threads)**, measured with `loadtest/quick_bench.py`. Locust users can run `loadtest/locustfile.py` instead:

```text
url=http://localhost:5000/predict concurrency=1 duration=15.0s
requests=698 errors=0 throughput=46.5 req/s
latency ms: mean=21.5 p50=20.4 p95=28.1 p99=36.9

url=http://localhost:5000/predict concurrency=4 duration=15.0s
requests=1046 errors=0 throughput=69.7 req/s
latency ms: mean=57.4 p50=58.3 p95=113.3 p99=133.1

url=http://localhost:5000/predict concurrency=8 duration=15.0s
requests=956 errors=0 throughput=63.7 req/s
latency ms: mean=126.0 p50=127.2 p95=222.7 p99=250.7
```

Throughput saturates at about 65–70 req/s per pod once concurrency reaches 4, and latency then grows linearly. This is the CPU-bound regime in which the HPA's 60% CPU target adds pods. With a 500m CPU limit per pod the saturation point is lower, so scale-out triggers early.

## 6. Part E — Verification & Resilience

- **E1 Self-healing:** `kubectl delete pod` on one replica makes the ReplicaSet create a replacement immediately. The new pod passes its readiness probe before it receives traffic, while the Service keeps routing to the 2 healthy pods, so there is no downtime. _[Paste docs/logs/k8s/05_self_healing.log]_
- **E2 End-to-end:** the Airflow manual run (Section 4) retrained and registered a new model and hot-reloaded the API. The CI workflow tests, builds and pushes the images. _[GitHub Actions screenshot; curl to the K8s NodePort/Ingress returning the 7-day forecast]_
- **Bugs found during verification and fixed:** (1) a model pickled with a newer scikit-learn/NumPy failed to load in the pinned image. Training now runs in the pinned `btc-pipeline` image and in CI. (2) `shutil.copy` failed on a shared volume (EPERM on chmod). It was replaced by an atomic `copyfile` + `os.replace`, which also stops the API from reading a half-written model.

## 7. Conclusion

The system automates the full lifecycle: ingestion → features → retraining with a quality gate → zero-downtime model hot-reload. It is containerized, CI-tested, and deployed on Kubernetes with probes, autoscaling, persistent model storage and externalized configuration. Future work: an LSTM or Prophet model compared against the GBM, MLflow model registry, Prometheus/Grafana monitoring, and drift detection that triggers retraining.
