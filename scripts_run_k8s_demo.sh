#!/usr/bin/env bash
# Part D + E evidence collector for Minikube. Run from the repo root on your machine:
#   bash scripts_run_k8s_demo.sh
# Every command's output is saved under docs/logs/k8s/ for the report.
set -uo pipefail
OUT=docs/logs/k8s; mkdir -p "$OUT"
run() { echo "\$ $*" | tee -a "$OUT/$LOG"; "$@" 2>&1 | tee -a "$OUT/$LOG"; echo | tee -a "$OUT/$LOG"; }

LOG=01_setup.log
run minikube start --cpus=4 --memory=6g
run minikube addons enable metrics-server
run minikube addons enable ingress
# Build the image directly inside Minikube's Docker so no registry pull is needed
run docker build -f app/Dockerfile -t btc-forecast-api:latest .
run minikube image load btc-forecast-api:latest

LOG=02_apply.log
run kubectl apply -f k8s/namespace.yaml
run kubectl apply -f k8s/configmap-secret.yaml -f k8s/pvc.yaml -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=180s
run kubectl apply -f k8s/ingress.yaml
run kubectl -n btc-mlops rollout status deploy/btc-api --timeout=180s

LOG=03_get_all.log
run kubectl -n btc-mlops get pods -o wide
run kubectl -n btc-mlops get svc
run kubectl -n btc-mlops get hpa
run kubectl -n btc-mlops get ingress
run kubectl -n btc-mlops get pvc
run kubectl -n btc-mlops get configmap,secret
run kubectl -n btc-mlops describe deploy btc-api

LOG=04_api_requests.log
# port-forward works on Windows/Mac/Linux (minikube ip is not reachable with the Docker driver)
kubectl -n btc-mlops port-forward svc/btc-api-nodeport 8081:80 >/dev/null 2>&1 &
PF1=$!
kubectl -n ingress-nginx port-forward svc/ingress-nginx-controller 8082:80 >/dev/null 2>&1 &
PF2=$!
sleep 5
run kubectl -n btc-mlops get svc btc-api-nodeport -o wide
run curl -s http://localhost:8081/health
run curl -s -X POST http://localhost:8081/predict -H "Content-Type: application/json" -d '{"forecast_days": 7}'
run curl -s http://localhost:8081/metrics
echo "# Through the NGINX Ingress, host btc-forecast.local:" | tee -a "$OUT/$LOG"
run curl -s -H "Host: btc-forecast.local" http://localhost:8082/health
run curl -s -H "Host: btc-forecast.local" -X POST http://localhost:8082/predict -H "Content-Type: application/json" -d '{"forecast_days": 7}'
kill $PF1 $PF2 2>/dev/null

LOG=05_self_healing.log
POD=$(kubectl -n btc-mlops get pods -l app=btc-api -o jsonpath='{.items[0].metadata.name}')
run kubectl -n btc-mlops get pods
run kubectl -n btc-mlops delete pod "$POD" --wait=false
sleep 2; run kubectl -n btc-mlops get pods
sleep 25; run kubectl -n btc-mlops get pods
run kubectl -n btc-mlops get events --sort-by=.lastTimestamp

LOG=06_hpa_scaling.log
(kubectl -n btc-mlops get hpa btc-api-hpa --watch 2>&1 | while read -r l; do echo "$(date +%T) $l"; done > "$OUT/06_hpa_watch.log") &
WATCH=$!
run bash loadtest/load.sh 6 240
for i in $(seq 1 18); do sleep 15; echo "--- t=$((i*15))s" >> "$OUT/$LOG"; kubectl -n btc-mlops get hpa,pods --no-headers >> "$OUT/$LOG" 2>&1; kubectl -n btc-mlops top pods >> "$OUT/$LOG" 2>&1; done
kubectl -n btc-mlops delete pod -l app=loadgen --ignore-not-found
sleep 90; run kubectl -n btc-mlops get hpa
kill $WATCH
echo "Done. Logs in $OUT/. Take screenshots of: kubectl get hpa --watch, Airflow graph, GitHub Actions run."
