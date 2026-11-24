#!/usr/bin/env bash
# In-cluster load generator (no external tools needed): N busybox-free python pods hammer /predict.
set -euo pipefail
WORKERS=${1:-6}
DURATION=${2:-180}
kubectl -n btc-mlops delete pod -l app=loadgen --ignore-not-found
for i in $(seq 1 "$WORKERS"); do
  kubectl -n btc-mlops run "loadgen-$i" -l app=loadgen --restart=Never --image=btc-forecast-api:latest \
    --image-pull-policy=IfNotPresent --command -- python -c "
import json,time,urllib.request
end=time.time()+$DURATION; n=0
while time.time()<end:
    r=urllib.request.Request('http://btc-api/predict',data=json.dumps({'forecast_days':30}).encode(),headers={'Content-Type':'application/json'})
    urllib.request.urlopen(r).read(); n+=1
print('requests',n)"
done
