"""Dependency-free concurrent load test: python loadtest/quick_bench.py URL CONCURRENCY SECONDS"""

import json
import statistics
import sys
import threading
import time
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000/predict"
conc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 20
lat, errors, lock = [], [0], threading.Lock()
body = json.dumps({"forecast_days": 7}).encode()


def worker(end):
    while time.time() < end:
        t = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            with lock:
                lat.append((time.perf_counter() - t) * 1000)
        except Exception:  # noqa: BLE001
            with lock:
                errors[0] += 1


end = time.time() + secs
threads = [threading.Thread(target=worker, args=(end,)) for _ in range(conc)]
[t.start() for t in threads]
[t.join() for t in threads]
lat.sort()
print(f"url={url} concurrency={conc} duration={secs}s")
print(f"requests={len(lat)} errors={errors[0]} throughput={len(lat) / secs:.1f} req/s")
print(
    f"latency ms: mean={statistics.mean(lat):.1f} p50={lat[len(lat) // 2]:.1f} "
    f"p95={lat[int(len(lat) * 0.95)]:.1f} p99={lat[int(len(lat) * 0.99)]:.1f}"
)
