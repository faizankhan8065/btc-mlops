# Step-by-step guide to finish Assignment 3

Commands are for **Git Bash** on Windows, or the normal Terminal on Mac/Linux.
Run every command from inside the `btc-mlops` folder unless a step says otherwise.

## Step 0: Install tools (skip what you already have)
- Docker Desktop: https://www.docker.com/products/docker-desktop (start it and wait until it shows "running")
- Git (Windows users get Git Bash with it): https://git-scm.com
- Minikube: https://minikube.sigs.k8s.io/docs/start
- kubectl: https://kubernetes.io/docs/tasks/tools
- Accounts: GitHub and Docker Hub

Check: `docker version`, `minikube version`, `kubectl version --client` should all print versions.

## Step 1: Unzip the project
Unzip `btc-mlops.zip`, open Git Bash/Terminal, and `cd` into the `btc-mlops` folder.

## Step 2: Train the model on REAL Bitcoin data (5 min)
```bash
docker build -f Dockerfile.pipeline -t btc-pipeline:latest .
docker run --rm --user root -v "$PWD/data:/opt/pipeline/data" -v "$PWD/models:/opt/pipeline/models" btc-pipeline:latest src.etl_fetch
docker run --rm --user root -v "$PWD/data:/opt/pipeline/data" -v "$PWD/models:/opt/pipeline/models" btc-pipeline:latest src.preprocess
docker run --rm --user root -v "$PWD/data:/opt/pipeline/data" -v "$PWD/models:/opt/pipeline/models" btc-pipeline:latest src.train_forecast
```
Windows Git Bash: if you get a path error, put `MSYS_NO_PATHCONV=1 ` in front of each `docker run`.
The last command prints MAE/RMSE/MAPE and `"data_source": "yfinance"` (or coingecko). **Copy this output and save it.**

## Step 3: Push to GitHub and set up CI/CD (10 min)
1. Docker Hub → Account Settings → Personal access tokens → create a token with Read & Write access. Copy it.
2. GitHub → New repository → name `btc-mlops`, Public, **no** README. Create it.
3. In the new repo: Settings → Secrets and variables → Actions → New repository secret:
   - `DOCKER_USERNAME` = your Docker Hub username
   - `DOCKER_PASSWORD` = the token from step 1
4. Push the code (replace YOUR_USER):
```bash
git init
git add -A
git commit -m "MLOps Assignment 3"
git branch -M main
git remote add origin https://github.com/YOUR_USER/btc-mlops.git
git push -u origin main
```
If the folder already has git history from me, `git init` just reinitialises it, which is fine.
5. GitHub → **Actions** tab → wait for both jobs ("Lint & Test", "Build & Push") to turn green.
   - **Screenshot 1:** the green workflow run.
   - **Screenshot 2:** Docker Hub → your `btc-forecast-api` repo showing the `latest` and commit-SHA tags.

## Step 4: Airflow with docker-compose (10 min)
```bash
docker compose up -d --build
```
Wait 2–3 minutes, then open http://localhost:8080 (login `admin` / `admin`).
1. Click `btc_etl_pipeline` → press ▶ **Trigger DAG**.
2. Open the **Graph** view and wait until all 5 boxes are green.
   - **Screenshot 3:** the Graph view, all green.
   - **Screenshot 4:** the log of the `reload_api_model` task (it shows "API now serving vXXXX").
3. Check the API picked up the new model:
```bash
curl http://localhost:5000/health
```
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"forecast_days": 7}'
```
Save that output. Then stop compose to free memory:
```bash
docker compose down
```

## Step 5: Kubernetes: deploy, self-healing, autoscaling (15 min, one command)
```bash
bash scripts_run_k8s_demo.sh
```
It starts Minikube, builds the image, deploys everything, calls the API, deletes a pod, and runs a 4-minute load test.
While the load test runs, open a **second** Git Bash window and run:
```bash
kubectl -n btc-mlops get hpa --watch
```
- **Screenshot 5:** that window once REPLICAS has gone up (for example 2 → 5).
- **Screenshot 6:** `kubectl -n btc-mlops get pods,svc,hpa,ingress,pvc`

All output is saved in `docs/logs/k8s/`.

## Step 6: Send me everything and I'll finish the report
Upload in this chat:
- the 6 screenshots
- the `docs/logs/k8s/` folder (zip it)
- the outputs from Steps 2 and 4
- your GitHub repo link, Docker Hub link, and your partner's name and roll number

I'll put them into the report and send back the final PDF.

## If something goes wrong
- `minikube start` fails → make sure Docker Desktop is running, then `minikube delete` and run the script again.
- HPA shows `<unknown>/60%` → metrics-server needs 1–2 minutes; wait.
- Pods stuck in `ErrImageNeverPull` / `ImagePullBackOff` → the image was built outside Minikube; rerun the script.
- CI "Build & Push" fails at login → recheck the two secret names and the token.
- Paste any error message here and I'll tell you the fix.
