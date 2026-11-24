# syntax=docker/dockerfile:1
# ---------- Stage 1: build wheels ----------
FROM python:3.10-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY app/requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.10-slim AS runtime
LABEL org.opencontainers.image.title="btc-forecast-api" \
      org.opencontainers.image.description="Flask + Gunicorn 7-day BTC-USD forecast service"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5000 MODEL_DIR=/app/models
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home app
COPY --chown=app:app src/__init__.py src/features.py ./src/
COPY --chown=app:app app/app.py app/gunicorn.conf.py ./
# Bundled baseline model so the image works standalone; in K8s the PVC
# mounted at /app/models holds the latest model written by the pipeline.
COPY --chown=app:app models/ ./models/
USER 10001
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').status==200 else 1)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
