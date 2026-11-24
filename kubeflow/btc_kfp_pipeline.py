"""Kubeflow Pipelines (KFP v2) definition: ingest -> features -> train.

Each step is a Container Component running the pipeline image
(Dockerfile.pipeline). Artifacts are passed between steps as KFP Dataset/Model
artifacts so they are tracked in the KFP metadata store.

Compile:  python kubeflow/btc_kfp_pipeline.py  -> kubeflow/btc_kfp_pipeline.yaml
"""

import os

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

PIPELINE_IMAGE = os.getenv("PIPELINE_IMAGE", "docker.io/YOUR_DOCKERHUB_USER/btc-pipeline:latest")


@dsl.container_component
def ingest_data(raw_data: Output[Dataset], allow_synthetic: str = "0"):
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["sh", "-c"],
        args=[
            'ALLOW_SYNTHETIC="$1" RAW_DATA_PATH="$0" python -m src.etl_fetch --out "$0"',
            raw_data.path,
            allow_synthetic,
        ],
    )


@dsl.container_component
def build_features(raw_data: Input[Dataset], features: Output[Dataset]):
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "src.preprocess"],
        args=["--raw", raw_data.path, "--out", features.path],
    )


@dsl.container_component
def train_model(features: Input[Dataset], model: Output[Model], metrics: Output[Metrics], mape_threshold: float):
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["sh", "-c"],
        args=[
            'python -m src.train_forecast --processed "$0" --model-dir "$1" --mape-threshold "$3" '
            '&& cp "$1/candidate_metrics.json" "$2"',
            features.path,
            model.path,
            metrics.path,
            mape_threshold,
        ],
    )


@dsl.pipeline(name="btc-7day-forecast", description="BTC-USD ingestion, feature engineering and training")
def btc_pipeline(allow_synthetic: str = "0", mape_threshold: float = 10.0):
    ingest = ingest_data(allow_synthetic=allow_synthetic)
    feats = build_features(raw_data=ingest.outputs["raw_data"])
    train_model(features=feats.outputs["features"], mape_threshold=mape_threshold)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_kfp_pipeline.yaml")
    compiler.Compiler().compile(btc_pipeline, out)
    print(f"Compiled pipeline -> {out}")
