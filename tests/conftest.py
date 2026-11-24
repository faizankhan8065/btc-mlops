import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """Train a small model on synthetic data (no network needed)."""
    from src import etl_fetch, preprocess, train_forecast

    d = tmp_path_factory.mktemp("pipeline")
    raw, proc, models = d / "raw.csv", d / "proc.csv", d / "models"
    etl_fetch.synthetic_prices(days=400).to_csv(raw, index=False)
    preprocess.run(str(raw), str(proc))
    train_forecast.train(str(proc), str(models), mape_threshold=100.0)
    return str(models)


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", model_dir)
    monkeypatch.setenv("API_KEY", "test-key")
    sys.path.insert(0, os.path.join(ROOT, "app"))
    import app as app_module

    importlib.reload(app_module)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
