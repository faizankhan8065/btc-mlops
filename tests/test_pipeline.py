import json
import os

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features, recursive_forecast
from src.train_forecast import metrics


def test_features_have_no_lookahead():
    close = pd.Series(np.linspace(100, 200, 60))
    full = build_features(close)
    partial = build_features(close.iloc[:40])
    pd.testing.assert_frame_equal(full.iloc[:40], partial)
    assert list(full.columns) == FEATURE_COLUMNS


def test_metrics_values():
    m = metrics([100, 200], [110, 180])
    assert m["mae"] == 15
    assert round(m["mape"], 2) == 10.0


def test_training_artifacts(model_dir):
    assert os.path.exists(os.path.join(model_dir, "model.pkl"))
    report = json.load(open(os.path.join(model_dir, "metrics.json")))
    assert report["passed_threshold"] is True
    assert len(report["validation_predicted"]) == 7


def test_recursive_forecast_rejects_short_history():
    class Dummy:
        def predict(self, x):
            return [0.0]

    try:
        recursive_forecast(Dummy(), [1.0] * 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
