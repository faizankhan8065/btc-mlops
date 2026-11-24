def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("v")
    assert body["uptime_seconds"] >= 0


def test_predict_default_7_days(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200
    body = r.get_json()
    assert body["forecast_days"] == 7
    assert len(body["predictions"]) == 7
    assert all(p["predicted_close"] > 0 for p in body["predictions"])
    dates = [p["date"] for p in body["predictions"]]
    assert dates == sorted(dates) and len(set(dates)) == 7


def test_predict_no_body(client):
    assert client.post("/predict").status_code == 200


def test_predict_custom_days_and_history(client):
    history = [30000 + i * 50 for i in range(60)]
    r = client.post("/predict", json={"forecast_days": 3, "history": history})
    assert r.status_code == 200
    body = r.get_json()
    assert body["input"] == "custom_history"
    assert len(body["predictions"]) == 3


def test_predict_invalid_days(client):
    for bad in [0, 31, -1, "7", 2.5, None, True]:
        r = client.post("/predict", json={"forecast_days": bad})
        assert r.status_code == 400, bad
        assert r.get_json()["status"] == "error"


def test_predict_invalid_history(client):
    assert client.post("/predict", json={"history": [1, 2, 3]}).status_code == 400
    assert client.post("/predict", json={"history": "abc"}).status_code == 400
    assert client.post("/predict", json={"history": [100.0] * 40 + [None]}).status_code == 400
    assert client.post("/predict", json={"history": [100.0] * 40 + [-5]}).status_code == 400


def test_predict_malformed_json(client):
    r = client.post("/predict", data="{not json", content_type="application/json")
    assert r.status_code == 400
    r = client.post("/predict", data="hello", content_type="text/plain")
    assert r.status_code == 400
    r = client.post("/predict", json=[1, 2, 3])
    assert r.status_code == 400


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.get_json()["metrics"]
    assert {"mae", "rmse", "mape"} <= set(m)


def test_reload_requires_key(client):
    assert client.post("/reload").status_code == 401
    r = client.post("/reload", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert r.get_json()["status"] == "reloaded"


def test_unknown_endpoint_and_method(client):
    assert client.get("/nope").status_code == 404
    assert client.get("/predict").status_code == 405
