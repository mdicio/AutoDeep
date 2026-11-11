import time

import pytest
from fastapi.testclient import TestClient

from autodeep.web.server import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "ws")))


def test_catalogue_endpoints(client):
    assert client.get("/").status_code == 200
    models = client.get("/api/models").json()
    assert {m["name"] for m in models["models"]} >= {"xgb", "fttransformer", "resnet"}
    metrics = client.get("/api/metrics").json()
    assert metrics["regression"]["default"] == "rmse"
    assert any(d["name"] == "iris" for d in client.get("/api/datasets").json()["builtin"])


def test_upload_profile_estimate(client, mixed_csv):
    with open(mixed_csv, "rb") as f:
        up = client.post("/api/datasets/upload", files={"file": ("my data.csv", f, "text/csv")}).json()
    assert up["name"] == "my_data" and up["profile"]["n_rows"] == 300
    prof = client.post("/api/datasets/profile", json={"path": up["path"], "target": "label"}).json()
    assert prof["inferred_problem_type"] == "binary_classification"
    job = {"datasets": [{"name": "m", "path": up["path"], "target": "label", "drop_columns": ["id"]}], "models": ["xgb", "gandalf"], "max_evals": 3}
    est = client.post("/api/estimate", json=job).json()
    assert len(est["jobs"]) == 2 and est["total_seconds"] > 0
    bad = client.post("/api/estimate", json={**job, "datasets": [{**job["datasets"][0], "target": "nope"}]})
    assert bad.status_code == 400 and "not found" in bad.json()["detail"]
    assert client.post("/api/datasets/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400


def test_job_lifecycle(client):
    r = client.post("/api/jobs", json={"datasets": [{"name": "iris", "path": "builtin:iris"}], "models": ["xgb"], "max_evals": 2})
    job_id = r.json()["id"]
    for _ in range(240):
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"]["state"] not in ("queued", "starting", "running"):
            break
        time.sleep(0.5)
    assert info["status"]["state"] == "finished", client.get(f"/api/jobs/{job_id}/log").text
    assert info["results"][0]["status"] == "ok"
    assert client.get(f"/api/jobs/{job_id}/results.csv").status_code == 200
    best = info["best_models"]
    assert len(best) == 1 and best[0]["dataset"] == "iris" and best[0]["model"] == "xgb"
    z = client.get(f"/api/jobs/{job_id}/models/iris/download")
    assert z.status_code == 200 and z.content[:2] == b"PK"
    rows = "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3\n"
    pred = client.post(f"/api/jobs/{job_id}/models/iris/predict", files={"file": ("rows.csv", rows.encode(), "text/csv")})
    assert pred.status_code == 200, pred.text
    assert pred.text.splitlines()[0].startswith("sepal length") and "prediction" in pred.text.splitlines()[0]
    assert client.get(f"/api/jobs/{job_id}/models/..%2F..%2Fx/download").status_code == 404
    assert any(j["id"] == job_id for j in client.get("/api/jobs").json()["jobs"])
    assert client.delete(f"/api/jobs/{job_id}").json()["ok"]
