import time

from fastapi.testclient import TestClient

from app.engine import DEFAULTS, TASKS, make_batch, run_job


def tiny(**kw):
    cfg = dict(DEFAULTS)
    cfg.update(hidden=6, iters=10, batch=4, train_len=20, test_len=30, drift=[0.1], repeats=1)
    cfg.update(kw)
    return cfg


def test_batches_shapes():
    for task, spec in TASKS.items():
        u, y = make_batch(task, 3, 25, 0.1, 0)
        assert u.shape == (25, 3, spec["nin"])
        assert y.shape == (25, 3, spec["nout"])
        assert set(y.ravel().tolist()) <= {-1.0, 1.0, 0.0}


def test_arms_match_params_and_signs():
    params = set()
    for arm in DEFAULTS["arms"]:
        r = run_job("flipflop", arm, tiny())
        params.add(r["params"])
        assert len(r["wells"]) == 96
        if arm == "coop":
            assert r["negative_edges"] == 0
        if arm == "contract":
            assert r["attractors"] == 1
    assert len(params) == 1


def test_api_run_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_DATA", str(tmp_path))
    import importlib
    import app.server as server
    importlib.reload(server)
    with TestClient(server.app) as client:
        assert client.get("/api/health").json()["ok"]
        body = tiny(tasks=["xor"], arms=["coop", "contract"])
        rid = client.post("/api/runs", json=body).json()["id"]
        for _ in range(240):
            run = client.get(f"/api/runs/{rid}").json()
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert run["status"] == "done", run.get("error")
        assert len(run["results"]) == 2
        assert client.get(f"/api/runs/{rid}/export.csv").text.startswith("task,arm")
        assert client.delete(f"/api/runs/{rid}").status_code == 200
