import time

from fastapi.testclient import TestClient

from app.engine import DEFAULTS, TASKS, make_batch, run_job


def tiny(**kw):
    cfg = dict(DEFAULTS)
    cfg.update(hidden=6, iters=10, batch=4, train_len=20, test_len=30, drift=[0.1], noise=[0.05], division=[0.2],
               inhibitor=[0.25], repeats=1)
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
        assert r["landscape"]["traj"] and len(r["noise"]) == 1 and len(r["division"]) == 1
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
        body = tiny(tasks=["xor", "commit", "antagonist"], membrane="gated", arms=["coop", "contract"])
        rid = client.post("/api/runs", json=body).json()["id"]
        for _ in range(240):
            run = client.get(f"/api/runs/{rid}").json()
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert run["status"] == "done", run.get("error")
        assert len(run["results"]) == 6
        assert client.get(f"/api/runs/{rid}/export.csv").text.startswith("task,arm")
        assert client.delete(f"/api/runs/{rid}").status_code == 200


def test_membranes_keep_params_matched_and_signs():
    for mode in ("transporter", "gated"):
        params = set()
        for arm in DEFAULTS["arms"]:
            r = run_job("commit", arm, tiny(membrane=mode))
            params.add(r["params"])
            assert r["membrane"] == mode and r["uptake"]["y"][0] == 0.0
            assert len(r["inhibitor"]) == 1
        assert len(params) == 1


def test_drug_tasks_balanced():
    u, y = make_batch("antagonist", 256, 400, 0.05, 3)
    assert 0.35 < (y > 0).mean() < 0.65
    u, y = make_batch("commit", 256, 400, 0.05, 3)
    assert ((y[1:] - y[:-1]) < 0).sum() == 0


def test_model_export_matches_engine(tmp_path):
    import importlib.util

    import numpy as np

    from app.model import CellModel, code_numpy, code_torch, graph, probe

    for mode, task in (("gated", "antagonist"), ("transporter", "commit"), ("direct", "flipflop")):
        r = run_job(task, "coop", tiny(membrane=mode))
        spec = r["model"]
        assert graph(spec, task, r)["total_params"] == r["params"]
        path = tmp_path / f"m_{mode}.py"
        path.write_text(code_numpy(spec, task, "t"))
        sp = importlib.util.spec_from_file_location(f"m_{mode}", path)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        u, _ = make_batch(task, 3, 40, 0.1, 5)
        a, _ = mod.CellModel().run(u)
        b, _ = CellModel(spec).run(u)
        assert np.abs(a - b).max() < 1e-4
        compile(code_torch(spec, task, "t"), "t", "exec")
        if task in ("antagonist", "commit"):
            assert probe(spec, task)["kind"]


def test_report_and_model_api(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_DATA", str(tmp_path))
    import importlib
    import app.server as server
    importlib.reload(server)
    with TestClient(server.app) as client:
        rid = client.post("/api/runs", json=tiny(tasks=["commit"], arms=["coop", "free"], membrane="transporter")).json()["id"]
        for _ in range(240):
            run = client.get(f"/api/runs/{rid}").json()
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert run["status"] == "done", run.get("error")
        rep = client.get(f"/api/runs/{rid}/report").json()
        assert rep["tasks"][0]["lines"]
        m = client.get(f"/api/runs/{rid}/model", params={"task": "commit"}).json()
        assert m["graph"]["nodes"][0]["type"] == "Input"
        sim = client.get(f"/api/runs/{rid}/model/simulate", params={"task": "commit", "seed": 2}).json()
        assert len(sim["output"][0]) == 200
        ex = client.get(f"/api/runs/{rid}/model/export", params={"task": "commit", "kind": "torch"})
        assert "fate-cell-model-commit" in ex.headers["content-disposition"]


def test_resistance_rule_and_therapy():
    import numpy as np

    from app.engine import resistance_truth
    from app.model import therapy

    x = np.zeros((80, 1, 1))
    x[:20, 0, 0] = 1.0
    x[30:34, 0, 0] = 0.8
    x[70:74, 0, 0] = 0.8
    y = resistance_truth(x)
    assert y[1, 0, 0] == 1.0
    assert y[32, 0, 0] == -1.0
    assert y[72, 0, 0] == 1.0
    r = run_job("resistance", "coop", tiny())
    th = therapy(r["model"], "resistance")
    assert th["n"] == 90 and len(th["rows"]) == 90 and len(th["by_inhibitor"]) == 2


def test_reachability_order_theorem_holds_for_sign_consistent_gated():
    from app.model import reachability

    r = run_job("flipflop", "coop", tiny(membrane="gated", iters=20))
    R = reachability(r["model"], "flipflop")
    assert R["attractors"]
    assert R["theory"] is not None and R["theory"]["holds"]
    assert R["theory"]["violations"] == 0
    assert all(len(a["escape"]) == len(R["sigmas"]) for a in R["attractors"])


def test_pathway_structure_and_simulation():
    from app.pathways import PATHWAYS, analyse, simulate_pathway

    for pid, p in PATHWAYS.items():
        a = analyse(p)
        assert a["structure"]["complexes"] > 0 and a["notes"]
        assert len(a["stoichiometry"]) == len(p["species"])
        s = simulate_pathway(p, steps=600)
        assert max(s["final"]) < 50
    gly = analyse(PATHWAYS["glycolysis"])
    pools = {tuple(sorted(sp for sp, _ in law)) for law in gly["conservation"]}
    assert ("NAD", "NADH") in pools
    assert ("ADP", "ATP") in pools
    base = simulate_pathway(PATHWAYS["glycolysis"], steps=2500)
    scale = [0.2 if r["id"] == "pfk" else 1.0 for r in PATHWAYS["glycolysis"]["reactions"]]
    ko = simulate_pathway(PATHWAYS["glycolysis"], steps=2500, enzyme_scale=scale)
    atp = PATHWAYS["glycolysis"]["species"].index("ATP")
    assert ko["final"][atp] < base["final"][atp]


def test_workspace_edit_simulate_export(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_DATA", str(tmp_path / "runs"))
    monkeypatch.setenv("FML_WORKSPACES", str(tmp_path / "ws"))
    import importlib
    import app.server as server
    importlib.reload(server)
    with TestClient(server.app) as client:
        w = client.post("/api/workspaces", json={"pathway": "glycolysis"}).json()
        assert w["compile"]["errors"] == []
        wid = w["id"]
        w = client.post(f"/api/workspaces/{wid}/layers", json={"type": "gate", "after": 0}).json()
        assert any(l["type"] == "gate" for l in w["layers"])
        layers = [l for l in w["layers"] if l["type"] != "gate"]
        w = client.put(f"/api/workspaces/{wid}", json={"layers": layers, "name": "edited"}).json()
        assert w["name"] == "edited"
        sim = client.post(f"/api/workspaces/{wid}/simulate", json={"length": 40}).json()
        assert len(sim["output"][0]) == 40 and sim["trace"]
        code = client.get(f"/api/workspaces/{wid}/code").json()["code"]
        compile(code, "ws", "exec")
        assert client.delete(f"/api/workspaces/{wid}").status_code == 200


def test_new_arms_match_params_and_conserve_pools():
    import numpy as np

    from app.engine import pool_projection

    params = set()
    for arm in ("coop", "feedback", "pool", "free"):
        r = run_job("resistance", arm, tiny())
        params.add(r["params"])
        if arm == "feedback":
            assert r["negative_edges"] == tiny()["feedback_edges"]
        if arm == "pool":
            assert r["model"]["pool_groups"]
    assert len(params) == 1
    proj, groups = pool_projection(8, 4)
    h = np.random.default_rng(0).random((3, 8))
    out = proj(h)
    for g in groups:
        assert abs(out[:, g].sum(axis=1) - len(g) * 0.5).max() < 1e-6


def test_pathway_deeper_analysis():
    from app.pathways import PATHWAYS, deeper

    d = deeper(PATHWAYS["glycolysis"])
    assert d["notes"] and d["control"]["coefficients"]
    assert abs(d["control"]["sum"]) > 0.2
    assert set(d["acr"]) == {"robust", "sensitive", "factors"}


def test_pathway_graph_cypher_and_whatif():
    from app.pathways import PATHWAYS, cypher, graph, what_if

    assert len(PATHWAYS) == 8
    for pid, p in PATHWAYS.items():
        g = graph(p)
        assert len(g["nodes"]) == len(p["species"]) + len(p["reactions"])
        assert all(e["from"].startswith(("s:", "r:")) for e in g["edges"])
        c = cypher(p)
        assert "MERGE (p:Pathway" in c and c.count(";") >= len(p["reactions"])
    w = what_if(PATHWAYS["purine"], "r:xo1")
    urate = {c["label"]: {x["species"]: x["after"] for x in c["changes"]} for c in w["cases"]}
    assert any(d["name"] == "Allopurinol" for d in w["drugs"])
    assert urate["0% activity"]["Urate"] < urate["50% activity"]["Urate"]
    ko = what_if(PATHWAYS["phe"], "r:pah")["cases"][-1]
    phe = next(x for x in ko["changes"] if x["species"] == "Phe")
    assert phe["after"] > phe["before"] * 10
    assert what_if(PATHWAYS["galactose"], "s:Gal")["cases"]
    assert what_if(PATHWAYS["urea"], "r:nope") is None


def test_neo4j_status_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_DATA", str(tmp_path / "r"))
    monkeypatch.setenv("FML_WORKSPACES", str(tmp_path / "w"))
    import importlib
    import app.server as server
    importlib.reload(server)
    with TestClient(server.app) as client:
        st = client.get("/api/neo4j/status").json()
        assert set(st) >= {"driver_installed", "configured"}
        assert client.get("/api/pathways/purine/graph").json()["nodes"]
        assert "MERGE" in client.get("/api/pathways/urea/cypher").json()["cypher"]
