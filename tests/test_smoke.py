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

    assert len(PATHWAYS) >= 8
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


def test_merged_metabolism_runs_and_links_pathways():
    from app.pathways import analyse, merged, simulate_pathway, what_if

    m = merged()
    assert len(m["species"]) > 50 and len(m["reactions"]) > 50
    run = simulate_pathway(m, steps=4000, record=False)
    assert run["steady"] and max(run["final"]) < 20
    a = analyse(m)
    pools = {tuple(sorted(sp for sp, _ in law)) for law in a["conservation"]}
    assert ("ADP", "ATP") in pools and ("NAD", "NADH") in pools
    w = what_if(m, "r:phe_pah")
    hit = w["cases"][-1]["pathways"]
    assert "phe" in hit and len(hit) > 1


def test_query_panel_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_DATA", str(tmp_path / "r"))
    monkeypatch.setenv("FML_WORKSPACES", str(tmp_path / "w"))
    import importlib
    import app.server as server
    importlib.reload(server)
    with TestClient(server.app) as client:
        assert client.post("/api/neo4j/query", json={"cypher": "MATCH (n) DETACH DELETE n"}).status_code == 400
        assert client.post("/api/neo4j/query", json={"cypher": ""}).status_code == 400
        assert client.get("/api/pathways").json()[-1]["id"] == "all"
        assert "IS]->(c)" in client.get("/api/pathways/purine/cypher").json()["cypher"]


def test_all_pathways_balanced_and_annotated():
    from app.pathways import PATHWAYS, merged, simulate_pathway

    assert len(PATHWAYS) >= 26
    annotated = sum(1 for p in PATHWAYS.values() for r in p["reactions"] if r.get("deficiency"))
    drugged = sum(1 for p in PATHWAYS.values() for r in p["reactions"] if r.get("drugs"))
    assert annotated > 40 and drugged > 20
    for pid, p in PATHWAYS.items():
        run = simulate_pathway(p, steps=4000, record=False)
        assert run["residual"] < 0.05, f"{pid} did not settle"
        assert max(run["final"]) < 10, f"{pid} blew up"
    m = simulate_pathway(merged(), steps=8000, record=False)
    assert m["steady"] and max(m["final"]) < 10


def test_strategy_engine_finds_substrate_reduction():
    from app.pathways import PATHWAYS, strategies

    r = strategies(PATHWAYS["sphingolipid"], "gba")
    assert r["toxin"] == "GlcCer" and r["toxin_disease"] > 10 * r["toxin_healthy"]
    moves = {s["move"]: s for s in r["strategies"]}
    assert any("Restore the missing enzyme fully" in m for m in moves)
    assert any("UGCG" in m for m in moves), "substrate reduction should rank in the top moves"
    assert strategies(PATHWAYS["phe"], "nope") is None


def test_benchmark_reports_chance_baseline(monkeypatch):
    import app.pathways as P

    subset = {k: P.PATHWAYS[k] for k in ("heme", "sphingolipid")}
    monkeypatch.setattr(P, "PATHWAYS", subset)
    b = P.benchmark(steps=800)
    assert b["scored"] > 0 and b["expected_top3"] > 0
    assert set(b) >= {"observed_top3", "expected_top3", "z", "cases"}
    gba = next(c for c in b["cases"] if c["enzyme"] == "GBA")
    assert gba["rank"] is not None and gba["rank"] <= 3


def test_strategies_report_combinations_and_side_effects():
    from app.pathways import PATHWAYS, strategies

    r = strategies(PATHWAYS["sphingolipid"], "gba")
    assert r["combinations"] and "best_single" in r["combinations"][0]


def test_research_templates_parse_read_and_build(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_RESEARCH", str(tmp_path / "research"))
    import importlib
    import app.research as RS
    importlib.reload(RS)
    from app.pathways import PATHWAYS, target_search

    for tid in RS.TEMPLATES:
        f = RS.feasibility(RS.template_pathway(tid))
        assert f["verdict"].startswith("A working model"), tid
    viral = RS.template_pathway("viral")
    run = __import__("app.pathways", fromlist=["x"]).simulate_pathway(viral, steps=4000, record=False)
    assert abs(dict(zip(viral["species"], run["final"]))["T"] - 3.0) < 0.05
    fixture = {"resultList": {"result": [{"source": "MED", "id": "1", "title": "HIV <i>viral</i> dynamics",
                                          "authorString": "A B", "pubYear": "2025", "abstractText": "<p>target cell</p>",
                                          "journalInfo": {"journal": {"title": "J"}}}]}}
    papers = RS.parse_europe_pmc(fixture)
    assert papers[0]["title"] == "HIV viral dynamics" and papers[0]["link"].endswith("/MED/1")
    rd = RS.read_keywords("A target cell limited model of HIV viral load with infected cells half-life of 1.4 days.")
    assert rd["family"] == "viral" and rd["stated_numbers"]
    rec, feas = RS.save_model({"title": "t"}, rd, rd["recipe"])
    assert rec["pid"] in PATHWAYS
    t = target_search(PATHWAYS[rec["pid"]], "V", "lower", steps=1500)
    assert t["moves"][0]["change"] < -0.5
    assert RS.delete_model(rec["id"]) and rec["pid"] not in PATHWAYS


def test_llm_output_is_sanitised():
    import app.research as RS

    raw = {"family": "viral", "template_overrides": {"clear": 23.0, "bogus": 5, "infect": "x"},
           "stated_parameters": [{"name": "c", "value": "23", "unit": "/day", "sentence": "s"}],
           "confidence": "medium", "caveats": ["scaled"]}
    out = RS.sanitise_llm(raw)
    assert out["recipe"]["overrides"] == {"clear": 23.0}
    custom = RS.sanitise_custom({"species": [{"name": "X!", "initial": 1}, {"name": "Y"}],
                                 "reactions": [{"id": "r", "substrates": {"X": 1}, "products": {"Y": 1, "Z": 1},
                                                "law": "weird", "rate": 1e9}]})
    assert custom["species"] == ["X", "Y"] and custom["reactions"][0]["law"] == "mass_action"
    assert custom["reactions"][0]["kcat"] == 1e4 and "Z" not in custom["reactions"][0]["products"]


def test_settings_store_masks_secrets_and_guards_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("FML_DATA", str(tmp_path / "r"))
    monkeypatch.setenv("FML_WORKSPACES", str(tmp_path / "w"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib
    import os
    import app.settings as ST
    importlib.reload(ST)
    import app.server as server
    importlib.reload(server)
    with TestClient(server.app) as client:
        v = client.get("/api/settings").json()
        assert v["can_edit"] and not v["anthropic_api_key"]["set"]
        key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz-1234"
        v = client.put("/api/settings", json={"values": {"anthropic_api_key": key, "anthropic_model": "claude-sonnet-5"}}).json()
        assert v["anthropic_api_key"]["set"] and key not in str(v) and v["anthropic_api_key"]["value"].endswith("1234")
        assert oct(os.stat(tmp_path / "settings.json").st_mode)[-3:] == "600"
        assert ST.get("anthropic_api_key") == key
        assert client.get("/api/research/status").json()["claude"] is True
        v = client.put("/api/settings", json={"values": {"anthropic_api_key": ""}}).json()
        assert v["anthropic_api_key"]["set"], "an empty secret field must not wipe the stored key"
        assert client.put("/api/settings", json={"values": {"anthropic_model": "nope"}}).status_code == 400
        tunnel = {"cf-connecting-ip": "203.0.113.9"}
        assert client.put("/api/settings", json={"values": {"neo4j_uri": "x"}}, headers=tunnel).status_code == 403
        assert client.get("/api/settings", headers=tunnel).json()["can_edit"] is False
        signed_in = dict(tunnel, **{"cf-access-authenticated-user-email": "me@example.com"})
        assert client.put("/api/settings", json={"values": {"neo4j_uri": "bolt://localhost:7687"}}, headers=signed_in).status_code == 200
        v = client.put("/api/settings", json={"clear": ["anthropic_api_key"]}).json()
        assert not v["anthropic_api_key"]["set"]
