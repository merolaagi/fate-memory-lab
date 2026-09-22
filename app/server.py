import csv
import io
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from typing import Literal

from .engine import ARMS, DEFAULTS, MEMBRANES, TASKS, Cancelled, run_job
from .explain import LABEL, explain
from .model import code_numpy, code_torch, graph, probe, simulate

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FML_DATA", ROOT / "data" / "runs"))
DATA.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"
WORKERS = int(os.environ.get("FML_WORKERS", max(1, (os.cpu_count() or 2) - 2)))

_lock = threading.Lock()
_queue = []
_wake = threading.Event()


class RunConfig(BaseModel):
    name: str = ""
    tasks: list[str] = Field(default_factory=lambda: list(DEFAULTS["tasks"]))
    arms: list[str] = Field(default_factory=lambda: list(DEFAULTS["arms"]))
    hidden: int = Field(DEFAULTS["hidden"], ge=4, le=128)
    substeps: int = Field(DEFAULTS["substeps"], ge=1, le=10)
    iters: int = Field(DEFAULTS["iters"], ge=10, le=20000)
    lr: float = Field(DEFAULTS["lr"], gt=0, le=1)
    batch: int = Field(DEFAULTS["batch"], ge=1, le=512)
    train_len: int = Field(DEFAULTS["train_len"], ge=10, le=1000)
    test_len: int = Field(DEFAULTS["test_len"], ge=10, le=5000)
    pulse_prob: float = Field(DEFAULTS["pulse_prob"], gt=0, lt=1)
    self_excitation: float = Field(DEFAULTS["self_excitation"], ge=0, le=20)
    coupling: float = Field(DEFAULTS["coupling"], ge=0, le=5)
    drift: list[float] = Field(default_factory=lambda: list(DEFAULTS["drift"]))
    noise: list[float] = Field(default_factory=lambda: list(DEFAULTS["noise"]))
    division: list[float] = Field(default_factory=lambda: list(DEFAULTS["division"]))
    inhibitor: list[float] = Field(default_factory=lambda: list(DEFAULTS["inhibitor"]))
    membrane: Literal["direct", "transporter", "gated"] = DEFAULTS["membrane"]
    seed: int = DEFAULTS["seed"]
    repeats: int = Field(DEFAULTS["repeats"], ge=1, le=10)


def run_path(rid):
    return DATA / f"{rid}.json"


def progress_dir(rid):
    return DATA / f"{rid}.progress"


def load(rid):
    p = run_path(rid)
    if not p.exists() or "/" in rid or ".." in rid:
        raise HTTPException(404, "Run not found")
    return json.loads(p.read_text())


def save(run):
    tmp = run_path(run["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(run))
    tmp.replace(run_path(run["id"]))


def _job(task, arm, rep, cfg, pdir):
    pdir = Path(pdir)
    pfile = pdir / f"{task}__{arm}__{rep}"
    stop = pdir / "CANCEL"

    def progress(frac):
        if stop.exists():
            raise Cancelled()
        pfile.write_text(f"{frac:.4f}")

    return run_job(task, arm, cfg, progress, rep=rep)


def _execute(rid):
    run = load(rid)
    if run["status"] == "cancelled":
        return
    cfg = run["config"]
    pdir = progress_dir(rid)
    pdir.mkdir(exist_ok=True)
    run["status"] = "running"
    run["started"] = time.time()
    save(run)
    jobs = [(t, a, r) for t in cfg["tasks"] for a in cfg["arms"] for r in range(cfg["repeats"])]
    status, error = "done", None
    try:
        with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as ex:
            futs = [ex.submit(_job, t, a, r, cfg, str(pdir)) for t, a, r in jobs]
            for f in as_completed(futs):
                try:
                    res = f.result()
                except Cancelled:
                    status = "cancelled"
                    continue
                with _lock:
                    run = load(rid)
                    run["results"].append(res)
                    save(run)
    except Exception:
        status, error = "failed", traceback.format_exc(limit=3)
    if (pdir / "CANCEL").exists():
        status = "cancelled"
    run = load(rid)
    run["status"] = status
    if error:
        run["error"] = error
    run["finished"] = time.time()
    save(run)
    shutil.rmtree(pdir, ignore_errors=True)


def _worker():
    while True:
        _wake.wait()
        while True:
            with _lock:
                if not _queue:
                    _wake.clear()
                    break
                rid = _queue.pop(0)
            try:
                _execute(rid)
            except Exception:
                traceback.print_exc()


def _recover():
    for p in sorted(DATA.glob("*.json")):
        run = json.loads(p.read_text())
        if run.get("status") in ("queued", "running"):
            run["status"] = "queued"
            run["results"] = []
            run["config"].setdefault("repeats", 1)
            save(run)
            shutil.rmtree(progress_dir(run["id"]), ignore_errors=True)
            _queue.append(run["id"])
    if _queue:
        _wake.set()


@asynccontextmanager
async def lifespan(app):
    _recover()
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="Fate Memory Lab", version=__version__, lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "queued": len(_queue)}


@app.get("/api/meta")
def meta():
    return {"version": __version__, "defaults": DEFAULTS, "arms": ARMS, "tasks": list(TASKS),
            "membranes": MEMBRANES, "workers": WORKERS}


@app.post("/api/runs")
def create_run(cfg: RunConfig):
    c = cfg.model_dump()
    c["tasks"] = [t for t in c["tasks"] if t in TASKS]
    c["arms"] = [a for a in c["arms"] if a in ARMS]
    if not c["tasks"] or not c["arms"]:
        raise HTTPException(400, "Pick at least one task and one arm")
    c["drift"] = [float(e) for e in c["drift"] if 0 <= float(e) <= 3][:6]
    c["noise"] = [float(e) for e in c["noise"] if 0 <= float(e) <= 1][:6]
    c["division"] = [float(e) for e in c["division"] if 0 <= float(e) <= 3][:6]
    c["inhibitor"] = [float(e) for e in c["inhibitor"] if 0 < float(e) <= 1][:6]
    rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    run = {"id": rid, "name": c.pop("name") or rid, "created": time.time(), "config": c,
           "status": "queued", "results": [], "version": __version__}
    save(run)
    with _lock:
        _queue.append(rid)
    _wake.set()
    return run


@app.get("/api/runs")
def list_runs():
    out = []
    for p in sorted(DATA.glob("*.json"), reverse=True):
        r = json.loads(p.read_text())
        c = r["config"]
        out.append({"id": r["id"], "name": r["name"], "status": r["status"], "created": r["created"],
                    "tasks": c["tasks"], "arms": c["arms"], "iters": c["iters"], "repeats": c.get("repeats", 1),
                    "membrane": c.get("membrane", "direct")})
    return out


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    run = load(rid)
    run["config"].setdefault("repeats", 1)
    prog = {}
    pdir = progress_dir(rid)
    if pdir.exists():
        for f in pdir.iterdir():
            if f.name == "CANCEL":
                continue
            try:
                prog[f.name] = float(f.read_text() or 0)
            except ValueError:
                pass
    for r in run["results"]:
        prog[f"{r['task']}__{r['arm']}__{r.get('rep', 0)}"] = 1.0
    run["progress"] = prog
    return run


@app.post("/api/runs/{rid}/cancel")
def cancel_run(rid: str):
    run = load(rid)
    with _lock:
        if rid in _queue:
            _queue.remove(rid)
            run["status"] = "cancelled"
            save(run)
            return run
    if run["status"] != "running":
        raise HTTPException(409, f"This run is {run['status']}, so there is nothing to stop.")
    pdir = progress_dir(rid)
    pdir.mkdir(exist_ok=True)
    (pdir / "CANCEL").write_text("1")
    return {"id": rid, "status": "stopping"}


@app.get("/api/runs/{rid}/export.json")
def export_json(rid: str):
    load(rid)
    return FileResponse(run_path(rid), media_type="application/json", filename=f"fate-memory-{rid}.json")


@app.get("/api/runs/{rid}/export.csv")
def export_csv(rid: str):
    run = load(rid)
    buf = io.StringIO()
    w = csv.writer(buf)
    c = run["config"]
    drift_eps, noise_eps, div_eps = c["drift"], c.get("noise", []), c.get("division", [])
    inh_eps = c.get("inhibitor", [])
    w.writerow(["task", "arm", "rep", "seed", "membrane", "params", "acc_train_len", "acc_test_len", "settled",
                "attractors", "settle_steps", "negative_edges", "seconds"]
               + [f"drift_{e}_acc" for e in drift_eps] + [f"drift_{e}_settled" for e in drift_eps]
               + [f"noise_{e}_acc" for e in noise_eps] + [f"division_{e}_acc" for e in div_eps]
               + [f"inhibitor_{e}_acc" for e in inh_eps])
    for r in sorted(run["results"], key=lambda r: (r["task"], r["arm"], r.get("rep", 0))):
        w.writerow([r["task"], r["arm"], r.get("rep", 0), r.get("seed", ""), r.get("membrane", "direct"), r["params"], r["acc_train_len"],
                    r["acc_test_len"], r["settled"], r["attractors"], r.get("settle_steps", ""),
                    r["negative_edges"], r["seconds"]]
                   + [d["acc"] for d in r["drift"]] + [d["settled"] for d in r["drift"]]
                   + [d["acc"] for d in r.get("noise", [])] + [d["acc"] for d in r.get("division", [])]
                   + [d["acc"] for d in r.get("inhibitor", [])])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="fate-memory-{rid}.csv"'})


@app.delete("/api/runs/{rid}")
def delete_run(rid: str):
    run = load(rid)
    if run["status"] == "running":
        raise HTTPException(409, "This run is still training. Stop it first, then delete it.")
    with _lock:
        if rid in _queue:
            _queue.remove(rid)
    run_path(rid).unlink(missing_ok=True)
    shutil.rmtree(progress_dir(rid), ignore_errors=True)
    return {"deleted": rid}


@app.get("/api/runs/{rid}/report")
def report(rid: str):
    return explain(load(rid))


def _pick(run, task, arm=None, rep=None):
    rs = [r for r in run["results"] if r["task"] == task]
    if not rs:
        raise HTTPException(404, f"No results for {task} in this run.")
    rs = [r for r in rs if "model" in r]
    if not rs:
        raise HTTPException(409, "This run was trained before version 0.5.0 and did not save its weights. Start a new run to build a model.")
    if arm is None:
        means = {}
        for r in rs:
            means.setdefault(r["arm"], []).append(r["acc_test_len"])
        arm = max(means, key=lambda a: sum(means[a]) / len(means[a]))
    pool = [r for r in rs if r["arm"] == arm]
    if not pool:
        raise HTTPException(404, f"No {arm} results for {task}.")
    if rep is None:
        return max(pool, key=lambda r: r["acc_test_len"]), rs
    for r in pool:
        if r.get("rep", 0) == rep:
            return r, rs
    raise HTTPException(404, "No such seed.")


def _title(run, r):
    return (f"Fate Memory Lab cell model: {r['task']} task, {LABEL[r['arm']]} arm, seed {r.get('rep', 0) + 1}, "
            f"{r.get('membrane', 'direct')} membrane, from run {run['id']} (v{run.get('version', '?')})")


@app.get("/api/runs/{rid}/model")
def model_info(rid: str, task: str, arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, rs = _pick(run, task, arm, rep)
    return {"task": task, "arm": r["arm"], "rep": r.get("rep", 0), "membrane": r.get("membrane", "direct"),
            "acc_test_len": r["acc_test_len"], "title": _title(run, r),
            "candidates": sorted([{"arm": x["arm"], "rep": x.get("rep", 0), "acc": x["acc_test_len"]} for x in rs],
                                 key=lambda x: (-x["acc"])),
            "graph": graph(r["model"], task, r), "has_probe": task in ("antagonist", "commit")}


@app.get("/api/runs/{rid}/model/simulate")
def model_simulate(rid: str, task: str, arm: str | None = None, rep: int | None = None, seed: int = 0, length: int = 200):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    return simulate(r["model"], task, T=max(20, min(length, 1000)), seed=seed, p=run["config"]["pulse_prob"])


@app.get("/api/runs/{rid}/model/probe")
def model_probe(rid: str, task: str, arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    out = probe(r["model"], task)
    if out is None:
        raise HTTPException(404, "No probe for this task.")
    return out


@app.get("/api/runs/{rid}/model/export")
def model_export(rid: str, task: str, kind: Literal["numpy", "torch", "json"] = "numpy", arm: str | None = None, rep: int | None = None):
    run = load(rid)
    r, _ = _pick(run, task, arm, rep)
    base = f"fate-cell-model-{task}-{r['arm']}-seed{r.get('rep', 0) + 1}-{rid}"
    title = _title(run, r)
    if kind == "json":
        body = json.dumps({"title": title, "task": task, "graph": graph(r["model"], task, r), "spec": r["model"]}, indent=1)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{base}.json"'})
    code = code_numpy(r["model"], task, title) if kind == "numpy" else code_torch(r["model"], task, title)
    suffix = "numpy" if kind == "numpy" else "torch"
    return Response(code, media_type="text/x-python",
                    headers={"Content-Disposition": f'attachment; filename="{base}-{suffix}.py"'})
