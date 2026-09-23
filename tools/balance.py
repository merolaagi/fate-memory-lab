"""Balance the illustrative rate constants so every pathway reaches a steady state.

Run: .venv/bin/python tools/balance.py
Writes app/rate_tuning.json, which app/pathways.py applies at import.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import pathways as P  # noqa: E402


def balance(path, rounds=80, steps=3000, cap=10.0, floor=0.05, target=6.0):
    for _ in range(rounds):
        run = P.simulate_pathway(path, steps=steps, record=False)
        levels = {s: v for s, v in zip(path["species"], run["final"]) if s not in path["clamped"]}
        worst = max(levels.items(), key=lambda kv: kv[1], default=(None, 0))
        if run["steady"] and worst[1] < target:
            return True
        sp = worst[0]
        consumers = [r for r in path["reactions"] if sp in r["substrates"]]
        producers = [r for r in path["reactions"] if sp in r["products"]]
        if not consumers and not producers:
            return False
        for r in consumers:
            r["kcat"] = round(min(r["kcat"] * 1.4, cap), 3)
        for r in producers:
            r["kcat"] = round(max(r["kcat"] * 0.8, floor), 3)
    run = P.simulate_pathway(path, steps=steps, record=False)
    return run["steady"]


def main():
    out = {}
    for pid, path in P.PATHWAYS.items():
        ok = balance(path)
        out[pid] = {r["id"]: r["kcat"] for r in path["reactions"]}
        run = P.simulate_pathway(path, steps=4000, record=False)
        print(f"{pid:16s} settled={run['steady']} residual={run['residual']:.4f} "
              f"max={max(run['final']):.2f} {'' if ok else '(did not fully balance)'}")
    m = P.merged()
    ok = balance(m, rounds=120, steps=4000)
    out["all"] = {r["id"]: r["kcat"] for r in m["reactions"]}
    run = P.simulate_pathway(m, steps=8000, record=False)
    print(f"{'all':16s} settled={run['steady']} residual={run['residual']:.4f} max={max(run['final']):.2f}")
    Path(__file__).resolve().parent.parent.joinpath("app/rate_tuning.json").write_text(json.dumps(out, indent=1))
    print("wrote app/rate_tuning.json")


if __name__ == "__main__":
    main()
