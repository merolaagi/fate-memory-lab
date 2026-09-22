"""Turn a trained arm into a standalone cell model: simulation, probes, layer graph and code export."""
import json

import numpy as np

from .engine import TASKS, make_batch

TASK_IO = {
    "flipflop": (["channel 1", "channel 2", "channel 3"], ["memory 1", "memory 2", "memory 3"]),
    "xor": (["channel 1", "channel 2"], ["product of memories"]),
    "parity": (["pulse"], ["parity"]),
    "dose": (["channel 1", "channel 2", "channel 3"], ["memory 1", "memory 2", "memory 3"]),
    "background": (["channel 1", "channel 2", "channel 3"], ["memory 1", "memory 2", "memory 3"]),
    "commit": (["drug concentration"], ["committed"]),
    "antagonist": (["agonist", "antagonist"], ["receptor active"]),
}
ARM_RULE = {
    "coop": "sign-consistent: W = S ⊙ softplus(V), S_ij = s_i s_j, no negative cycles",
    "broken": "sign-consistent except one flipped off-diagonal sign (one negative cycle)",
    "contract": "contraction: W = 3.6 V / ‖V‖_F, unique equilibrium",
    "free": "unconstrained W",
}


class CellModel:
    def __init__(self, spec):
        self.spec = spec
        self.mode = spec["membrane"]
        self.dt, self.K = spec["dt"], spec["substeps"]
        for k in ("W", "U", "b", "R", "c", "vmax", "km", "G", "g0"):
            if k in spec:
                setattr(self, k, np.asarray(spec[k], dtype=float))

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    def drive(self, x, h):
        if self.mode == "direct":
            return x @ self.U.T + self.b
        xs = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=-1)
        intake = self.vmax * xs / (self.km + xs)
        if self.mode == "gated":
            intake = intake * self._sig(h @ self.G.T + self.g0)
        return intake @ self.U.T + self.b

    def run(self, u, h0=None):
        u = np.asarray(u, dtype=float)
        if u.ndim == 2:
            u = u[:, None, :]
        h = np.full((u.shape[1], self.W.shape[0]), 0.5) if h0 is None else np.asarray(h0, dtype=float)
        outs, states = [], []
        for t in range(u.shape[0]):
            for _ in range(self.K):
                h = h + self.dt * (-h + self._sig(h @ self.W.T + self.drive(u[t], h)))
            outs.append(h @ self.R.T + self.c)
            states.append(h.copy())
        return np.stack(outs), np.stack(states)


def simulate(spec, task, T=200, seed=0, p=0.05):
    u, y = make_batch(task, 1, T, p, 50000 + seed)
    out, _ = CellModel(spec).run(u)
    acc = float(np.mean(np.sign(out[10:]) == y[10:]))
    return {"inputs": np.round(u[:, 0, :], 4).T.tolist(), "target": y[:, 0, :].T.tolist(),
            "output": np.round(out[:, 0, :], 4).T.tolist(), "accuracy": acc,
            "input_names": TASK_IO[task][0], "output_names": TASK_IO[task][1]}


def probe(spec, task):
    m = CellModel(spec)
    if task == "antagonist":
        ag = np.linspace(0, 2, 21)
        curves = []
        for b in (0.0, 0.5, 1.0, 1.5):
            u = np.zeros((80, len(ag), 2))
            u[:, :, 0] = ag
            u[:, :, 1] = b
            out, _ = m.run(u)
            truth = ag / (ag + 0.5 * (1 + b)) > 0.5
            curves.append({"antagonist": b, "response": np.round(out[-1, :, 0], 4).tolist(),
                           "threshold_true": round(0.5 * (1 + b), 3), "truth": truth.astype(int).tolist()})
        return {"kind": "dose_response", "agonist": np.round(ag, 3).tolist(), "curves": curves}
    if task == "commit":
        amps = np.round(np.linspace(0.1, 1.6, 16), 3)
        durs = np.arange(1, 17)
        grid, truth = [], []
        for d in durs:
            u = np.zeros((d + 60, len(amps), 1))
            u[:d, :, 0] = amps
            out, _ = m.run(u)
            grid.append(np.round(out[-1, :, 0], 3).tolist())
            acc = np.zeros(len(amps))
            hit = np.zeros(len(amps), dtype=bool)
            for t in range(d):
                acc = 0.85 * acc + amps
                hit |= acc > 3.2
            truth.append(hit.astype(int).tolist())
        return {"kind": "commit_map", "amplitude": amps.tolist(), "duration": durs.tolist(), "output": grid, "truth": truth}
    return None


def graph(spec, task, meta):
    ins, outs = TASK_IO[task]
    n, mode = spec["hidden"], spec["membrane"]
    W = np.asarray(spec["W"])
    off = ~np.eye(n, dtype=bool)
    nodes = [{"id": "in", "type": "Input", "title": "Signals", "eq": "x_t ∈ R^%d" % spec["nin"],
              "detail": "Inputs: " + ", ".join(ins), "params": 0}]
    edges = []
    prev = "in"
    if mode != "direct":
        vmax, km = np.asarray(spec["vmax"]), np.asarray(spec["km"])
        nodes.append({"id": "split", "type": "SignSplit", "title": "Molecule split",
                      "eq": "m = [max(x, 0), max(−x, 0)]",
                      "detail": "Each signal becomes a positive and a negative molecule, since concentrations cannot be negative.", "params": 0})
        edges.append([prev, "split"])
        nodes.append({"id": "transport", "type": "MichaelisMenten", "title": "Transporters",
                      "eq": "intake = Vmax ⊙ m / (Km + m)",
                      "detail": "Saturating membrane uptake. Vmax %.2f–%.2f, Km %.2f–%.2f across %d transporters." % (vmax.min(), vmax.max(), km.min(), km.max(), len(km)),
                      "params": int(2 * len(km)), "vectors": {"Vmax": vmax.round(3).tolist(), "Km": km.round(3).tolist()}})
        edges.append(["split", "transport"])
        prev = "transport"
        if mode == "gated":
            nodes.append({"id": "gate", "type": "StateGate", "title": "Gated channels",
                          "eq": "intake ← intake ⊙ σ(G h + g₀)",
                          "detail": "The cell opens and closes its own channels based on its current state.",
                          "params": int(np.asarray(spec["G"]).size + len(spec["g0"])), "matrix": {"G": np.asarray(spec["G"]).round(3).tolist()}})
            edges.append([prev, "gate"])
            prev = "gate"
    U = np.asarray(spec["U"])
    nodes.append({"id": "proj", "type": "Linear", "title": "Input projection", "eq": "d = U·intake + b" if mode != "direct" else "d = U·x + b",
                  "detail": "Maps what gets in onto the %d circuit units." % n, "params": int(U.size + n), "matrix": {"U": U.round(3).tolist()}})
    edges.append([prev, "proj"])
    nodes.append({"id": "cell", "type": "CellCircuit", "title": "Cell circuit",
                  "eq": "h ← h + Δt(−h + σ(W h + d)), ×%d per input step" % spec["substeps"],
                  "detail": "%s. %d units, %d excitatory and %d inhibitory connections. %d resting states found." % (
                      ARM_RULE[spec["arm"]].capitalize(), n, int((W[off] > 0).sum()), int((W[off] < 0).sum()), meta.get("attractors", 0)),
                  "params": int(W.size), "matrix": {"W": W.round(3).tolist()}})
    edges.append(["proj", "cell"])
    edges.append(["cell", "cell", "recurrent"])
    if mode == "gated":
        edges.append(["cell", "gate", "feedback"])
    R = np.asarray(spec["R"])
    nodes.append({"id": "readout", "type": "Linear", "title": "Readout", "eq": "y = R h + c",
                  "detail": "Reads the circuit state out as: " + ", ".join(outs) + ".", "params": int(R.size + len(spec["c"])),
                  "matrix": {"R": R.round(3).tolist()}})
    edges.append(["cell", "readout"])
    nodes.append({"id": "decide", "type": "Sign", "title": "Decision", "eq": "on if y > 0, off otherwise",
                  "detail": "Outputs: " + ", ".join(outs), "params": 0})
    edges.append(["readout", "decide"])
    return {"nodes": nodes, "edges": edges, "total_params": int(sum(nd["params"] for nd in nodes))}


def _arr(name, a):
    return "%s = np.array(%s)" % (name, json.dumps(np.asarray(a).round(6).tolist()))


def code_numpy(spec, task, title):
    ins, outs = TASK_IO[task]
    lines = ['"""', title, "",
             "Standalone cell model exported from Fate Memory Lab. Needs only NumPy.",
             "Inputs per time step: " + ", ".join(ins) + ".",
             "Outputs per time step: " + ", ".join(outs) + " (positive means on).",
             "",
             "Usage:",
             "    from cell_model import CellModel",
             "    m = CellModel()",
             "    outputs, states = m.run(inputs)   # inputs shape (T, %d) or (T, batch, %d)" % (spec["nin"], spec["nin"]),
             '"""', "import numpy as np", "",
             "MEMBRANE = %r" % spec["membrane"], "DT = %r" % spec["dt"], "SUBSTEPS = %d" % spec["substeps"]]
    for k in ("W", "U", "b", "R", "c", "vmax", "km", "G", "g0"):
        if k in spec:
            lines.append(_arr(k, spec[k]))
    lines += ["", "", "def _sig(z):", "    return 1.0 / (1.0 + np.exp(-z))", "", "",
              "class CellModel:",
              "    def drive(self, x, h):",
              '        if MEMBRANE == "direct":',
              "            return x @ U.T + b",
              "        m = np.concatenate([np.maximum(x, 0.0), np.maximum(-x, 0.0)], axis=-1)",
              "        intake = vmax * m / (km + m)",
              '        if MEMBRANE == "gated":',
              "            intake = intake * _sig(h @ G.T + g0)",
              "        return intake @ U.T + b",
              "",
              "    def run(self, inputs, h0=None):",
              "        u = np.asarray(inputs, dtype=float)",
              "        if u.ndim == 2:",
              "            u = u[:, None, :]",
              "        h = np.full((u.shape[1], W.shape[0]), 0.5) if h0 is None else np.asarray(h0, dtype=float)",
              "        outs, states = [], []",
              "        for t in range(u.shape[0]):",
              "            for _ in range(SUBSTEPS):",
              "                h = h + DT * (-h + _sig(h @ W.T + self.drive(u[t], h)))",
              "            outs.append(h @ R.T + c)",
              "            states.append(h.copy())",
              "        return np.stack(outs), np.stack(states)",
              "", "",
              'if __name__ == "__main__":',
              "    x = np.zeros((50, %d))" % spec["nin"],
              "    x[5] = 1.0",
              "    y, _ = CellModel().run(x)",
              '    print("output at last step:", y[-1, 0])', ""]
    return "\n".join(lines)


def code_torch(spec, task, title):
    ins, outs = TASK_IO[task]
    mode = spec["membrane"]
    L = ['"""', title, "",
         "PyTorch version of a cell model exported from Fate Memory Lab.",
         "Weights load as trainable parameters, so the model can be fine-tuned or dropped into a larger network.",
         "Note: fine-tuning W freely does not keep the sign pattern of a sign-consistent circuit.",
         "Inputs: tensor (T, batch, %d): %s." % (spec["nin"], ", ".join(ins)),
         "Outputs: tensor (T, batch, %d): %s (positive means on)." % (spec["nout"], ", ".join(outs)),
         '"""', "import torch", "import torch.nn as nn", "",
         "INIT = {"]
    for k in ("W", "U", "b", "R", "c", "vmax", "km", "G", "g0"):
        if k in spec:
            L.append("    %r: %s," % (k, json.dumps(np.asarray(spec[k]).round(6).tolist())))
    L += ["}", "", "",
          "class MichaelisMentenMembrane(nn.Module):",
          "    def __init__(self, vmax, km):",
          "        super().__init__()",
          "        self.vmax = nn.Parameter(torch.tensor(vmax))",
          "        self.km = nn.Parameter(torch.tensor(km))",
          "",
          "    def forward(self, x):",
          "        m = torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)",
          "        return self.vmax * m / (self.km.clamp_min(1e-3) + m)",
          "", "",
          "class CellCircuit(nn.Module):",
          "    def __init__(self, dt=%r, substeps=%d):" % (spec["dt"], spec["substeps"]),
          "        super().__init__()",
          "        self.dt, self.substeps = dt, substeps",
          "        self.W = nn.Parameter(torch.tensor(INIT['W']))",
          "        self.U = nn.Parameter(torch.tensor(INIT['U']))",
          "        self.b = nn.Parameter(torch.tensor(INIT['b']))",
          "        self.R = nn.Parameter(torch.tensor(INIT['R']))",
          "        self.c = nn.Parameter(torch.tensor(INIT['c']))"]
    if mode != "direct":
        L.append("        self.membrane = MichaelisMentenMembrane(INIT['vmax'], INIT['km'])")
    if mode == "gated":
        L += ["        self.G = nn.Parameter(torch.tensor(INIT['G']))",
              "        self.g0 = nn.Parameter(torch.tensor(INIT['g0']))"]
    L += ["", "    def drive(self, x, h):"]
    if mode == "direct":
        L.append("        return x @ self.U.T + self.b")
    else:
        L.append("        intake = self.membrane(x)")
        if mode == "gated":
            L.append("        intake = intake * torch.sigmoid(h @ self.G.T + self.g0)")
        L.append("        return intake @ self.U.T + self.b")
    L += ["",
          "    def forward(self, u, h=None):",
          "        if h is None:",
          "            h = torch.full((u.shape[1], self.W.shape[0]), 0.5, dtype=u.dtype, device=u.device)",
          "        outs = []",
          "        for t in range(u.shape[0]):",
          "            for _ in range(self.substeps):",
          "                h = h + self.dt * (-h + torch.sigmoid(h @ self.W.T + self.drive(u[t], h)))",
          "            outs.append(h @ self.R.T + self.c)",
          "        return torch.stack(outs), h",
          "", "",
          'if __name__ == "__main__":',
          "    model = CellCircuit()",
          "    x = torch.zeros(50, 1, %d)" % spec["nin"],
          "    x[5] = 1.0",
          "    y, h = model(x)",
          '    print("output at last step:", y[-1, 0].tolist())', ""]
    return "\n".join(L)
