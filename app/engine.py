"""
Fate-memory bench engine.

Arms (matched parameter count, identical starting weights):
  coop      W = S * softplus(V), S_ij = s_i s_j   sign-consistent, monotone (Hirsch / Angeli-Sontag)
  broken    same with one off-diagonal sign flipped -> exactly one negative cycle
  contract  W = (3.6 / ||V||_F) V                  contraction -> unique equilibrium
  free      W = V                                  unconstrained

Tasks:
  flipflop  3 channels, output = sign of last pulse on each channel
  xor       2 latched channels, output = product of the two latched signs
  parity    1 channel of identical pulses, output flips on every pulse
  dose      flipflop, but each sequence's pulses are scaled by a random gain (0.3x to 3x)
  background flipflop on top of a slowly wandering background level on every channel

Cell-inspired stress tests on the trained circuit:
  drift     sign-preserving multiplicative drift of weight magnitudes (enzyme levels vary)
  noise     Langevin noise on the state during the test rollout (gene-expression noise)
  division  every 40 steps the state is multiplied by random partition noise (cell division)

Landscape: 2D principal-component map of settling trajectories (a Waddington landscape view).
"""
import time
import autograd.numpy as np
from autograd import grad
import numpy as onp

ARMS = ["coop", "broken", "contract", "free"]
TASKS = {
    "flipflop": {"nin": 3, "nout": 3},
    "xor": {"nin": 2, "nout": 1},
    "parity": {"nin": 1, "nout": 1},
    "dose": {"nin": 3, "nout": 3},
    "background": {"nin": 3, "nout": 3},
}
DT = 0.5

DEFAULTS = {
    "tasks": ["flipflop", "xor", "parity"],
    "arms": ARMS,
    "hidden": 24,
    "substeps": 3,
    "iters": 500,
    "lr": 0.01,
    "batch": 32,
    "train_len": 60,
    "test_len": 400,
    "pulse_prob": 0.05,
    "self_excitation": 5.0,
    "coupling": 0.05,
    "drift": [0.1, 0.2, 0.3],
    "noise": [0.02, 0.05, 0.1],
    "division": [0.1, 0.25, 0.5],
    "seed": 0,
    "repeats": 3,
}


class Cancelled(Exception):
    pass


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def sign_matrices(n, seed):
    r = onp.random.default_rng(seed)
    s = r.choice([-1.0, 1.0], size=n)
    coop = onp.outer(s, s)
    broken = coop.copy()
    broken[0, 1] *= -1.0
    return coop, broken


def recurrent(arm, V, S):
    if arm == "coop":
        return S["coop"] * softplus(V)
    if arm == "broken":
        return S["broken"] * softplus(V)
    if arm == "contract":
        return 3.6 * V / np.sqrt(np.sum(V ** 2))
    return V


def init_params(arm, cfg, S, nin, nout):
    n = cfg["hidden"]
    r = onp.random.default_rng(cfg["seed"] + 1)
    mag = onp.abs(r.normal(cfg["coupling"], cfg["coupling"] * 0.4 + 1e-6, (n, n)))
    onp.fill_diagonal(mag, cfg["self_excitation"])
    w0 = S["coop"] * mag
    V = onp.log(onp.expm1(mag)) if arm in ("coop", "broken") else w0.copy()
    return {
        "V": V,
        "U": r.normal(0, 1.0, (n, nin)),
        "b": -0.5 * onp.diag(w0).copy(),
        "R": r.normal(0, 0.1, (nout, n)),
        "c": onp.zeros(nout),
    }


def make_batch(task, B, T, p, seed):
    r = onp.random.default_rng(seed)
    nin = TASKS[task]["nin"]
    mask = r.random((T, B, nin)) < p
    if task == "parity":
        u = mask.astype(float)
        y = onp.zeros((T, B, 1))
        state = onp.ones((B, 1))
        for t in range(T):
            state = onp.where(u[t] > 0, -state, state)
            y[t] = state
        return u, y
    mask[0] = True
    base = "flipflop" if task in ("dose", "background") else task
    u = onp.zeros((T, B, nin))
    u[mask] = r.choice([-1.0, 1.0], size=int(mask.sum()))
    latched = onp.zeros((T, B, nin))
    last = onp.zeros((B, nin))
    for t in range(T):
        last = onp.where(u[t] != 0, u[t], last)
        latched[t] = last
    if task == "dose":
        gain = onp.exp(r.uniform(onp.log(0.3), onp.log(3.0), size=(1, B, 1)))
        return u * gain, latched
    if task == "background":
        steps = r.normal(0, 0.04, (T, B, nin))
        bg = onp.zeros((T, B, nin))
        level = r.uniform(-0.5, 0.5, (B, nin))
        for t in range(T):
            level = onp.clip(level + steps[t], -0.5, 0.5)
            bg[t] = level
        return u + bg, latched
    if base == "flipflop":
        return u, latched
    return u, latched[:, :, :1] * latched[:, :, 1:2]


def rollout(P, W, u, K):
    h = 0.5 * np.ones((u.shape[1], W.shape[0]))
    outs = []
    for t in range(u.shape[0]):
        drive = u[t] @ P["U"].T + P["b"]
        for _ in range(K):
            h = h + DT * (-h + sig(h @ W.T + drive))
        outs.append(h @ P["R"].T + P["c"])
    return np.stack(outs)


def rollout_perturbed(P, W, u, K, noise=0.0, division=0.0, seed=11, every=40):
    r = onp.random.default_rng(seed)
    h = 0.5 * onp.ones((u.shape[1], W.shape[0]))
    outs = []
    for t in range(u.shape[0]):
        if division > 0 and t > 0 and t % every == 0:
            h = onp.clip(h * onp.exp(division * r.normal(size=h.shape)), 0.0, 1.0)
        drive = u[t] @ P["U"].T + P["b"]
        for _ in range(K):
            h = h + DT * (-h + 1.0 / (1.0 + onp.exp(-(h @ W.T + drive))))
            if noise > 0:
                h = onp.clip(h + noise * onp.sqrt(DT) * r.normal(size=h.shape), 0.0, 1.0)
        outs.append(h @ P["R"].T + P["c"])
    return onp.stack(outs)


def perturbed_accuracy(task, P, W, cfg, noise=0.0, division=0.0, B=48, seed=17):
    u, y = make_batch(task, B, cfg["test_len"], cfg["pulse_prob"], seed)
    accs = []
    for k in range(3):
        out = rollout_perturbed(P, W, u, cfg["substeps"], noise, division, seed=seed + 101 * k)
        accs.append(onp.mean(onp.sign(out[10:]) == y[10:]))
    return float(onp.mean(accs))


def landscape(P, W, runs=96, traj_runs=24, steps=600, every=20, seed=3, ids=None):
    r = onp.random.default_rng(seed)
    h = r.random((runs, W.shape[0]))
    frames = [h.copy()]
    for s in range(1, 2501):
        h = h + DT * (-h + 1.0 / (1.0 + onp.exp(-(h @ W.T + P["b"]))))
        if s <= steps and s % every == 0:
            frames.append(h.copy())
    ends = h
    X = onp.concatenate(frames + [ends], axis=0)
    mu = X.mean(axis=0)
    _, sv, Vt = onp.linalg.svd(X - mu, full_matrices=False)
    comps = Vt[:2]
    var = (sv[:2] ** 2) / max(float((sv ** 2).sum()), 1e-12)
    proj = lambda A: onp.round((A - mu) @ comps.T, 4)
    traj = [proj(onp.stack([f[i] for f in frames])).tolist() for i in range(traj_runs)]
    return {"traj": traj, "ends": proj(ends).tolist(), "ids": ids or [], "var": [round(float(v), 3) for v in var]}


def settle_test(P, W, runs=96, steps=2500, tol=1e-6, seed=3):
    r = onp.random.default_rng(seed)
    h = r.random((runs, W.shape[0]))
    speed = onp.zeros(runs)
    first_still = onp.full(runs, -1)
    for s in range(steps):
        hn = h + DT * (-h + 1.0 / (1.0 + onp.exp(-(h @ W.T + P["b"]))))
        step_speed = onp.abs(hn - h).max(axis=1)
        moving = step_speed >= tol
        first_still = onp.where(moving, -1, onp.where(first_still < 0, s, first_still))
        if s >= steps - 300:
            speed = onp.maximum(speed, step_speed)
        h = hn
    ids, table = [], {}
    for end, sp in zip(onp.round(h, 2), speed):
        if sp >= tol:
            ids.append(-1)
            continue
        key = tuple(end)
        if key not in table:
            table[key] = len(table)
        ids.append(table[key])
    settled_mask = speed < tol
    settled = float(onp.mean(settled_mask))
    times = first_still[settled_mask & (first_still >= 0)]
    settle_steps = float(onp.median(times)) if times.size else None
    return {"settled": settled, "attractors": len(table), "wells": ids, "settle_steps": settle_steps}


def accuracy(task, P, W, cfg, T, B=128, seed=7):
    u, y = make_batch(task, B, T, cfg["pulse_prob"], seed)
    out = onp.asarray(rollout(P, W, u, cfg["substeps"]))
    return float(onp.mean(onp.sign(out[10:]) == y[10:]))


def negative_cycle_edges(W, S):
    off = ~onp.eye(W.shape[0], dtype=bool)
    return int(((S["coop"] * W)[off] < 0).sum())


def run_job(task, arm, cfg, progress=None, rep=0):
    cfg = dict(cfg)
    cfg["seed"] = cfg["seed"] + 7919 * rep
    t0 = time.time()
    tk = TASKS[task]
    Sc, Sb = sign_matrices(cfg["hidden"], cfg["seed"])
    S = {"coop": Sc, "broken": Sb}
    P = init_params(arm, cfg, S, tk["nin"], tk["nout"])
    K = cfg["substeps"]

    def loss(P, u, y):
        W = recurrent(arm, P["V"], S)
        return np.mean((rollout(P, W, u, K) - y) ** 2)

    g = grad(loss)
    m = {k: onp.zeros_like(v) for k, v in P.items()}
    v2 = {k: onp.zeros_like(v) for k, v in P.items()}
    curve = []
    lr = cfg["lr"]
    for i in range(1, cfg["iters"] + 1):
        u, y = make_batch(task, cfg["batch"], cfg["train_len"], cfg["pulse_prob"], cfg["seed"] * 100003 + 1000 + i)
        G = g(P, u, y)
        for k in P:
            gk = onp.nan_to_num(G[k])
            m[k] = 0.9 * m[k] + 0.1 * gk
            v2[k] = 0.999 * v2[k] + 0.001 * gk ** 2
            P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** i)) / (onp.sqrt(v2[k] / (1 - 0.999 ** i)) + 1e-8)
        if i % 10 == 0 or i == 1:
            curve.append([i, float(loss(P, u, y))])
            if progress:
                progress(i / cfg["iters"])

    W = onp.asarray(recurrent(arm, P["V"], S))
    result = {
        "task": task,
        "arm": arm,
        "rep": rep,
        "seed": cfg["seed"],
        "params": int(sum(v.size for v in P.values())),
        "acc_train_len": accuracy(task, P, W, cfg, cfg["train_len"]),
        "acc_test_len": accuracy(task, P, W, cfg, cfg["test_len"]),
        "negative_edges": negative_cycle_edges(W, S),
        "curve": curve,
    }
    result.update(settle_test(P, W))
    result["landscape"] = landscape(P, W, ids=result["wells"])
    drift = []
    for eps in cfg["drift"]:
        fs, accs = [], []
        for d in range(4):
            r = onp.random.default_rng(100 + d)
            Wd = W * onp.exp(eps * r.normal(size=W.shape))
            fs.append(settle_test(P, Wd, runs=48, steps=1500)["settled"])
            accs.append(accuracy(task, P, Wd, cfg, cfg["test_len"], B=48))
        drift.append({"eps": eps, "settled": float(onp.mean(fs)), "acc": float(onp.mean(accs))})
    result["drift"] = drift
    result["drift_base"] = accuracy(task, P, W, cfg, cfg["test_len"], B=48)
    result["stress_base"] = perturbed_accuracy(task, P, W, cfg)
    result["noise"] = [{"eps": e, "acc": perturbed_accuracy(task, P, W, cfg, noise=e)} for e in cfg.get("noise", [])]
    result["division"] = [{"eps": e, "acc": perturbed_accuracy(task, P, W, cfg, division=e)} for e in cfg.get("division", [])]
    result["seconds"] = round(time.time() - t0, 1)
    return result
