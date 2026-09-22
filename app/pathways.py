"""Metabolic pathway library, structural analysis and simulation.

Stoichiometry follows standard textbook pathways. Rate constants are illustrative, not fitted to
measurements, so structural results (conservation laws, deficiency, sign-consistency, which theorems
apply) are meaningful while specific concentrations and timings are not.
"""
import numpy as np

PATHWAYS = {}


def _p(pid, name, description, species, clamped, reactions, reference):
    PATHWAYS[pid] = {"id": pid, "name": name, "description": description, "species": species,
                     "clamped": clamped, "reactions": reactions, "reference": reference}


def R(rid, name, enzyme, subs, prods, kcat=1.0, km=0.5, reversible=False, regulators=()):
    return {"id": rid, "name": name, "enzyme": enzyme, "substrates": subs, "products": prods,
            "kcat": kcat, "km": km, "reversible": reversible, "regulators": list(regulators)}


_p("glycolysis", "Glycolysis (Embden-Meyerhof-Parnas)",
   "Glucose to pyruvate in ten enzyme steps, with ATP use and NADH reoxidation attached so the "
   "pathway can reach a steady state. Reads as: glucose to pyruvate, spending two ATP and recovering four. Includes the classic "
   "allosteric controls: ATP inhibits phosphofructokinase and pyruvate kinase, AMP relieves it. AMP is held fixed as a regulator rather than tracked, so the adenine pool here is ATP plus ADP.",
   ["Glc", "G6P", "F6P", "F16BP", "DHAP", "GAP", "BPG", "3PG", "2PG", "PEP", "Pyr",
    "ATP", "ADP", "AMP", "NAD", "NADH", "Pi"],
   ["Glc", "Pi", "AMP"],
   [R("hk", "Hexokinase", "HK", {"Glc": 1, "ATP": 1}, {"G6P": 1, "ADP": 1}, kcat=0.5, km=0.3,
      regulators=[{"species": "G6P", "effect": "inhibit", "k": 1.5}]),
    R("pgi", "Phosphoglucose isomerase", "PGI", {"G6P": 1}, {"F6P": 1}, kcat=2.0, km=0.6, reversible=True),
    R("pfk", "Phosphofructokinase", "PFK", {"F6P": 1, "ATP": 1}, {"F16BP": 1, "ADP": 1}, kcat=2.5, km=0.3,
      regulators=[{"species": "ATP", "effect": "inhibit", "k": 2.0}, {"species": "AMP", "effect": "activate", "k": 0.3}]),
    R("ald", "Aldolase", "ALDO", {"F16BP": 1}, {"DHAP": 1, "GAP": 1}, kcat=1.5, km=0.6, reversible=True),
    R("tpi", "Triose phosphate isomerase", "TPI", {"DHAP": 1}, {"GAP": 1}, kcat=4.0, km=0.6, reversible=True),
    R("gapdh", "Glyceraldehyde-3-P dehydrogenase", "GAPDH", {"GAP": 1, "NAD": 1, "Pi": 1}, {"BPG": 1, "NADH": 1},
      kcat=3.0, km=0.6, reversible=True),
    R("pgk", "Phosphoglycerate kinase", "PGK", {"BPG": 1, "ADP": 1}, {"3PG": 1, "ATP": 1}, kcat=3.0, km=0.6, reversible=True),
    R("pgm", "Phosphoglycerate mutase", "PGM", {"3PG": 1}, {"2PG": 1}, kcat=2.0, km=0.6, reversible=True),
    R("eno", "Enolase", "ENO", {"2PG": 1}, {"PEP": 1}, kcat=2.0, km=0.6, reversible=True),
    R("pk", "Pyruvate kinase", "PK", {"PEP": 1, "ADP": 1}, {"Pyr": 1, "ATP": 1}, kcat=3.0, km=0.3,
      regulators=[{"species": "ATP", "effect": "inhibit", "k": 5.0}, {"species": "F16BP", "effect": "activate", "k": 0.4}]),
    R("atpase", "ATP consumption (cell work)", "ATPase", {"ATP": 1}, {"ADP": 1, "Pi": 1}, kcat=1.2, km=0.5),
    R("pdh", "Pyruvate consumption (PDH or export)", "PDH", {"Pyr": 1}, {}, kcat=1.5, km=0.4),
    R("nox", "NADH reoxidation (respiration or fermentation)", "NOX", {"NADH": 1}, {"NAD": 1}, kcat=1.2, km=0.3),
    R("oxphos", "ATP regeneration from NADH (oxidative phosphorylation)", "OXPHOS", {"NADH": 1, "ADP": 1, "Pi": 1},
      {"NAD": 1, "ATP": 1}, kcat=1.5, km=0.3)],
   "Berg, Tymoczko and Stryer, Biochemistry, glycolysis chapter")

_p("fermentation", "Lactate fermentation",
   "The anaerobic branch: lactate dehydrogenase regenerates NAD from NADH so glycolysis can keep running "
   "without oxygen. This is the Warburg-style branch point in tumour metabolism.",
   ["Pyr", "Lac", "NAD", "NADH", "H"], ["H"],
   [R("ldh", "Lactate dehydrogenase", "LDH", {"Pyr": 1, "NADH": 1}, {"Lac": 1, "NAD": 1}, kcat=2.0, km=0.3, reversible=True),
    R("mct", "Lactate export", "MCT", {"Lac": 1}, {}, kcat=1.0, km=0.6)],
   "Standard anaerobic glycolysis branch")

_p("ppp", "Pentose phosphate pathway (oxidative branch)",
   "The branch that makes NADPH for antioxidant defence and biosynthesis. Cells shifting flux here is one "
   "known route to surviving oxidative stress and some drugs.",
   ["G6P", "6PGL", "6PG", "Ru5P", "CO2", "NADP", "NADPH"], ["CO2"],
   [R("g6pd", "Glucose-6-P dehydrogenase", "G6PD", {"G6P": 1, "NADP": 1}, {"6PGL": 1, "NADPH": 1}, kcat=1.5, km=0.3,
      regulators=[{"species": "NADPH", "effect": "inhibit", "k": 1.0}]),
    R("pgls", "6-phosphogluconolactonase", "PGLS", {"6PGL": 1}, {"6PG": 1}, kcat=3.0, km=0.4),
    R("pgd", "6-phosphogluconate dehydrogenase", "PGD", {"6PG": 1, "NADP": 1}, {"Ru5P": 1, "CO2": 1, "NADPH": 1},
      kcat=1.5, km=0.3)],
   "Berg, Tymoczko and Stryer, pentose phosphate pathway chapter")

_p("tca", "Citric acid cycle (TCA)",
   "The cycle that oxidises acetyl-CoA to CO2, feeding NADH to the respiratory chain. Its conserved cycle "
   "structure makes it a good test of conservation-law analysis.",
   ["AcCoA", "OAA", "Cit", "IsoCit", "AKG", "SucCoA", "Suc", "Fum", "Mal", "CoA", "NAD", "NADH", "CO2", "GDP", "GTP", "Pi"],
   ["CO2", "Pi", "GDP"],
   [R("cs", "Citrate synthase", "CS", {"AcCoA": 1, "OAA": 1}, {"Cit": 1, "CoA": 1}, kcat=1.5, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 1.0}]),
    R("acon", "Aconitase", "ACO", {"Cit": 1}, {"IsoCit": 1}, kcat=2.0, km=0.4, reversible=True),
    R("idh", "Isocitrate dehydrogenase", "IDH", {"IsoCit": 1, "NAD": 1}, {"AKG": 1, "NADH": 1, "CO2": 1}, kcat=1.2, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 0.8}]),
    R("kgdh", "Alpha-ketoglutarate dehydrogenase", "KGDH", {"AKG": 1, "NAD": 1, "CoA": 1}, {"SucCoA": 1, "NADH": 1, "CO2": 1},
      kcat=1.2, km=0.3),
    R("scs", "Succinyl-CoA synthetase", "SCS", {"SucCoA": 1, "GDP": 1, "Pi": 1}, {"Suc": 1, "CoA": 1, "GTP": 1}, kcat=1.5, km=0.3),
    R("sdh", "Succinate dehydrogenase", "SDH", {"Suc": 1}, {"Fum": 1}, kcat=1.5, km=0.4, reversible=True),
    R("fum", "Fumarase", "FUM", {"Fum": 1}, {"Mal": 1}, kcat=2.5, km=0.4, reversible=True),
    R("mdh", "Malate dehydrogenase", "MDH", {"Mal": 1, "NAD": 1}, {"OAA": 1, "NADH": 1}, kcat=1.5, km=0.4, reversible=True)],
   "Berg, Tymoczko and Stryer, citric acid cycle chapter")


def _complex_key(d):
    return tuple(sorted(d.items()))


def stoichiometry(path):
    sp = path["species"]
    idx = {s: i for i, s in enumerate(sp)}
    cols = []
    for r in path["reactions"]:
        v = np.zeros(len(sp))
        for s, k in r["substrates"].items():
            v[idx[s]] -= k
        for s, k in r["products"].items():
            v[idx[s]] += k
        cols.append(v)
    return np.stack(cols, axis=1) if cols else np.zeros((len(sp), 0))


def conservation_laws(N, species, clamped=(), max_support=4, max_coef=3):
    """Small integer conserved pools: y >= 0 with y @ N = 0, over species that are not held fixed."""
    from itertools import combinations, product
    free = [i for i, sp in enumerate(species) if sp not in clamped]
    rows = {i: N[i] for i in free}
    laws, supports = [], []
    for size in range(2, max_support + 1):
        for combo in combinations(free, size):
            if any(set(prev) <= set(combo) for prev in supports):
                continue
            base = np.stack([rows[i] for i in combo])
            for coefs in product(range(1, max_coef + 1), repeat=size):
                if np.gcd.reduce(coefs) != 1:
                    continue
                if np.abs(np.asarray(coefs) @ base).max() < 1e-9:
                    laws.append([(species[i], int(c)) for i, c in zip(combo, coefs)])
                    supports.append(combo)
                    break
            if len(laws) >= 8:
                return laws
    return laws


def deficiency(path):
    complexes, links = [], []
    for r in path["reactions"]:
        a, b = _complex_key(r["substrates"]), _complex_key(r["products"])
        for c in (a, b):
            if c not in complexes:
                complexes.append(c)
        links.append((complexes.index(a), complexes.index(b)))
    n = len(complexes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    l = len({find(i) for i in range(n)})
    N = stoichiometry(path)
    s = int(np.linalg.matrix_rank(N)) if N.size else 0
    reach = {i: set() for i in range(n)}
    for a, b in links:
        reach[a].add(b)
        if any(r["reversible"] for r in path["reactions"]):
            pass
    for r, (a, b) in zip(path["reactions"], links):
        if r["reversible"]:
            reach[b].add(a)
    changed = True
    while changed:
        changed = False
        for i in list(reach):
            for j in list(reach[i]):
                new = reach[j] - reach[i]
                if new:
                    reach[i] |= new
                    changed = True
    weakly_reversible = all(i in reach[j] for i in range(n) for j in reach[i])
    return {"complexes": n, "linkage_classes": l, "rank": s, "deficiency": n - l - s,
            "weakly_reversible": bool(weakly_reversible)}


def influence_signs(path):
    """Sign of each species' effect on each other species, from substrate use, product formation and regulation."""
    sp = path["species"]
    idx = {s: i for i, s in enumerate(sp)}
    edges = {}
    for r in path["reactions"]:
        actors = [(s, +1) for s in r["substrates"]] + [(g["species"], +1 if g["effect"] == "activate" else -1) for g in r["regulators"]]
        targets = [(s, -1) for s in r["substrates"]] + [(s, +1) for s in r["products"]]
        for a, sa in actors:
            for b, sb in targets:
                if a == b:
                    continue
                edges.setdefault((idx[a], idx[b]), set()).add(sa * sb)
        if r["reversible"]:
            for a in r["products"]:
                for b, sb in [(s, +1) for s in r["substrates"]] + [(s, -1) for s in r["products"]]:
                    if a == b:
                        continue
                    edges.setdefault((idx[a], idx[b]), set()).add(sb)
    return edges


def sign_consistency(path):
    """Is the influence graph sign-consistent (no negative undirected cycle)? Monotone if it is."""
    edges = influence_signs(path)
    n = len(path["species"])
    parent, par = list(range(n)), [0] * n

    def find(x):
        p = 0
        while parent[x] != x:
            p ^= par[x]
            x = parent[x]
        return x, p

    conflicts, ambiguous = [], []
    for (a, b), signs in edges.items():
        if len(signs) > 1:
            ambiguous.append((path["species"][a], path["species"][b]))
            continue
        sgn = next(iter(signs))
        ra, pa = find(a)
        rb, pb = find(b)
        need = 0 if sgn > 0 else 1
        if ra == rb:
            if (pa ^ pb) != need:
                conflicts.append((path["species"][a], path["species"][b]))
        else:
            parent[ra] = rb
            par[ra] = pa ^ pb ^ need
    return {"consistent": not conflicts and not ambiguous, "conflicts": conflicts[:8], "ambiguous": ambiguous[:8],
            "edges": len(edges)}


def analyse(path):
    N = stoichiometry(path)
    d = deficiency(path)
    sc = sign_consistency(path)
    laws = conservation_laws(N, path["species"], path["clamped"])
    notes = []
    if d["deficiency"] == 0 and d["weakly_reversible"]:
        notes.append("Deficiency zero and weakly reversible: by the Deficiency Zero Theorem, with mass-action kinetics this "
                     "network has exactly one positive steady state in each conserved class, and it is stable, for every "
                     "choice of rate constants.")
    elif d["deficiency"] == 0:
        notes.append("Deficiency zero but not weakly reversible. For the closed network with mass-action kinetics, the "
                     "theorem rules out a positive steady state whatever the rate constants. That does not contradict the "
                     "simulation below, which holds some pools fixed and uses saturating enzyme kinetics, so it is a "
                     "different, open system.")
    elif d["deficiency"] == 1:
        notes.append("Deficiency one: the Deficiency One Theorem may apply, and absolute concentration robustness becomes "
                     "possible, where one species holds the same steady-state level no matter how much total material there is.")
    else:
        notes.append(f"Deficiency {d['deficiency']}: too high for the deficiency theorems, so steady-state behaviour depends "
                     "on the rate constants and has to be studied numerically.")
    if sc["consistent"]:
        notes.append("The influence graph is sign-consistent, so this network is monotone: it cannot sustain oscillations, "
                     "and almost every trajectory settles. This is the same structure the bench's sign-consistent arm enforces.")
    elif sc["conflicts"]:
        notes.append(f"The influence graph has {len(sc['conflicts'])} sign conflict(s), so the network contains negative "
                     "feedback cycles. Oscillation and multiple stable states are possible.")
    else:
        notes.append("Some species affect each other in both directions depending on the reaction, so the network is not "
                     "sign-consistent and no monotonicity guarantee applies.")
    if laws:
        pools = "; ".join(" + ".join((f"{c} {sp}" if c > 1 else sp) for sp, c in law) for law in laws[:4])
        notes.append(f"{len(laws)} conserved pool(s), quantities the reactions move around but never create or destroy: {pools}.")
    return {"species": path["species"], "reactions": [{"id": r["id"], "name": r["name"], "enzyme": r["enzyme"],
                                                       "substrates": r["substrates"], "products": r["products"],
                                                       "reversible": r["reversible"], "regulators": r["regulators"]}
                                                      for r in path["reactions"]],
            "stoichiometry": N.astype(int).tolist(), "conservation": laws, "structure": d, "signs": sc,
            "notes": notes, "clamped": path["clamped"], "name": path["name"], "description": path["description"],
            "reference": path["reference"]}


def rates(path, c, idx, enzyme_scale=None):
    v = np.zeros((c.shape[0], len(path["reactions"])))
    for j, r in enumerate(path["reactions"]):
        f = np.full(c.shape[0], r["kcat"])
        for s, k in r["substrates"].items():
            x = c[:, idx[s]]
            f = f * (x ** k) / (r["km"] ** k + x ** k)
        for g in r["regulators"]:
            x = c[:, idx[g["species"]]]
            f = f * (g["k"] / (g["k"] + x) if g["effect"] == "inhibit" else 0.3 + 0.7 * x / (g["k"] + x))
        if r["reversible"]:
            b = np.full(c.shape[0], r["kcat"] * 0.4)
            for s, k in r["products"].items():
                x = c[:, idx[s]]
                b = b * (x ** k) / (r["km"] ** k + x ** k)
            f = f - b
        if enzyme_scale is not None:
            f = f * enzyme_scale[j]
        v[:, j] = f
    return v


def initial_state(path, level=1.0):
    c = {s: 0.05 for s in path["species"]}
    for s in path["clamped"]:
        c[s] = level
    for s in ("ATP", "NAD", "NADP", "CoA", "OAA", "GDP", "Pi", "Glc", "AcCoA", "G6P", "Pyr"):
        if s in c:
            c[s] = max(c[s], 0.6)
    for s in ("ADP", "AMP", "NADH", "NADPH", "GTP"):
        if s in c:
            c[s] = 0.15
    return c


def simulate_pathway(path, steps=400, dt=0.05, inputs=None, enzyme_scale=None, c0=None, record=True):
    sp = path["species"]
    idx = {s: i for i, s in enumerate(sp)}
    c = np.array([[(c0 or initial_state(path))[s] for s in sp]], dtype=float)
    N = stoichiometry(path)
    clamp = np.array([s in path["clamped"] for s in sp])
    trace = []
    for t in range(steps):
        v = rates(path, c, idx, enzyme_scale)
        dc = (N @ v[0])[None, :]
        if inputs:
            for s, amt in inputs.items():
                dc[0, idx[s]] += amt
        c = np.clip(c + dt * dc, 0.0, None)
        c[:, clamp] = np.array([[(c0 or initial_state(path))[s] for s in sp]])[:, clamp]
        if record and t % max(1, steps // 200) == 0:
            trace.append(np.round(c[0], 4).tolist())
    flux = rates(path, c, idx, enzyme_scale)[0]
    residual = float(np.abs((N @ flux)[~clamp]).max())
    return {"species": sp, "trace": trace, "final": np.round(c[0], 4).tolist(), "residual": round(residual, 5),
            "flux": {r["id"]: round(float(f), 4) for r, f in zip(path["reactions"], flux)},
            "steady": bool(residual < 5e-3)}
