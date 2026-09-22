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


def R(rid, name, enzyme, subs, prods, kcat=1.0, km=0.5, reversible=False, regulators=(), deficiency=None, drugs=()):
    return {"id": rid, "name": name, "enzyme": enzyme, "substrates": subs, "products": prods,
            "kcat": kcat, "km": km, "reversible": reversible, "regulators": list(regulators),
            "deficiency": deficiency, "drugs": list(drugs)}


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
   ["Pyr", "Lac", "NAD", "NADH", "H"], ["H", "Pyr"],
   [R("ldh", "Lactate dehydrogenase", "LDH", {"Pyr": 1, "NADH": 1}, {"Lac": 1, "NAD": 1}, kcat=2.0, km=0.3, reversible=True),
    R("mct", "Lactate export", "MCT", {"Lac": 1}, {}, kcat=1.0, km=0.6),
    R("supply", "NADH supply from glycolysis (GAPDH upstream)", "GAPDH", {"NAD": 1}, {"NADH": 1}, kcat=0.6, km=0.4)],
   "Standard anaerobic glycolysis branch")

_p("ppp", "Pentose phosphate pathway (oxidative branch)",
   "The branch that makes NADPH for antioxidant defence and biosynthesis. Cells shifting flux here is one "
   "known route to surviving oxidative stress and some drugs.",
   ["G6P", "6PGL", "6PG", "Ru5P", "CO2", "NADP", "NADPH"], ["CO2", "G6P"],
   [R("g6pd", "Glucose-6-P dehydrogenase", "G6PD", {"G6P": 1, "NADP": 1}, {"6PGL": 1, "NADPH": 1}, kcat=1.5, km=0.3,
      regulators=[{"species": "NADPH", "effect": "inhibit", "k": 1.0}]),
    R("pgls", "6-phosphogluconolactonase", "PGLS", {"6PGL": 1}, {"6PG": 1}, kcat=3.0, km=0.4),
    R("pgd", "6-phosphogluconate dehydrogenase", "PGD", {"6PG": 1, "NADP": 1}, {"Ru5P": 1, "CO2": 1, "NADPH": 1},
      kcat=1.5, km=0.3),
    R("nadph_use", "NADPH consumption (biosynthesis and antioxidant defence)", "NADPH-use", {"NADPH": 1}, {"NADP": 1},
      kcat=1.0, km=0.3),
    R("r5p_use", "Ribose-5-P consumption (nucleotide synthesis)", "R5P-use", {"Ru5P": 1}, {}, kcat=1.0, km=0.3)],
   "Berg, Tymoczko and Stryer, pentose phosphate pathway chapter")

_p("tca", "Citric acid cycle (TCA)",
   "The cycle that oxidises acetyl-CoA to CO2, feeding NADH to the respiratory chain. Its conserved cycle "
   "structure makes it a good test of conservation-law analysis.",
   ["Pyr", "AcCoA", "OAA", "Cit", "IsoCit", "AKG", "SucCoA", "Suc", "Fum", "Mal", "CoA", "NAD", "NADH", "CO2",
    "GDP", "GTP", "Pi", "PEP", "ADP", "ATP"],
   ["CO2", "Pi", "GDP", "Pyr", "ADP"],
   [R("pdh", "Pyruvate dehydrogenase (entry into the cycle)", "PDH", {"Pyr": 1, "CoA": 1, "NAD": 1},
      {"AcCoA": 1, "NADH": 1, "CO2": 1}, kcat=1.2, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 1.0}]),
    R("cs", "Citrate synthase", "CS", {"AcCoA": 1, "OAA": 1}, {"Cit": 1, "CoA": 1}, kcat=1.5, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 1.0}]),
    R("acon", "Aconitase", "ACO", {"Cit": 1}, {"IsoCit": 1}, kcat=2.0, km=0.4, reversible=True),
    R("idh", "Isocitrate dehydrogenase", "IDH", {"IsoCit": 1, "NAD": 1}, {"AKG": 1, "NADH": 1, "CO2": 1}, kcat=1.2, km=0.3,
      regulators=[{"species": "NADH", "effect": "inhibit", "k": 0.8}]),
    R("kgdh", "Alpha-ketoglutarate dehydrogenase", "KGDH", {"AKG": 1, "NAD": 1, "CoA": 1}, {"SucCoA": 1, "NADH": 1, "CO2": 1},
      kcat=1.2, km=0.3),
    R("scs", "Succinyl-CoA synthetase", "SCS", {"SucCoA": 1, "GDP": 1, "Pi": 1}, {"Suc": 1, "CoA": 1, "GTP": 1}, kcat=1.5, km=0.3),
    R("sdh", "Succinate dehydrogenase", "SDH", {"Suc": 1}, {"Fum": 1}, kcat=1.5, km=0.4, reversible=True),
    R("fum", "Fumarase", "FUM", {"Fum": 1}, {"Mal": 1}, kcat=2.5, km=0.4, reversible=True),
    R("mdh", "Malate dehydrogenase", "MDH", {"Mal": 1, "NAD": 1}, {"OAA": 1, "NADH": 1}, kcat=1.5, km=0.4, reversible=True),
    R("etc", "Respiration: NADH reoxidation with ATP synthesis", "ETC", {"NADH": 1, "ADP": 1, "Pi": 1},
      {"NAD": 1, "ATP": 1}, kcat=2.0, km=0.3),
    R("gtp_use", "GTP consumption", "GTP-use", {"GTP": 1}, {"GDP": 1}, kcat=1.5, km=0.3),
    R("atp_use", "ATP consumption (cell work)", "ATPase", {"ATP": 1}, {"ADP": 1, "Pi": 1}, kcat=1.2, km=0.5),
    R("pepck", "Phosphoenolpyruvate carboxykinase (exit to gluconeogenesis)", "PEPCK",
      {"OAA": 1}, {"PEP": 1, "CO2": 1}, kcat=1.2, km=0.4),
    R("pep_out", "PEP into gluconeogenesis", "Transfer", {"PEP": 1}, {}, kcat=1.5, km=0.4)],
   "Berg, Tymoczko and Stryer, citric acid cycle chapter")


_p("urea", "Urea cycle",
   "How the body turns toxic ammonia into urea for excretion. Every step has a known inherited deficiency, and "
   "blocking any of them backs ammonia up into the blood.",
   ["NH3", "CO2", "CP", "Orn", "Citrul", "Asp", "ASA", "Arg", "Urea", "Fum"], ["CO2", "Asp"],
   [R("nh3_in", "Ammonia from protein turnover", "Supply", {}, {"NH3": 1}, kcat=0.3),
    R("cps1", "Carbamoyl phosphate synthetase I", "CPS1", {"NH3": 1, "CO2": 1}, {"CP": 1}, kcat=0.6, km=0.4,
      deficiency="CPS1 deficiency: ammonia builds up from birth, the most severe urea cycle disorder"),
    R("otc", "Ornithine transcarbamylase", "OTC", {"CP": 1, "Orn": 1}, {"Citrul": 1}, kcat=1.5, km=0.3,
      deficiency="OTC deficiency: the commonest urea cycle disorder, X-linked; carbamoyl phosphate spills into orotic acid"),
    R("ass", "Argininosuccinate synthetase", "ASS", {"Citrul": 1, "Asp": 1}, {"ASA": 1}, kcat=1.2, km=0.3,
      deficiency="Citrullinemia type I: citrulline accumulates"),
    R("asl", "Argininosuccinate lyase", "ASL", {"ASA": 1}, {"Arg": 1, "Fum": 1}, kcat=1.5, km=0.3,
      deficiency="Argininosuccinic aciduria: argininosuccinate accumulates"),
    R("arg1", "Arginase 1", "ARG1", {"Arg": 1}, {"Urea": 1, "Orn": 1}, kcat=1.5, km=0.3,
      deficiency="Argininemia: arginine accumulates, with spastic diplegia rather than acute ammonia crises"),
    R("urea_out", "Urea excretion", "Excretion", {"Urea": 1}, {}, kcat=2.0, km=0.4),
    R("fum_out", "Fumarate to the TCA cycle", "Transfer", {"Fum": 1}, {}, kcat=2.0, km=0.4)],
   "Standard urea cycle; deficiencies as described in clinical genetics texts")

_p("phe", "Phenylalanine and tyrosine catabolism",
   "The route that breaks down phenylalanine. Blocks along it cause several of the best known inherited metabolic "
   "diseases, and one of them is treated with a drug that deliberately blocks a step further down.",
   ["Phe", "Tyr", "HPP", "HGA", "MAA", "Fum", "AcAc"], [],
   [R("phe_in", "Dietary phenylalanine", "Diet", {}, {"Phe": 1}, kcat=0.5),
    R("pah", "Phenylalanine hydroxylase", "PAH", {"Phe": 1}, {"Tyr": 1}, kcat=1.2, km=0.4,
      deficiency="Phenylketonuria (PKU): phenylalanine accumulates and harms the developing brain; treated by dietary restriction",
      drugs=[{"name": "Sapropterin (BH4)", "effect": "activates", "note": "cofactor analogue that boosts residual enzyme in some patients"}]),
    R("tat", "Tyrosine aminotransferase", "TAT", {"Tyr": 1}, {"HPP": 1}, kcat=1.5, km=0.4,
      deficiency="Tyrosinemia type II: tyrosine accumulates, with eye and skin lesions"),
    R("hpd", "4-hydroxyphenylpyruvate dioxygenase", "HPD", {"HPP": 1}, {"HGA": 1}, kcat=1.5, km=0.4,
      deficiency="Tyrosinemia type III",
      drugs=[{"name": "Nitisinone", "effect": "inhibits", "note": "blocks this step on purpose to stop toxic metabolites forming further down in tyrosinemia type I"}]),
    R("hgd", "Homogentisate 1,2-dioxygenase", "HGD", {"HGA": 1}, {"MAA": 1}, kcat=1.5, km=0.4,
      deficiency="Alkaptonuria: homogentisic acid accumulates, darkens urine and damages cartilage"),
    R("fah", "Fumarylacetoacetate hydrolase", "FAH", {"MAA": 1}, {"Fum": 1, "AcAc": 1}, kcat=1.5, km=0.4,
      deficiency="Tyrosinemia type I: toxic intermediates build up and damage liver and kidney"),
    R("fum_out", "Fumarate to the TCA cycle", "Transfer", {"Fum": 1}, {}, kcat=2.0, km=0.4),
    R("acac_out", "Acetoacetate to ketone metabolism", "Transfer", {"AcAc": 1}, {}, kcat=2.0, km=0.4)],
   "Phenylalanine to fumarate and acetoacetate; diseases as in clinical genetics texts")

_p("galactose", "Galactose metabolism (Leloir pathway)",
   "How milk sugar enters glycolysis. A block at the second step is classic galactosemia, where galactose-1-phosphate "
   "accumulates inside cells.",
   ["Gal", "Gal1P", "UDPGlc", "UDPGal", "G1P", "G6P", "ATP", "ADP"], ["ATP", "ADP"],
   [R("gal_in", "Dietary galactose (lactose)", "Diet", {}, {"Gal": 1}, kcat=0.15),
    R("galk", "Galactokinase", "GALK", {"Gal": 1, "ATP": 1}, {"Gal1P": 1, "ADP": 1}, kcat=0.6, km=0.3,
      deficiency="Galactokinase deficiency: galactose accumulates and forms cataracts, but without the systemic illness of classic galactosemia"),
    R("galt", "Galactose-1-phosphate uridylyltransferase", "GALT", {"Gal1P": 1, "UDPGlc": 1}, {"G1P": 1, "UDPGal": 1},
      kcat=1.5, km=0.3,
      deficiency="Classic galactosemia: galactose-1-phosphate accumulates and is toxic to liver, brain and ovary"),
    R("gale", "UDP-galactose 4-epimerase", "GALE", {"UDPGal": 1}, {"UDPGlc": 1}, kcat=2.0, km=0.4, reversible=True,
      deficiency="Epimerase deficiency galactosemia, usually milder"),
    R("pgm", "Phosphoglucomutase", "PGM", {"G1P": 1}, {"G6P": 1}, kcat=2.0, km=0.4, reversible=True),
    R("g6p_out", "Glucose-6-phosphate into glycolysis", "Transfer", {"G6P": 1}, {}, kcat=1.5, km=0.4)],
   "Leloir pathway; galactosemias as in clinical genetics texts")

_p("purine", "Purine salvage and degradation",
   "Recycling purines instead of making them from scratch, and the breakdown route that ends in uric acid. It holds "
   "two famous enzyme deficiencies and the target of the commonest gout drug.",
   ["PRPP", "Hx", "IMP", "Xan", "Urate", "Ado", "Ino"], ["PRPP", "Ado"],
   [R("hgprt", "Hypoxanthine-guanine phosphoribosyltransferase", "HGPRT", {"Hx": 1, "PRPP": 1}, {"IMP": 1},
      kcat=1.5, km=0.3,
      deficiency="Lesch-Nyhan syndrome when complete: salvage fails, purines are shunted to uric acid, causing gout and severe neurological disease"),
    R("ada", "Adenosine deaminase", "ADA", {"Ado": 1}, {"Ino": 1}, kcat=1.5, km=0.3,
      deficiency="ADA deficiency: a form of severe combined immunodeficiency; toxic metabolites kill developing lymphocytes",
      drugs=[{"name": "Pentostatin", "effect": "inhibits", "note": "used in hairy cell leukaemia, exploiting the same lymphocyte toxicity"}]),
    R("pnp", "Purine nucleoside phosphorylase", "PNP", {"Ino": 1}, {"Hx": 1}, kcat=2.0, km=0.3,
      deficiency="PNP deficiency: T-cell immunodeficiency"),
    R("xo1", "Xanthine oxidase, first step", "XO", {"Hx": 1}, {"Xan": 1}, kcat=1.0, km=0.4,
      drugs=[{"name": "Allopurinol", "effect": "inhibits", "note": "the standard gout drug, lowering uric acid production"},
             {"name": "Febuxostat", "effect": "inhibits", "note": "a selective alternative"}]),
    R("xo2", "Xanthine oxidase, second step", "XO", {"Xan": 1}, {"Urate": 1}, kcat=1.0, km=0.4,
      drugs=[{"name": "Allopurinol", "effect": "inhibits", "note": "same enzyme, same block"}]),
    R("imp_use", "IMP into nucleotide synthesis", "Transfer", {"IMP": 1}, {}, kcat=1.5, km=0.4),
    R("urate_out", "Uric acid excretion", "Excretion", {"Urate": 1}, {}, kcat=1.0, km=0.5,
      drugs=[{"name": "Probenecid", "effect": "activates", "note": "increases urinary excretion of uric acid"}])],
   "Purine salvage and degradation; diseases and drugs as in clinical pharmacology texts")


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


def conservation_laws(N, species, clamped=(), max_support=None, max_coef=3):
    """Small integer conserved pools: y >= 0 with y @ N = 0, over species that are not held fixed."""
    from itertools import combinations, product
    free = [i for i, sp in enumerate(species) if sp not in clamped]
    if max_support is None:
        max_support = 4 if len(free) <= 30 else 3
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
                                                       "reversible": r["reversible"], "regulators": r["regulators"],
                                                       "deficiency": r.get("deficiency"), "drugs": r.get("drugs", [])}
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


COFACTORS = ("ATP", "NAD", "NADP", "CoA", "OAA", "GDP", "Pi", "Glc", "AcCoA", "G6P", "Pyr", "Orn", "UDPGlc",
             "PRPP", "Asp", "Ado", "Hx", "Phe", "Gal", "NH3", "CO2")


def initial_state(path, level=1.0):
    c = {s: 0.05 for s in path["species"]}
    for s in path["clamped"]:
        c[s] = level
    big = 1.5 if path.get("id") == "all" else 0.6
    for s in COFACTORS:
        if s in c:
            c[s] = max(c[s], big)
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


def backbone_sign_consistency(path):
    """Is the pathway monotone once the allosteric regulation is removed?"""
    bare = dict(path)
    bare["reactions"] = [dict(r, regulators=[]) for r in path["reactions"]]
    return sign_consistency(bare)


def acr_scan(path, factors=(0.7, 1.0, 1.4), steps=2500, tol=0.02):
    """Species whose steady level barely moves when every free pool is scaled: candidates for
    absolute concentration robustness."""
    sp = path["species"]
    finals = []
    for f in factors:
        c0 = initial_state(path)
        c0 = {k: (v * f if k not in path["clamped"] else v) for k, v in c0.items()}
        finals.append(np.array(simulate_pathway(path, steps=steps, c0=c0, record=False)["final"]))
    F = np.stack(finals)
    lo, hi = F.min(axis=0), F.max(axis=0)
    spread = (hi - lo) / np.maximum(hi, 1e-6)
    robust, sensitive = [], []
    for i, s in enumerate(sp):
        if s in path["clamped"] or hi[i] < 1e-3:
            continue
        (robust if spread[i] < tol else sensitive).append(
            {"species": s, "spread": round(float(spread[i]), 4), "level": round(float(F[1, i]), 4)})
    sensitive.sort(key=lambda d: -d["spread"])
    return {"robust": robust, "sensitive": sensitive[:6], "factors": list(factors)}


def steady_states(path, tries=8, steps=2500, seed=2, merge=0.05):
    """Distinct steady states from different starting points *within the same conserved pools*, so that
    differences mean multistability rather than different amounts of material."""
    rng = np.random.default_rng(seed)
    N = stoichiometry(path)
    pools = conservation_laws(N, path["species"], path["clamped"])
    base = initial_state(path)
    ends = []
    for k in range(tries):
        c0 = {s: (v if s in path["clamped"] else float(np.clip(v * rng.uniform(0.2, 2.5), 1e-6, None)))
              for s, v in base.items()}
        for law in pools:
            want = sum(c * base[sp] for sp, c in law)
            have = sum(c * c0[sp] for sp, c in law)
            if have > 1e-9:
                for sp, _ in law:
                    c0[sp] *= want / have
        out = simulate_pathway(path, steps=steps, c0=c0, record=False)
        ends.append((np.array(out["final"]), out["residual"]))
    distinct = []
    unsettled = 0
    for e, res in ends:
        if res > 5e-3:
            unsettled += 1
            continue
        if not any(np.abs(e - d).max() < merge for d in distinct):
            distinct.append(e)
    return {"tries": tries, "distinct": len(distinct), "unsettled": unsettled}


def control_coefficients(path, target=None, steps=2500, delta=0.15):
    """Flux control coefficients: the fractional change in the pathway's output flux per fractional change
    in each enzyme. This is the measured drug-target ranking."""
    run = simulate_pathway(path, steps=steps, record=False)
    flux = run["flux"]
    if target is None:
        exits = [r["id"] for r in path["reactions"] if not r["products"]]
        target = max(exits, key=lambda k: flux[k]) if exits else max(flux, key=lambda k: flux[k])
    base = flux[target]
    rows = []
    for j, r in enumerate(path["reactions"]):
        out = []
        for f in (1 - delta, 1 + delta):
            scale = [f if k == j else 1.0 for k in range(len(path["reactions"]))]
            out.append(simulate_pathway(path, steps=steps, enzyme_scale=scale, record=False)["flux"][target])
        c = ((out[1] - out[0]) / max(abs(base), 1e-6)) / (2 * delta)
        rows.append({"reaction": r["id"], "enzyme": r["enzyme"], "control": round(float(c), 3)})
    rows.sort(key=lambda d: -abs(d["control"]))
    name = next(r["enzyme"] for r in path["reactions"] if r["id"] == target)
    return {"target": target, "target_name": name, "base": round(float(base), 4), "coefficients": rows,
            "sum": round(float(sum(x["control"] for x in rows)), 3)}


def deeper(path):
    bb = backbone_sign_consistency(path)
    acr = acr_scan(path)
    ss = steady_states(path)
    cc = control_coefficients(path)
    notes = []
    full = sign_consistency(path)
    if bb["consistent"] and not full["consistent"]:
        notes.append("Strip the allosteric regulation and the pathway becomes sign-consistent, so its backbone is "
                     "monotone and the regulation is what breaks it. That is the weaker property real pathways share: "
                     "a monotone skeleton plus a handful of feedback loops. The bench's feedback arm enforces exactly this.")
    elif bb["consistent"]:
        notes.append("The pathway is sign-consistent with or without its regulation, so it is monotone as it stands.")
    else:
        notes.append("Even with the regulation removed, the reaction wiring itself is not sign-consistent, so this "
                     "pathway is not monotone at any level.")
    if acr["robust"]:
        notes.append("Species that held their level when every free pool was scaled by 0.7 and 1.4: "
                     + ", ".join(d["species"] for d in acr["robust"][:8])
                     + ". These are candidates for absolute concentration robustness, where the steady level does not "
                       "depend on how much material there is. This is a numerical test, not a proof.")
    else:
        notes.append("No species held its level when the pools were scaled, so nothing here looks concentration-robust.")
    if ss["distinct"] > 1:
        notes.append(f"From {ss['tries']} different starting points the pathway settled into {ss['distinct']} distinct "
                     "steady states, so it is multistable: where it starts decides where it ends.")
    elif ss["unsettled"]:
        notes.append(f"{ss['unsettled']} of {ss['tries']} starting points had not settled by the end of the run.")
    else:
        notes.append("Every starting point settled to the same steady state, so no multistability showed up here.")
    if ss["distinct"] > 1:
        notes.append("Because the pathway is multistable, treat the control coefficients below with care: a small "
                     "change in an enzyme can tip it into a different steady state, which shows up as a huge coefficient.")
    top = cc["coefficients"][0] if cc["coefficients"] else None
    if top:
        notes.append(f"Control over the pathway's output flux ({cc['target_name']}) sits mostly with {top['enzyme']} "
                     f"(flux control coefficient {top['control']}; the coefficients sum to {cc['sum']}, and for a "
                     "pathway at steady state the sum should be close to 1). The enzyme with most control is the one "
                     "worth inhibiting, and the ones near zero are the ones a drug would barely move. These are "
                     "measured here, and depend on the illustrative rate constants.")
    return {"backbone": bb, "acr": acr, "steady_states": ss, "control": cc, "notes": notes}


def graph(path):
    """Bipartite graph of species and reactions, ready for drawing or for loading into a graph database."""
    nodes = [{"id": f"s:{s}", "kind": "species", "label": s, "clamped": s in path["clamped"]} for s in path["species"]]
    nodes += [{"id": f"r:{r['id']}", "kind": "reaction", "label": r["enzyme"], "name": r["name"],
               "deficiency": r.get("deficiency"), "drugs": r.get("drugs", []), "reversible": r["reversible"]}
              for r in path["reactions"]]
    edges = []
    for r in path["reactions"]:
        for sp, k in r["substrates"].items():
            edges.append({"from": f"s:{sp}", "to": f"r:{r['id']}", "type": "SUBSTRATE_OF", "stoich": k})
        for sp, k in r["products"].items():
            edges.append({"from": f"r:{r['id']}", "to": f"s:{sp}", "type": "PRODUCT_OF", "stoich": k})
        for g in r["regulators"]:
            edges.append({"from": f"s:{g['species']}", "to": f"r:{r['id']}", "type": "REGULATES", "effect": g["effect"]})
    return {"id": path["id"], "name": path["name"], "nodes": nodes, "edges": edges}


def _cy(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def cypher(path):
    """Cypher statements that build this pathway in a graph database such as Neo4j."""
    L = [f"// {path['name']} — generated by Fate Memory Lab",
         f"MERGE (p:Pathway {{id: {_cy(path['id'])}}}) SET p.name = {_cy(path['name'])}, "
         f"p.description = {_cy(path['description'])}, p.reference = {_cy(path['reference'])};"]
    for sp in path["species"]:
        L.append(f"MERGE (m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}) "
                 f"SET m.held_fixed = {_cy(sp in path['clamped'])} "
                 f"WITH m MATCH (p:Pathway {{id: {_cy(path['id'])}}}) MERGE (m)-[:IN_PATHWAY]->(p);")
        L.append(f"MERGE (c:Compound {{name: {_cy(sp)}}}) WITH c "
                 f"MATCH (m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}) MERGE (m)-[:IS]->(c);")
    for r in path["reactions"]:
        L.append(f"MERGE (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                 f"SET x.name = {_cy(r['name'])}, x.enzyme = {_cy(r['enzyme'])}, x.kcat = {_cy(r['kcat'])}, "
                 f"x.km = {_cy(r['km'])}, x.reversible = {_cy(r['reversible'])}, x.deficiency = {_cy(r.get('deficiency'))} "
                 f"WITH x MATCH (p:Pathway {{id: {_cy(path['id'])}}}) MERGE (x)-[:IN_PATHWAY]->(p);")
        L.append(f"MERGE (e:Enzyme {{name: {_cy(r['enzyme'])}}}) WITH e "
                 f"MATCH (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) MERGE (e)-[:CATALYSES]->(x);")
        for sp, k in r["substrates"].items():
            L.append(f"MATCH (m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}), "
                     f"(x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (m)-[:SUBSTRATE_OF {{stoichiometry: {_cy(k)}}}]->(x);")
        for sp, k in r["products"].items():
            L.append(f"MATCH (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}), "
                     f"(m:Metabolite {{name: {_cy(sp)}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (x)-[:PRODUCES {{stoichiometry: {_cy(k)}}}]->(m);")
        for g in r["regulators"]:
            L.append(f"MATCH (m:Metabolite {{name: {_cy(g['species'])}, pathway: {_cy(path['id'])}}}), "
                     f"(x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (m)-[:REGULATES {{effect: {_cy(g['effect'])}, constant: {_cy(g['k'])}}}]->(x);")
        for d in r.get("drugs", []):
            L.append(f"MERGE (d:Drug {{name: {_cy(d['name'])}}}) WITH d "
                     f"MATCH (x:Reaction {{id: {_cy(r['id'])}, pathway: {_cy(path['id'])}}}) "
                     f"MERGE (d)-[:ACTS_ON {{effect: {_cy(d['effect'])}, note: {_cy(d.get('note'))}}}]->(x);")
        if r.get("deficiency"):
            L.append(f"MERGE (c:Condition {{name: {_cy(r['deficiency'].split(':')[0])}}}) "
                     f"SET c.description = {_cy(r['deficiency'])} WITH c "
                     f"MATCH (e:Enzyme {{name: {_cy(r['enzyme'])}}}) MERGE (c)-[:CAUSED_BY_LOSS_OF]->(e);")
    return "\n".join(L)


def what_if(path, node_id, levels=(0.5, 0.2, 0.0), steps=3000, boost=3.0, tol=0.05, attribute=True):
    """What happens if one thing changes: an enzyme loses activity, or a metabolite is supplied in excess."""
    sp = path["species"]
    base = simulate_pathway(path, steps=steps, record=False)
    kind, key = node_id.split(":", 1)
    out = {"node": node_id, "kind": kind, "label": key, "baseline_flux": base["flux"], "cases": []}
    if kind == "r":
        r = next((x for x in path["reactions"] if x["id"] == key), None)
        out["in_pathways"] = path.get("origin_reactions", {}).get(key, [path["id"]])
        if r is None:
            return None
        out["label"] = r["enzyme"]
        out["name"] = r["name"]
        out["deficiency"] = r.get("deficiency")
        out["drugs"] = r.get("drugs", [])
        for lv in levels:
            scale = [lv if x["id"] == key else 1.0 for x in path["reactions"]]
            run = simulate_pathway(path, steps=steps, enzyme_scale=scale, record=False)
            case = _compare(sp, base, run, f"{int(lv * 100)}% activity", tol)
            if attribute and path.get("origin_species"):
                case["pathways"] = affected_pathways(path, case["changes"])
            out["cases"].append(case)
    else:
        if key not in sp:
            return None
        out["name"] = key
        for f in (boost, 1 / boost):
            c0 = initial_state(path)
            c0[key] = c0[key] * f
            held = dict(path)
            held["clamped"] = sorted(set(path["clamped"]) | {key})
            run = simulate_pathway(held, steps=steps, c0=c0, record=False)
            case = _compare(sp, base, run, f"{key} held {'high' if f > 1 else 'low'} ({round(f, 2)}x)", tol)
            if attribute and path.get("origin_species"):
                case["pathways"] = affected_pathways(path, case["changes"])
            out["cases"].append(case)
    return out


def _compare(sp, base, run, label, tol):
    changes = []
    for i, s in enumerate(sp):
        b, k = base["final"][i], run["final"][i]
        if abs(k - b) > tol and max(b, k) > 1e-3:
            changes.append({"species": s, "before": round(float(b), 3), "after": round(float(k), 3),
                            "fold": round(float((k + 1e-6) / (b + 1e-6)), 2)})
    changes.sort(key=lambda d: -abs(np.log(max(d["fold"], 1e-6))))
    fl = []
    for rid, b in base["flux"].items():
        k = run["flux"][rid]
        if abs(k - b) > 0.02:
            fl.append({"reaction": rid, "before": round(float(b), 3), "after": round(float(k), 3)})
    fl.sort(key=lambda d: -abs(d["after"] - d["before"]))
    return {"label": label, "changes": changes[:8], "fluxes": fl[:6], "settled": run["steady"]}


def merged(pids=None):
    """All pathways joined into one network through the metabolites they share."""
    pids = list(pids or PATHWAYS)
    species, clamped_votes, holders, producers = [], {}, {}, set()
    reactions, seen = [], {}
    origin_r, origin_s = {}, {}
    for pid in pids:
        p = PATHWAYS[pid]
        for sp in p["species"]:
            if sp not in species:
                species.append(sp)
            holders.setdefault(sp, []).append(pid)
            clamped_votes.setdefault(sp, []).append(sp in p["clamped"])
            origin_s.setdefault(sp, []).append(pid)
        for r in p["reactions"]:
            key = (tuple(sorted(r["substrates"].items())), tuple(sorted(r["products"].items())), r["enzyme"])
            if key in seen:
                origin_r[seen[key]].append(pid)
                continue
            rid = f"{pid}_{r['id']}"
            seen[key] = rid
            origin_r[rid] = [pid]
            reactions.append(dict(r, id=rid))
            producers.update(r["products"])
    environment = {"Glc", "Pi", "CO2", "H", "AMP", "GDP", "Asp", "PRPP", "Ado", "O2"}
    clamped = [sp for sp in species if (all(clamped_votes[sp]) and sp not in producers) or sp in environment]
    return {"id": "all", "name": "Whole metabolism (all pathways joined)",
            "description": "Every pathway in the app merged into one network through the metabolites they share, so a "
                           "change in one pathway can be followed into the others. Reactions that appear in more than "
                           "one pathway are kept once.",
            "species": species, "clamped": clamped, "reactions": reactions,
            "reference": "Union of the individual pathways listed in this app",
            "origin_reactions": origin_r, "origin_species": origin_s, "members": pids}


def affected_pathways(path, changes):
    """Which source pathways contain the metabolites that moved."""
    org = path.get("origin_species", {})
    out = {}
    for c in changes:
        for pid in org.get(c["species"], []):
            out.setdefault(pid, []).append(c["species"])
    return {k: sorted(set(v)) for k, v in out.items()}
