# Changelog

## 1.5.1
- The Neo4j driver now installs with the app instead of only with the developer tools
- Push all pathways button, loading every pathway as its own graph for cross-pathway queries
- Two new built-in queries: conditions linked through a shared metabolite, and drugs one shared metabolite away from a condition
- Clearer message when a query returns no rows
- README: Neo4j setup corrected, with the password quoted and set before the first start, and a reliable reset

## 1.5.0
- Settings page: enter the Anthropic API key and model, and the Neo4j address, user, password and database, inside the app
- Stored in data/settings.json with owner-only permissions, excluded from git, kept across updates; keys never returned to the browser in full
- Test buttons confirm the Anthropic key and the Neo4j connection
- Settings can only be changed from the Mac itself or through Cloudflare Access; the open internet sees them read-only
- Values saved in the app take precedence over environment variables, which fixes the background service not passing them through

## 1.4.0
- Research tab: search Europe PMC (PubMed, PMC, bioRxiv, medRxiv), read a paper's math, build a model from it, and test whether it works
- Reading by Claude when ANTHROPIC_API_KEY is set (structured recipe, stated parameters with source sentences, caveats, validated before use), or by a transparent keyword reader otherwise
- Seven model families for drug and vaccine work: viral dynamics, vaccine antibody response, PK/PD, enzyme inhibition, gene switch, drug-resistant tumour, epidemic with vaccination
- Feasibility test for every built model, including how many rate constants the paper actually supplied
- Research models join Pathways (network, what-if, therapy search, Cypher) and open on the Models canvas
- Drug-target search: block and boost every step to move a chosen quantity, with side effects
- Kinetics engine gains mass-action and Hill rate laws, cooperative regulators and explicit initial states

## 1.3.0
- Strategy search now tries combinations of the best single moves and reports whether each pair helps or interferes
- On the joined network, reports what the best drug-like move does to other pathways: the model's side effects
- Large networks restrict the search to enzymes within two steps of the failure, keeping it to about a minute
- Test the method: runs the strategy search on all 66 known deficiencies and compares the model's ranking with an exact random-ranking baseline and a z-score
- Scoring re-weighted toward clearing the toxin (85%) over restoring output (15%), which took the benchmark from no better than chance (22 vs 21.3) to better than chance (33 vs 27.1, z = 2.26)

## 1.2.0
- Eighteen more pathways, 26 in total: gluconeogenesis, glycogen, fructose, the polyol pathway, pyruvate, beta-oxidation, ketone bodies, propionate, branched-chain amino acids, catecholamines, methionine and homocysteine, folate, pyrimidines, haem, bilirubin, glutathione, sphingolipids, cholesterol and alcohol
- Almost every enzyme step now carries its inherited deficiency and the drugs acting on it
- What would help: a therapy search that drops an enzyme to 5% activity, tries the standard treatment moves (enzyme restoration, substrate reduction, upstream or downstream block, scavenger, product supplementation, dietary restriction, alternative route) and ranks them, flagging matches with real drugs
- tools/balance.py automatically balances rate constants so every pathway and the joined network reach a steady state; results are stored in app/rate_tuning.json

## 1.1.0
- Whole metabolism: all eight pathways joined into one network through shared metabolites, with structure, simulation, network view and what-if all working on the joined system
- A what-if on the joined network reports which source pathways the change reaches
- Respiration now makes ATP and a PEPCK exit drains oxaloacetate, so the joined network reaches a steady state
- Citrulline renamed so it no longer collides with citrate when the urea and citric acid cycles are joined
- Read-only Cypher query panel with built-in cross-pathway queries, and Compound nodes so queries can cross pathway boundaries
- Conserved-pool search adapts its depth to network size, keeping the joined analysis fast

## 1.0.0
- Four more pathways with clinical annotations: the urea cycle, phenylalanine and tyrosine catabolism, galactose metabolism, and purine salvage and degradation
- Every enzyme step carries its known inherited deficiency and the drugs that act on it
- Network view: metabolites and enzyme steps as a draggable graph, with regulation, deficiencies and drug targets marked
- What happens if this changes: click an enzyme to re-simulate it at 50%, 20% and 0% activity, or a metabolite to hold it high or low, and see which metabolites move and by how much
- Cypher export for every pathway, building Pathway, Metabolite, Reaction, Enzyme, Drug and Condition nodes with their relationships
- Optional direct push into Neo4j with the neo4j driver and NEO4J_URI / NEO4J_PASSWORD

## 0.9.0
- Two new arms drawn from what real pathways actually satisfy: monotone backbone (sign-consistent wiring plus a few negative feedback edges) and conserved pools (unit groups whose totals never change)
- Both at matched parameter count, with the number of feedback edges and the pool size as settings
- Deeper pathway analysis: backbone monotonicity with regulation stripped out, a numerical scan for concentration-robust species, multistability tested within the same conserved pools, and flux control coefficients per enzyme
- Control coefficients are flagged as unreliable when the pathway is multistable
- Each pathway now carries real throughput, so fluxes and control coefficients are meaningful; the coefficients sum to about 1 as the theory requires

## 0.8.0
- Three tabs: Runs, Models and Pathways
- Models: an editable workspace. Build a model layer by layer on a canvas, starting blank or from a pathway or a trained circuit; edit, insert and delete layers; run it on inputs you choose; read and download the generated NumPy code
- Layers include transporters, gated channels, linear maps, the cell circuit with a selectable wiring rule, readout, decision, and a metabolic pathway as a layer
- Pathways: glycolysis, lactate fermentation, the oxidative pentose phosphate pathway and the citric acid cycle, with textbook stoichiometry and illustrative rate constants
- Exact structural analysis per pathway: conserved pools, complexes, linkage classes, rank, deficiency with what the theorems say, and sign-consistency with the specific regulations that break it
- Pathway simulation with one-click enzyme knockdown to 20%, the in-model equivalent of an inhibitor

## 0.7.0
- Reachability map for every built model: stable states, which states each input kick can reach, traps no kick can leave, and states found only by kicking
- Escape barriers: the noise level needed to knock the cell out of each state
- Theory check for sign-consistent circuits behind gated channels: counts kicks that move the cell against its channel's direction in the circuit's order (the theorem says none can)
- When every input pushes one way, flags dead-end states that no schedule of that drug can ever reverse

## 0.6.0
- New task: resistance. Sustained drug exposure makes the cell tolerant; a 25-step drug holiday resensitizes it (reversible drug tolerance)
- Therapy search for resistance and commit models: induces the unwanted state, tries 90 schedules of dose, time on, time off and an inhibitor, and compares every result with the true rule
- Reports the gentlest working schedule, whether a drug holiday suffices, whether the inhibitor helps or hurts, and counts false and missed cures
- The run explanation covers resistance and points to the therapy search

## 0.5.0
- What this run shows: a plain-language explanation of every finished run, generated locally by fixed rules, with suggested next steps
- Build model: turns any trained arm and seed into a standalone cell model
- Layer canvas: draggable graph of the model's mathematical layers, with equations, parameter counts and weight heatmaps
- Try it: runs the model on fresh inputs against the right answer
- Probes against the biology: dose-response curves for antagonist, commitment map for commit
- Export as NumPy-only Python, a PyTorch module, or a JSON layer graph with weights; file names are unique per task, arm, seed and run
- Runs now save each circuit's trained weights

## 0.4.1
- Fix: Start run silently did nothing in the browser. The learning-rate field's step setting made the default 0.01 count as invalid, so the browser blocked the form without showing why (the field is inside the collapsed More settings). Fields now accept any decimal, and the server does the validation

## 0.4.0
- Membrane input stage for every arm: none, saturating Michaelis-Menten transporters, or transporters with gates the circuit controls itself
- Sign-consistent arm keeps its guarantee through gated channels, by construction of the gate and input signs
- Two drug tasks: commit (irreversible commitment after sustained exposure, not brief spikes) and antagonist (competitive receptor binding)
- Inhibitor stress test: a drug partly blocks a random fraction of units
- Membrane uptake chart showing each arm's learned intake curve
- CSV exports include membrane and inhibitor results

## 0.3.0
- Two new cell-inspired tasks: dose (random signal strength per sequence) and background (slowly wandering baseline)
- Two new stress tests: expression noise on the state, and uneven partitioning at cell division every 40 steps
- Waddington landscape view: settling trajectories projected onto their two main directions, colored by attractor
- Noise and division levels are adjustable under More settings and included in CSV exports
- README explains the new tasks, landscape and stress tests

## 0.2.0
- Multiple seeds per arm, with mean and standard deviation in every table column
- Steps-to-settle metric: median time for a random starting state to come to rest
- Stop a running or queued run; finished parts are kept
- Download any run as CSV or JSON
- Health endpoint at /api/health
- Default port 47431
- update.sh installs a newer package from ~/Downloads while keeping runs, .venv and logs
- Tests and GitHub Actions workflow; MIT license

## 0.1.0
- First web bench: flipflop, xor and parity tasks; four arms; plate view of attractors; weight-drift stress test
