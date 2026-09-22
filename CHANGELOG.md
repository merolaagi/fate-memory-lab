# Changelog

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
