# Fate Memory Lab

**Can a neural network remember things the way a cell remembers what it is?**

A skin cell divides and makes more skin cells. A liver cell stays a liver cell for years. Cells hold that identity
with small gene-regulatory circuits, such as the toggle switch, in which genes switch each other on and off until
the system locks into one stable state. Mathematicians have known for decades why such circuits are so dependable:
when a circuit's wiring has **no negative cycles**, it is *monotone*. A monotone circuit cannot get stuck in an endless
oscillation. Almost every starting point settles into a resting state, and there can be many resting states to choose
from. That combination is exactly what a memory needs: many states to store, plus a guarantee that the circuit
settles into one of them.

Fate Memory Lab is a small web app that asks whether that biological design principle makes a better memory layer
for a trained recurrent network. It trains four versions of the same network side by side, gives them memory tasks,
and shows you what each one learned, where its internal states end up, and how well its memory survives when its
weights are disturbed.

It runs locally on your own machine, in your browser, with no GPU and no cloud account.

---

## What it compares

Every task trains four **arms**. They have exactly the same number of parameters and start from identical weights,
so any difference in the results comes from the one structural rule each arm follows.

| Arm | Rule it follows | What the theory predicts |
|---|---|---|
| **Sign-consistent** | Connection signs follow a fixed pattern with no negative cycles, like a gene-regulatory network | Always settles; can store many states |
| **One cycle flipped** | Identical, except one connection's sign is reversed, creating exactly one negative cycle | The settling guarantee no longer holds |
| **Contraction** | Weights are scaled down so the network always pulls toward a single point | Always settles, but only ever to one state, so it cannot remember |
| **Unconstrained** | No rule at all | No guarantee either way |

The *one cycle flipped* arm is the key control. It differs from the sign-consistent arm by a single sign, so if the
two behave differently, that one broken condition is the reason.

## The memory tasks

| Task | What the network sees | What it must output | Why it's interesting |
|---|---|---|---|
| **flipflop** | Occasional +1 or −1 pulses on three channels | The sign of the most recent pulse on each channel, held until the next one | Pure memory: three independent switches |
| **xor** | Pulses on two channels | The product of the two stored signs | The stored memories have to interact |
| **parity** | Identical pulses on one channel | An output that flips on every pulse | The same input must push the state up one time and down the next, which is the hardest case for a sign-consistent circuit |
| **dose** | Flip-flop pulses scaled by a random strength per sequence, from 0.3× to 3× | The same as flipflop | Cells respond to relative change rather than absolute amount. Can the circuit ignore the dose? |
| **background** | Flip-flop pulses on top of a slowly wandering background level | The same as flipflop | Cells adapt to steady backgrounds yet still react to sharp signals. Can the circuit tell a pulse from a drifting baseline? |
| **commit** | A drug arriving as brief strong spikes or longer, weaker exposures | Off until the accumulated exposure crosses a threshold, then on for good | Cells commit to dividing, differentiating or dying only after sustained signals, and never go back. Brief spikes alone must not trigger it |
| **antagonist** | An agonist and a competing antagonist, both drifting slowly | On while the receptor is more than half occupied by the agonist | Classic receptor pharmacology. It needs no memory, so it shows where even the contraction arm is competitive |

## The membrane

Real cells don't let signals straight in. Molecules cross the membrane through transporters that fill up at high
concentrations, and through channels the cell opens and closes itself. Each run picks one membrane, applied
identically to every arm so the comparison stays fair:

| Membrane | How signals get in |
|---|---|
| **None** | Straight into the circuit, as in earlier versions |
| **Transporter** | Each signal is split into a positive and a negative "molecule", and each crosses through a saturating transporter: intake = capacity × concentration / (affinity constant + concentration), the Michaelis–Menten law. The network learns every transporter's capacity and affinity |
| **Gated channel** | Transporters plus gates controlled by the circuit's own state, so the cell regulates its own intake. For the sign-consistent arm and its control, the signs of the gates and inputs are chosen so the loop through the membrane adds no negative cycle and the settling guarantee still holds |

Networks train on short sequences and are then tested on sequences several times longer, to check whether their
memory actually holds over time rather than just fitting the training length.

---

## Quick start

You need macOS (or Linux) with Python 3.10 or newer.

```bash
git clone https://github.com/merolaagi/fate-memory-lab.git ~/Sites/fate-memory-lab
cd ~/Sites/fate-memory-lab
bash install.sh
./service.sh install
open http://127.0.0.1:47431
```

`install.sh` creates a private Python environment in `.venv` and installs everything the app needs.
`service.sh install` runs the app in the background, starts it automatically when you log in, and restarts it
if it ever crashes. To run it in the terminal instead, use `./run.sh`.

---

## Using the app

### 1. Start a run

The panel on the left is the run form.

- **Tasks**: pick one or more. Each task trains its own set of arms.
- **Membrane**: how signals enter the circuit. See *The membrane* above.
- **Arms**: all four are selected by default. Keep them all for a fair comparison.
- **Training steps**: how long each network trains. 500 is a quick look; 3000 or more gives results worth reading,
  especially for parity.
- **Hidden units**: the size of each network. The default of 24 is plenty for these tasks.
- **Seeds per arm**: how many times to repeat each arm with a different random start. Use at least 3. A single seed
  can be lucky or unlucky, and seeds are how you tell a real difference from noise.

**More settings** holds the finer controls: learning rate, batch size, training and test sequence lengths, how often
pulses arrive, the starting strength of self-connections and cross-connections, the drift levels for the stress test,
and the base random seed.

Click **Start run**. Every task, arm and seed trains in its own process, so a machine with many cores finishes much
faster. The number of parallel workers is shown in the bottom-left corner.

### 2. Watch it train

While a run is active, the main panel shows a row of progress bars for each task, one colored bar per arm.
Results appear as each job finishes, so you can start reading them before the whole run is done.
**Stop run** cancels it and keeps whatever has already finished.

### 3. Read the results

Each task gets its own section with three parts.

**The results table.** With several seeds, each cell shows the average plus or minus the spread across seeds.

| Column | What it means |
|---|---|
| Accuracy, train length | How often the output has the right sign on sequences as long as the training ones |
| Accuracy, test length | The same on much longer sequences. This is the real test of memory. 50% is chance |
| Runs that settled | Of 96 random starting states, the share that came fully to rest |
| Attractors | How many distinct resting states the trained network has. More means more room to store things |
| Steps to settle | How long a random starting state typically takes to come to rest |
| Negative edges | How many connections break the sign pattern. Always 0 for the sign-consistent arm and 1 for its control |
| Params | Parameter count, identical across arms by design |

**The plates.** Each arm gets a picture laid out like a 96-well lab plate. The app drops 96 random starting states
into the trained network, lets each one run until it stops moving, and colors each well by the resting state it
ended in. Many colors mean many attractors, so a network with lots of room for memory. One solid color means
everything collapses to a single state, as the contraction arm always does. A dashed empty ring marks a starting
state that never settled; for the sign-consistent arm, the theory says you should essentially never see one.

**The Waddington landscape.** Biologists picture cell fate as a ball rolling down a hilly landscape into one of
several valleys. The landscape view draws the trained circuit the same way: it flattens the network's internal state
onto its two most important directions, draws 24 random starting states as faint lines as they settle, and marks
where all 96 end, colored to match the plate. Separate clusters are separate stored memories. The caption says how
much of the full picture the flat map captures; when it is low, clusters that look merged may be separate.

**The stress tests.** Each is borrowed from something real cells survive, and each chart shows test accuracy as the
disturbance grows. The leftmost point is the undisturbed circuit.

- *Weight drift*: every connection's strength is randomly scaled up or down, with its sign kept. In a cell, the
  amounts of the molecules in a circuit vary from cell to cell while the wiring stays the same.
- *Expression noise*: random jitter is added to the state at every step, like the random bursts in which genes are
  actually expressed.
- *Cell division*: every 40 steps the state is split unevenly, the way proteins are shared unequally between two
  daughter cells. A good memory should survive the split.
- *Inhibitor drug*: a drug partly blocks a random fraction of the circuit's units, cutting their activity to 30%,
  the way an inhibitor knocks down a protein. The level is the fraction of units blocked.

With a transporter or gated membrane, a *membrane uptake* chart also shows the learned intake curve for each arm:
how much signal gets in at each outside concentration.

*Training loss* shows how each arm learned, one line per seed.

### 4. Save and share results

Every run is saved automatically in `data/runs` and stays in the Runs list after restarts. Use **Download CSV** for
a spreadsheet of every task, arm and seed, or **Download JSON** for the complete raw record, including loss curves
and plate data.

---

## What we've seen so far

These are early, small-scale observations, not conclusions.

- The contraction arm never remembers anything. It sits at chance on every task, exactly as the theory predicts.
- On flipflop, the sign-consistent, one-cycle-flipped and unconstrained arms all learn the task well. The
  unconstrained arm settled just as reliably as the sign-consistent one, so on easy tasks the guarantee costs
  nothing but also shows no benefit.
- On xor, the unconstrained arm was slightly more accurate than the sign-consistent one in early tests.
- Parity is the open question. Short training runs haven't solved it for any arm.
- Results vary noticeably between seeds, which is why the app defaults to three.
- The dose, background, expression-noise and cell-division tests are new in 0.3.0; the membrane, the drug tasks and
  the inhibitor test are new in 0.4.0. None have full-length results yet.

---

## Keeping it updated

New versions ship as a zip named like `fate-memory-lab-v0.3.0-20261001-1415.zip`. Download it to `~/Downloads`, then:

```bash
cd ~/Sites/fate-memory-lab
./update.sh
```

`update.sh` installs the newest package from Downloads and restarts the app. Your saved runs, Python environment,
logs and git history are left untouched. If you cloned from GitHub, `git pull` followed by `bash install.sh` and
`./service.sh restart` works too.

## Managing the background service

```bash
./service.sh status
./service.sh logs
./service.sh restart
./service.sh uninstall
```

## Settings

| Variable | Default | What it does |
|---|---|---|
| `FML_PORT` | `47431` | Port the app listens on |
| `FML_HOST` | `127.0.0.1` | Address it binds to. The default is reachable only from this machine |
| `FML_WORKERS` | CPU cores minus 2 | How many training jobs run at once |
| `FML_DATA` | `data/runs` | Where runs are saved |

Set these before `./service.sh install` or `./run.sh`, for example `FML_PORT=50123 ./service.sh install`.
To reach the app from other devices, put it behind a tunnel such as Cloudflare Tunnel rather than opening the port.

## API

Everything the page does is available over HTTP.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Is the server up, and which version |
| GET | `/api/meta` | Default settings, arms and tasks |
| GET | `/api/runs` | List all runs |
| POST | `/api/runs` | Start a run; the body takes the same fields as the form |
| GET | `/api/runs/{id}` | Full results and live progress |
| POST | `/api/runs/{id}/cancel` | Stop a queued or running run |
| GET | `/api/runs/{id}/export.csv` | Results as CSV |
| GET | `/api/runs/{id}/export.json` | Complete raw run record |
| DELETE | `/api/runs/{id}` | Delete a run |

Example, starting a run from the terminal:

```bash
curl -X POST http://127.0.0.1:47431/api/runs -H 'Content-Type: application/json' -d '{"tasks":["parity"],"iters":3000,"repeats":3}'
```

## Project layout

```
app/engine.py        The model: arms, tasks, training, settling test, landscape, stress tests
app/server.py        FastAPI server: run queue, parallel workers, storage, exports
app/static/index.html  The whole browser interface in one file
install.sh           Creates .venv and installs dependencies
run.sh               Runs the server in the foreground
service.sh           Installs and manages the macOS background service
update.sh            Installs a newer package from ~/Downloads
tests/               Smoke tests, also run by GitHub Actions on every push
```

The model code is plain NumPy with the small `autograd` library for gradients, so it has no heavy dependencies and
is easy to read and modify.

## Running the tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```

## Background reading

- M. W. Hirsch, "Systems of differential equations which are competitive or cooperative," *SIAM Journal on
  Mathematical Analysis*, 1985, and the series that follows it. Why monotone systems settle.
- H. L. Smith, *Monotone Dynamical Systems*, American Mathematical Society, 1995. The standard book on the theory.
- D. Angeli and E. D. Sontag, "Monotone control systems," *IEEE Transactions on Automatic Control*, 2003.
  The "no negative cycles" condition for systems with inputs.
- T. S. Gardner, C. R. Cantor and J. J. Collins, "Construction of a genetic toggle switch in *Escherichia coli*,"
  *Nature*, 2000. The biological circuit that inspired the sign-consistent arm.
- D. Sussillo and O. Barak, "Opening the black box," *Neural Computation*, 2013. Source of the flip-flop task.

## License

MIT. See `LICENSE`.
