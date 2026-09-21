# Fate Memory Lab

A small web bench asking one question: does a recurrent circuit built like a gene-regulatory network make a better
memory layer?

Cells hold their identity with circuits such as the toggle switch. When a circuit's interaction graph has no negative
cycles it is *monotone*: it cannot sustain attracting oscillations, and almost every trajectory settles to an
equilibrium (Hirsch; Angeli and Sontag), while many equilibria can coexist. That is memory with a settling guarantee.
The bench tests whether enforcing that structure buys anything in a trained network.

## Arms

All arms have the same parameter count and start from identical weights.

| Arm | Recurrent weights | What the theory says |
|---|---|---|
| Sign-consistent | `W = S * softplus(V)`, `S_ij = s_i s_j` | No negative cycles, generic convergence, many attractors allowed |
| One cycle flipped | Same, one off-diagonal sign flipped | Breaks exactly one condition |
| Contraction | `W = 3.6 V / ‖V‖_F` | Unique equilibrium, so it cannot store state |
| Unconstrained | `W = V` | No guarantee |

## Tasks

- **flipflop**: three channels; each output holds the sign of the last pulse on its channel.
- **xor**: two latched channels; the output is their product, so stored memories must interact.
- **parity**: identical pulses on one channel; the output flips on every pulse.

## Measurements

Accuracy at the training length and at a longer test length; the plate view (96 random starting states per trained
circuit, colored by the attractor each one settles into); runs that settled; attractor count; median steps to settle;
negative edges relative to the sign pattern; and memory under sign-preserving multiplicative drift of the weights.
Every task, arm and seed trains in its own process.

## Install on macOS

Requires Python 3.10 or newer.

```bash
git clone https://github.com/merolaagi/fate-memory-lab.git ~/Sites/fate-memory-lab
cd ~/Sites/fate-memory-lab
bash install.sh
./service.sh install
open http://127.0.0.1:47431
```

`service.sh install|uninstall|restart|status|logs` manages a launchd agent that starts at login and restarts on crash.
`./run.sh` runs in the foreground instead. `./update.sh` installs the newest `fate-memory-lab-v*.zip` from
`~/Downloads`, keeping saved runs, the virtual environment and logs.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `FML_PORT` | `47431` | Port |
| `FML_HOST` | `127.0.0.1` | Bind address |
| `FML_WORKERS` | CPU cores minus 2 | Parallel training processes |
| `FML_DATA` | `data/runs` | Where runs are saved |

## API

`GET /api/health`, `GET /api/meta`, `GET /api/runs`, `POST /api/runs`, `GET /api/runs/{id}`,
`POST /api/runs/{id}/cancel`, `GET /api/runs/{id}/export.csv`, `GET /api/runs/{id}/export.json`,
`DELETE /api/runs/{id}`.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```

## License

MIT
