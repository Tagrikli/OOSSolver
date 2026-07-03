# OOSKiller

A discrete-event **simulator** and a **deterministic plan solver** for an **OOS
automated car park** — a facility where carriers (lifts and shuttles) move
pallet-borne cars between LIFO shelves and customer rooms. The system and its
dynamics are specified in [`docs/PROBLEM.md`](docs/PROBLEM.md).

**Control is solved by the V3 plan solver** (no learning anywhere in the
system): `oos/plan/planner.py` + `oos/plan/solver.py` + `oos/plan/runtime.py`,
specified by [`docs/SOLUTION_V3.md`](docs/SOLUTION_V3.md) with the behavior
contract in [`docs/AGENT_BEHAVIOR.md`](docs/AGENT_BEHAVIOR.md). Its acceptance
battery runs with `python -m oos.plan.battery --gate all`. An earlier RL
training pipeline was removed once the solver passed the full battery; it
survives in git history and in the docs under `docs/`.

## The system

An OOS facility stores and retrieves cars (sedans and SUVs), each riding on a
fungible pallet, across a network of independently-moving carriers that hand off to
one another and to LIFO shelves; cars enter and leave only at customer rooms.
Layouts range from a single carrier with one room up to many rooms with multiple
serving and non-serving carriers. The full description — entities, state, the
`GOTO` / `TAKE` / `GIVE` / `WAIT` primitives, automatic handoffs, customer
load/unload, and concurrency — is in [`docs/PROBLEM.md`](docs/PROBLEM.md).

## Code structure

| Part | Module | Responsibility |
|------|--------|----------------|
| **SimEngine** | `oos/sim/facility.py` | The discrete-event world: topology, state, scheduler, task queue, dynamics, `advance_until`. |
| **Plan solver** | `oos/plan/` | The control brain: `oracle.py` (solvability/admission oracle), `planner.py` (retrieval planning), `solver.py` (plan + rung dispatch), `runtime.py` (headless pump loop), `battery.py` (acceptance battery). |
| **Move layer** | `oos/env/moves.py` | Pallet-move primitives (`Move`, `MoveExecutor`): compiles a move into carrier scripts and executes them against the live engine. |
| **Environment** | `oos/env/env.py` | The per-carrier decision loop over one SimEngine: action enumeration/decoding (`oos/env/action.py`), `advance_until` / `submit_action` / `needs_decision`. Used by the viz to drive the sim frame by frame. |
| **Viz** | `oos/viz/` | Interactive DearPyGui visualizer: `Session` (headless logic) + `SolverBridge` (solver ↔ decision loop) + `app` (rendering glue). |

## Repository layout

```
oos/
├── sim/         discrete-event SimEngine (state, scheduler, queue, dynamics, motion)
├── plan/        the deterministic V3 plan solver: oracle, planner, solver, runtime, battery
├── env/         decision-loop Environment + action enumeration + the pallet-move executor
├── viz/         interactive DearPyGui visualizer / driver (solver-driven)
├── facilities/  hand-authored facility registry (name -> factory)
├── dsl/         facility-definition builder DSL (authors + validates topologies)
└── config/      experiment-config dataclasses (durations, task stream)

tests/           pytest suite: smoke, kinematics, primitives, shuffle, plan solver, viz
docs/            PROBLEM.md — the system spec; SOLUTION_V3.md — the solver spec;
                 AGENT_BEHAVIOR.md — the behavior contract; earlier docs are history
```

## Getting started

Uses **uv**; requires **Python ≥ 3.12**.

```bash
uv sync
```

### Tests

```bash
uv run python -m pytest tests/ -q
```

### Acceptance battery

The solver's full acceptance battery (retrieval gates, continuous operation,
full-state conformance):

```bash
uv run python -m oos.plan.battery --gate all
```

### Visualizer

```bash
uv run python -m oos.viz [facility]
```

Opens a DearPyGui app: a canvas renders the live facility (carrier tracks, shelves,
room docks, the customer queue) beside a sidebar to play / pause / step, randomize
state, tune a set-point auto-world, queue customer interactions by hand, and click
pallets to request a car. The V3 plan solver drives the carriers. With no
positional facility it reopens the last-used one (or `tiny_medipol`).

## Facilities

Ten facilities are registered (look up via `get_facility(name)`):

`mini` · `tiny` · `tiny_tall` · `tiny_wide` · `tiny_medipol` · `stacker` ·
`stacker_deep` · `stacker_wide` · `dibaji` · `campus`

New facilities are authored with the `oos.dsl` builder, which validates and
compiles a topology to a frozen `(Topology, SeedingConfig)`:

```python
from oos.dsl import Carrier, Facility, Handoff, Room, Shelf

def make_facility():
    fac = Facility("example")
    fac.register_carriers(...)   # carriers, their tracks, shelves, rooms
    fac.pair(...)                # declare handoff poses between carriers
    fac.seed_pool()              # seed initial empty pallets
    return fac.build()           # -> (Topology, SeedingConfig)
```
