# OOSKiller

A discrete-event **simulator** and supporting tooling for an **OOS automated car
park** — a facility where carriers (lifts and shuttles) move pallet-borne cars
between LIFO shelves and customer rooms. The system and its dynamics are specified
in [`docs/PROBLEM.md`](docs/PROBLEM.md).

This repository provides the simulator, an environment / observation / reward
framework, an interactive visualizer, and neutral building blocks for
learning-based control. It deliberately does **not** fix an objective, a reward, or
a control approach — those are open, to be defined separately.

## The system

An OOS facility stores and retrieves cars (sedans and SUVs), each riding on a
fungible pallet, across a network of independently-moving carriers that hand off to
one another and to LIFO shelves; cars enter and leave only at customer rooms.
Layouts range from a single carrier with one room up to many rooms with multiple
serving and non-serving carriers. The full description — entities, state, the
`GOTO` / `TAKE` / `GIVE` / `WAIT` primitives, automatic handoffs, customer
load/unload, and concurrency — is in [`docs/PROBLEM.md`](docs/PROBLEM.md).

## Code structure

The runtime is layered into single-responsibility parts. A driver (the visualizer,
or any training loop you write) owns an **Agent**, which pairs an injected policy
with an **Environment**. The Environment runs episode control and the per-carrier
decision loop, delegating the world to a **SimEngine** and scoring to an injected
**RewardSystem**.

| Part | Module | Responsibility |
|------|--------|----------------|
| **SimEngine** | `oos/sim/facility.py` | The discrete-event world: topology, state, scheduler, task queue, dynamics, `advance_until`. No RL, no torch. |
| **Environment** | `oos/env/env.py` | Episode control, typed-graph observation + action encoding/masking, the per-carrier decision loop, and an injected RewardSystem. `reset`/`step` for training; `advance_until` / `submit_action` / `needs_decision` / `from_name` for embedding and the viz. |
| **RewardSystem** | `oos/env/reward_system.py` | A pluggable suite of reward terms, each a pure function of `(s, a, s')`, injected into an Environment. |
| **Agent** | `oos/agent/agent.py` | One class with the policy injected; drives an Environment step by step. UI-free and embeddable. |

A separation contract is enforced: **`oos.sim` and `oos.env` never import from
`oos.learn`** — torch is isolated to `oos.learn`.

## Repository layout

```
oos/
├── sim/         discrete-event SimEngine (state, scheduler, queue, dynamics, motion) — no RL, no torch
├── env/         Environment + typed-graph observation, action enumeration/masking, pluggable reward suite
├── agent/       embeddable, UI-free Agent runtime (policy + Environment, step by step)
├── learn/       learning building blocks (torch): a graph policy/value network, a PPO update, rollout
│                collection, batching, checkpoint/normalize, and a checkpoint-loading policy adapter
├── viz/         interactive DearPyGui visualizer / driver
├── facilities/  hand-authored facility registry (name -> factory)
├── dsl/         facility-definition builder DSL (authors + validates topologies)
└── config/      experiment-config dataclasses (durations, task stream, episode budgets)

tests/           pytest suite (47 tests): smoke, kinematics, network, reward-system, reward-potential,
                 primitives, shuffle, viz
docs/            PROBLEM.md — the system-and-dynamics specification
```

The `oos/learn/` modules are provided as a starting point only; the model design,
the reward, and the training procedure are **not** prescribed here.

## Getting started

Uses **uv**; requires **Python ≥ 3.12**.

```bash
uv sync
```

### Tests

```bash
SDL_VIDEODRIVER=dummy uv run python -m pytest tests/ -q
```

### Visualizer

```bash
python -m oos.viz [facility] [--runs DIR]
```

Opens a DearPyGui app: a canvas renders the live facility (carrier tracks, shelves,
room docks, the customer queue) beside a sidebar to play / pause / step, randomize
state, queue customer interactions by hand, click pallets to request a car, and
load a policy checkpoint discovered under `--runs` (default `runs/`). With no
positional facility it reopens the last-used one (or `tiny_medipol`). torch is
imported lazily, so it runs with just the random policy when torch is absent.

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
