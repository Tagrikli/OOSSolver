# OOSKiller

A simulator, DSL, Gymnasium environment, and pygame visualizer for the
**OOS (Optimal Order Servicing) planner problem** — scheduling automated
storage / retrieval over a network of carriers, shelves, rooms, and
handoffs to minimize expected per-task customer wait time.

The end goal is a single trained policy that controls a multi-carrier
facility end-to-end. This repo is the substrate that policy will train
on and run inside.

## Status

What ships today:

- **Discrete-event sim** ([oos/sim/](oos/sim/)) — pure physics, no RL deps.
- **Embedded Python DSL** ([oos/dsl/](oos/dsl/)) for authoring facilities.
- **Gymnasium env** ([oos/env/](oos/env/)) with standard `step` and a
  split-step API (`submit_action` + `advance(time_limit)`) for smooth
  sub-event animation.
- **Pygame visualizer** ([oos/viz/](oos/viz/)) — cyberpunk themed live
  view of the facility with controls for stepping, speeding, and
  toggling between animated and step-through modes. Pluggable policy.
- **One hand-authored reference facility** ([oos/facilities/dev.py](oos/facilities/dev.py)):
  3 carriers, 24 shelves (cap 4 each), 2 rooms, 2 handoffs.
- **Smoke tests** ([tests/test_smoke.py](tests/test_smoke.py)) — DSL
  build, env contract, pallet conservation, admission control,
  determinism.

What's deferred (clearly defined, not built yet):

- **Heuristic baseline** (FIFO + nearest-shelf) for comparison.
- **PPO trainer** + GNN policy network ([docs/SOLUTION_1.md](docs/SOLUTION_1.md)).
- **Replay recorder / player** for deterministic re-runs.
- **SubprocVecEnv** for parallel rollouts.
- **Randomized topology generator** for cross-facility generalization.

## Quick start

```bash
# install (uses uv, https://docs.astral.sh/uv/)
uv sync

# run the visualizer with a random policy on the dev facility
uv run python -m oos.viz

# run the test suite
uv run pytest
```

In the visualizer:

| Key | Action |
|---|---|
| `space` | pause / resume |
| `→` | step one decision instant |
| `m` | toggle anim ↔ step mode |
| `+` / `-` | speed up / slow down |
| `r` | reset env |
| `q` / `esc` | quit |
| mouse wheel | scroll the pending-tasks panel |

## The problem

A facility has:

- **Carriers** — 1D movers that travel a track of integer-numbered slots, holding at most one pallet.
- **Shelves** — LIFO stacks (capacity 1–5) accessible by one carrier (or two, for transfer shelves).
- **Rooms** — customer interface points served by exactly one carrier.
- **Handoff poses** — synchronous swap points where two carriers exchange a pallet.
- **Transfer shelves** — single-slot buffers shared by two carriers, time-decoupled.
- **Pallets** — physical, conserved objects. Empty pallets become loaded when a customer drops an item; loaded pallets revert to empty when a customer collects one.

The exogenous task stream brings two kinds of work:

- **Store** — a customer with an item of a given size to deposit; the planner picks which room to direct them to.
- **Retrieve** — a request for a specific item by id. Each stored item schedules its own retrieval after a per-item dwell time, so retrieves track real inventory lifecycle.

The planner controls:

- **Room choice** for each incoming store (via the `STAGE_ROOM` action).
- **Per-carrier routing**: `TAKE`/`GIVE` to shelves, `MOVE_TO_PARTNER`/`HANDOFF` for cross-carrier transfers, `DELIVER_ITEM` to fulfill retrieves, `PARK` to idle.
- **Storage placement** for each newly-stored item.
- **Eviction destinations** for blockers during retrieves.
- **Background reshuffling** during idle time.

The objective is the long-run expected per-task wait time. Details and formal model: [docs/PROBLEM.md](docs/PROBLEM.md).

## Layout

```
oos/
├── sim/          # discrete-event physics; no RL or torch deps
├── dsl/          # facility-authoring DSL (Python embedded)
├── config/       # ExperimentConfig dataclasses
├── facilities/   # hand-authored facilities (dev.py is the reference)
├── env/          # gymnasium.Env wrapper
└── viz/          # pygame live viewer with pluggable policy

docs/
├── PROBLEM.md           # formal problem statement
├── SOLUTION_1.md        # GNN + PPO design (not built yet)
├── SOLUTION_1_DSL.md    # DSL design + API reference
└── SOLUTION_1_ENV.md    # sim/env design + API reference

tests/
└── test_smoke.py        # 5 passing tests
```

## Key concepts at a glance

- **Pallets are conserved.** Same physical pallet morphs between empty and loaded states during customer interactions — never created, never destroyed.
- **Carrier kind doesn't exist.** All carriers are 1D movers. Shuttle vs lift is a real-world detail the planner doesn't model.
- **One customer queue.** Store arrivals go into a single global queue. The agent picks the room per store via the `STAGE_ROOM` action.
- **Per-item dwell-time retrieval.** When an item is stored, the facility samples a delay from `Gamma(mean=300s, std=120s)` and schedules a retrieve for that specific item. Retrieves correlate with stores rather than being an independent Poisson process.
- **Big-item admission control.** When every slot on every big shelf holds a big item, pending and incoming big stores are silently dropped from the queue — natural capacity behavior, not a rejection.
- **Bundled movement.** `TAKE(A1)` means "move to A1's position, then take" — the policy never picks raw moves. The only standalone movement is `MOVE_TO_PARTNER` for handoff prep.
- **Pessimistic shelf reservation.** Two carriers can't both `TAKE` the last pallet from a shared transfer shelf — in-flight takes/gives are subtracted/added when checking preconditions.
- **Pallet conservation guard.** A regression test exercises 2000 random steps at high store rate and asserts the pallet count never changes from initial.
- **Determinism.** Same seed + same action trace = byte-identical observations and rewards.

## Action space

Seven action types, all with bundled movement where applicable:

| Action | Target | Effect |
|---|---|---|
| `TAKE` | shelf | move to shelf, pop top pallet onto carrier |
| `GIVE` | shelf | move to shelf, push current pallet onto top (size-compatible) |
| `HANDOFF` | partner carrier | instant transfer; both carriers must be co-located |
| `MOVE_TO_PARTNER` | partner carrier | position at the handoff pose with this partner |
| `STAGE_ROOM` | room | move to room with an empty pallet; locks in store cost |
| `DELIVER_ITEM` | room | move to room with a loaded pallet; locks in retrieve cost if a pending retrieve matches |
| `PARK` | — | brief no-op |

Realized as `Discrete(N_max)` with an action mask in the observation;
illegal indices have their logit set to `−∞` so the policy can never
pick them.

## Observation

A typed-graph dict with four node types and four edge types:

- **Carriers** (8 features each): position, load type, busy state, ETA, "is querying" flag.
- **Shelves** (26 features each: 6 base + 5 slots × 4 one-hot): size class, capacity, depth, transfer flag, and the full per-slot stack content (one-hot of `{empty, empty pallet, small item, big item}`, padded with all-zeros for unused slots beyond capacity).
- **Rooms** (4 features each): ready, carrier present, carrier busy at room, time since last use.
- **Global** (6 features): queue mix (small / big / total), retrieve count, oldest pending age, normalized sim time.
- **Edges**: `accesses` (carrier↔shelf, carrier↔room), `handoff` (carrier↔carrier), `transfer` (carrier↔transfer-shelf), `committed` (carrier→in-flight target).

Full feature catalog and rationale: [docs/SOLUTION_1_ENV.md](docs/SOLUTION_1_ENV.md) §5.3.

## Reward

```
r = − pending_weight        × dt × n_pending
    − responsiveness_weight × stranding_penalty
    + completion_bonus      × n_completions
```

`−dt × n_pending` is the workload-integrated cost — in expectation it
sums to the negative of mean per-task wait time. Responsiveness is a
soft penalty for being unable to stage any room before the next likely
store arrival. Completion bonus is cosmetic; defaults to 0.

The store/retrieve credit assignment problem is discussed in
[docs/SOLUTION_1.md](docs/SOLUTION_1.md) §5 — the per-item dwell-time
retrieval model gives the policy a real lifecycle signal that
uncorrelated retrievals didn't.

## Authoring a facility

Hand-author in Python via the DSL:

```python
from oos.dsl import Facility

fac = Facility("my_facility")

C1 = fac.carrier("C1", positions=12)
C2 = fac.carrier("C2", positions=12)
C3 = fac.carrier("C3", positions=12)  # mediator (no room)

# 8 shelves per carrier, alternating big/small, capacity 4
sizes = ("big", "small") * 4
for i, (slot, size) in enumerate(zip(range(2, 10), sizes), start=1):
    C1.shelf(f"A{i}", at=slot, capacity=4, size=size)
    C2.shelf(f"B{i}", at=slot, capacity=4, size=size)
    C3.shelf(f"M{i}", at=slot, capacity=4, size=size)

C1.room("R1", at=0)
C2.room("R2", at=0)

# C1↔C3 and C2↔C3 (no direct C1↔C2)
fac.handoff(between=(C1, C3), at={C1: 1, C3: 1})
fac.handoff(between=(C2, C3), at={C2: 2, C3: 2})

# Seed up to (total_capacity − biggest_big_shelf_cap) empties
fac.seed_pool()

topology, seeding = fac.build()
```

Validation runs at `build()`. Full DSL reference: [docs/SOLUTION_1_DSL.md](docs/SOLUTION_1_DSL.md).

## Plugging in a policy

The visualizer takes a callable `policy_fn(obs, info) → action_idx`:

```python
import numpy as np
from oos.viz.app import run_app
from oos.env.env import OOSEnv
from oos.facilities import make_facility

def my_policy(obs, info):
    # mask out illegal actions
    legal = np.flatnonzero(obs["action_mask"])
    # ... your logic here ...
    return int(legal[0])

env = OOSEnv(facility_factory=make_facility)
run_app(env, policy=my_policy, seed=0)
```

When you have a trained agent, the same callable shape works — wrap
your model's `predict` call.

## Docs

- [docs/PROBLEM.md](docs/PROBLEM.md) — formal problem statement and constraints
- [docs/SOLUTION_1.md](docs/SOLUTION_1.md) — GNN encoder + pointer attention + PPO design (not built yet)
- [docs/SOLUTION_1_ENV.md](docs/SOLUTION_1_ENV.md) — sim + env design + API reference
- [docs/SOLUTION_1_DSL.md](docs/SOLUTION_1_DSL.md) — DSL design + API reference

## Tests

```bash
uv run pytest -q
```

5 passing tests:

1. `test_dsl_builds_dev_facility` — DSL → sim contract
2. `test_env_reset_and_random_rollout` — random policy runs to truncation
3. `test_pallet_count_conserved` — 2000-step rollout, pallet count invariant
4. `test_big_stores_dropped_when_big_capacity_exhausted` — admission control
5. `test_determinism_across_seeds` — same seed = same trajectory

## Dependencies

- Python ≥ 3.12
- `numpy`, `gymnasium`, `pygame` (runtime)
- `pytest` (dev)
- `uv` for environment management

Managed via `pyproject.toml` + `uv.lock`. Run anything with `uv run ...`.

## Conventions worth knowing

- **No legacy code.** When the model has changed (shuttle/lift dropped, transfer width dropped, `Room.accepted_sizes` dropped, `Store.room` dropped, etc.), all references are removed across sim, env, viz, tests, and docs. No backward-compat shims; nothing is deployed.
- **Sim has no RL deps.** `oos.sim` does not import `oos.env`, `oos.config`, `oos.viz`, or any RL library. Hard rule.
- **Pallet conservation is invariant.** Don't add code paths that create or destroy pallets — the regression test will catch it.
- **Determinism is invariant.** Don't introduce `random.seed()` or `np.random.seed()` calls. Spawn child generators from the env's seeded generator.
