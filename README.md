# OOSKiller

OOSKiller trains a single reinforcement-learning agent to plan a Parkolay automated storage/retrieval system (ASRS) end-to-end. Customers arrive at rooms to **store** an item onto a staged empty pallet or to **retrieve** a specific stored item; fungible pallets move through a network of carriers (lifts = vertical, shuttles = horizontal) that hand off to each other and to LIFO shelves. The planner must, continuously and concurrently across every carrier, service the incoming store/retrieve stream so that the **expected per-task wait time** is minimized — reshuffling pallets between tasks to lower future cost while never stranding a room. The facility state is encoded as a typed graph and fed through a graph-attention trunk; the policy picks **one primitive per decision instant** — `GOTO` (dock at a shelf / room / handoff pose), `TAKE`, `GIVE`, or `WAIT` — with action masking, so one trained agent generalizes across facility layouts. A store/retrieve completes against the load a carrier physically holds while docked and WAITing at a room (rooms are interaction docks, not storage). Training is PPO against a torch-free discrete-event simulator; the current focus is a minimal **retrieve-only** spine (`scripts/train_retrieve.py`): one buried retrieve per episode at a configurable depth, a single delivery reward, and a deterministic greedy eval.

## The problem

A planner must service a store/retrieve stream over an ASRS while minimizing expected per-task wait (store cost = arrival → room ready with empty pallet; retrieve cost = request → item delivered to a room). What makes it hard: LIFO burial and blockers make relocation NP-hard, multiple carriers act concurrently with synchronized handoffs, the task distribution is unknown and estimated online, and a hard responsiveness constraint forbids ever stranding a room. The formal specification lives in [`docs/PROBLEM.md`](docs/PROBLEM.md); the design write-ups are in [`docs/`](docs/): [`SOLUTION_1.md`](docs/SOLUTION_1.md) (overall approach), [`SOLUTION_1_ENV.md`](docs/SOLUTION_1_ENV.md) (env design), and [`SOLUTION_1_DSL.md`](docs/SOLUTION_1_DSL.md) (topology DSL).

## Architecture

The core is decomposed into four single-responsibility parts. A driver (a trainer or the visualizer) owns an **Agent**, which pairs an injected policy with an **Environment**. The Environment owns episode control and the per-carrier decision loop, and delegates to a **SimEngine** (the world) and an injected **RewardSystem** (the scoring).

```
   driver (trainer / viz)
        │
        ▼
   Agent ── policy (random / LearnedPolicy / MCTSPolicy)
        │   pure (obs, info) -> action_idx
        ▼
   Environment            episode control, obs/action encoding,
        │                 per-carrier decision loop, info["events"]
        ├──────────────► SimEngine     discrete-event world engine:
        │                              topology, state, scheduler, queue,
        │                              dynamics, advance_until  (no torch)
        └──────────────► RewardSystem  injected, pluggable reward suite:
                                       StepEvents -> RewardContext -> (total, breakdown)
```

| Part | Module | Responsibility |
|------|--------|----------------|
| **Environment** | `oos/env/env.py` | Base class. Owns obs/action encoding, the per-carrier decision loop, episode control, and an injected RewardSystem. Exposes a training API (`reset`/`step`/`advance`) **and** an embedding/viz API (`apply_action`/`advance_until`/`submit_action`/`needs_decision`/`from_name`/…). Action count is `env.n_actions`. |
| **SimEngine** | `oos/sim/facility.py` | The world engine: state, scheduler, task queue, dynamics, `advance_until`. RL-free and torch-free. Reached as `environment.engine`. |
| **RewardSystem** | `oos/env/reward_system.py` | Pluggable reward suite (`RewardContext` + `RewardTerm` + factories), injectable into envs. Every term is a pure function of `(s, a, s')`. |
| **Agent** | `oos/agent/agent.py` | One class, policy injected. Drives an Environment episode-by-episode, recording each decision as an `AgentStep`. UI-free and embeddable. |

Env subclasses live in `oos/env/` — **`RetrieveEnv`** (the training spine) and **`SingleTaskEnv`** (the viz's "Generate" scenario env). They're thin: they override only the `setup_episode(facility, seed)` and `finalize_reset(obs, info)` hooks plus their own `step()` for success-termination; none override `reset()`. The reward path is built from a typed `StepEvents` struct surfaced as `info["events"]`.

## Repository layout

```
oos/
├── sim/         discrete-event facility SimEngine (state, scheduler, queue, dynamics) — no RL, no torch
├── env/         Environment layer: base env + task envs (RetrieveEnv, SingleTaskEnv), typed-graph
│                observation, action enum/masking, pluggable reward suite, retrieve-target helpers
├── agent/       embeddable, UI-free Agent runtime (policy + Environment, step-by-step)
├── learn/       the RL agent: GAT+pointer network, PPO, the reusable training harness
│                (net / checkpoint / progress / console), viz policy adapters (the only torch package)
├── viz/         pygame visualizer / interactive driver (canvas + tabbed sidebar)
├── facilities/  hand-authored facility registry (FACILITIES name -> factory)
├── dsl/         facility-definition builder DSL (authors + validates topologies)
└── config/      experiment-config dataclasses (durations, task stream, episode budgets)

scripts/         train_retrieve.py — the single self-contained trainer (config as globals, no CLI)
tests/           pytest suite (50 tests / 7 files): smoke, network, reward-system, reward-potential,
                 primitives, shuffle, state-sampler
docs/            PROBLEM.md (formal spec) + SOLUTION_1{,_ENV,_DSL}.md design write-ups
```

A separation contract is enforced: `oos.sim` and `oos.env` must **never** import from `oos.learn` (torch is isolated to `oos.learn`).

## Neural architecture (brief)

- **Typed-graph observation** — the facility is encoded as nodes in a canonical `[carriers | shelves | rooms]` index space (carrier / shelf / room / global features) with four typed edge families (`accesses`, `handoff`, `transfer`, `docked`). The `docked` edge points each carrier at the shelf / room / partner-carrier node it is currently parked at, so the trunk can fold a carrier's dock into its embedding. Loads live on carriers (no in-flight overlay), and shelves are padded to a hard cap of 5 LIFO slots (index 0 = top).
- **GAT trunk** — `TypedGATLayer` applies typed multi-head graph attention with per-edge-type weights, segment-softmax over destinations, residual + LayerNorm. `NetworkConfig` defaults to `hidden=64`, `n_heads=4`, and a library default of `n_gat_layers=0` (0 bypasses the trunk for tiny graphs). The retrieve trainer sets `n_gat_layers=2` for multi-hop facilities like `tiny_medipol`.
- **Heads** — a mean-pool **value head**; a **`GOTO` single-pointer head** that scores each reachable destination node (shelf / room / handoff-partner) against the ego-carrier embedding; and scalar **`TAKE` / `GIVE` / `WAIT`** heads off the ego carrier (they act on its current dock, so they need no target). Illegal slots are masked to `-inf`.
- Because the network scores *node embeddings* via pointer attention and uses **no absolute IDs**, one trained agent transfers across facility layouts. The model is small (well under ~100k–110k parameters at the default width).

## Getting started

The project uses **uv** and requires **Python ≥ 3.12**.

```bash
uv sync
```

Core dependencies: `numpy`, `pygame`, `torch`, `tensorboard` (and `pytest` as a dev dependency group).

### Run the tests

```bash
SDL_VIDEODRIVER=dummy uv run python -m pytest tests/ -q
```

50 tests across 7 files (smoke / network / reward-system / reward-potential / primitives / shuffle / state-sampler). The `SDL_VIDEODRIVER=dummy` prefix runs headless.

### Visualizer

```bash
python -m oos.viz --facility tiny_medipol --seed 0
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--facility` | persisted last-opened, else `tiny_medipol` | Which hand-authored facility to visualize (choices = registered facilities). |
| `--seed` | `0` | RNG seed passed to `run_app`. |

With no flags, `python -m oos.viz` opens your last-viewed facility (or `tiny_medipol` on a fresh checkout).

## Training

One trainer: **`scripts/train_retrieve.py`** — a single self-contained script with all configuration as module-level globals at the top (no CLI flags). Edit the `CONFIG` block and run it directly:

```bash
python scripts/train_retrieve.py        # or .venv/bin/python scripts/train_retrieve.py
```

Each episode is **one retrieve**: a target pallet at a depth drawn from `[MIN_DEPTH, MAX_DEPTH]` on a **direct** shelf (reachable to a room without a handoff), over a random `shuffle_state` layout (one `FULLNESS` knob; `-1` = a fresh `U[0,1]` per episode). The reward is a single flat **delivery bonus** (`delivery_system`) — no potential, movement, time, or idle penalty; the episode ends the moment the requested item is delivered to a room. Every `EVAL_EVERY` iterations a **deterministic argmax** greedy eval runs on `EVAL_EPISODES` fixed-seed layouts at the fixed `EVAL_DEPTH`, so the eval curve is directly comparable across iterations — watch the GREEDY rate, not the (entropy-noisy) sampled rate.

Outputs land in `runs/<RUN_NAME>/`: `ckpt_latest.pt` (checkpoint), `progress.md` (the greedy-eval curve), and `metrics.jsonl` (per-eval records). Set `RESUME = "runs/<RUN_NAME>/ckpt_latest.pt"` to restore net + optimizer and continue from the saved iteration — e.g. train depth 0 to mastery, then bump `MAX_DEPTH` and resume for a hand-rolled curriculum.

### The training harness

The plumbing every trainer needs lives in `oos/learn/` as four reusable modules, so the loop never re-implements it (and the viz checkpoint contract can't drift):

| Module | Provides |
|--------|----------|
| `net.py` | `build_net(...) -> (net, net_cfg, feat_dims)` — the one place feature dims + `NetworkConfig` are derived. |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint` / `restore_into` in the **one schema the viz speaks** (`network_config` + `feat_dims` + `net_state_dict` + `iteration`), so a checkpoint always loads in the visualizer. |
| `progress.py` | `ProgressWriter` — rewrites `progress.md` (a stats table) and appends `metrics.jsonl`. |
| `console.py` | the styled banner + per-iteration episode / PPO / eval logging (wraps `oos.learn._style`). |

### Reward

`RetrieveEnv` uses **`delivery_system`** — a single flat `DELIVER` term, nothing else (sparse by design; the delivery is the whole objective). The base `Environment` (and the viz's `SingleTaskEnv`) default to **`base_system`** — `DELIVER` + `SERVE` over a four-term shaping potential `Φ = −w_ret·Σ(steps_to_deliver) + w_ready·#staged − w_wrong·#parked − w_empty·shallowest_empty_depth`, applied either as policy-invariant PBRS (`γ·Φ(s′)−Φ(s)`) or, with `RewardConfig.dense_progress=True`, as the un-discounted `Φ(s′)−Φ(s)` (which drops the `γ<1` idle-drip). See `oos/env/reward.py` and `Environment._potential`.

### Inspecting results

`scripts/train_retrieve.py` writes `runs/<RUN_NAME>/progress.md` (the greedy-eval curve, rewritten each eval) and `metrics.jsonl` (one record per eval). Checkpoints in `runs/` are discoverable by the visualizer's policy picker.

### Training environments

| Env | Module | What an episode is |
|-----|--------|--------------------|
| **`RetrieveEnv`** *(the training spine)* | `oos/env/retrieve_env.py` | One buried `Retrieve` per episode at a depth in `[MIN_DEPTH, MAX_DEPTH]` on a **direct** shelf, over a `shuffle_state` random layout. Reward = one flat `DELIVER` (`delivery_system`); terminates on delivery. Depth is the only scenario axis. |
| **`SingleTaskEnv`** *(viz only)* | `oos/env/single_task_env.py` | One atomic goal — `retrieve` a marked pallet, or `bring_empty` (stage an empty at a room and WAIT). The env the viz's "Generate" feature spawns to watch a loaded policy attempt a configured scenario; not used in training. |

## Visualizer

`python -m oos.viz` launches a pygame app: a left canvas renders the live facility (carrier strips, shelves, room docking ports a carrier parks into, the customer queue, and a solvability overlay) alongside a tabbed sidebar (status, randomize, auto-queue, and an action-distribution panel). You can play/pause/step the sim, **swap policy and facility at runtime** (the policy picker discovers checkpoints under `runs/`, with an optional inference-time MCTS toggle), queue tasks and randomize state by hand, click pallets to request retrieves, and edit pallet/shelf contents directly. It boots in manual mode (auto-arrivals off; `m` toggles); press `h` for the full keybinding/colour legend. The session (facility, policy path, zoom, speed, knob dicts) persists in `runs/.viz_state.json`. torch is imported lazily, so the viz runs with only the random policy when torch is absent.

## Facilities

Ten hand-authored facilities are registered (look up via `get_facility(name)`; `--facility` choices are these names sorted):

`mini` · `tiny` · `tiny_tall` · `tiny_wide` · `tiny_medipol` · `stacker` · `stacker_deep` · `stacker_wide` · `dibaji` · `campus`

`tiny_medipol` is the eventual deployment target — the only facility mixing **direct** (lift) and **handoff** (shuttle) retrieve routes; the retrieve spine currently trains on direct shelves (the `train_retrieve.py` default facility is `tiny`). New facilities are authored with the `oos.dsl` builder, which validates and compiles to a frozen `(Topology, SeedingConfig)`:

```python
from oos.dsl import Carrier, Facility, Handoff, Room, Shelf

def make_facility():
    fac = Facility("tiny_medipol")
    fac.register_carriers(L1, L2, S1, S2)   # carriers, their tracks, shelves, rooms
    fac.pair(h_L1_S1, h_S1_L1)              # declare handoff poses between carriers
    fac.seed_pool()                         # seed initial empty pallets
    return fac.build()                      # -> (Topology, SeedingConfig)
```
