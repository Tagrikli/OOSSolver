# OOSKiller

OOSKiller trains a single reinforcement-learning agent to plan a Parkolay automated storage/retrieval system (ASRS) end-to-end. Customers arrive at rooms to **store** an item onto a staged empty pallet or to **retrieve** a specific stored item; fungible pallets move through a network of carriers (lifts = vertical, shuttles = horizontal) that hand off to each other and to LIFO shelves. The planner must, continuously and concurrently across every carrier, service the incoming store/retrieve stream so that the **expected per-task wait time** is minimized — reshuffling pallets between tasks to lower future cost while never stranding a room. The facility state is encoded as a typed graph and fed through a graph-attention trunk; the policy picks **one primitive per decision instant** — `GOTO` (dock at a shelf / room / handoff pose), `TAKE`, `GIVE`, or `WAIT` — with action masking, so one trained agent generalizes across facility layouts. A store/retrieve completes against the load a carrier physically holds while docked and WAITing at a room (rooms are interaction docks, not storage). Training is PPO against a torch-free discrete-event simulator, with the current research focus being continuous truncated-episode **Prioritized Level Replay (PLR)** on the `tiny_medipol` facility.

## The problem

A planner must service a store/retrieve stream over an ASRS while minimizing expected per-task wait (store cost = arrival → room ready with empty pallet; retrieve cost = request → item delivered to a room). What makes it hard: LIFO burial and blockers make relocation NP-hard, multiple carriers act concurrently with synchronized handoffs, the task distribution is unknown and estimated online, and a hard responsiveness constraint forbids ever stranding a room. The formal specification lives in [`docs/PROBLEM.md`](docs/PROBLEM.md); the design write-ups are in [`docs/`](docs/): [`SOLUTION_1.md`](docs/SOLUTION_1.md) (overall approach), [`SOLUTION_1_ENV.md`](docs/SOLUTION_1_ENV.md) (env design), [`SOLUTION_1_DSL.md`](docs/SOLUTION_1_DSL.md) (topology DSL), and [`SINGLE_TASK_ENV.md`](docs/SINGLE_TASK_ENV.md) (single-task env notes).

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

Env subclasses (`ContinuousEnv`, `SingleTaskEnv`, `EpisodeEnv` in `oos/learn/`) are thin: they override only the `setup_episode(facility, seed)` and `finalize_reset(obs, info)` hooks (plus their own `step()` for dense-reward recomputation); none override `reset()`. The reward path is built from a typed `StepEvents` struct surfaced as `info["events"]`.

## Repository layout

```
oos/
├── sim/         discrete-event facility SimEngine (state, scheduler, queue, dynamics) — no RL, no torch
├── env/         Environment layer: typed-graph observation, action enum/masking, pluggable reward suite
├── agent/       embeddable, UI-free Agent runtime (policy + Environment, step-by-step)
├── learn/       the RL agent: GAT+pointer network, PPO, three trainers, PLR scheduler, viz policy adapters (the only torch package)
├── viz/         pygame visualizer / interactive driver (canvas + tabbed sidebar)
├── facilities/  hand-authored facility registry (FACILITIES name -> factory)
├── dsl/         facility-definition builder DSL (authors + validates topologies)
└── config/      experiment-config dataclasses (durations, task stream, episode budgets)

scripts/         shell launchers (train_*.sh) + standalone diagnostics (diagnose_zero.py, test_seeded.py)
tests/           pytest suite (50 tests / 8 files): smoke, network, reward-system, reward-potential, primitives, shuffle, state-sampler, PLR
docs/            PROBLEM.md (formal spec) + SOLUTION_1{,_ENV,_DSL}.md + SINGLE_TASK_ENV.md design write-ups
```

A separation contract is enforced: `oos.sim` and `oos.env` must **never** import from `oos.learn` (torch is isolated to `oos.learn`).

## Neural architecture (brief)

- **Typed-graph observation** — the facility is encoded as nodes in a canonical `[carriers | shelves | rooms]` index space (carrier / shelf / room / global features) with four typed edge families (`accesses`, `handoff`, `transfer`, `docked`). The `docked` edge points each carrier at the shelf / room / partner-carrier node it is currently parked at, so the trunk can fold a carrier's dock into its embedding. Loads live on carriers (no in-flight overlay), and shelves are padded to a hard cap of 5 LIFO slots (index 0 = top).
- **GAT trunk** — `TypedGATLayer` applies typed multi-head graph attention with per-edge-type weights, segment-softmax over destinations, residual + LayerNorm. `NetworkConfig` defaults to `hidden=64`, `n_heads=4`, and a library default of `n_gat_layers=0` (0 bypasses the trunk for tiny graphs). The continuous and single-task trainers raise the CLI `--n-gat-layers` default to **2** for multi-hop facilities like `tiny_medipol`; the episodic trainer leaves it at 0.
- **Heads** — a mean-pool **value head**; a **`GOTO` single-pointer head** that scores each reachable destination node (shelf / room / handoff-partner) against the ego-carrier embedding; and scalar **`TAKE` / `GIVE` / `WAIT`** heads off the ego carrier (they act on its current dock, so they need no target). Illegal slots are masked to `-inf`.
- Because the network scores *node embeddings* via pointer attention and uses **no absolute IDs**, one trained agent transfers across facility layouts. The model is small (well under ~100k parameters at the default width).

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

50 tests across 8 files (smoke / network / reward-system / reward-potential / primitives / shuffle / state-sampler / continuous-PLR). The `SDL_VIDEODRIVER=dummy` prefix runs headless.

### Visualizer

```bash
python -m oos.viz --facility tiny_medipol --seed 0
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--facility` | persisted last-opened, else `tiny_medipol` | Which hand-authored facility to visualize (choices = registered facilities). |
| `--seed` | `0` | RNG seed passed to `run_app`. |

With no flags, `python -m oos.viz` opens your last-viewed facility (or `tiny_medipol` on a fresh checkout).

### Trainers

**Continuous PLR (current focus)** — `python -m oos.learn.train_continuous`. Each iteration is one truncated continuous episode; a `LevelScheduler` picks a hardness level and value-loss regret feeds back. Outputs `runs/<run-name>/{config.json, tb/, ckpt_latest.pt, metrics.jsonl, progress.md}`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--facility` | `stacker` | Facility to train on. |
| `--total-iterations` | `500` | Number of training iterations (each = one rollout-episode). |
| `--steps-per-iter` | `1024` | Steps per episode (== `max_steps`); episode truncated at this cap. |
| `--episodes-per-iter` | `1` | Episodes batched into one PPO update (each gets its own PLR level + regret). |
| `--store-rate` | `0.1` | Poisson store arrival rate (tasks/sim-sec). |
| `--replay-prob` | `0.5` | PLR probability of replaying a buffered level vs sampling a fresh one. |
| `--reward-deliver` | `50.0` | + per requested item delivered (flat; depth is rewarded by the potential). |
| `--reward-serve` | `20.0` | + per store served onto a staged empty (must exceed `room-ready + wrong-car`). |
| `--potential-item-retrieval` | `1.0` | PBRS `w_ret`: Φ drops by `w_ret·(depth+1)` per requested item; digging it shallower raises Φ. |
| `--potential-room-ready` | `2.0` | PBRS `w_ready`: + per carrier docked at a room holding an empty pallet. |
| `--potential-wrong-car` | `2.0` | PBRS `w_wrong`: − per carrier at a room holding a non-requested car (restored on leaving). |
| `--potential-shallowest-empty` | `1.0` | PBRS `w_empty`: − `w_empty·(burial depth of the shallowest empty anywhere)`. |
| `--regret-metric` | `l1_value_loss` | PLR scoring: `l1_value_loss` or `positive_value_loss`. |
| `--n-gat-layers` | `2` | GAT trunk depth (library default is 0; this trainer raises it for multi-hop facilities). |
| `--device` | `cpu` | Torch device. |
| `--run-name` | auto `cont_<timestamp>` | Run name → `runs/<run-name>/`. |
| `--resume` | — | Path to `ckpt_latest.pt` to continue from (restores net/optim/PLR buffer/counters). |

The reward is two pump-safe outcome rewards (`DELIVER`, `SERVE`) over a four-term **PBRS potential** `Φ = −w_ret·Σ(depth+1) + w_ready·#staged − w_wrong·#parked − w_empty·shallowest_empty_depth`; the shaped reward is `γ·Φ(s′) − Φ(s)`. There are no movement/idle/time penalties — urgency comes from `γ`. See `oos/env/reward.py` and `Environment._potential`. The curated launcher `./scripts/train_cont_easy.sh` wraps this with a tuned flag set.

```bash
uv run python -m oos.learn.train_continuous \
  --facility tiny_medipol --total-iterations 100 --steps-per-iter 1024 \
  --store-rate 0.008 --replay-prob 0.5 --reward-deliver 50 --reward-serve 20 \
  --potential-item-retrieval 1 --potential-room-ready 2 --potential-wrong-car 2 \
  --n-gat-layers 2 --run-name medipol_cont1 --device cpu
```

**Two-phase store→retrieve** — `python -m oos.learn.train` (`EpisodeEnv`). Pallets start empty; phase 1 stores, phase 2 retrieves all; terminates on full clear or step cap. Supports `VecEnv` collection when `--n-envs > 1`, plus an `--eval-only` branch.

| Flag | Default | Meaning |
|------|---------|---------|
| `--facility` | `tiny` | Facility to train on. |
| `--total-iterations` | `200` | Number of training iterations. |
| `--steps-per-iter` | `2048` | Transitions collected per iteration. |
| `--n-envs` | `1` | Parallel envs; `>1` uses VecEnv collection. |
| `--big-prob` | `0.15` | Probability a sampled Store is big (gated; may downgrade to small). |
| `--reward-retrieve` | `50.0` | Reward per Retrieve completion. |
| `--max-episode-steps` | `400` | Step cap per full store+retrieve cycle. |
| `--n-gat-layers` | `0` | GAT trunk depth. |
| `--eval-only` | `False` | Skip training; run `--eval-episodes` and report. |
| `--device` | `cpu` | Torch device. |
| `--run-name` | auto `episode_<timestamp>` | Run name → `runs/<run-name>/`. |
| `--resume` | — | Checkpoint path to resume from (`ckpt_best.pt` / `ckpt_latest.pt` / `ckpt_iter_*.pt`). |

```bash
uv run python -m oos.learn.train --total-iterations 200 --run-name v1 --n-envs 10 --facility tiny --device cpu
uv run python -m oos.learn.train --eval-only --eval-episodes 100 --resume runs/v1/ckpt_best.pt --facility tiny
```

**Single atomic task** — `python -m oos.learn.train_single_task` (`SingleTaskEnv`). Each episode is one atomic task (`retrieve` or `bring_empty`), sampled per reset; single-env only. Has an `--eval-only` branch reporting per-task success.

| Flag | Default | Meaning |
|------|---------|---------|
| `--facility` | `stacker` | Facility to train on. |
| `--total-iterations` | `200` | Number of training iterations. |
| `--steps-per-iter` | `1024` | Transitions collected per iteration. |
| `--task` | `retrieve` | Task type: `retrieve` or `bring_empty`. |
| `--retrieve-from` | `big` | Shelf class the retrieve target is drawn from (`big`/`small`). |
| `--retrieve-route` | `direct` | Delivery route of target shelf: `direct` or `handoff` (handoff is harder). |
| `--target-depth` | `0` | Retrieve target's stack depth (0 = top/accessible). |
| `--big-shelf-fullness` | `0.5` | Fraction of big-shelf slots occupied; dominant retrieve-hardness lever. |
| `--reward-success` | `2.0` | Reward paid once per successful episode (terminates same step). |
| `--n-gat-layers` | `2` | GAT trunk depth (library default is 0; this trainer raises it). |
| `--eval-only` | `False` | Skip training; run `--eval-episodes` and report per-task success. |
| `--device` | `cpu` | Torch device. |
| `--run-name` | auto `single_task_<timestamp>` | Run name → `runs/<run-name>/`. |
| `--resume` | — | Checkpoint path to resume from. |

```bash
uv run python -m oos.learn.train_single_task \
  --facility tiny_medipol --total-iterations 500 --steps-per-iter 1024 \
  --task bring_empty --target-depth 0 --reward-success 20 \
  --no-reward-scaling --n-gat-layers 1 --run-name st1 --device cpu
```

All three trainers draw `--facility` from the registered facilities and write to `runs/<run-name>/` with `config.json`, a `tb/` TensorBoard directory, and checkpoints. `train_continuous` saves `ckpt_latest.pt` (resume from it); `train` and `train_single_task` additionally save periodic `ckpt_iter_<NNNNNN>.pt` and a best-so-far `ckpt_best.pt`. Curated shell launchers wrap these with hyperparameter sets and forward extra flags: `./scripts/train_cont.sh`, `./scripts/train_cont_easy.sh`, `./scripts/train_st.sh`.

### Inspecting results

```bash
tensorboard --logdir runs/
```

Every trainer streams scalars to `runs/<run-name>/tb/`. `train_continuous` also writes `runs/<run-name>/progress.md` — a human-readable live summary rewritten every `--log-every` iterations — and `metrics.jsonl`, a machine-readable per-iteration log.

### Training environments

| Env | Trainer | What an episode is |
|-----|---------|--------------------|
| **`ContinuousEnv`** *(current focus)* | `train_continuous` | One truncated continuous episode reset into a PLR-scheduler-chosen hardness *level* (via `InitialStateSampler`), with a buried `Retrieve` seeded and the Poisson store stream + dwell retrievals kept on. **Never terminates, only truncates.** Reward = `DELIVER` + `SERVE` outcomes over a four-term PBRS potential (retrieval depth / room-ready / wrong-car / shallowest-empty); no movement/idle penalty — urgency comes from the discount `gamma`. |
| **`EpisodeEnv`** | `train` | Two-phase: phase 1 fills the facility from empty via gated random Store sampling; phase 2 drains all stored pallets one at a time. Terminates on full clear. The only env wired into the multiprocess `VecEnv` (so only `train` can use `--n-envs > 1`). |
| **`SingleTaskEnv`** | `train_single_task` | One atomic goal per episode — `retrieve` a marked pallet, or `bring_empty` (stage an empty at a room and WAIT). Initial state from `InitialStateSampler`; reward via `single_task_system`. The curriculum baseline — design notes in [`docs/SINGLE_TASK_ENV.md`](docs/SINGLE_TASK_ENV.md). |

## Visualizer

`python -m oos.viz` launches a pygame app: a left canvas renders the live facility (carrier strips, shelves, room docking ports a carrier parks into, the customer queue, and a solvability overlay) alongside a tabbed sidebar (status, randomize, auto-queue, and an action-distribution panel). You can play/pause/step the sim, **swap policy and facility at runtime** (the policy picker discovers checkpoints under `runs/`, with an optional inference-time MCTS toggle), queue tasks and randomize state by hand, click pallets to request retrieves, and edit pallet/shelf contents directly. It boots in manual mode (auto-arrivals off; `m` toggles); press `h` for the full keybinding/colour legend. The session (facility, policy path, zoom, speed, knob dicts) persists in `runs/.viz_state.json`. torch is imported lazily, so the viz runs with only the random policy when torch is absent.

## Facilities

Ten hand-authored facilities are registered (look up via `get_facility(name)`; `--facility` choices are these names sorted):

`mini` · `tiny` · `tiny_tall` · `tiny_wide` · `tiny_medipol` · `stacker` · `stacker_deep` · `stacker_wide` · `dibaji` · `campus`

`tiny_medipol` is the canonical training target — the only facility mixing **direct** (lift) and **handoff** (shuttle) retrieve routes. New facilities are authored with the `oos.dsl` builder, which validates and compiles to a frozen `(Topology, SeedingConfig)`:

```python
from oos.dsl import Carrier, Facility, Handoff, Room, Shelf

def make_facility():
    fac = Facility("tiny_medipol")
    fac.register_carriers(L1, L2, S1, S2)   # carriers, their tracks, shelves, rooms
    fac.pair(h_L1_S1, h_S1_L1)              # declare handoff poses between carriers
    fac.seed_pool()                         # seed initial empty pallets
    return fac.build()                      # -> (Topology, SeedingConfig)
```
