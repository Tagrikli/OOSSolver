# Solution 1 — Environment design

The simulation environment that backs the RL agent from
[SOLUTION_1.md](SOLUTION_1.md). This document specifies the physics
simulator, the Gymnasium wrapper, the task-stream sampler, the config
layer, and the testing harness. It reflects the in-tree state — every
type and feature listed here exists in the codebase today.

Goals:

- Discrete-event facility physics consistent with [PROBLEM.md](PROBLEM.md).
- A `gymnasium.Env` whose `reset` / `step` return
  `(observation, reward, terminated, truncated, info)` tuples consumable
  by standard PPO implementations.
- Deterministic given `(facility, config, seed, action_trace)`.
- A split-step API (`submit_action` + `advance(time_limit)`) so the
  visualizer can drive the env at sub-event resolution for smooth
  animation.

Out of scope: the GNN, the PPO trainer, the policy network.

## 1. Top-level architecture

```
oos/
├── sim/                      # pure physics, no RL, no torch
│   ├── topology.py           # static facility description
│   ├── state.py              # dynamic state dataclasses
│   ├── actions.py            # action primitives + preconditions
│   ├── durations.py          # move/handoff/customer duration models
│   ├── scheduler.py          # event queue, clock
│   ├── tasks.py              # store arrival stream (Poisson)
│   └── facility.py           # Facility = topology + state + scheduler + dwell sampler
│
├── dsl/                      # authoring DSL (see SOLUTION_1_DSL.md)
│   ├── builder.py
│   ├── compile.py
│   ├── refs.py
│   └── validate.py
│
├── config/
│   └── schema.py             # ExperimentConfig (durations, task stream, episode)
│
├── facilities/
│   └── dev.py                # hand-authored 3-carrier reference facility
│
├── env/                      # Gymnasium wrapper
│   ├── env.py                # OOSEnv (split-step: submit_action + advance)
│   ├── observation.py        # state → typed-graph dict
│   ├── action.py             # action enumeration + masking
│   ├── reward.py             # reward computation
│   └── responsiveness.py     # stranding-risk penalty
│
└── viz/                      # pygame visualization (cyberpunk themed)
    ├── components.py         # reusable widgets (shelves, rooms, panels, toasts)
    ├── layout.py             # auto-layout from topology
    ├── renderer.py           # frame composition
    ├── player.py             # env-stepping driver + pluggable policy
    └── app.py                # main loop + controls
```

Hard rule: `oos.sim` does not import `oos.env`, `oos.config`, `oos.viz`,
or any RL library. The sim is a library; the env and viz are consumers.

## 2. Sim core

### 2.1 Static topology ([sim/topology.py](../oos/sim/topology.py))

```python
@dataclass(frozen=True)
class Carrier:
    id: CarrierId
    positions: int            # number of 1D slots on its track
    default_position: int = 0
    speed: float = 1.0

@dataclass(frozen=True)
class Shelf:
    id: ShelfId
    size_class: Literal["small", "big"]
    capacity: int                                  # ≤ SHELF_MAX_CAPACITY (5)
    access: tuple[CarrierId, ...]                  # 1 carrier (normal); 2 (transfer shelf)
    position_for: Mapping[CarrierId, Position]
    is_transfer: bool = False                      # if True, capacity is always 1

@dataclass(frozen=True)
class Room:
    id: RoomId
    served_by: CarrierId
    position: Position
    # All rooms accept all sizes — no accepts field.

@dataclass(frozen=True)
class Handoff:
    carriers: tuple[CarrierId, CarrierId]
    positions: Mapping[CarrierId, Position]

@dataclass(frozen=True)
class Topology:
    carriers: Mapping[CarrierId, Carrier]
    shelves:  Mapping[ShelfId, Shelf]
    rooms:    Mapping[RoomId, Room]
    handoffs: tuple[Handoff, ...]
    # Derived caches:
    accessible_shelves: Mapping[CarrierId, frozenset[ShelfId]]
    accessible_rooms:   Mapping[CarrierId, frozenset[RoomId]]
    handoff_partners:   Mapping[CarrierId, frozenset[CarrierId]]
    handoff_positions:  Mapping[tuple[CarrierId, CarrierId], tuple[Position, Position]]
```

Frozen, hashable, validated. `validate_topology` raises on dangling
handoffs, transfer shelves with `len(access) ≠ 2`, transfer shelf
capacity ≠ 1, position out-of-range, etc.

**Carrier model**: there is no shuttle/lift distinction. Carriers are
1D movers; physical orientation is real-world detail the planner
doesn't model.

**Transfer shelf**: single-slot buffer accessible by 2 carriers. Time-
decoupled exchange (carrier A gives, carrier B takes later). Capacity
always 1. For synchronous co-located swap, use a `Handoff` pose.

### 2.2 Dynamic state ([sim/state.py](../oos/sim/state.py))

```python
@dataclass(frozen=True)
class Pallet:                          # value object; conserved
    item: Optional[ItemId] = None
    item_size: Optional[SizeClass] = None

@dataclass
class CarrierState:
    position: Position
    load: Optional[Pallet] = None
    busy_until: Optional[SimTime] = None
    command_started_at: Optional[SimTime] = None
    command_start_position: Optional[Position] = None
    current_command: Optional[object] = None

@dataclass
class ShelfState:
    stack: list[Pallet]                # top = stack[-1] (LIFO)

@dataclass
class RoomState:
    # Pallets stay on the serving carrier through the entire customer
    # interaction. No `content` field — the room itself holds nothing.
    customer_interaction_until: Optional[SimTime] = None

@dataclass
class FacilityState:
    time: SimTime
    carriers: dict[CarrierId, CarrierState]
    shelves:  dict[ShelfId, ShelfState]
    rooms:    dict[RoomId, RoomState]
```

**Pallets are conserved.** Empty pallets and loaded pallets are the
same physical objects — when a customer loads, the pallet on the
carrier morphs from empty to carrying an item; when a customer unloads,
it reverts to empty. The pallet itself is never created or destroyed.

### 2.3 Action primitives ([sim/actions.py](../oos/sim/actions.py))

Seven commands. All movement-involving commands **bundle the move** with
the operation — the policy says "take from A1" and the system handles
the move to A1's position first.

| Command | Effect | Duration |
|---|---|---|
| `Take(carrier, shelf)` | move to shelf, pop top pallet onto carrier | `move + shelf_op` |
| `Give(carrier, shelf)` | move to shelf, push current pallet onto top | `move + shelf_op` |
| `Handoff(giver, receiver)` | instant swap between two co-located carriers | `handoff_time` |
| `MoveToPartner(carrier, partner)` | position at the handoff pose with `partner` | `move` |
| `StageRoom(carrier, room)` | move to room with an empty pallet (locks in store cost) | `move`; then `customer_load_time` runs as a sentinel |
| `DeliverItem(carrier, room)` | move to room with a loaded pallet (locks in retrieve cost if a match) | `move`; then `customer_unload_time` runs as a sentinel |
| `Park(carrier)` | no-op, brief stay | configurable, default 1.0 |

Each command has `check_preconditions`, `start` (returns `busy_until`),
and `complete` (applies effects on event-fire).

**Concurrent shelf access guarded by pessimistic reservation**: when a
transfer shelf is targetable by two carriers simultaneously, `Take` /
`Give` precondition checks subtract pending takes from depth and add
pending gives to occupancy — so two carriers can't both successfully
`Take` from a 1-pallet shelf.

**Handoff is symmetric in lockout**: when one carrier submits `Handoff`,
the *receiver* is also locked (set busy) until completion. Precondition
requires both carriers already co-located, idle, with compatible loads.

### 2.4 Durations ([sim/durations.py](../oos/sim/durations.py))

```python
class DurationModel(Protocol):
    def move(self, carrier: Carrier, frm: int, to: int) -> SimTime: ...
    def shelf_op(self, kind: Literal["give", "take"], shelf: Shelf) -> SimTime: ...
    def handoff(self) -> SimTime: ...
    def customer_load(self) -> SimTime: ...
    def customer_unload(self) -> SimTime: ...
```

`LinearDurations` is the default: `move = |Δpos| / carrier.speed`;
shelf-op / handoff / customer-load / customer-unload are constants
(configurable via `DurationsConfig`).

### 2.5 Scheduler ([sim/scheduler.py](../oos/sim/scheduler.py))

Min-heap of `Event(when, seq, kind, payload)` keyed by `(when, seq)`
for deterministic tiebreak. Five event kinds:

- `command_done` — a carrier's current command completes
- `task_arrival` — store arrival from the task stream
- `retrieve_arrival` — a per-item scheduled retrieve fires
- `customer_load_done` — customer finishes loading the staged pallet
- `customer_unload_done` — customer finishes unloading a delivered item

### 2.6 Facility ([sim/facility.py](../oos/sim/facility.py))

The top-level handle. Owns topology, state, scheduler, task stream,
and a per-item dwell sampler.

```python
class Facility:
    def __init__(
        self, topology, seeding, durations,
        task_stream=None, rng=None,
        dwell_sampler: Callable[[ItemId, SizeClass], SimTime] | None = None,
    ): ...

    def submit(self, cmd: Command) -> None:
        """Apply preconditions, mark carrier busy, schedule command_done."""

    def advance(self) -> AdvanceResult:
        """Process events until the next decision instant (no time cap)."""

    def advance_until(self, time_limit: SimTime | None) -> AdvanceResult:
        """Process events until decision instant OR next event > time_limit.
        Used by the visualizer for sub-event interpolation."""

    def idle_carriers(self) -> list[CarrierId]: ...
```

`AdvanceResult` exposes `dt`, `completions`, `arrivals`, `dropped`
(capacity-evicted tasks), and a `terminal` flag.

**Per-item retrieval**: when a `customer_load_done` event fires (an
item materializes), the facility calls `dwell_sampler(item_id, size)`
to get a delay, then schedules a `retrieve_arrival` event for that
specific item id at `now + delay`. The event fires later and pushes
a `Retrieve(item=X)` into the queue.

**Big-item admission control**: after every event, [`_sweep_unservable_bigs`](../oos/sim/facility.py)
checks `_can_accept_big_item()`. If every slot on every big shelf
holds a big item (no empties, no smalls, no free slots), all pending
`Store(size="big")` tasks are removed from the queue and added to
`AdvanceResult.dropped`. New big arrivals are dropped the same way.
Customer leaves the line — not a rejection, just capacity behavior.

## 3. Task stream ([sim/tasks.py](../oos/sim/tasks.py))

```python
@dataclass(frozen=True)
class Store(Task):
    arrived_at: SimTime
    size: SizeClass            # no `room` field — planner picks the room

@dataclass(frozen=True)
class Retrieve(Task):
    arrived_at: SimTime
    item: ItemId

class PoissonTaskStream:
    """Generates Store arrivals as a Poisson process.

    Retrieves are NOT generated here — each Store, once fulfilled,
    schedules its own retrieve after a per-item dwell delay (see
    Facility.dwell_sampler).
    """
    store_rate: float
    size_mix: dict[SizeClass, float]
```

The stream gives us a single global store queue. The agent's
`STAGE_ROOM` choice is what implicitly picks which room the next
customer is directed to.

## 4. Configuration ([config/schema.py](../oos/config/schema.py))

```python
@dataclass(frozen=True)
class TaskStreamConfig:
    store_rate: float = 0.0
    size_mix: dict[SizeClass, float] = {"small": 0.85, "big": 0.15}
    mean_dwell_seconds: float = 300.0          # 5 minutes (per-item retrieve delay)
    std_dwell_seconds: float = 120.0           # CV = 0.4 (Gamma distribution)

@dataclass(frozen=True)
class DurationsConfig:
    shelf_op_time: SimTime = 2.0
    handoff_time: SimTime = 1.0
    customer_load_time: SimTime = 10.0
    customer_unload_time: SimTime = 10.0

@dataclass(frozen=True)
class EpisodeConfig:
    max_sim_time: SimTime = 3600.0
    max_steps: int = 10_000
    warmup_sim_time: SimTime = 0.0

@dataclass(frozen=True)
class ExperimentConfig:
    durations: DurationsConfig
    task_stream: TaskStreamConfig
    episode: EpisodeConfig
```

The dwell sampler in the env constructs `Gamma(shape=(mean/std)², scale=std²/mean)`,
which reduces to exponential when `std == mean`.

## 5. Gymnasium environment ([env/env.py](../oos/env/env.py))

### 5.1 Public API

```python
class OOSEnv(gymnasium.Env):
    def __init__(
        self,
        facility_factory: Callable[[], tuple[Topology, SeedingConfig]],
        experiment_config: ExperimentConfig | None = None,
        reward_config: RewardConfig | None = None,
        observation_config: ObservationConfig | None = None,
    ): ...

    # Standard Gym
    def reset(self, seed=None, options=None) -> tuple[obs, info]: ...
    def step(self, action: int) -> tuple[obs, reward, terminated, truncated, info]: ...

    # Split-step API (used by the viz for smooth animation)
    def submit_action(self, action: int) -> bool:
        """Submit an action; advance pending_idle queue if more carriers
        need to decide at this same instant. Does NOT advance time."""
    def advance(self, time_limit: SimTime | None = None):
        """Advance scheduler until next decision OR time_limit.
        Returns the standard step tuple."""
    def needs_decision(self) -> bool:
        """True iff a carrier is currently waiting for an action."""
```

`step(action)` is `submit_action(action) + advance(None)` composed.
Trainers use `step`; the viz alternates `submit_action` and `advance`
to render mid-command states.

### 5.2 Action space

`Discrete(N_max)` where `N_max` is computed once from the facility as
the upper bound on legal `(type, target)` pairs for any carrier. The
**action mask** in the observation indicates which indices are legal
at the current decision instant; everything else has its logit set to
`-∞` (the network never picks an illegal action).

Seven action types match the seven commands in §2.3. The `action_entries`
list in `info` gives the `(type, target)` decoding for each index.

### 5.3 Observation space

A `gymnasium.spaces.Dict`:

```python
{
    "carrier_features": Box(n_carriers, 8),
    "shelf_features":   Box(n_shelves, 26),
    "room_features":    Box(n_rooms, 4),
    "global_features":  Box(6,),
    "action_mask":      Box(N_max, int8),
    "querying_carrier": Discrete(n_carriers),
}
```

#### Carrier features (8 per carrier)

```
position_norm, load_empty, load_pallet_empty, load_pallet_small,
load_pallet_big, busy, eta_norm, is_querying
```

`is_querying` flags which carrier this query is for (the policy head
attends to this node specifically).

#### Shelf features (6 base + 5 slots × 4 = 26 per shelf)

Base:
```
size_small, size_big, capacity_norm, depth_norm, depth_frac, is_transfer
```

Per slot (slot 0 = LIFO top = carrier-accessible), 4-way one-hot:
```
slot_empty            — slot exists on shelf but no pallet here
slot_pallet_empty     — slot holds an empty pallet
slot_pallet_small     — slot holds a small item
slot_pallet_big       — slot holds a big item
```

Slots beyond a shelf's capacity → all four bits zero (distinguishable
padding). `SHELF_MAX_CAPACITY = 5` is the hard cap, enforced at DSL
build time.

#### Room features (4 per room)

```
ready                       — carrier present, idle, holding empty pallet
carrier_present             — carrier physically at room.position
carrier_busy_at_room        — present AND mid-command
time_since_last_use_norm    — recency of last customer interaction
```

There are no per-room "pending store" features — the customer queue
is global, not per-room.

#### Global features (6)

```
n_pending_stores_norm
n_pending_stores_small_norm
n_pending_stores_big_norm
n_pending_retrieves_norm
oldest_pending_age_norm     — wait time of the front-of-queue customer
time_norm
```

#### Edges (info dict, not in observation_space)

```
edges_accesses              — carrier ↔ shelf, carrier ↔ room
edges_handoff               — carrier ↔ carrier (handoff partners)
edges_transfer              — carrier ↔ shelf (transfer shelves)
edges_committed             — carrier → in-flight-target (dynamic)
```

### 5.4 Reward ([env/reward.py](../oos/env/reward.py))

```python
r = − pending_weight        × dt × n_pending
    − responsiveness_weight × stranding_penalty(state, task_cfg, dt)
    + completion_bonus      × n_completions
```

`-pending × dt` is the workload-integrated cost — for every second
that passes, each pending task adds 1 unit of cost. Summed over an
episode, this is the negative of mean per-task wait time (modulo the
warmup window). Completion bonus is cosmetic; the optimum is unchanged.

### 5.5 Responsiveness penalty ([env/responsiveness.py](../oos/env/responsiveness.py))

```
eta_to_ready(r) ≈ time for ρ(r) to be back at the room with empty pallet
strand_risk    = 1 − exp(−store_rate × min(eta_to_ready over all rooms))
penalty        = strand_risk × dt
```

Aggregated across rooms because the queue is global: as long as ANY
room is ready (or about to be), strand risk is low.

### 5.6 Decision-instant semantics

When `advance` produces N idle carriers at the same `SimTime`, the env
queries the policy N times back-to-back with the clock frozen. Between
queries, the observation is rebuilt so that `committed_target` edges
reflect prior queries in the same instant. Reward is only emitted on
the first query of an instant (`dt = 0` for subsequent queries).

Order of queries within an instant: deterministic, by carrier id.

### 5.7 Episode termination

- `truncated = True` when `state.time >= episode.max_sim_time` OR
  `step_count >= episode.max_steps`.
- `terminated = True` never under v1 — the task stream is unbounded.

## 6. Determinism

Single seed (`reset(seed=...)`) drives a `numpy.random.Generator`.
Child generators are spawned for each consumer:

- Task stream
- Dwell sampler
- Facility (scheduler tiebreaks)
- Duration model (when stochastic — currently constants)

No global `random.seed()`. No global state. Two envs with the same
seed and action sequence produce byte-identical observations and rewards.

`test_determinism_across_seeds` ([tests/test_smoke.py](../tests/test_smoke.py))
asserts this.

## 7. Vectorization (deferred)

Plan for when PPO is wired up: `SubprocVecEnv` (one process per env).
Variable-shape obs over the pipe via pickled dicts. 16 workers is the
target for the first training run.

The batcher (lives with the trainer, not the env) accepts a list of
variable-size graph observations and builds a single PyG `Batch` or a
manually-padded tensor stack.

Not built yet — first priority is shipping the env, the viz, and a
heuristic baseline.

## 8. Testing ([tests/test_smoke.py](../tests/test_smoke.py))

Five tests, all passing:

1. **`test_dsl_builds_dev_facility`** — the DSL compiles the dev facility
   correctly: 3 carriers (C1, C2, C3), no room on C3, exactly 2 handoffs
   (C1↔C3 and C2↔C3, no C1↔C2).
2. **`test_env_reset_and_random_rollout`** — env resets, random rollout
   runs to truncation, observation shapes match.
3. **`test_pallet_count_conserved`** — 2000-step random rollout at high
   store rate; pallet count never drifts from initial. Guards against
   pallet-conservation regressions.
4. **`test_big_stores_dropped_when_big_capacity_exhausted`** —
   admission control: pending big stores are evicted from the queue
   when big-shelf slots fill with big items; small stores unaffected.
5. **`test_determinism_across_seeds`** — same seed + actions = same
   reward trajectory.

## 9. Visualization ([oos/viz/](../oos/viz/))

A pygame-based facility viewer (the original "out of scope" guess
turned out to be a major build).

- **Run**: `uv run python -m oos.viz`
- **Theme**: "indigoshell" cyberpunk palette — magenta/cyan/yellow on
  deep blue-violet; signature beveled chrome with 45° cuts.
- **Controls**: `space` pause, `→` step one decision, `m` toggle anim/step,
  `+`/`-` speed up/down, `r` reset, `q`/`esc` quit. Scroll wheel
  scrolls the pending-tasks panel.
- **Smooth interpolation**: carriers slide linearly between
  `command_start_position` and `command_target_position` over
  `move_dur` (not the full command duration — shelf_op and customer
  interactions are stationary at the target).
- **Requested-item pulse**: items with a pending retrieve glow with a
  yellow halo at 2.5 Hz, on every shelf slot and every carrier load.
- **Customer queue strip**: top-of-canvas chip array; cyan=small,
  magenta=big; front-of-queue chip glows and shows wait time.
- **Room states**: ready (green) / busy (yellow only during real
  customer interaction) / idle (dim grey).
- **Handoff badges** at actual track positions; vertical dashed
  connectors between paired badges.
- **Dropped tasks** disappear silently from the queue strip (no toast).

Pluggable policy: `run_app(env, policy=my_callable, seed=...)`.
`my_callable(obs, info) → action_idx`. A trained agent slots in here.

## 10. Open decisions (deferred, not blocking)

1. **Customer interaction modeling** — constant vs sampled. Sampled
   adds learning variance; constant is current.
2. **`Park` vs implicit idle** — could drop `Park` if "no command
   submitted" means idle, but the explicit commitment is useful for
   handoff coordination.
3. **Warmup design** — hardcoded vs sampled stationary distribution.
4. **Hazard estimator location** — in the env (so reward sees it) or
   in the policy (so observation sees it)?
5. **Action mask in observation vs in info** — currently in observation
   (PyTorch-RL convention). Gym convention prefers info.
6. **Slot existence bit** — currently padding is "all 4 slot bits = 0";
   could add an explicit `slot_exists` bit for unambiguous padding.

## 11. What this enables next

- **Heuristic baselines** (FIFO + nearest, greedy, etc.) consume the
  env directly via the pluggable policy.
- **PPO trainer** (SOLUTION_1 §6) consumes the env directly. Replace
  the random policy in `viz/player.py` with an inference call.
- **Hazard-based reward shaping** drops into [env/reward.py](../oos/env/reward.py)
  without touching the sim. Already discussed in the credit-assignment
  thread.
- **Online fine-tuning** (SOLUTION_1 §8): swap `TaskStream` for one
  that replays logged traffic.
- **Multiple facilities / domain randomization**: change `facility_factory`
  to a sampler. The action space's `N_max` would need a global upper
  bound across configs.
