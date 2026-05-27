# SingleTaskEnv — design notes

Code: [`oos/learn/single_task_env.py`](../oos/learn/single_task_env.py)

A "one goal, one episode" environment. Each episode is exactly one of two
atomic tasks; the agent doesn't see which task it's on except through the
usual observation features (specifically, whether any `slot_pallet_requested`
bit is set).

## Two task types

| Task | Probability (default) | Observation hint | Success condition | Termination |
|---|---|---|---|---|
| **retrieve** | 0.8 | One pallet has `slot_pallet_requested=1` (because a `Retrieve` is queued for it) | A `Retrieve` completion fires with `task.pallet == target_id` | `terminated=True` immediately on success |
| **bring_empty** | 0.2 | No pallet has `slot_pallet_requested=1` (no `Retrieve` queued) | First `free → empty` room transition fires (i.e., the agent placed any empty pallet on a free room) | `terminated=True` immediately on success |

Knob: `SingleTaskConfig.bring_empty_prob ∈ [0, 1]`.

The "agent doesn't know about the tasks" is realized by **not adding any new
observation feature for task type**. The agent has to infer it from whether
the requested-bit is set anywhere.

## Random initial state

Shared by both task types. Built in `_place_pallets` and `_randomize_carriers`.

### Pallet counts (per-episode uniform sample over configured ranges)

Each episode draws fresh `big_ratio` and `small_ratio` values uniformly
from the configured ranges, then computes deterministic counts from
those samples:

```
big_ratio   ~ Uniform(big_ratio_range[0],   big_ratio_range[1])
small_ratio ~ Uniform(small_ratio_range[0], small_ratio_range[1])
big_count   = round(max_big_capacity * big_ratio)
remaining   = total_capacity - big_count
small_count = round(remaining * small_ratio)
empty_count = total_capacity - big_count - small_count
```

Where:
- `max_big_capacity` = sum of `shelf.capacity` over all big shelves.
- `total_capacity` = sum of `shelf.capacity` over all shelves.

**To pin a ratio to a fixed value**, set the range's low and high equal
— e.g. `big_ratio_range=(0.5, 0.5)`. To explore the full distribution,
set `(0.0, 1.0)`. To restrict to a band, set e.g. `(0.2, 0.8)`.

**Invariants (per episode, after sampling):**
- `big_ratio = 0` ⇒ 0 bigs that episode.
- `big_ratio = 1` ⇒ all big-shelf slots hold a big item.
- `small_ratio = 0` ⇒ 0 smalls; remaining slots are all empty.
- `small_ratio = 1` ⇒ no empties left (every non-big slot is a small).
- `big_ratio = 1, small_ratio = 1` ⇒ **zero empty pallets in the facility**
  (so a bring-empty draw will fall back to retrieve — see edge cases).

The sampled values are exposed via `info["episode_big_ratio"]` and
`info["episode_small_ratio"]` so trainers can log the distribution.

### Pallet placement (uniform random)

After deciding counts:
1. Run `shuffle_state(fullness=0)` → wipes carriers/rooms/scheduler and
   distributes all pallets to shelves with **random within-shelf order**, all
   contents=empty.
2. Collect all `(shelf_id, stack_index)` positions, partitioned by the
   shelf's size class.
3. Shuffle big-shelf positions; first `big_count` of them become content=big.
4. The leftover big-shelf positions go into a small/empty pool together with
   all small-shelf positions. Shuffle that pool; first `small_count` become
   content=small.
5. Everything else stays empty (from step 1).

Within-shelf stack order comes entirely from step 1's random shuffle —
content assignment only overwrites the `contents` field at chosen positions,
never reorders the stack.

### Room initial state (categorical per episode)

The room is 1-capacity. Each episode samples its initial load
categorically over three states, default 1/3 each via
`room_state_probs=(p_empty, p_small, p_big)`:

| Sampled state | Room.load | Net change vs all-empty room |
|---|---|---|
| `empty` | `None` | none |
| `small_item` | a Pallet with `contents="small"` | one empty pallet leaves the shelves; pallet count preserved |
| `big_item` | a Pallet with `contents="big"` | one empty pallet leaves the shelves; pallet count preserved |

**Conservation invariant.** When small/big is drawn, the env picks *any*
empty pallet across all shelves uniformly at random, reissues its ID
into the room with `contents` set to small/big, and removes it from
the source shelf. The relative order of pallets above the chosen empty
is preserved — `list.pop(idx)` is the atomic equivalent of "pop above
into a buffer, pop the empty, push buffer back." No gaps; LIFO holds.

The total pallet count in the system (shelves + room + carriers) is
identical to an empty-room episode of the same ratios.

**Fallback.** If no empty pallet exists on any shelf at sampling time
(e.g. both ratios collapsed to 1.0), the draw silently flips to
`empty`. The trainer can detect this from `info["episode_room_state"]`
mismatched against the configured probs.

**Interaction with bring-empty task.** If the room starts filled with
an item, the agent must clear it (`filled → free` transition, no
reward) before placing an empty (`free → empty` → success). That
adds one mandatory clear-step to the bring-empty task whenever the
room sampled non-empty.

### Carrier positions (uniform random)

Per carrier, on every reset:

```
carrier.position ~ Uniform[carrier.min_pos, carrier.max_pos]
```

No constraints on which carrier ends up where; positions are independent.

## Retrieve target selection (retrieve task only)

Code: `_pick_retrieve_target`.

Stratified pick at the configured depth (`SingleTaskConfig.target_depth`).
Stack convention: `stack[-1]` = top (carrier-accessible). `stack[-1 - depth]`
is at depth `depth` from the top. Shelves with stack length ≤ `depth` are
skipped (no candidate from that shelf).

**Stratification is 50/50 between size classes, not weighted by shelf count.**

Example with `target_depth=1`, facility with 1 big shelf and 4 small shelves
where every shelf has ≥ 2 pallets:
- Big-shelf candidates pool: 1 pallet (depth-1 of the big shelf).
- Small-shelf candidates pool: 4 pallets (depth-1 of each small shelf).
- P(target ∈ big pool) = 0.5; P(target ∈ small pool) = 0.5.
- Conditional on pool, pick uniformly within the pool.
- ⇒ The single big-shelf depth-1 pallet is 4× more likely than any single
  small-shelf depth-1 pallet. This forces the agent to practice big-shelf
  retrieves with equal frequency to small-shelf retrieves regardless of
  shelf-class imbalance in the layout.

Fallback chain when stratification can't produce a candidate:

1. Both classes have candidates at depth → 50/50 between classes.
2. Only one class has candidates at depth → pick uniformly within that class.
3. No class has a candidate at the requested depth → pick uniformly from
   **any pallet at any depth across all shelves**. (Lets shallow facilities
   or aggressive depth values still run.)
4. No pallets exist at all in the facility → return `None`. Reset falls back
   to bring-empty if any empty exists; otherwise raises (pathological).

**Empty pallets are valid retrieve targets.** This is intentional: the
"empty pallet, small item, big item" universe was specified explicitly.

## Edge cases on `reset()`

- **Sampled `bring_empty`, no empties exist (ratios=1):** task is silently
  re-rolled to `retrieve`. The effective `bring_empty_prob` becomes
  approximate but every episode is feasible. (Alternative behaviors —
  forcing one empty into the state, or running the unsolvable episode to
  truncation — were considered and rejected; see commit history.)
- **Sampled `retrieve`, no item at requested depth in either class:** falls
  through to any-pallet-anywhere (see step 3 above).
- **Sampled `retrieve`, no pallets exist at all:** falls back to `bring_empty`
  if an empty exists; otherwise raises `RuntimeError` (pathological topology).

## Reward shape

Defined by `SingleTaskRewardConfig`:

| Term | Sign | Default | Fires when |
|---|---|---|---|
| `reward_success` | + | +10.0 | Episode succeeds (either task). Paid **once**; episode terminates the same step. |
| `penalty_wrong_item_to_room` | − | −5.0 per event | Agent places a filled, non-target pallet at a free room. Active in both tasks. |
| `penalty_idle_with_retrieve` | − | −1.0 per step | A `Retrieve` is pending AND no carrier is mid-command. Catches WAIT-spam during retrieve task. Never fires during bring-empty (no retrieve queued). |
| `movement_weight` | − | −0.01 × distance | Per-millimetre carrier travel each step. |

What was **dropped** from the legacy reward (`oos/env/reward.py`):
- `reward_retrieve`, `reward_stage_room`, `penalty_unstage_room` — unified
  into `reward_success`. The legacy "stage means task done in phase 1, but
  not phase 2" distinction goes away because each episode is one task.

### Why not just keep the legacy reward function?

Two reasons:
1. **Double-counting on retrieve success.** When the agent delivers the
   retrieve target to a room, the auto-serve transitions the room
   `free → empty` within the same advance window. With the legacy reward,
   that would fire both `reward_retrieve` (target served) AND
   `reward_stage_room` (no-retrieve-pending gate passes post-serve). The
   subclass dodges this by passing `_ZERO_REWARD_CFG` to the base and
   recomputing reward itself, where `reward_success` pays once on the
   detected success regardless of which event triggered it.
2. **Spec clarity.** Configs map to user intent. "I want one success
   reward" maps to one config field.

## Carrier-position randomization (every episode)

Always applied on `reset()`. There is no knob to disable it — the agent
should always train on randomized starting positions per the spec.

## Implementation notes

- `SingleTaskEnv` subclasses `OOSEnv`. It inherits the observation builder,
  action decoder, scheduler advance, and event bookkeeping.
- The base env is passed `_ZERO_REWARD_CFG` so its `compute_reward` returns
  0 every step. The subclass re-derives reward from `info` in its own
  `step()`.
- `info["movement_distance"]` (added in `oos/env/env.py`) exposes the
  per-step travel distance the base env already computes internally, so
  the subclass can compute the movement penalty without duplicating the
  position-snapshot logic.
- Task state is in `self._task` ∈ {"retrieve", "bring_empty"} and
  `self._target_id`. Both populated on `reset()` and surfaced in
  `info["task"]` / `info["target_pallet_id"]`.

## What's NOT in this env (deliberately)

- **No curriculum scheduling.** The ratios and target_depth are static for
  the lifetime of the env. If you want a curriculum, sample these
  externally and rebuild the env (or add `set_task_config()` later).
- **No store stream / Poisson arrivals.** Auto-arrivals are disabled in
  `reset()`. The only `Retrieve` queued is the one we add for the retrieve
  task; nothing else generates work.
- **No multi-task episodes.** One goal per episode, end of story.
- **No partial-credit on retrieve.** Delivering the wrong pallet to the
  room doesn't terminate; it pays the wrong-item penalty and the episode
  continues until the target arrives or `max_steps` hits.
