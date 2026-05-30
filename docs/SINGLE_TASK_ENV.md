# SingleTaskEnv — design notes

Code: [`oos/learn/single_task_env.py`](../oos/learn/single_task_env.py),
initial-state generation in
[`oos/sim/state_sampler.py`](../oos/sim/state_sampler.py).

A "one goal, one episode" environment. Each episode is exactly one atomic
task. **Every knob is an explicit value** — the sampler defines one point in
hardness-space, not a distribution. A higher layer (a curriculum or
regret-based scheduler) can sweep these values to generate variety; that is
not this env's job. The only randomness in an episode is *placement* (which
slots, within-shelf tie-breaks, carrier positions, which eligible pallet is
the target).

## Two task types (explicit, `SingleTaskConfig.task`)

| Task | Observation hint | Success condition | Termination |
|---|---|---|---|
| **retrieve** | One pallet has `slot_pallet_requested=1` (a `Retrieve` is queued for it) | A `Retrieve` completion fires with `task.pallet == target_id` | `terminated=True` on success |
| **bring_empty** | No pallet has `slot_pallet_requested=1` | The submitted action is `WAIT` while a room holds an empty pallet and no `Retrieve` is pending | `terminated=True` on success |

The agent isn't told the task type directly — it infers it from whether any
requested-bit is set. If `task=bring_empty` is configured but the sampled
state has no empties at all, the env switches to retrieve to keep the episode
feasible.

## Initial-state model (sequential: occupancy → content → ordering → room)

The total pallet (tray) count `N` is **fixed** — whatever the facility was
seeded with. The sampler never creates or destroys pallets; it redistributes
them and re-labels contents. Let `B` = total big-shelf slots, `S` = total
small-shelf slots.

### 1. Occupancy — `big_shelf_fullness`

`trays_on_big = round(big_shelf_fullness * B)`, clamped so the remainder fits
on small shelves (`trays_on_small = N - trays_on_big`). Big-shelf air =
`(1 - big_shelf_fullness) * B` is the **eviction headroom** — the dominant
retrieve-hardness lever. "Fullness" here is physical slot occupancy: a slot
holds a pallet (of any contents) or is air.

### 2. Content — `big_ratio`, `system_fullness`

- `n_big = round(big_ratio * trays_on_big)` — `big_ratio` is the big-item
  saturation of the *occupied big-shelf slots*, so the same value means the
  same big-shelf congestion on any facility (independent of total slot
  counts).
- `n_small = round(system_fullness * (N - n_big))` — `system_fullness` fills
  the non-big trays with smalls; the rest (`n_empty`) stay empty.

Bigs land on big-shelf trays, smalls on any remaining trays. There is no
separate `small_ratio`.

### 3. Ordering — `big_disorder`, `small_disorder`

Layers 1–2 fix the content multiset on each shelf; the two disorder knobs set
the within-stack order. Depth convention: depth 0 = top of stack (shaft side,
accessible); deeper = buried. `0` = the ordered state where larger items are
**most accessible**.

- `big_disorder ∈ [0,1]` — fraction of big items buried **deeper** than the
  smalls/empties on the same shelf. `0` = bigs shallowest (most accessible).
- `small_disorder ∈ [0,1]` — fraction of small items buried deeper than the
  empties on the same shelf. `0` = smalls above empties.

At `(0,0)` a stack reads top→bottom as **big, small, empty**; at `(1,1)` as
**empty, small, big**. Implemented by ranking each pallet (big=0, small=1,
empty=2; a buried big → 3, a buried small → 2.5) with a random tie-break,
then sorting deepest-first. See `_order_by_disorder`.

### 4. Room — `room_state` (`empty` | `small_item` | `big_item`)

For small/big, one empty pallet is pulled off the shelves and re-issued as the
room load (pallet count preserved). Falls back to `empty` if no empty exists.

### 5. Carriers

Each carrier's start position is drawn uniformly over its track. Always
applied; no knob to disable.

### Solvability

If `require_solvable=True` (default) the placement is retried (up to
`max_solvable_retries`, default 50) until `_layout_is_solvable` passes — the
same retrievability check the live Store-gate uses. If the budget is
exhausted, the layout is **repaired** instead: big items are converted to
empty pallets, shallowest (depth 0) first, until the check passes (worst case
every big becomes empty, which is trivially solvable).

`info["episode_*"]` exposes the realised `big_shelf_fullness`,
`system_fullness`, `big_ratio`, `room_state`, `target_depth`, and
`retrieve_from` for logging.

## Retrieve target selection (retrieve task)

Three explicit target axes: shelf class (`retrieve_from ∈ {big, small}`),
delivery route (`retrieve_route ∈ {direct, handoff}`), and depth
(`target_depth`, 0 = top; `stack[-1 - depth]` is the pallet at that depth).

`retrieve_route` is derived from topology by `_route_class_map` — a
multi-source BFS over the handoff graph from the room-serving carriers gives
each shelf its minimum handoffs-to-a-room: `direct` (0, the shelf's carrier
serves a room) or `handoff` (≥1, the carrier has no room so the pallet must
be handed off — a distinctly harder retrieve).

To land a target at *exactly* the requested depth on a shelf matching
(`retrieve_from`, `retrieve_route`), `reset()` **re-samples layouts** rather
than picking a random depth (code: `_sample_retrieve_layout`):

1. Try up to 50 fresh layouts at the requested depth, looking for a candidate
   on a shelf of the requested class.
2. If none has one, step the depth down by one and try 50 more.
3. Continue to depth 0. The realised depth is surfaced as
   `info["episode_target_depth"]`.

Only if the requested class has no pallets at *any* depth across every attempt
does it fall back — to bring_empty if an empty exists, else a retrieve over
any pallet (last resort, pathological config).

Empty pallets are valid retrieve targets — the `{empty, small, big}` universe
is intentional.

## Reward shape

Defined by `SingleTaskRewardConfig`:

| Term | Sign | Default | Fires when |
|---|---|---|---|
| `reward_success` | + | +10.0 | Episode succeeds (either task). Paid once; episode terminates the same step. |
| `penalty_wrong_item_to_room` | − | −5.0/event | Agent places a filled, non-target pallet at a free room. |
| `penalty_idle_with_retrieve` | − | −1.0/step | A `Retrieve` is pending and no carrier is mid-command. Never fires during bring_empty. |
| `time_weight` | − | 0.0/sim-sec | Per-sim-second penalty on non-success steps (the success step is exempt so a long-dt WAIT can't swamp `reward_success`). |
| `movement_weight` | − | −0.01 × dist | Per-millimetre carrier travel each step. |

The base `OOSEnv` is passed an all-zero `RewardConfig`, so its `compute_reward`
returns 0; the subclass re-derives reward from `info` in its own `step()`,
where `reward_success` pays once on the detected success regardless of which
event triggered it (avoids double-counting the auto-serve `free→empty`).

## What's NOT in this env (deliberately)

- **No per-episode distributions.** Every knob is a fixed value; sampling over
  them is a higher layer's job.
- **No store stream / Poisson arrivals.** Auto-arrivals are disabled in
  `reset()`; the only `Retrieve` queued is the one for the retrieve task.
- **No multi-task episodes.** One goal per episode.
- **No partial credit on retrieve.** Delivering the wrong pallet pays the
  wrong-item penalty and continues until the target arrives or the cap hits.
