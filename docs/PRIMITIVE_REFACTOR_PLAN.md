# Primitive-Actions Refactor Plan

Replace the two macro actions (`Relocate`, `MultiRelocate`) with four primitives
(`GOTO`, `TAKE`, `GIVE`, `WAIT`), make rooms physical (delete `room.load`), and
do it in one coherent pass. Derived from a 9-subsystem audit; every file:line
below is verified against the current tree.

---

## 0. Locked design decisions (resolutions to the audit's open questions)

These are the "sane defaults" chosen up front so implementation is mechanical.

- **D1 — Handoff poses are NOT new graph nodes. A GOTO to a handoff pose is
  encoded as GOTO targeting the *partner carrier node*.** For a carrier `C` and
  partner `P`, `handoff_positions[(C,P)]` already uniquely identifies `C`'s pose,
  and each carrier has exactly one pose per partner. So GOTO targets =
  `accessible_shelves ∪ accessible_rooms ∪ handoff_partners` (partners addressed
  by their carrier node). This reuses the existing `[carriers | shelves | rooms]`
  node space with **zero new node type, zero offset shift** — avoids the
  high-blast-radius fork the audit flagged.

- **D2 — New `CarrierState.docked_at: Optional[DockRef]`.** `DockRef` is a small
  frozen value `{kind: 'shelf'|'room'|'handoff', id: str}` where `id` is the
  shelf id, room id, or (for handoff) the **partner carrier id**. Set on `GOTO`
  complete; cleared (`None`) on `GOTO` submit (in-transit) and at reset. This is
  the disambiguator for up/down shelves that share a track position — position
  in mm is NOT sufficient.

- **D3 — TAKE/GIVE are targetless scalar actions** (operate on `docked_at`). In
  the net they are scalar heads `[h_query ; global]`. `source_per_slot` and
  `partner_per_slot` are **deleted** from `Batch`; only `target_per_slot`
  survives, used by GOTO.

- **D4 — Carrier→carrier transfer = receiver-initiated, atomic at the transfer
  instant, no new lock field.** When taker `X` (empty, docked at handoff pose
  toward `Y`) executes `TAKE` while `Y` is WAITing at the matching pose holding
  an item: the `Take` command sets the **same command instance** on both `X` and
  `Y` (mirroring `MultiRelocate`'s dual-lock at facility.py:161-169), `busy_until
  = now + handoff()` on both. On complete: `Y.load → X.load`, both cleared, both
  woken. Reuses the `id()`-dedup (actions.py:126-136) and idempotent
  `_on_command_done` guard (facility.py:421-422). GIVE-to-carrier is never a
  chosen action.

- **D5 — Store/retrieve fire ON the `WAIT` primitive, never on `GOTO`-arrival.**
  `Facility.wait(carrier)` is the sole serve trigger: if the carrier is docked at
  a room, holds the matching load, and a compatible task is pending → swap the
  held pallet's contents (empty→item for a store; item→empty for a retrieve),
  record completion, schedule the dwell retrieve for a fresh store, then wake all
  waiting carriers. Task arrivals and post-serve state changes only **wake**
  waiting carriers (clear `waiting`), so they re-decide and a re-chosen `WAIT`
  performs the next serve. This makes the `WAIT` always the creditable action
  (per the reward intent) and uniformly routes every serve through one path.

- **D6 — `room.load` fully removed.** `RoomState` and `FacilityState.rooms` are
  deleted (rooms are static topology only). Room *nodes* in the observation
  remain (built from `topology.rooms`); their dynamic info comes from the queue
  + docked carriers. `_pallet_exists` scans shelves + carrier loads.

- **D7 — `must_relocate_from` deleted entirely** (no room-as-storage to clean
  up). Removes the field (state.py:51-61), the `forced_src` gate
  (action.py:118-126,161-165), the maintenance block (facility.py:444-458), and
  the shuffle wipe (shuffle.py:132).

- **D8 — Parked-car exploit is structurally gone.** With no `room.load`, a
  retrieve can only complete by a carrier physically delivering the target and
  WAITing. `agent_delivered` is always true; `n_free_deliveries` → always 0. The
  reward layer can drop the distinction (deferred).

- **D9 — `ActionType` renumbered:** `GOTO=0, TAKE=1, GIVE=2, WAIT=3`. Applied
  atomically across every `int(ActionType)` compare (batching.py:185-186, the
  network head selectors, and the env/sim_driver/single_task/episode WAIT
  checks).

- **D10 — Reservation guard kept, minimal.** A lightweight `_pending_take_count`
  / `_pending_give_count` over in-flight `Take`/`Give` on a shelf replaces the
  macro `_pending_src/dst_count`, used by the masker + preconditions so two
  carriers can't double-book a shared/transfer shelf. The observation shows pure
  **physical** state (no overlay); correctness lives in the mask.

- **D11 — Edge families:** delete `committed` / `in_flight_src` /
  `in_flight_partner` (macro residue); add `docked` (carrier → its docked
  shelf/room/partner node). Keep static `accesses` / `handoff` / `transfer`.

- **D12 — Move (viz-only free positioning) deleted** — never constructed
  anywhere (confirmed by inventory); GOTO subsumes node-targeted motion.

- **D13 — Rewards: minimal now, redesign later.** Keep `base_system` as the
  worked example. Strip only the terms/`StepEvents` fields that read the
  now-deleted `room.load` (WRONG, EVAC, STAGE, UNSTAGE, IDLE_ROOM and the
  room-transition counters). Emit a minimal event surface
  (`n_deliveries`, `n_stores_served`, `n_moves`) so training runs. Full reward
  redesign is a separate follow-up pass.

- **D14 — All existing `runs/*.pt` checkpoints are dead** (policy-head and
  EDGE_TYPES shapes change). Clean break; resume must start fresh; `policy_swap`
  surfaces the load failure as a clean toast.

- **D15 — `episode_code._VERSION` bumped.** The sampler's room handling + RNG
  draw order change, so old codes would silently decode to different layouts.

---

## THE load-bearing invariant (do not break)

`enumerate_actions` order  →  `ActionDecoder.mask()` (prefix)  →  per-slot
tensors in `batching.collate`  →  `network.forward` scatter  →  sampled flat
index  →  `ActionDecoder.decode(i)`  →  Command.

The ordering is authoritative **only** in `enumerate_actions`. Nothing else
re-derives it; alignment is by construction. The new canonical order is:

> all legal **GOTO** entries (one per reachable target node, in a fixed node
> iteration order) → **TAKE** (0 or 1) → **GIVE** (0 or 1) → **WAIT** (always 1,
> last).

A silent permutation here trains the wrong action with no error. **Mitigation:**
a single canonical ordering function + a unit test asserting, for a random state,
that `decode(i).type` equals the `type_per_slot[i]` the collator wrote, for all i
(see §3 tests).

---

## 1. Implementation phases (bottom-up; compiles only at the end — one pass)

### Phase A — sim state (`oos/sim/state.py`)
- Add `DockRef` frozen dataclass `{kind, id}`.
- `CarrierState`: add `docked_at: Optional[DockRef] = None`; add
  `last_take_give: Optional[tuple[str, DockRef]] = None` (for the no-immediate-
  inverse guard, see D-rules below); **delete** `must_relocate_from` (51-61).
- **Delete** `RoomState` (91-103). Remove `rooms` from `FacilityState` (106-111).
- Update `_pallet_exists`/`pallet_depth` neighbours to scan carrier loads instead
  of `room.load`.

### Phase B — sim commands (`oos/sim/actions.py`)
- **Delete** `Relocate` (224-320), `MultiRelocate` (328-473), `Move` (187-216),
  `_pending_src_count`/`_pending_dst_count`, and the room branches of the
  location helpers.
- Add `Goto(carrier_id, target: DockRef)`:
  - precond: `target` reachable (shelf.access / accessible_rooms /
    handoff_partners), carrier idle.
  - start: `dur = durations.move(carrier, pos, _dockref_position(target))`.
  - complete: set `cs.position`, `cs.docked_at = target`, `cs.last_take_give =
    None` (GOTO clears the inverse guard).
- Add `Take(carrier_id)`:
  - precond: `cs.load is None`, idle, and either (a) `docked_at.kind=='shelf'`
    with a non-empty stack and no pending take claiming the top, **or** (b)
    `docked_at.kind=='handoff'` with partner `Y` WAITing at its matching pose
    holding a non-None load. NOT for rooms.
  - start: `shelf_op("take")` for (a), `handoff()` for (b); for (b) dual-lock `Y`.
  - complete: pop docked shelf top → `cs.load`; or `Y.load → cs.load`, clear `Y`,
    wake `Y`. Set `cs.last_take_give = (TAKE, docked_at)`; for (b) also set
    `Y.last_take_give = (GIVE, handoff→X)`.
- Add `Give(carrier_id)`:
  - precond: `cs.load is not None`, idle, `docked_at.kind=='shelf'`, the shelf
    `accepts(load.size_for_shelf)`, has capacity (minus pending gives), and NOT
    the immediate inverse of a `last_take_give==(TAKE, this shelf)`.
  - start: `shelf_op("give")`. complete: push `cs.load` onto docked shelf;
    `cs.load=None`; `cs.last_take_give=(GIVE, docked_at)`. GIVE targets only shelves.
- Add `_pending_take_count(shelf)` / `_pending_give_count(shelf)` (D10).
- Rewrite `short_action_label` for goto/take/give/wait.
- Add `_dockref_position(DockRef, carrier, topo)` resolving shelf/room/handoff →
  mm position (`handoff_positions[(C, partner)][0]` for handoff).

### Phase C — sim engine (`oos/sim/facility.py`)
- `submit` (153-170): dual-lock branch becomes "Take resolved to partner-
  transfer" instead of `isinstance(MultiRelocate)`.
- `_on_command_done` (415-458): drop the Relocate/MultiRelocate auto-serve +
  `must_relocate_from` blocks; clear both carriers for a partner-transfer Take;
  no room auto-serve here (serve is WAIT-triggered, D5).
- `wait` (227-231): after `waiting=True`, run the new serve check (D5): if docked
  at a room with matching held load + pending compatible task → consume/fill,
  append completion (tag `agent_delivered`), schedule dwell retrieve for stores,
  then `wake_waiting_carriers`.
- Rewrite `_try_auto_serve_room` / `_scan_all_for_auto_serve_rooms` (585-665) to
  operate on "carrier docked+waiting at room R with relevant load" instead of
  `rs.load`. Called from `wait` (for the acting carrier) and on task arrival (to
  wake candidate carriers). Remove the `_pending_src_count` skip.
- `_initial_state` (120-139): drop `RoomState` construction.
- Scheduler/durations/motion: unchanged.

### Phase D — DSL/topology (`oos/sim/topology.py`, `oos/dsl/*`)
- No structural change required: handoff poses already declared per-carrier;
  up/down shelves already distinct `ShelfId`s; rooms unchanged. **Verify only.**
- `validate.py`: reword the "no empty seeded → rooms cannot be staged" message
  (196-199); keep orientation-aware slot uniqueness + chain-depth.
- Optionally add a `Topology` helper to resolve a `DockRef` → position (or keep
  it in actions.py).

### Phase E — action enumeration (`oos/env/action.py`)
- `ActionType` → `GOTO=0, TAKE=1, GIVE=2, WAIT=3`.
- `ActionEntry` → `{type, target: Optional[DockRef]}` (GOTO sets target;
  TAKE/GIVE/WAIT leave None). `to_command` builds `Goto/Take/Give`; WAIT still not
  a Command.
- Rewrite `enumerate_actions` in the canonical order (§invariant):
  - GOTO: for each target in `accessible_shelves + accessible_rooms +
    handoff_partners`, gated by the **room-GOTO mask** (D-rule R1) and reachability.
  - TAKE: 0/1 per preconditions (incl. pending-take claim + no-inverse guard).
  - GIVE: 0/1 per preconditions (size/capacity via `Shelf.accepts` + pending-give
    + no-inverse guard).
  - WAIT: always, appended last.
- **Delete** `_room_dst_allowed`, the `forced_src`/`must_relocate_from` block,
  `_top_pallet`, and any now-unused helper.
- Rewrite `max_actions_per_carrier` = max over carriers of
  `|accessible_shelves| + |accessible_rooms| + |handoff_partners| + 3`.

**Masking rules (sane defaults, confirmed with user):**
- **R1 — GOTO(room)** legal iff the carrier holds an **empty pallet** OR the
  **requested retrieve target**. Empty-handed (load None) and non-requested
  small/big loads are masked.
- **R2 — no immediate inverse:** mask `GIVE→L` if `last_take_give==(TAKE,L)`;
  mask `TAKE→L` if `last_take_give==(GIVE,L)`; for partners, mask the reverse
  transfer. Cleared by GOTO, **not** by WAIT.
- **R3** — never produce an all-`-inf` mask: WAIT always present.

### Phase F — observation (`oos/env/observation.py`)
- **Delete** `compute_in_flight_overlay` + `_peek_src_pallet`; replace `eff_load`
  with `cs.load`, `eff_depth/eff_stack` with `ss.depth/ss.stack`.
- `CARRIER_FEATURE_NAMES`: append `docked_none, docked_shelf, docked_room,
  docked_handoff, at_handoff_with_item` (indices 11-15). Keep 0-10 as-is.
- `ROOM_FEATURE_NAMES`: replace `has_load/load_empty/load_small/load_big` with
  `has_pending_store, has_pending_retrieve, empty_staged_here,
  target_staged_here` (all computed from queue + docked carriers; explicit so the
  `n_gat_layers=0` path still sees staging).
- Edges: delete `edges_committed/in_flight_src/in_flight_partner` and
  `_target_node`; add `edges_docked` (carrier→docked node) using `_location_node`
  extended for the partner-carrier case. Keep `accesses/handoff/transfer`.

### Phase G — batching (`oos/learn/batching.py`)
- `EDGE_TYPES`: drop committed/in_flight_* (+ `_rev`); add `docked` + `docked_rev`.
- `Sample`: swap the three deleted edge arrays for `edges_docked`.
- `Batch`: keep `type_per_slot`, `target_per_slot`; **delete** `source_per_slot`,
  `partner_per_slot`.
- Collator inner loop: `target_per_slot[i] = node(entry.target)` for GOTO only
  (resolve DockRef → shelf/room/**partner-carrier** node); TAKE/GIVE/WAIT leave 0.
- Update `_edge_spec/_edge_attrs` and docstrings; keep `sample_from_env_step`
  reading the new edge key set.

### Phase H — network (`oos/learn/network.py`)
- Replace `TARGETED_ACTION_TYPES` and the `action_heads` ModuleDict: GOTO =
  single-pointer head `[h_query ; h_target ; global]` (2h+Fg); TAKE/GIVE/WAIT =
  scalar heads `[h_query ; global]` (h+Fg).
- `forward` (241-294): GOTO block gathers `target_per_slot` node embeddings and
  scatters; TAKE/GIVE/WAIT mirror the old WAIT block (`valid_mask & type==T`).
  Keep the `-inf` init + per-type scatter + `if sel.any()` guards.
- Value head unchanged. EDGE-type GAT layers auto-track `EDGE_TYPES`.

### Phase I — env (`oos/env/env.py`)
- `submit_action`: dispatch GOTO/TAKE/GIVE via `to_command`; WAIT via
  `facility.wait` (now serve-bearing).
- **Delete** the room.load transition accounting (447-547) and the `_potential`
  room reads (668-691).
- Emit the minimal `StepEvents` (D13). Keep `info["events"]`,
  `info["reward_breakdown"]`, `info["reward_events"]`, `info["completions"]`,
  `info["retrieves_completed"]`, `info["action_entries"]`, `info["action_mask"]`.
- Update the obs/info edge split + `_zero_obs` for the new edge set.

### Phase J — reward (`oos/env/reward_system.py`, `reward.py`)
- Delete WRONG, EVAC, STAGE, UNSTAGE, IDLE_ROOM terms and the room-transition
  `StepEvents` fields. Keep DELIVER, SERVE, MOVE(default 0), SUCCESS, TIME, IDLE,
  IDLE_RETR, SHAPE(pending_retrieve). Keep `base_system`; mark
  `continuous_system`/`single_task_system` for consolidation. (Deeper redesign
  deferred.)

### Phase K — training envs
- `continuous_env.py` (282-301): stop consuming removed room StepEvents fields.
- `single_task_env.py`: `bring_empty` success = "carrier docked+waiting at room
  holding an empty pallet AND WAIT" (replaces `_room_has_empty_pallet`,
  384-390); keep `ActionType.WAIT`.
- `episode_env.py`: move phase logic off `rs.load` (194-236); keep WAIT mask
  override + all-zero guard.

### Phase L — samplers / solvability / episode code
- `shuffle.py`: delete `must_relocate_from` reset (132) and `rs.load` reset
  (133-134); add `docked_at=None`, `last_take_give=None` to the wipe. Keep
  `_layout_is_solvable` logic **unchanged** (conservative, primitive-agnostic);
  refresh prose only.
- `state_sampler.py`: redesign `_apply_room_state` (352-381) to express the
  `room_state` knob as "the serving carrier starts docked+waiting at the room
  holding {empty|small|big} pallet"; drop the room-fold (266-271); fix
  `has_empty_pallet_anywhere` (453-456). Keep occupancy/disorder/solvability
  untouched. **Re-express the buffer-on-target benchmark** (378-380) as a carrier
  docked at the room holding the loaded pallet.
- `episode_code.py`: keep tables append-only; bump `_VERSION` (D15).

### Phase M — viz
- `animation.py`: delete `relocate_visual_state` + `multi_relocate_visual_position`;
  GOTO = single move (`interpolated_position`/`command_end_position` read the
  GOTO target via `location_visual_pos`); TAKE/GIVE/WAIT = stationary hold; drop
  macro imports.
- `facility_canvas.py`: replace command-type dispatch with primitive
  interpolation; remove `room.load` draw + ready/idle state; read `cs.load` /
  `ss.stack` directly (overlay deleted); animate **both** carriers as busy during
  a partner-transfer (detect via the shared `current_command`); primitive-aware
  action chip.
- `sim_driver.py`: update the WAIT/label branch + ActionType import; raise
  `MAX_ITERS_PER_FRAME` for the ~3–4× finer cadence.
- `distribution.py`: `_draw_tooltip` for `{GOTO target | TAKE | GIVE | WAIT}`.
- `components/widgets/room.py` + `carrier_panel.py`: room becomes a bare docking
  marker (no pallet slot). `shelf.py`: drop `set_hidden_count`. `carrier_icon.py`:
  unchanged (already draws held load).

### Phase N — runtime / exports / scripts
- `agent.py`: build labels for the 4 primitives; ensure a WAIT entry never calls
  `to_command`.
- `oos/sim/__init__.py`: drop `Relocate`/`MultiRelocate` exports; export
  `Goto/Take/Give` if surfaced.
- `scripts/test_seeded.py:131`, `scripts/diagnose_zero.py:187`: `env.action_space.n`
  → `env.n_actions`.
- `mcts.py`: no structural change; refresh docstring; consider `n_sims`/`gamma`
  retune for finer granularity (optional).

---

## 2. Cross-module contracts to keep (checklist)
1. Positional alignment GOTO…/TAKE/GIVE/WAIT (the invariant above).
2. `n_max = env.n_actions` is the single logit width; `ActionDecoder` raises if a
   carrier ever exceeds it — keep `max_actions_per_carrier` an exact upper bound.
3. WAIT always legal + appended **last** (sim_driver fallback, single_task peek,
   episode mask all assume it).
4. Carriers first in node space; `querying_carrier` = local carrier idx =
   concat idx; `is_querying` feature agrees.
5. Mask never all-`-inf` (NaN softmax).
6. One stored `Sample` fully regenerates the forward pass (PPO replay): new
   per-slot/edge fields must be pure functions of `Sample` contents.
7. Determinism/seeding: stable enumerate iteration order; vec_env worker seeds;
   episode-code reproducibility (hence the `_VERSION` bump).
8. `is_busy == (current_command is not None)`; a WAITing carrier is not busy and
   stays recruitable as a transfer partner.
9. Pallet conservation: a pallet formerly in `room.load` now lives on a carrier —
   conservation counting must follow it (`count_pallets` in tests).

## 3. Validation strategy
- **New test — index/type alignment:** for several random states, build the
  decoder + a collated Batch and assert `decode(i).type == type_per_slot[i]` for
  all `i < n_legal`, and that order is `[GOTO*, TAKE?, GIVE?, WAIT]`.
- **New test — handoff transfer:** giver WAITs holding an item at a pose, taker
  GOTO+TAKE → both end idle, item moved, both woken; no deadlock; no-inverse
  guard blocks immediate take-back.
- **New test — WAIT-triggered serve:** carrier holds empty at room, store queued,
  `WAIT` → held pallet filled + completion emitted; retrieve symmetric.
- **Update** `test_smoke` (count_pallets follows pallet onto carrier),
  `test_state_sampler::test_room_state_applied` (assert carrier-held, not
  `rs.load`), `test_reward_system` (prune dead terms).
- **Verify unchanged:** `test_shuffle_smoke`, route/depth sampler tests,
  `test_continuous_plr` topology/admission, `test_network_smoke` shapes.
- **Run:** `SDL_VIDEODRIVER=dummy uv run python -m pytest tests/ -q`, then a short
  `train_continuous` smoke (a few iters) + a viz boot with the random policy.

## 4. Items wanting a final nod before coding
- **D1** (handoff pose = partner-carrier node, no new node type) — the deepest
  structural choice; everything else follows from it.
- **D6** (full `RoomState`/`FacilityState.rooms` removal vs. keeping an empty
  placeholder) — removal is cleaner but touches more readers.
- **D5 chaining** — whether one `WAIT` may perform a retrieve-consume *and* a
  subsequent store-fill in the same instant, or each serve needs its own `WAIT`
  (default chosen: each serve needs its own WAIT, for clean per-WAIT reward).
