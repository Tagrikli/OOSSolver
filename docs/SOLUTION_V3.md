# SOLUTION V3 — Plan-Based Classical Solver

**Status: IMPLEMENTED (2026-07-03).** The solver lives in
`oos/plan/planner.py` (RetrievalPlanner), `oos/plan/solver.py`
(PlanSolver), `oos/plan/runtime.py` (standalone runtime + watchdog), and
`oos/plan/battery.py` (this document's §6 battery, runnable as
`python -m oos.plan.battery --gate all`). §8 below records what the
implementation learned beyond this handoff. The viz's `(classical solver)`
dropdown now drives the V3 solver.

This document was the complete handoff for a fresh
implementation session. It distills everything learned across the V2
campaign (move-level RL) and the V2.5 experiment (rule-cascade classical
solver): the physics, the assets worth keeping, the named root cause of
every hard failure, the V3 design, and the acceptance battery that a V3
implementation must pass before anything else is declared done.

The direction is decided: **deterministic classical solver, no RL.** The
operator wants a system that runs continuously, never gets stuck, never
fails to deliver a requested car, keeps rooms staged when at rest, does not
unstage rooms without need, and behaves identically and explainably on any
solvable layout — including hand-authored ones. Throughput is capped by
physics (~50–60 intakes/hour on campus-class layouts; symmetric delivery
rates) and that cap is accepted. Latency target: **head-of-queue delivery
in ~90–120 s** where air permits (see §6 for the honest conditional form).
Demand distributions are *evaluation scenarios, not training/design
targets* — the solver must be distribution-agnostic by construction.

---

## 1. The domain in one page (verified physics)

- **Topology**: rails with carriers — *lifts* (each serves exactly one
  room) and *shuttles*; shelves are LIFO stacks with a `size_class`
  (big/small) and integer capacity; carriers exchange pallets at handoff
  poses (rendezvous of a loaded + an empty carrier). Rooms are not storage:
  a pallet is "in a room" iff the room's serving lift is docked there
  holding it.
- **Pallets are conserved.** Every car rides a pallet; empties + cars =
  constant. `air(class) = slots(class) − pallets on that class's shelves`.
  A store consumes the staged empty (it becomes the car's pallet); a
  delivery returns one (customer drives off, empty remains with the lift).
  The empty-pool size is the capacity knob; class air is the scarce
  resource (big air especially).
- **Fullness is pool occupancy, not slot occupancy**: fullness = cars /
  pallets. The pallet pool is sized at facility definition so that shelf
  air exists even at fullness 1.0 — "enough pallets to solve any problem,
  no more." Consequence for the solver: **total air > 0 is an invariant
  of every reachable (and every validly seeded) state**, so "no air"
  is never an acceptable failure explanation. The only genuine physics
  corner is *class-local* exhaustion (big air 0 while bigs are buried),
  and the placement floor + big admission exist to keep it unreachable.
- **Staged** (AGENT_BEHAVIOR §5 rest state): serving lift docked at its
  room holding an empty. Stores are only admitted into staged rooms.
  §7.2: un-staging is legitimate only for in-flight need — one request
  should open one room, not five.
- **Store flow**: car arrives on the staged empty → lift carries the loaded
  pallet to a shelf. **Retrieve flow**: car's pallet is dug out and carried
  to an open (pallet-free) room; customer departs; empty remains → room is
  staged again as a side effect. Dwell samplers schedule each stored car's
  own future retrieve.
- **Admission gating** (keep as-is): smalls always admissible (net-zero
  service cycle); bigs strictly gated by the oracle at arrival so every
  stored big remains retrievable. Deployment-side gate exposed to the viz.

## 2. Assets that are proven and stay

| Asset | Where | Evidence |
|---|---|---|
| `SolvabilityOracle` — exact closed-form retrievability + admission counting | `oos/plan/oracle.py` | differential-validated 100% vs exhaustive DFS |
| Carrier role scripts: goto/take/give/send/recv, atomic all-free chain claims, exclusive src/dst shelf locks, single-authored rendezvous | `oos/env/moves.py` (`MoveExecutor`) | zero executor-level corruption across every campaign |
| Rule cascade (priority-rung dispatcher) | `oos/plan/classical.py` (v2.5 reference) | mass-drain 164-cars-all-requested at 192/h; full day-cycle pass; 96/100 dibaji@1.0 |
| Placement invariants: never bury a requested car; depth-k blocker cap; class-air floor; top-empty reserve | `classical.py::dst_score` | fixed the deep-SUV dead state by prevention |
| FIFO request window (show the policy/planner only the K oldest requests; K≈5–10) | `move_env.py` obs window | broke the mass-rush-out collapse |
| Liveness triggers (no-completion-for-N-minutes ⇒ escalate) | env valves | rescued every freeze; keep as watchdog telemetry in V3 (should never fire) |
| Gated admission engine | `move_env.py::GatedEngine` | wedge-free continuous operation |
| Viz seam: `PolicyFn` bridge + `(classical solver)` dropdown entry | `oos/viz/move_bridge.py::ClassicalPolicyBridge`, `session.py` | user-tested |

The **day-cycle harness** (`oos/learn/day_cycle_eval.py`) and the test
scripts listed in §7 are the evaluation suite. The move-level RL stack
(nets, PPO, checkpoints) is retired as a driver; keep `runs/` as baselines.

## 3. Root cause of every hard failure (the named ceiling)

**The Move grammar is too small.** A `Move` is atomic
`src(shelf-top|carrier) → dst(shelf|room)` with the destination locked at
start and no intermediate state. Two consequences:

1. **No temporary spaces.** Plans that the oracle *counts* as feasible —
   hold a blocker airborne on a carrier while the target's slot frees
   ("lead-blocker own-air"); use a room as a short-lived buffer with a
   round-trip; chain into slots that only free mid-plan — cannot be
   expressed. Rooms cannot even be a move *source*, so a room-buffered
   pallet is unreachable. Evidence: the seeded state with big-air = 0 and
   every big-shelf top a big car is *provably* undeliverable in this
   grammar while the oracle (max_holds=1) counts it solvable.
2. **Chooser limit-cycles.** Any *reactive* selector over single moves —
   greedy RL argmax or rule rungs — can circle in tight-air multi-blocker
   digs: 4/100 dibaji@1.0 trials stalled in `extract`/`hold` *with air
   still available*. All four RL freeze modes and both cascade livelocks
   of the V2.5 campaign are instances of the same class.

Everything else that went wrong was scaffolding to compensate for these
two facts. Stop compensating; fix the layer.

## 4. V3 design — plan, then execute

### 4.1 Retrieval planner (the core new piece)
For the head-of-queue request, compute a **complete extraction schedule**
before moving anything:

- Inputs: target stack, blocker list (classes), per-class air map, open /
  staged rooms, idle carriers, travel/op durations (`LinearDurations`).
- Search: the state is tiny (≤ ~5 blockers × a handful of candidate
  destinations + ≤ n_rooms buffers + ≤ n_carriers holds). Depth-bounded
  DFS/branch-and-bound with the oracle's counting as an exactness oracle
  and est-makespan as cost. Deterministic tie-breaks ⇒ reproducible plans.
- Vocabulary the planner may use (supersedes the Move grammar):
  - `RELOCATE(blocker → shelf slot)` — classic move;
  - `HOLD(blocker on carrier c until slot s frees)` — new;
  - `BUFFER(blocker → room r, round-trip back to a named slot)` — new;
  - `DELIVER(target → room r)`.
- Output: per-carrier **role scripts** (the executor's existing step
  language) plus two new step kinds:
  - `WAIT_SLOT(shelf)` — hold position/load until the shelf has air, then
    give (submit-when-ready in the pump; never submit a failing Give);
  - `TAKE_ROOM(room)` — lift takes the buffered pallet back out of its
    room (the primitive exists — store pickups do exactly this — it just
    was never reachable from a plan).
- **Resource contract**: the plan claims its carriers/locks up front
  (extend the existing atomic-claim discipline to plan scope); admission
  of new stores continues in parallel through unclaimed resources.

### 4.2 Dispatcher (the proven rungs, now emitting plans)
Same strict priority ladder, unchanged semantics, but each rung may emit a
multi-step plan instead of a single move:

`deliver-plan (head of FIFO window) > store placement > re-stage (§5) >
make-air / groom (load-gated)`

with the v2.5 hard rules kept verbatim: open rooms capped at
`min(pending_requests, n_rooms)`; unloaded serving lifts reserved for
delivery work only; anti-undo memory; placement scored by the invariant
table (§2). HOLD is only legal when no rung fires — and with a planner,
"no rung fires while a request waits" is a *bug by definition*, caught by
the watchdog.

### 4.3 Capacity contract (honest bounds)

- Total shelf air > 0 is already guaranteed by pool sizing (§1) — the
  planner gets no "no air" excuse, ever. **Completeness is unconditional
  on validly seeded layouts**: a finite plan exists for every request
  (oracle-checkable) and the planner must find it.
- Placement + admission additionally maintain the *class-local* margins:
  **class-air ≥ 1 whenever any car of that class is buried** (the
  SUV/big corner), and **≥ 1 free-able room** at all times. The only
  legitimate refusal in the system is the admission gate declining a
  store that would break these margins — surfaced in the UI; a retrieve
  is never refused and never fails.
- Head-of-queue latency: `T_head ≤ t_unstage + k·(t_extract + t_place) +
  t_travel_max + n_handoffs·t_handoff + t_dock` — all layout constants;
  on dibaji/campus geometry with k ≤ 3 this lands in the 90–180 s band
  (v2.5 measured median 72 s, p95 192 s at fullness 1.0).
- Under overload the guarantee degrades to FIFO queueing:
  `T(i) ≈ queue_position(i) / service_rate` — measured service ≈ 150–190
  deliveries/h in mass drain.

### 4.4 Explicit non-goals
No RL in the control path. No demand-distribution assumptions anywhere in
the solver. No per-layout tuning: everything derives from topology
constants at load time.

## 5. Traps (hard-won; each cost real debugging time)

1. Queued-but-unadmitted stores are invisible if you build views from
   admitted tasks only — starvation states *look* like rest. Any "am I
   idle legitimately?" check must consult the raw queue.
2. All empties can end up buried; then staging needs an uncover step
   first. (V2 freeze-2.)
3. Mass simultaneous retrieves must reach the solver through the FIFO
   window, and metrics must still use true arrival times.
4. §5 staging pressure vs delivery: a staged room is an *occupied* room.
   Deliver-enablement must un-stage exactly as many rooms as requests.
5. Reservation deadlock: reserving lifts for delivery must exempt the dig
   moves of the very target being served (lift-local burials).
6. Livelock via permissive fallbacks: when a restricted move-set is
   momentarily empty but work is in flight, WAIT — do not fall back to
   the full move-set (it hands reserved resources to junk moves).
7. Dispatch batching: several moves start per decision epoch; capacity
   counters must count *claimed* resources as consumed (a just-started
   park still looks staged).
8. Grooming fights physics above ~45% occupancy (avg stack depth > k+1).
   Load-gate it.
9. `TierSpec` seeding bypasses admission — seeded states can be legal yet
   outside the capacity contract; `_assert_solvable_start` uses oracle
   counting which includes plans the executor must actually be able to
   run (in V3 they finally can be).
10. Engine details: `PoissonTaskStream(rng, store_rate, size_mix)` (stores
    only; retrieves come from dwell samplers); stop arrivals with
    `engine.auto_arrivals_enabled = False`, never `task_stream = None`
    (a scheduled arrival event would hit an assert); `state` has no rooms
    dict — room contents live on the serving lift; pallets are uniform,
    only shelves have `size_class`.
11. Long evals: hourly progress lines + a no-completion watchdog, always;
    a silent 40-minute run is indistinguishable from a wedge.

## 6. Acceptance battery (all must pass before anything ships)

Fresh implementations must run these exact gates (scripts to be recreated
in-repo; v2.5 scratchpad versions were the prototypes).

**Early-abort rule for every gate**: a trial is declared STUCK the moment
no task completes for 3 sim-minutes while work is pending (plans in
progress extend the window by their own est-makespan). Never wait out a
fixed budget on a wedged run — wall time is evidence, not a fee. A stuck
trial must dump: last rung/plan, per-class air, legal-move count, carrier
states.

1. **dibaji @ fullness 1.0, capacity-5 SUV shelf, deepest item, 100
   seeds**: 100/100 delivered within the §4.3 contract (v2.5 cascade:
   96/100 — the 4 limit-cycles are the planner's reason to exist);
   budget 40 sim-min each, med ≤ 120 s.
2. **Deep-SUV seeded battery** (dibaji, fullness 0.85, 15 seeds): 15/15,
   med ≤ 90 s.
3. **Fill-then-dig** (fill through the solver to 0.85 with 35% bigs, then
   deepest-SUV): 10/10 — proves the placement contract keeps dig-ability.
4. **Mass drain**: 164 stored cars all requested at once (campus-class
   layout): 100% delivered, ≥ 150/h sustained, zero stalls.
5. **Day cycle** (campus, pallet_frac 0.87, target 0.8): fill from empty
   through morning rush → operate ≥ 0.8 with churn → rush-out → fully
   drained by night; zero stalls, zero watchdog escalations.
6. **7 consecutive day-cycles**, one continuous run, no resets: no drift
   in any daily metric, facility returns to clean rest nightly.
7. **Layout sweep**: every facility in `oos/facilities/` + operator-authored
   layouts: batteries 1–5 pass unmodified (bounds recomputed per layout).
8. **Viz manual regression** (operator's own failure reports): single
   request un-stages at most one room; no visible move loops ever; deep
   digs at high fullness deliver or the UI shows the capacity refusal.

## 7. Files superseded by V3

- `oos/plan/classical.py` — v2.5 rule cascade; reference for rung
  semantics and invariant scoring; delete after battery passes.
- Move-level RL training/eval stack (`oos/learn/move_*`,
  `train_move*`, checkpoints) — retired from the control path; keep as
  baseline generators for comparison plots if desired.
- The env-side liveness valves in `move_env.py::_refresh_legal` — became
  scaffolding for a chooser that no longer exists; V3 keeps only the
  watchdog *telemetry* (escalation should be unreachable).
- `MoveExecutor` stays, extended with `WAIT_SLOT` / `TAKE_ROOM` steps and
  plan-scoped claims.

## 8. Implementation notes (what the build taught us, 2026-07-03)

The design above held. Deviations and additions, each earned by a failing
seed:

- **WAIT_SLOT / TAKE_ROOM were not needed as executor steps.** Submit-when-
  ready lives one layer up: plan intents start only when their preconditions
  hold *right now* (pallet accessible, chain hands-free, destination
  fundable), so no illegal primitive can ever be submitted. Carrier-source
  moves already cover taking a buffered pallet onward. The one true executor
  extension is the HOLD move (`dst_kind="carrier"`), plus a claim-time
  anti-ping-pong stamp clear and a role-sync inside `inflight_effects`
  (stale `popped` flags otherwise crash the admission gate mid-advance).
- **The planner is a virtual simulation, not a search.** Working stacks +
  air ledger + carrier-hands timeline; disposal vocabulary = scored real
  air → HOLD on a spare carrier → extraction. Extractions mirror the
  oracle's closure exactly: eager (before disposals can eat hop space),
  need-based, with **return-hops** and **own-air hops onto the dig shelf**
  (they become new blockers the loop re-pops). Every plan is validated by
  an end-state oracle check before activation.
- **Per-shelf stack-op sequences** (`src_seq`/`dst_seq`, assigned by the
  sim) are the execution-ordering mechanism. A blanket pop-before-push
  rule self-blocks matched hop/return pairs; per-shelf order enforcement
  is deadlock-free because the sim's emission order is one consistent
  global schedule. Pops count at TAKE-completion, so digs pipeline.
- **Scoring hierarchy** (do not reorder casually):
  never-bury-protected (HARD, **head-window only** — protecting the whole
  queue deadlocks mass drains) > big-air starvation (RESERVE_BIG) >
  bury-a-protected-peer as last resort (BURY_PROTECTED — when refusing
  would make planning impossible on mutually-protecting stacks) >
  last-stageable-empty burial > depth-k (unmaintainable at high fullness;
  must never dominate correctness).
- **Plans beyond retrieval:** `plan_store` (recovers orphaned held cars by
  opening its own route) and `plan_stage` (staging = retrieving an empty;
  used when every empty is buried deeper than one uncover move). Staging
  runs regardless of demand — §5's rest is unconditional.
- **Full-state ruling (AGENT_BEHAVIOR §5.1):** staged rooms =
  min(rooms, free empties); at zero free empties a just-parked car stays
  on its lift (instantly deliverable); a full facility with queued stores
  is overload-quiescent rest, not a wedge.
- **Concurrency:** one plan per lift (`max_concurrent_plans` dial exists;
  measurements showed robustness comes from the reservation contract, not
  smaller batches — cap=1 costs ~74% drain time). Plan assignment scans
  past carrier-blocked FIFO heads (bounded) so clustered queues cannot
  head-of-line-block idle lifts. Mandatory-path hand-freeing iterates to a
  fixed point (a member's park may route through another member's hands).
- **Rung enumeration is TARGETED, never a full sweep.** The rungs use
  three narrow move families (store-from-carrier, stage-to-room, uncover);
  a full `iter_startable()` per tick is O(shelves²) oracle work and hits a
  wall-time cliff at high fullness. Staging moves (popping a TOP empty)
  provably cannot break the counting view — no oracle call at all.
- **The §6 stuck verdict is self-verifying**: when the completion-gap
  fires, the runtime grants one bounded probation window (120 s); any
  completion clears it. Coarse event sampling can span idle→work
  transitions in a single advance, and a raw gap check false-aborts
  exactly when a request lands after a quiet spell.
- **Battery results (final, all PASS):**
  gate 1 100/100 (med 99 s); gate 2 15/15 (med 85 s); gate 3 10/10;
  gate 4 164/164 at 163–170/h; gate 5 full day 397/397 stores+deliveries,
  drained to zero (wall 27 s); gate 6 seven consecutive days ~400
  deliveries/day, drained nightly to ≤1 leftover, zero stuck, no drift;
  gate 7 all 10 facilities. Unit layer: `tests/test_plan_solver.py`
  (+ executor fuzz, 57 tests total). `oos/plan/classical.py` deleted per
  §7.
