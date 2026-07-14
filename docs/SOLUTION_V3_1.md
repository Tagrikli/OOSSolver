# SOLUTION V3.1 — Revisions: Service Dwell, Charger Ops, Big-Air Groom, Concurrency-Aware Planning

**Status: SPECIFIED (2026-07-13), implementation in progress.** This
document is the formal specification for four operator-driven revisions to
the V3 plan solver ([SOLUTION_V3.md](SOLUTION_V3.md)). Each section states
the problem, the decided behavior, the invariants it must preserve, and
the acceptance criteria. The V3 battery remains the regression gate; §5
records the threshold adjustments the new physics (§1) requires.

Decisions in this document were made with the operator on 2026-07-13 and
are **not** open design questions.

---

## 1. Customer service dwell (room occupancy time)

### 1.1 Problem

V3 serves are **instant**: the same sim event that docks a lift at a room
with a requested car completes the Retrieve, converts the held pallet to
an empty, and leaves the lift staged and free. Symmetrically, a queued
Store is absorbed in the instant a staged lift is scanned. Real customers
take time to get into a delivered car and drive out, and to drive in and
park. Instant serves let the planner assume a lift is reusable the moment
it reaches a room — systematically optimistic for back-to-back operations
through the same room.

### 1.2 Decided behavior

- Every **Retrieve serve** occupies the serving lift at the room for a
  fixed **exit dwell** `serve_exit_s`; every **Store serve** occupies it
  for a fixed **entry dwell** `serve_entry_s`. Both are **fixed constants
  per facility** (no sampling — determinism is a design pillar).
- The dwell models the *customer interaction*: it begins at the instant
  the serve would previously have fired (all serve gates are consulted at
  that same instant, unchanged), and the state mutation (pallet contents
  flip, task completion, dwell-retrieve scheduling) happens at dwell
  **end**. During the dwell the lift is **busy** — docked, holding the
  pallet, unavailable to the executor, the rungs, and other plans.
- A serve, once started, is **committed**: the customer is physically
  present. Canceling the underlying task mid-dwell does not stop the
  serve (a canceled Retrieve mid-dwell still hands the car over; the
  completion is simply not recorded if the task was already removed).
- Task latency (`cost`) includes the dwell: a Retrieve completes when the
  customer has driven off, not when the car reached the room.
- `serve_exit_s = serve_entry_s = 45.0` by default for all DSL-built
  facilities; hand-built `Topology` objects default to `0.0` (exact
  backward compatibility for unit fixtures).

### 1.3 Where it lives

- `Topology.serve_exit_s` / `Topology.serve_entry_s` (static facility
  constants, compiled from the DSL `Facility(serve_exit_s=…,
  serve_entry_s=…)`).
- Engine: `_try_serve_at_room` starts a timed serve (carrier busy,
  `serve_done` scheduler event) instead of completing in place. A task
  being served is *in service*: it stays in the pending queue (so the
  solver still sees the request/store as live) but is excluded from serve
  matching, from the unservable-big sweep, and from cancellation.
- Planner/executor cost model: a delivery's makespan estimate includes
  `serve_exit_s` (the lift is not free until the customer leaves).

### 1.4 Invariants preserved

- **Future view correctness**: an in-flight delivery still removes its
  car from the view (it leaves the system — just later). No oracle
  change.
- **Serve-gate semantics**: gates run at serve *start* against the same
  state they used to see. A store that is not fundable still waits at the
  door; it never becomes an in-service store.
- **Liveness**: the dwell (45 s) is far below every watchdog window
  (120 s replan, 180 s stuck-gap); a serving lift is busy with a
  scheduled completion event, so the runtime never sees a quiet
  scheduler mid-serve.

### 1.5 Acceptance

- All battery gates pass with the adjusted bounds of §5.
- A delivery immediately followed by a store through the same room shows
  the store serve starting no earlier than `t_dock + serve_exit_s`.
- Zero-dwell topologies behave bit-identically to V3.

---

## 2. Charger-shelf service operations (Evict / Place)

### 2.1 Problem addition

Some shelves carry an **automatic EV charger**. Physically they are
ordinary shelves (regular size classes, LIFO stacks, same carrier
access); the charger serves **any slot**, so there is **no position or
burial constraint** on a charging car. An external policy (outside the
solver) rotates cars through charger shelves. The solver's job is only to
execute two new relocation primitives, with no knowledge of charging
semantics.

### 2.2 Decided behavior

Two new task kinds, issued by the external policy:

- **`Evict(pallet)`** — remove the specific car from wherever it is and
  store it at any acceptable ordinary placement (scored by the standard
  disposal table, oracle-gated). No room is involved; the car stays in
  the system. **The source shelf ends unchanged apart from the removed
  car** (operator revision, 2026-07-14): blockers above the target are
  never disposed permanently — each is HELD on a spare carrier or
  temp-hopped to another shelf, and all are pushed back in their original
  order once the target is out. (An empty restored to the top may
  afterwards be consumed by normal staging — empties remain
  infrastructure; *cars* stay put.) Big-air extractions may still run to
  fund the target's own landing.
- **`Place(pallet, shelf)`** — bring the specific car to the specific
  destination shelf, landing on top of its current stack. The
  destination's existing occupants are **untouchable**: the plan may not
  relocate them, may not pop the destination, and may not use the
  destination as temporary hop space. If the destination has no free slot
  (net of reservations), the solver returns **no solution** — it is the
  policy's job to first issue an `Evict` of a specific car from that
  shelf. The solver never manufactures air on the destination.
- A charger swap is one `Evict` + one `Place` with **no ordering
  constraint** between them; they interact only through air, which the
  reservation machinery already arbitrates.
- The moved car itself may be buried at its **source**; source-side
  digging is unrestricted (standard machinery).

### 2.3 EV-shelf deprioritization

Shelves gain a static `is_ev` flag. The placement scoring table gains

    EV_SHELF = 250.0   # soft: prefer any non-EV alternative

applied to every *scored* placement (disposals, stores, empty parking,
groom) onto an EV shelf. It ranks with `POLLUTE_BIG` — a pure preference,
decisively below every correctness cost, so EV slots stay available for
charge work but remain full citizens of the storage pool under pressure.
An explicit `Place` onto an EV shelf pays no penalty (the destination is
fixed, not scored).

### 2.4 Scheduling and priority

- Evict/Place tasks queue alongside Retrieves/Stores and are planned by
  the same plan machinery (plan kinds `"evict"` / `"place"`), with the
  same reservations, watchdogs, and end-state oracle validation.
- **Priority**: strictly below customer work. An Evict/Place plan is
  assigned only when every pending Retrieve in the head window already
  has an active plan (or none is pending). Customer retrieves never wait
  on charger rotation.
- Completion: the plan's terminal relocation completes the task (the
  solver removes it from the queue). A canceled task drops its plan;
  orphaned holds fall to the store rung as usual.
- A `Place` whose destination is full **fails fast** (no plan, noted in
  telemetry) rather than waiting; re-issuing after an `Evict` is the
  policy's responsibility.

### 2.5 Invariants preserved

- The end state of every evict/place plan passes `oracle.check_view` —
  charger rotation can never make the world unsolvable.
- The destination shelf of a `Place` is locked against foreign pushes for
  the plan's duration (its reserved slot is phantom-filled in everyone
  else's view), and its stack composition is never altered by the plan
  itself.
- "Never bury a protected request" applies to the dug blockers exactly as
  in retrieval plans.

### 2.6 Acceptance

- Evict of a buried car (including under SUVs on a big shelf, requiring
  an extraction) completes and leaves a solvable world.
- Place onto a shelf with one free slot lands the car on top with the
  original occupants in their original order.
- Place onto a full shelf returns no-solution without moving anything.
- Evict+Place swap between two shelves completes regardless of issue
  order.

---

## 3. Grooming, repurposed: proactive big-air decluttering

### 3.1 Problem

The V3 groom rung (idle-time depth-k tidying + empty uncovering) had no
shared termination potential across its move families and was observed to
shuffle in circles; depth-k tidying at that is cosmetic. Meanwhile a real
resource problem goes unaddressed: **non-big pallets (sedans, empties)
clogging big shelves suppress SUV admission**. The admission gate needs a
*raw* free big slot in the future view — the extraction closure is only
credited for the retrievability of cars already stored, not for admission
placement. A facility whose big shelves are physically full of non-bigs
refuses every arriving SUV even though idle-time relocations could open
slots.

### 3.2 Decided behavior

The groom rung is **replaced**. The new groom, running only when the
solver is otherwise fully idle (no plans, nothing else started this
tick — same slot as before, load gate removed). "Idle" tolerates one
kind of pending work: **big Stores the door currently refuses** (oracle
admission false, i.e. big air exhausted). Those customers are waiting
for exactly the air a declutter mints, so they must not lock out their
own remedy — six queued SUVs against zero big air starved the groom for
hours in the campus endurance run (day 20) before this carve-out.
Anything else pending (a small, a retrieve, an admissible big) still
silences the groom:

1. **Parks floating empties**: an unclaimed non-serving carrier stuck
   holding an empty (e.g. after a replan) gets one scored, oracle-gated
   parking move. (Kept from V3 — nothing else re-shelves those, and a
   permanently loaded carrier blocks every chain through it.)
2. **Declutters big shelves**: relocates one non-big pallet from a big
   shelf to a scored **small-shelf** placement per activation —
   - top-of-stack non-bigs move as single oracle-gated moves;
   - a non-big buried under SUVs is recovered with an **evict plan**
     (§2) restricted to small-shelf destinations, one at a time.
3. Refuses any declutter whose placement score reaches correctness
   levels: it must not spend the last small slot(s)
   (`CLASS_FLOOR`/`EMPTY_FLOOR`), must not starve the held-SUV extraction
   reserve (`RESERVE_BIG` via `small_need`), must not bury a protected
   request, and must not kill the staging pipeline (`TOP_EMPTY_LAST`).
   The depth-k term is a *preference*, not a correctness cost: the guard
   ignores it (ranking still prefers violation-free destinations), else
   the declutter stalls at moderate fullness.

Two additional rules, both learned from observed loops during
implementation (2026-07-13):

- **Staged lifts are untouchable to the groom.** A groom-evict plan must
  never park a staged room's empty to open its corridor: un-staging
  forces a re-stage whose uncover move pushes a small back onto the big
  shelf, undoing the declutter in a perpetual carousel. Rest-state
  infrastructure outranks tidying; big shelves reachable only through a
  staged lift simply wait.
- **The monotone potential is enforced on whole plans, not just single
  moves**: a groom-evict plan is committed only if its intents strictly
  reduce the non-big-on-big-shelves count — a plan whose own hand-freeing
  parks an empty onto a big shelf while extracting one nets zero and
  would loop.

A master switch (`PlanSolver.groom_enabled`) disables the rung wholesale.

### 3.3 Termination (replaces the depth-k potential)

Every declutter move strictly increases free big air (a non-big leaves a
big shelf for a small shelf; nothing the groom does moves pallets onto
big shelves), and free big air is bounded by total big capacity. The
floating-empty park strictly decreases the number of loaded idle
carriers. Both potentials are monotone and bounded ⇒ the groom cannot
loop. Depth-k tidying and idle empty-uncovering are **deleted** (the
depth-k *placement preference* in the scoring table stays — it shapes
disposals for free and generates no moves).

### 3.4 Stopping condition

Full declutter with guards: groom until no non-big on any big shelf has a
guard-passing move (rule 3 above). No headroom target parameter — idle
time is free and the guards prevent small-air starvation. (If a future
facility legitimately overflows sedans onto big shelves at high fullness,
revisit with a headroom cap; the store rung placing sedans back onto big
shelves scores `POLLUTE_BIG`+`EV_SHELF`-adjacent costs and will not
ping-pong against the groom guards.)

### 3.5 Acceptance

- A facility with big shelves full of empties/sedans and idle time
  declutters until SUV admission passes, without violating any guard.
- Rest state with nothing to declutter starts **zero** moves at any
  fullness (strictly stronger than V3's load-gated rest test).
- No groom sequence ever repeats a state (implied by the potential; the
  old operator loop report scenario must converge).

---

## 4. Concurrency-aware planning

### 4.1 Problem

Plan *execution* is submit-when-ready, but plan *construction* is
serial-minded, so committed plans often serialize physically parallel
work:

1. `_chain_free_sim` models hands at the plan's sequence point, so the
   planner freely picks disposal chains through carriers that are still
   loaded in the live world and only become free via an earlier intent of
   the same plan — the dig then waits on the hand-freeing even when a
   disjoint-chain alternative existed.
2. `est_cost` is a **serial sum** of per-intent makespans: a
   two-carriers-working-at-once plan scores the same as the same moves
   one-after-another, so nothing selects for parallel structure.
3. Per-shelf sequence numbers order **push-after-push** even when the
   two pushes commute, adding false execution-order constraints.

### 4.2 Decided behavior

- **Critical-path cost model.** `PlanSim` gains a per-carrier ready-time
  ledger and a per-shelf op-completion ledger. Every emitted intent is
  virtually scheduled: `start = max(ready(chain carriers), ready(shelf
  predecessors))`, `end = start + est`. The plan's `est_cost` becomes the
  schedule **makespan** (plus the existing flat structural surcharges),
  so candidate selection prefers plans that overlap work.
- **Wait-aware destination scoring.** The destination pickers add the
  candidate's *start wait* (how long its chain would idle before it can
  begin, per the ledger) to the score at a weight comparable to the
  chain-length penalty — a destination reachable *now* through one extra
  hop beats one that waits a minute for a carrier this plan still has to
  unload. This directly fixes the observed "everyone waits for the first
  carrier to empty its hands" pathology.
- **Push-push commutation.** `_shelf_ops_ready` lets a push onto shelf S
  skip waiting for an *earlier planned push* onto S when (a) the plan
  performs no later pop from S (composition above the old stack is then
  inconsequential to the plan's own schedule), and (b) both pushed
  pallets have identical contents class (the terminal per-position
  composition — what the end-state oracle validated — is then identical
  under either order). Pushes always wait for earlier pops; pops stay
  strictly ordered.

### 4.3 Invariants preserved

- The virtual schedule is a ranking device only; execution readiness
  remains live-checked, so nothing can start illegally.
- Sequence-number semantics stay deadlock-free: the relaxation only
  *removes* wait edges, and only where the terminal state is provably
  order-independent; the remaining edges are a subgraph of the original
  acyclic schedule order.
- Determinism: ledgers are computed from the same deterministic
  emission order; ties break as before.

### 4.4 Acceptance

- The motivating scenario — dig carrier free, delivery-chain member still
  holding its staging empty — starts the first disposal concurrently with
  the hand-freeing when a disjoint destination exists.
- Battery gate 4 (mass drain) throughput does not regress; single-dig
  latency medians (gates 1–2) do not regress beyond noise.

---

## 5. Battery threshold adjustments (dwell physics)

The V3 bounds were calibrated on instant serves. With `serve_exit_s = E`
and `serve_entry_s` per facility:

- **Gates 1–2 (single-dig latency)**: each delivery pays exactly one
  exit dwell ⇒ bounds become `med ≤ 120 + E` (gate 1) and
  `med ≤ 90 + E` (gate 2).
- **Gate 4 (mass drain rate)**: each lift's delivery cycle lengthens by
  `E`. With the V3 requirement `R₀ = 150/h` over `n` lifts, the adjusted
  requirement is `R = 0.95 · R₀ / (1 + R₀·E / (3600·n))` (for campus
  n=5, E=45: ≈ 104/h). The 0.95 factor is a measured
  secondary-contention allowance: a lift pinned at its room during the
  dwell also delays its region's staging and neighboring digs, which the
  per-delivery model ignores (~4% on campus; the zero-dwell rate is
  unchanged at ~148–150/h, confirming the solver itself did not regress).
- **Gates 5–7**: unchanged criteria (drain-by-night, zero stuck,
  delivered counts); dwell consumes existing slack.

These are physics corrections, not relaxations: the same solver work is
being measured with customer time added on top.

---

## 6. Explicit non-goals of V3.1

- No modeling of charger occupancy, charge levels, or rotation policy —
  the external policy owns all of it.
- No sampled service times; dwells are per-facility constants.
- No depth-k grooming; tidiness remains a placement-time preference only.
- No change to oracle semantics; all four revisions sit above it.

### 4.5 Staging prefetch (added 2026-07-13, operator report)

Atomic all-free chain claims mean a relay move cannot start while any
member is busy — so the shuttle leg of a re-stage used to wait for the
lift to finish shelving the just-parked car. The prefetch rung starts the
far leg early: while a room's lift is busy with short-horizon work
(including the entry dwell itself), a partner shuttle fetches the next
staging empty as a **HOLD parked at the handoff pose** (`Move.park_at`);
when the lift frees, only the rendezvous + room leg remain. One prefetch
per room; the stage rung defers to an in-flight prefetch instead of
racing it; skipped when the lift's own shelves have a top empty, when the
lift is mid-exit-dwell (the delivery re-stages the room by itself), when
a plan owns the room, or at zero free empties. The groom yields entirely
while any room is unstaged — a held empty is staging material.
