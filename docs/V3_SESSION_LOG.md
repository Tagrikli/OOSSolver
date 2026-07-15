# V3 implementation session log — 2026-07-03

One continuous session took `docs/SOLUTION_V3.md` from **proposed** to
**implemented, battery-certified, and viz-integrated**. This log is the
narrative record; the technical deltas live in `SOLUTION_V3.md §8` and the
behavior addendum in `AGENT_BEHAVIOR.md §5.1`.

## What was built

| Piece | File | Role |
|---|---|---|
| RetrievalPlanner | `oos/plan/planner.py` | Virtual-simulation dig planner: scored real-air disposals, HOLDs on spare carriers, the §9 apex extraction closure (return-hops + own-air hops), per-shelf stack-op sequencing, end-state oracle validation. Also `plan_store` (orphan recovery) and `plan_stage` (staging = retrieving an empty). |
| PlanSolver | `oos/plan/solver.py` | Dispatcher: plans + the v2.5 rungs under plan-scoped reservations (lift, dig carrier, holders, shelves, air slots); targeted rung enumeration; admission + serve gates; keep-on-lift full-state rule. |
| SolverRuntime | `oos/plan/runtime.py` | Standalone loop, §6 stuck watchdog (idle-rebaselined, probation-verified), seeding helpers, metrics. |
| Battery | `oos/plan/battery.py` | The §6 acceptance gates as a CLI: `python -m oos.plan.battery --gate all`. |
| Executor extension | `oos/env/moves.py` | HOLD moves (`dst_kind="carrier"`), claim-time transfer sanctioning, role-sync in `inflight_effects`, plan view hooks. |
| Engine hooks | `oos/sim/facility.py` | `admission_check` / `store_serve_gate` on the base engine (the solver's oracle gates apply everywhere, viz included). |
| Viz | `oos/viz/{move_bridge,session,app}.py` | `(classical solver)` dropdown drives the PlanSolver; 64× speed; no episode caps; solver heartbeat; cockpit STATUS panel (capacity, sedan/SUV counts, admission verdicts, waiting queues, since-reset retrieval latency min/med/avg/max per class); 28-row scrollable event log colored per event (green ✓ deliveries); deterministic toggle greyed for the classical brain; set-point auto-world (target fullness + change rate + dynamicity + SUV rate: a correction flow marches the pool to the target at a facility-relative pace, a balanced churn flow exchanges cars at rest; visit durations emerge via Little's law and a live hint shows in/out per min + avg stay), demand bursts, rush-out. |
| Tests | `tests/test_plan_solver.py` | Apex digs, concurrency, §10.3 rest-point invariant, full-state spec, HOLD executor, ping-pong + rest-churn regressions. |

Retired: `oos/plan/classical.py` (v2.5 cascade) — deleted per §7 after the
battery passed. The RL stack remains only as baseline material.

## Final battery (all PASS, ~3 min wall for the full run)

1. dibaji @ fullness 1.0, deepest on the cap-5 SUV shelf, 100 seeds —
   **100/100, med 99 s** (v2.5: 96/100).
2. Deep-SUV battery — **15/15, med 85 s**.
3. Fill-then-dig — **10/10**.
4. Mass drain, 164 cars at once — **164/164 at 163–170/h**.
5. Full day cycle — **397/397 stores + deliveries, drained to zero**.
6. Seven consecutive days — **~400/day, ≤1 leftover nightly, no drift**.
7. Layout sweep — **all 10 facilities**.

## Bugs found and fixed along the way (each has a regression guard)

- Planner completeness vs the oracle: eager + need-based extraction
  closure; **return-hops** (one-way hops pollute future hop space);
  own-air hops onto the dig shelf; `placed` guard must un-mark returns.
- Scoring hierarchy inversions at high fullness: big-air reservation
  (RESERVE_BIG) > last-empty burial (TOP_EMPTY_LAST) > depth-k; head-window
  never-bury (whole-queue protection deadlocks mass drains); BURY_PROTECTED
  as a last resort on mutually-protecting stacks.
- Concurrency contract: per-shelf stack-op sequences (a blanket
  pop-before-push rule self-blocks hop/return pairs); hands-timeline
  chain checks; volatile foreign lifts; mandatory-path freeing to a fixed
  point; store plans that open their own routes; delivered plans must
  finish their cleanup (dropping on request-completion orphaned holds).
- **Staged-empty steal relay** (operator-reported): a staged room's empty
  must never be a staging source — two lifts ping-ponged one empty through
  a shuttle forever at 0.9 fullness. Fixed + `_any_top_empty` aligned so
  the uncover/stage-plan fallback engages; campus rest state now starts
  zero moves over hours.
- Rendezvous deadlock: a holder that never moved since receiving carried a
  stale anti-ping-pong stamp that vetoed its own hand-back; claims now
  sanction transfers.
- Admission-gate crash window: `future_view` between a pop's event and the
  next pump read stale flags; roles now sync inside `inflight_effects`.
- Performance cliff: full `iter_startable()` per tick is O(shelves²)
  oracle work — day-1 of gate 6 went from 28+ min (aborted) to 26 s wall
  after switching the rungs to targeted enumeration.
- Watchdog false-positives: idle time must rebaseline the liveness clock
  (edge-triggered), and verdicts get a 120 s probation window.

## Operator rulings recorded this session

- `AGENT_BEHAVIOR.md §5.1`: staged rooms = min(rooms, free empties); at
  zero free empties a just-parked car stays on the lift (instantly
  deliverable); a full facility with queued stores is overload-quiescent
  rest, not a wedge.
- Delivery ordering: assignment is FIFO (per-carrier service strictly so);
  completions may overtake across lifts by design. `max_concurrent_plans=1`
  would give strict global order at ~74 % drain-time cost.

## Operator-reported fixes, round 2 (same day)

- **Groom empty-shuffle loop** (campus @0.35, ~200 pointless moves/hour,
  reproduced 10/10): the depth-k score charged only CAR placements, so
  parking an empty on top of a buried car was "free" — groom fixed one
  violation by creating another, forever. Fixes: any pallet (empty
  included) now scores as a blocker for cars beneath it, and the groom
  rung only starts moves that STRICTLY reduce the global violation count
  (bounded potential ⇒ provable termination). Result: 1-2 moves then
  silence.
- **All-lifts-wedged-under-SUVs** (rate 0.26/s, 40 % SUV, 17-min dwell on
  the raw campus pool; reproduced 4/4 by ~800 s): three coordinated fixes —
  (1) the storability DFS's staging-turnover credit is bounded by lifts
  actually free of held cars (it conjured phantom air with all five lifts
  loaded, approving the fatal fifth serve); (2) store plans can now CREATE
  big air via extractions (the gate credits them, so execution must be
  able to perform them); (3) small air is reserved as extraction fuel for
  held SUVs, and held bigs store before sedans. Result: 5/5 seeds survive
  the full overload (~260 stores served, ~210 delivered), refused SUVs
  wait at the door.
- **SUV admission requires a spare empty** (operator rule): an SUV
  consumes the staged empty it parks on — admitted only if ≥1 more empty
  remains in the system, else the room could never re-stage.
- **Dwell never requests empty pallets**: a scheduled retrieve whose car
  already left (manual retrieve raced the dwell) is dropped at fire time.

Battery re-certified ALL PASS after each round; regression tests:
`test_groom_converges_at_low_fullness`, `test_suv_overload_never_wedges`,
`test_dwell_never_requests_empties`, plus the earlier ping-pong and
rest-churn guards.

## Operator-reported fixes, round 3: the held-SUV freeze

**Symptom** (tiny_medipol, set-point world, SUV-heavy): an admitted SUV
sits on its serving lift and NOTHING moves until some retrieval frees big
air — while "accepts SUV" showed true and free empties existed. Reproduced
live 4-6/8 configs; the oracle confirmed a safe placement existed every
time, so `plan_store` was incomplete, in three nested layers:

1. **First-candidate give-up + sim corruption**: `_emit_extraction` picked
   one best shelf and returned None if its destination chains refused —
   even mid-emission, leaving the virtual sim corrupted. Now: candidates
   iterate in (k, mule-busy) order with snapshot/rollback per attempt.
2. **No hop-holds**: with zero big air and every big shelf topped by bigs,
   creating big air needs the top bigs HELD on spare carrier hands (the
   oracle's own_temp counts exactly this, so the gate admits these SUVs).
   Extractions can now hop bigs onto free hands (`dispose`→`land`, the dig
   planner's own vocabulary) — the holders join `plan.holders` and stay
   reserved until the land completes.
3. **Single-placement validation + the holder's own hand**: the terminal
   oracle check (`_finish`) can fail for the best-scored placement while
   an alternative passes, and in the deepest family every passing variant
   needs the extraction dump chain to run THROUGH the holder itself — so
   the car is first relayed to a spare shuttle (`dispose` to carrier),
   freeing the lift's hand, and placements retry with failures excluded.

Result: 10/10 live set-point trials clean (previously 4-6/8 frozen), worst
idle-with-held-SUV 0 s. Regression: the exact captured trap state
(`test_plan_store_escapes_holder_blocked_extraction`) + a 3-seed ×
3-sim-hour steady-state watchdog run (`test_suv_steady_state_never_freezes`).
Battery re-certified ALL PASS (188 s). Also this round: solver "no plan"
retry notes collapse to one line + 60 s heartbeat (log spam), DejaVu font
for the viz glyphs, set-point auto-world, random-room serve order.

**Round 3b — the sedan-side follow-up** ("happens with sedans too, from 0
fullness"): two more layers under the same symptom.

- `plan_store`'s escalations were big-gated; sedans now get the same
  spare-carrier relay and extraction escalation (both only fire after
  every direct placement failed — no cost in normal operation).
- **Door-side serve starvation**: room serves are arrival/dock/completion
  -triggered, but the serve GATE's verdict is time-varying (it consults
  in-flight effects and reservations). A store refused at its arrival
  instant whose gate later cleared waited for an unrelated event —
  visibly "frozen until a request comes". New `engine.retry_serves()`
  re-attempts serves at QUIESCENT instants only — the viz heartbeat and
  the runtime's standstill branch. (First attempt put it inside
  `wake_waiting_carriers`; that fires mid-advance and broke pump
  invariants — 5 tests red. Layering matters: serve retries belong at
  idle boundaries, not inside the event loop.)
- Diagnosis caveat for future hunts: the viz solver heartbeat is WALL-
  clock-throttled (0.5 s), so a tight harness loop compresses hundreds of
  sim-seconds between beats and reports "freezes" the real UI never
  shows. Force `s._hb_t = 0` per tick to emulate real frame pacing.

Verified: 8/8 hunt matrix clean at realistic cadence (bursts @0.85-0.9 +
zero-fullness set-point fills, the operator's exact recipe), 68/68 tests,
battery ALL PASS (208 s).

## Round 4 — the rigorous overnight campaign ("it still happens")

The operator re-reported freezes (sedans too) plus "looping". A proper
harness was built (`scratchpad/rigor_hunt2.py` pattern): faithful UI
pacing (heartbeat every ~32 sim-s, small frame ticks), three detectors —
FREEZE (work continuously present, zero move completions, not
overload-quiescent), LATENCY (retrieve waiting too long), LOOP (same-
shelf round trips with no task progress between) — and a 29-run
config × facility matrix. Detector design matters: v1 compared against
the last-move time, which goes stale during legitimate rest, flagging
every rest→work transition; v1's loop detector counted chain transit and
extraction hop-returns as "bouncing". Verdicts only count with a
continuous-work clock and task-progress resets.

Two real bugs found and fixed:

- **Rest-state staging wedge — the operator's sedan freeze.**
  `_rung_stage` skipped its dig escalation whenever ANY top empty
  existed (`_any_top_empty`) — but the only top empty can be PERMANENTLY
  unreachable (it sits in the other, staged lift's region; staged lifts
  are rest-state infrastructure and their chains are off-limits). The
  room never re-staged; after one serve consumed a room's empty, every
  later sedan waited at the door until an unrelated event. Fix: escalate
  when the executor is quiet (nothing in flight ⇒ no completion will
  unblock the single move) or after 60 s blocked. Regression:
  `test_stage_escalates_past_unreachable_top_empty` (live-captured
  layout).
- **Keep-on-lift must yield to pending retrievals.** At absolute
  saturation (zero free empties AND zero air margin) with retrieves
  queued, the keep-on-lift rule pinned both lifts while the retrieval
  planner needed exactly those lifts freed — mutual wait. A pending
  Retrieve now overrides the keep (that state is not rest); the solver
  stores the kept car into the last air and digs. Battery re-certified —
  the full-state gate still passes (keep-on-lift semantics intact when
  only stores are queued).

"Looping" explained: extraction hop-bigs go OUT to a spare hand/shelf
and land BACK by design (net +1 air), and holder-relays move a car
lift→shuttle→shelf; both read as back-and-forth to the eye. The
corrected loop detector (task-progress resets) found zero unproductive
loops across the matrix.

Final sweep: 18-config matrix + repeats — all FREEZE species dead;
worst-case retrieve tail ≈10 min at 0.9-0.95 fullness under continuous
churn (system actively working, 2-4 moves in flight — queueing depth,
not a stall). Viz gained a STATUS line that names the state (WORKING /
PLANNING / RESTING-full / RETRYING / IDLE) so "why is nothing moving"
is answered on screen. 69/69 tests, battery ALL PASS.

## Round 4b — event starvation (the deepest liveness hole)

Operator: "stuck when there is nothing on it" + "rooms don't stage for a
long time, one shuttle does all the work" (campus, their exact saved
config: target 0.88, change 1.0, churn 0). Root cause, proven by
counter-instrumentation: **the engine is event-driven and only queries
carriers at events; the bridge (and therefore the solver) runs only on
queries.** With an empty scheduler — churn 0, arrivals sparse — a held
car with a trivial store plan, or an unstaged room, sat until the next
unrelated event. Two layered fixes in the Session heartbeat:

1. The heartbeat now **ticks the solver directly** (exactly how the
   headless SolverRuntime loop drives it) instead of hoping wake +
   zero-advance materializes a query. Started moves schedule real engine
   events, so the world wakes.
2. **Stale-claim release ordering**: a claim whose move completed but was
   never query-synced hides its carrier from `work_pending` — the sync
   must run BEFORE the work gate (gating sync on work_pending is
   circular; trace showed `wp: False` heartbeats for minutes while a
   store plan existed).

Verification: staging-at-rest audit (persistence-corrected semantics —
a just-served arrival flips a room "unstaged" for one tick and must not
count) **24/24 trials clean**; operator's exact campus config over 8
sim-h: **98% staged-room uptime, zero rest-while-unstaged seconds, all
ten carriers active** (was: single-shuttle monopoly); 18-config matrix
15/18 clean with residuals classified: extreme-fullness rest where the
free empties are physically undiggable (correct behavior; staging
resumes after the next retrieval), one transient snapshot with a 72-s-
old retrieve mid-planning, and a single 15-min latency tail at the
heaviest campus config while 3 plans + 2 moves were in flight. 70/70
tests (new: `test_heartbeat_ticks_solver_without_events`), battery ALL
PASS. Detector-artifact catalogue for future hunts: stale freeze clocks
across rest, chain-transit "loops", extraction round-trips, fresh-work
flips at sample instants — every one produced a convincing false
"FROZEN" at some point this session.

---

# V3.1 revision session — 2026-07-13

Operator-driven revisions, specified in `docs/SOLUTION_V3_1.md` and
implemented end-to-end in one session. Battery ALL PASS (7/7 gates,
dwell-adjusted bounds), 58/58 tests.

## What changed

| Revision | Where | Essence |
|---|---|---|
| Customer service dwell | `oos/sim/facility.py` (`_ServeInteraction`, `serve_done` event), `Topology.serve_exit_s/entry_s`, DSL defaults 45 s | Serves occupy the lift at the room for a fixed per-facility dwell; the state mutation happens at dwell END; in-service tasks are committed (uncancelable, excluded from matching/sweeps). Zero-dwell topologies keep exact V3 semantics. |
| Evict / Place service ops | `oos/sim/tasks.py` (`Evict`, `Place`), `planner.plan_evict/plan_place` (generalized `_build` with `dest_any` / `dest_shelf`), solver `_assign_service_plans` | Charger-shelf rotation primitives: dig a specific car out and store it anywhere (evict) / land it on a specific shelf without touching that shelf's occupants (place; full destination = fast REJECT note). Strictly below customer retrieves. |
| EV shelves | `Shelf.is_ev`, DSL `Shelf(ev=True)`, `EV_SHELF=250` in the scoring table, `tiny_medipol_ev` facility | Charger shelves deprioritized as scored destinations; explicit Place targets pay nothing. |
| Groom repurposed | solver `_rung_groom` | Depth-k tidying + idle uncovering DELETED (the loop family). New groom: park floating empties + DECLUTTER big shelves (non-bigs → small air) so SUV admission stays open; buried non-bigs escalate to small-restricted evict plans. |
| Concurrency | `PlanSim.schedule` ledgers, `_sched` at every emission, wait-aware pickers (+2/s), `est_cost = makespan + surcharges`, push-push commutation in `_shelf_ops_ready` | Plans are virtually scheduled; cost is the critical path; destinations reachable NOW beat ones waiting on a carrier the plan still has to unload; commuting pushes skip false ordering. |

## Rules that took a fight (again)

- **Groom termination must be a monotone resource, enforced on whole
  plans.** First loop: the evict's own hand-freeing parked an empty onto
  a big shelf while extracting one — net zero, forever. Guard: a groom
  plan commits only if it strictly reduces non-bigs-on-big-shelves.
- **Staged lifts are untouchable to the groom.** Second loop: parking a
  staged room's empty to open a corridor forces a re-stage whose uncover
  pushes a small back onto the big shelf — a perpetual carousel
  (A1/A4/R1). Rest-state infrastructure outranks tidying.
- **The depth-k term is a preference, not a correctness cost** — the
  declutter guard must see past it or it stalls at moderate fullness.
- **Anchors must be freed and exempted.** Shelf-destination plans thread
  cross-region chains through an anchor lift: it must be exempt from the
  volatile set (it is plan-reserved), on the mandatory hand-freeing list
  even when off the exit chain, and the dig carrier itself is never
  volatile in shelf-destination modes.
- **A dropped diverged plan at rest is not a wedge**: the runtime's
  empty-scheduler path must re-check `work_pending` after
  force-replanning before declaring stuck.
- **Gate 4 physics**: dwell costs more than the per-delivery model —
  a pinned lift also delays staging and neighboring digs (~4% on
  campus); bound = 0.95 × R₀/(1 + R₀·E/(3600·n)). Zero-dwell rate
  unchanged (~148–150/h): the solver itself did not regress.

## Viz visibility pass (same session, operator's-father request)

`oos/viz/app.py`: ~1.7× bigger pallets/carriers/rooms (dense layouts fall
back to compact metrics), 3 px shelf borders, intuitive item palette
(turquoise empty / dark-green sedan / dark-red SUV — saturated shades so
they pop on the dark background), S/X letter glyphs (red-green colorblind
insurance), steel-gray carriers with a same-ratio centered cargo
rectangle, working carriers pulse (steady when paused), room borders =
live state (green ready / red unstaged / orange customer-at-door with a
serve-dwell countdown), EV shelves get a yellowish second border, bigger
labels, thicker requested-car outline, and a toggleable on-canvas legend.

## Staging prefetch (same session, operator concurrency report)

Observed: after a store, the shuttle that will re-stage the room idles
until the lift finishes shelving the car, then runs the WHOLE relay —
because moves claim their chains atomically (the deadlock-impossibility
cornerstone), a relay cannot start while any member is busy. Structural,
not a plan-layout accident; the fix uses the architecture's own safe
primitive. **Staging prefetch** (`_rung_stage_prefetch` + `Move.park_at`):
while a room's lift is busy with short-horizon work (from the ENTRY DWELL
onward — the customer is still parking), a partner shuttle fetches the
next staging empty as a HOLD (claims only the shuttle) and parks at the
handoff pose toward the lift; when the lift frees, only the rendezvous +
room leg remain. Guards learned while building it: one prefetch per room
(in-flight prefetches count as "provided for" — first cut dispatched BOTH
shuttles); the stage rung must not race an in-flight prefetch with a
fresh shelf fetch (offer only carrier-sourced candidates until it lands);
skip when a lift-own top empty exists (single-move stage beats it), when
the lift is mid-EXIT-dwell (the delivery re-stages by itself), or when a
plan owns the room; groom yields entirely while any room is unstaged
(held empties are staging material). Crafted benchmark: re-stage at ~78 s
vs ~103 s serialized.

Latent bug exposed and fixed: `GatedEngine._find_pending_store` predates
the serve dwell and skipped the in-service check — EVERY staged lift
started serving the SAME store simultaneously (self-healing at dwell end,
but each extra lift was pointlessly pinned for the full dwell). Also
switched in-service tracking to task IDENTITY: frozen-dataclass value
equality made two same-instant equal stores block each other.

Prefetch round 2 (operator repro: fresh tiny_medipol, 4 sequential
sedans — 4th store's re-stage still serialized): three stacked causes.
(1) The store buried the lift's LAST local top empty; the own-empty
guard now ignores executor-locked shelves (a dst-locked shelf's top is
about to be buried by the in-flight push). (2) The anti-undo guard
blocked storing the parked car back onto the shelf its pallet was staged
from, forcing a pointless store-PLAN escalation on air-tight fresh
layouts — records are now contents-aware (a serve changed the pallet's
cargo: progress, not churn). (3) That store plan's room reservation
blocked the prefetch. Hardening from the stress suite: prefetch yields
entirely to plan work (any active plan or pending retrieve — a parked
loaded shuttle can starve a struggling plan's chains into a wedge;
observed at 3-h SUV steady state), and a patience-gated RELEASE VALVE
(`_rung_release_stranded`, 60 s) parks any held empty nothing consumed —
an expired prefetch or dropped-plan orphan at pool-full wedged the
facility with the old idle-only groom park path gated off.

Deep-retrieve wedge (operator repro: tiny_medipol, dwell slider 0,
re-roll ~0.78, several deep big-shelf retrieves → "only retries, no
plan"): a LATENT false-completion bug in the plan advancer, present
since V3. `_already_at_dst` (the rung-race absorber) marked a pending
LAND intent done because its pallet was "already on" the dig shelf —
where a blocker STARTS before its hold has popped it. Whenever the hold
move could not start in the assignment tick (busy chains; the dwell-0
fast regime makes this common), the land falsely completed, the plan
finished, and the blocker orphaned on its holder. At zero small air the
orphan was oracle-unstorable, permanently wedging the dig carrier —
every later plan for that shelf failed. Fix: the shortcut fires only
when every EARLIER intent moving the same pallet is done ("already
returned" vs "never left"). Regression:
test_land_intent_never_falsely_completes (exact captured seed).

## E/P canvas ops + the hardening they surfaced

Viz: hover a car + **E** = evict; **P** = arm place (orange outline +
hint), then click the destination shelf; pending Evict/Place cars carry
an ORANGE border (relocation, not a yellow customer request); dedupe
guard on repeat keys; legend row added.

Testing the flow surfaced four solver fixes, battery re-certified 7/7:
1. **In-flight room-tail hands projection** (PlanSim): a plan racing an
   in-flight stage saw the lift empty-handed, emitted no park intent for
   the arriving staged empty, and wedged until the stall watchdog. Only
   the room-destination tail is projected — a BLANKET projection (chain
   members → None, hold tails) measurably broke the day-cycle drain
   (gate-6 day-1 leftover 107).
2. **Land-chain unblock valve**: at executor quiescence a delivered
   plan's cleanup blocked by a lift-held empty gets that empty parked
   (retrieval cleanup outranks staged-rest) instead of pacing the
   1200 s drop-replan crawl that stranded day-cycle leftovers.
3. **Cleanup parks avoid plan-touched shelves**: park_delivered chose a
   shelf the plan still POPS — the push-after-pop sequence edge plus the
   pop-chain needing the very hands the park frees = a three-way
   circular cleanup wait (captured live: extract:B2 ↔ park_delivered:22
   → B2 ↔ L2's hands). The unblock valve also generalized from land-only
   to all pending cleanup intents.
4. **Serve-time big solvability re-check** (`store_serve_ok`): admission
   checks solvability at ARRIVAL; the customer walks in later — a big
   whose every remaining placement had become solvability-breaking
   stranded unstorable on dibaji's only lift (gate-3 seed 2). The gate
   re-checks `oracle.admission_ok` at the door; the customer waits.

## Evict restore-semantics (operator revision, 2026-07-14)

An evicted car's shelf must end UNCHANGED apart from the removed car.
The dest_any dig no longer disposes blockers permanently: every blocker
is HELD (spare carrier) or TEMP-HOPPED (extract_hop to another shelf,
sim-tracked), and after the target's relocate they are pushed back in
reverse pop order (land / extract_return, requires_target_off) — the
original composition minus the target. Extraction hops onto the dig
shelf are disabled in this mode (x_air=0: a hop onto X would be
mistaken for a blocker and "restored"); the target's exit pick excludes
active temp shelves (landing on one would bury a pending return); big
air is only grown for the target's own landing. Also benefits the
groom's declutter evicts (bigs go home, only the non-big leaves).
Campus @0.8 evict sweep: 20/20 planned and completed with cars
preserved in order (the operator's earlier "no plan" reports trace to
the pre-fix planner races patched this session); the two "unrestored"
sweep hits were restored EMPTIES legitimately consumed by staging
moments later — empties remain infrastructure, cars stay put.
Regression: test_evict_buried_car_restores_shelf. Battery 7/7,
65/65 tests.

## Characterization campaign (reports/tiny_medipol/, 2026-07-14)

Operator asked for a comprehensive tiny_medipol test — every feature,
every difficulty, plus a 30-day endurance run — with a report + graphs.
Harness: `oos/plan/characterize.py` (experiments A-G + month); output:
`reports/tiny_medipol/{report.md, plots/, data/}`. Headlines: dig
latency flat in fullness 0.3→0.95 (depth dominates, ~30 s/blocker);
drain knee k≈4 → ~50 cars/h; intake dwell-bounded; SUV acceptance knee
at 0.70→0.85; evict/place contracts 15/15; groom converges ≤7 moves;
prefetch −16 % restage; 30 days: 1293=1293 tasks, 0 stuck, flat
latency, 30/30 rotations, 11 s wall. The campaign surfaced and fixed
four real defects: one-path routing blindness (→ equal-length
availability/avoid-aware BFS in free_chain; unbounded rerouting was
measured collapsing staging uptime and rejected), serve-dwell hands
race (PlanSim projects dwells + room-tails), double-booked cleanup
parks (owned-pallet guard on land-route emission), stall-drop/replan
atomicity (one-tick planning hold). Gate-4 contention allowance
0.95→0.93 with the zero-dwell sentinel measured at 152/h (above the
original 150/h bar). Battery 7/7, 65/65 tests.

## Characterization campaign, campus (reports/campus/, 2026-07-14)

Operator: "do the same report thing for campus as well." The harness
grew per-facility configs (`CONFIGS` + `--facility`; drain ks and burst
scale with lift/room count; the groom/prefetch worlds are now built
programmatically instead of tiny-hardcoded), and exp_month dumps
incrementally so a killed run keeps its completed days. Campus
headlines land in `reports/campus/report.md`.

The campus month was the payoff: it surfaced two deep defects that
tiny_medipol is too small and too empty to reach.

1. **Zombie-big starvation + liveness false-positive (days 20-25 of
   the first run).** Campus runs at 411 pallets / 420 slots, so big
   air is structurally ~0 for long stretches. Six SUV stores arrived
   against zero big air and became zombies: unservable (the admission
   oracle rightly refuses), unsweepable (`_can_accept_big_item` sees
   in-principle-evictable non-bigs on big shelves), and — the defect —
   they LOCKED OUT the groom (`_groom_allowed` required an empty
   queue), i.e. the queue starved its own remedy. On top, the
   endurance runner's stuck detector read "work pending + a natural
   post-rush arrival lull" as a wedge and aborted five healthy days;
   deterministic replay showed the solver processing 695- and
   1471-delivery bursts the moment a dense stretch let a run survive.
   Fixes: `_groom_allowed` tolerates a queue of currently-refused bigs
   (declutter mints exactly the air they wait for);
   `overload_quiescent` extended to big-air overload (all-big pending
   + admission false = legitimate rest, not a stall). Tests:
   test_pending_unservable_bigs_do_not_block_groom,
   test_bigair_overload_is_quiescent_not_stuck.
2. **Three-plan air deadlock (day 1 evening of the fixed run).** At
   ~98 % pallet occupancy the facility owns ~9 free slots TOTAL; three
   concurrent retrieval plans locked/reserved eight of them as their
   own dig shelves and slot reservations, the in-view air fell below
   the oracle floor (every `move_ok` false facility-wide), every
   remaining land/extract chain needed a lift, and every lift was
   staged holding an empty it could not legally park. Stall-drops
   rebuilt the identical plans (livelock); the land-chain unblock
   valve found zero oracle-passing parks and gave up. Fix:
   `_force_park_for_land` — terminal recovery inside the valve. When a
   delivered plan stalls at full executor quiescence and no
   oracle-gated park exists anywhere, force-park one staged empty
   WITHOUT the oracle gate (dig shelves that nothing pops from anymore
   become legal push targets; the land's own slot stays protected by
   the reservation margin). Rationale: the oracle is refusing moves
   out of an already-failing view, and a transient solvability debt is
   strictly better than the only alternative — a permanent deadlock.
   Deterministic repro: seed-42 campus month, day 1 evening (135 del /
   232 leftover before; 372 del / clean after).

3. **Liveness verdict blind to customer interactions (tiny month
   day 22 of the re-run).** A quiet afternoon lull + one small store
   transiently gate-refused at its arrival event: no move completes,
   no event re-asks the gate, and the stuck detector's progress clock
   (completed moves only) declared a wedge at the very instant the
   store's entry dwell began. Deterministic replay showed a fully
   healthy world — the day's 43 stores all served, zero deliveries
   simply because none were due yet. Fixes in the runtime liveness
   path, tried in order before escalating: any busy carrier (a serve
   dwell IS progress) rebaselines the clock; then `retry_serves()`
   (gates are time-varying — service, not a verdict, is the answer to
   a lull); then the overload excuses.

Also this session: serve gates now count in-flight store dwells as
committed held cars (`_free_held_cars` virtual entries — closes a
narrow race where two concurrent dwells could both claim the last
storable slot; unit-tested contract); `plan_store` failures feed
`planner_failures` so a stranded held car reads as a failure storm
instead of silence; and `_groom_allowed` also yields to a stranded
plan-less held big at zero raw big air (defensive — reachable if a
dispose consumes the last big slot during the post-serve hands-off
window; predicate unit-tested). One earlier misread corrected: the
`[.B]` glyphs in solver dumps are busy-flags, not held bigs — no
stranded-big state was actually observed in any month run. The full
campaign (campus month + A-G, tiny all) re-ran on the final build for
the published reports. Battery 7/7, 69/69 tests.

## Viz parking carousel (operator report, 2026-07-15)

"Severe loops in tiny_medipol and campus when i try to park a car" —
carriers bouncing shelf ↔ handoff ↔ shelf forever, surviving zero
demand. Reproduced deterministically (Session, dwell 0, fullness 0.779,
seed 7, three sedan parks → 398 moves per idle 2.5 sim-h, unbounded):

- **Root cause (planner):** the sedan parks consumed the big-shelf
  dispose air, so the stage plan for the next room dug its buried empty
  out from under a big by HOLDING the big on a shuttle. The hold's land
  chain relays through the delivery lift — which, post-delivery, holds
  the staged empty. The land-route freeing logic emitted
  `park_delivered` for it, and `_pick_empty_dst(... exclude={X}) or X`
  fell back to the excluded dig shelf itself. Net plan: dig the empty,
  stage the room, un-stage the empty back onto the dig shelf, land the
  big on top — the exact starting world, so the stage rung rebuilt the
  identical plan forever (zero replans; the executor's anti-undo memory
  only covers rung moves, and per-move each step was legal). Fix, refined after a first blunt
  attempt wedged tiny month day 12 (refusing the plan outright left
  both rooms unstageable all morning): the plan shape is only
  pathological through the `or X` FALLBACK — parked to any OTHER
  shelf, the "un-staging" plan productively uncovers the buried empty
  so the next stage is a single move. Stage targets now forbid only
  the dig-shelf fallback: with a real alternative destination the
  two-phase uncover-then-stage runs; with none, no plan — the room
  waits for a retrieve to free real air
  (test_stage_plan_never_parks_its_own_staging).
- **Second carousel closed while hunting (groom licensing):** the
  campus-month groom unlock (`_only_unservable_bigs`) licensed grooming
  whenever a queued big was refused — but at a packed pool the oracle
  refuses bigs no matter how much raw big air a declutter mints, so a
  zombie SUV + a live store stream re-polluting the big shelves =
  perpetual declutter↔placement ping-pong. `_mint_would_admit` now
  builds the groom's capacity-bounded FIXPOINT view (all non-bigs off
  big shelves, bounded by free small slots) and asks the oracle once:
  if even a completed grooming campaign leaves bigs inadmissible, the
  groom stays quiet. (One-move simulation was too weak — the zombie
  regression world needs multi-step grooming through buried empties.)

- **Staging-starved rest excuse (found by the month re-run):** with
  the carousel plan gone, the day-12 geometry (four empties, all
  buried at depth 2, air too tight for the end-state oracle to fund
  any stage dig) becomes an honest wait — retrieves due within the
  hour mint the staging source — but the liveness verdict aborted it.
  `overload_quiescent` now also recognizes all-pending-stores + no
  staged room + no top empty anywhere as legitimate rest
  (test_staging_starved_rest_is_not_a_wedge). The pre-fix months
  "passed" this geometry only because the carousel accidentally
  served stores during its momentary staged flickers.

Diagnosis notes: `last_rung` is stale on plan-advance moves (rung
starters only) — don't trust it when attributing loop moves; and the
first "reproduction" (314 moves/5 sim-h at target 0.918) was actually
the configured setpoint churn (~65 moves/h of legitimate exchange
traffic) — always compare observed motion against configured demand
before calling it a loop. Battery 7/7, 70/70 tests; both months re-run
clean on the final build.
