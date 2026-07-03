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
