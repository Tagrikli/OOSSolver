# How the V3 Plan Solver Works

*A tutorial for someone who knows A\* or RL but has never seen an
industrial planner-executor. No hard math — just counting. Code references
point at the real implementation.*

---

## 1. The problem, in one minute

An automated car park stores cars on **pallets** in **LIFO stacks**
(shelves). **Carriers** (lifts and shuttles) move one pallet at a time
along fixed tracks, and can hand a pallet to a neighboring carrier at a
**handoff pose**. Cars enter and leave only at **rooms** (customer doors),
each served by one lift.

Two size classes exist: sedans (`small`) fit anywhere, SUVs (`big`) fit
only on big shelves. Pallets are **conserved objects**: when a customer
drives away, the pallet stays behind as an *empty*; when a car arrives, it
parks *onto* an empty that must already be waiting at the room ("staged").

The core difficulty: a requested car may be **buried** under other pallets
in its stack. LIFO means you must relocate every pallet above it — the
*blockers* — somewhere else first, using carriers that also have to do all
the other work (staging rooms, storing arrivals), without ever painting
yourself into a corner.

If you know the operations-research literature: retrieving one buried item
from LIFO stacks is the **Block Relocation Problem** (a.k.a. Container
Relocation Problem), which is NP-hard in general. Our variant adds size
classes, fungible empties, multiple concurrent requests, and a *physical
multi-agent execution layer* — the relocations are performed by real
carriers with travel times, handoffs, and contention.

## 2. What kind of algorithm is this?

It is **not search** in the A\* sense and **not learning** in the RL sense
(an RL pipeline was tried first and retired; it lives in this repo's git
history). It is a **deterministic, online, hierarchical
plan-and-execute architecture**. If you want one sentence:

> **Simulate, commit, verify, act, repair.** For each request, build a
> complete relocation schedule in a private *virtual simulation*, verify
> its end state with an exact feasibility check, commit resources to it,
> execute it step-by-step against the live world, and — if the world
> drifts — throw the plan away and rebuild it from what is actually true.

Placing it on the map of things a CS grad knows:

| Familiar thing | Relationship |
|---|---|
| **A\*** | A\* searches a state graph for an optimal path. State here is astronomically large (every stack permutation × every carrier pose × continuous time), and we need *good soon*, not *optimal eventually*. Instead of searching, the planner **constructs** one plan greedily and **validates** it. There is no open list and no backtracking across plans — only bounded retries with different candidate parameters. |
| **RL** | RL learns a policy from reward. This system replaces the learned policy with an engineered one; every rule in it is legible and was added in response to a specific observed failure. Determinism is a feature: the same world always produces the same behavior, so bugs reproduce. |
| **MPC / receding horizon** | The dispatcher re-decides every "tick" from live state, like model-predictive control — but instead of re-optimizing a horizon, it advances committed plans and only re-plans on failure. |
| **Greedy + feasibility oracle** | The closest classification. Every decision (where to put a blocker, whether to admit an SUV) is a *greedy scored choice*, filtered by an *exact feasibility check* ("would the world still be solvable?"). Think bin-packing with a safety invariant. |
| **Resource reservation scheduling** | Concurrent plans coexist by *reserving* carriers, shelves, and air slots — like railway block signalling. Deadlock is avoided by construction, not detection. |

## 3. The big idea and the four layers

The system is four layers, each with exactly one job, each depending only
on the layers below it:

```
┌───────────────────────────────────────────────────────────────┐
│ PlanSolver   (oos/plan/solver.py)   "the dispatcher"          │
│   Runs a priority ladder every tick; owns reservations,       │
│   watchdogs, and the admission gates.                         │
├───────────────────────────────────────────────────────────────┤
│ RetrievalPlanner (oos/plan/planner.py)  "the strategist"      │
│   Builds ONE complete plan for one request in a private       │
│   virtual simulation. Never touches the real world.           │
├───────────────────────────────────────────────────────────────┤
│ MoveExecutor (oos/plan/moves.py)  "the choreographer"         │
│   Turns one pallet relocation into per-carrier primitive      │
│   scripts (GOTO/TAKE/GIVE) and runs them closed-loop.         │
├───────────────────────────────────────────────────────────────┤
│ SolvabilityOracle (oos/plan/oracle.py)  "the safety check"    │
│   Answers, exactly and fast: "is this world still solvable?"  │
└───────────────────────────────────────────────────────────────┘
```

The single load-bearing invariant, enforced everywhere:

> **The world is always solvable.** Every stored car must remain
> retrievable, and every car currently held by a carrier must remain
> storable — at every future rest point. No move, no plan, and no
> admitted arrival may ever break this.

Everything else — scoring tables, gates, reservations — exists to keep
this invariant *without* strangling throughput.

## 4. The oracle: solvability by counting

The oracle is the intellectual core, and it is beautifully cheap. The key
observation about this move space (pop a stack top → push it onto another
stack with free space; one pallet airborne per move):

- Shelf-to-shelf moves **conserve total free space** ("air"). Moving a
  pallet frees one slot and fills another.
- Nothing can ever be inserted *below* a buried car.

So ask: when car X finally surfaces, where did its blockers go? They must
be sitting on *other* shelves. That gives an exact condition
([oracle.py](../oos/plan/oracle.py), `_car_retrievable_stacks`):

> Car X, buried under `d` blockers of which `b` are big, is retrievable
> **iff**
> 1. all `d` blockers fit in the free slots *off X's own shelf*, and
> 2. the `b` big blockers fit in the free *big-shelf* slots off X's shelf.

That's it. Necessity is the observation above. Sufficiency is
constructive: pop the blockers top-down straight into the counted slots
(the carrier graph is connected, so any pallet can reach any shelf). No
search — one counting pass, O(depth) per car.

**Small example.** Shelves (top is rightmost), capacities 3:

```
A (small): [car X, sedan, sedan]     ← X buried under d=2, b=0
B (small): [sedan]                   ← 2 free slots
C (big):   [SUV, SUV, SUV]           ← 0 free slots
```

Total air off shelf A = 2, blockers = 2 → condition (1) holds; no big
blockers → condition (2) holds. X is retrievable. Now push one more sedan
onto B: air off A drops to 1 < 2 → **not** retrievable → the oracle
forbids whatever move would have done that.

### 4.1 The extraction closure — growing big air

Condition (2) has an escape hatch. Big air can be **created**: find a big
shelf holding a non-big pallet (sedans and empties end up on big shelves
all the time), pull that non-big out to a small shelf, and one big slot
appears. If the non-big is itself buried under `k` SUVs, those SUVs hop to
existing big air *temporarily* and hop back after — a net conversion of
one small-air slot into one big-air slot.

The oracle counts the maximum number of such extractions with a monotone
fixpoint loop (`_max_extractions`): each completed extraction grows big
air, which can fund the next extraction's hops, and so on. Two details
that took real debugging to get right:

- **Reserve small air** for the target's own non-big blockers — the
  closure must not spend slots that inequality (1) already promised away.
- **The target shelf's own freed slots count as temporary hop space**
  (`own_temp`): as the dig's top blockers depart, X's shelf gains air an
  SUV can *briefly* sit in. This is the "apex maneuver" — parking an SUV
  *above the very car you are digging for*, because you can prove it will
  leave again (§10 has a worked example).

### 4.2 The future view

The oracle never checks the *instantaneous* state — that would be wrong
in both directions. It checks the **FutureView**: the shelf stacks as
they will be *once every in-flight move completes*, plus every car held
by a resting carrier, plus (by projection) every store waiting in queue.
Executor liveness guarantees started moves finish, so future views are
exactly the reachable rest states. This closes an entire class of
concurrency races — e.g. "spend the last big slot while a delivery is one
second from freeing another" is judged on the state where the slot *is*
free.

### 4.3 A fast path that provably changes nothing

Checking hundreds of candidate moves per tick was too slow, so there is
an O(1) screening (`move_ok_ctx`): precompute per-car *slack margins*
once per tick; a candidate move that can shift class air by at most δ and
has global slack ≥ δ provably can't flip anyone's condition and is
accepted without recomputation. Anything not provably safe falls through
to the exact check — semantics are identical (differential-tested in
`tests/test_oracle_differential.py`).

## 5. High-level pseudocode

The dispatcher runs this **every tick** (a tick fires whenever any
carrier asks "what should I do?", plus a heartbeat):

```
tick():
    sync all carrier roles; release finished moves

    # Rung 0 — ADVANCE committed plans (highest priority)
    for each active plan:
        mark completed intents done; drop finished plans
        drop canceled plans (abort their impossible moves)
        start every intent whose preconditions hold RIGHT NOW
        if stalled too long: drop and replan from live state

    # Rung 1 — ASSIGN new plans
    for each pending Retrieve, oldest first (bounded scan):
        if a free lift+room candidate exists:
            plan = RetrievalPlanner.plan(target, candidates, reservations)
            if plan: commit it (reserve its resources), start ready intents

    # Rung 2 — STORE: park cars held by idle carriers (oracle-gated,
    #           scored placement; escalate to a store-plan if no single
    #           move can place it)

    # Rung 3 — STAGE: keep every room supplied with an empty pallet
    #           (top empty → room; escalate to uncover / a stage-plan)

    # Rung 4 — GROOM: only when totally idle and lightly loaded,
    #           tidy depth violations one move at a time
```

And the planner, for one request:

```
plan(target):
    for each (lift, room) candidate, each ordering heuristic:
        sim = virtual copy of all stacks + an air ledger
              (minus other plans' reservations and locked shelves)

        1. free the hands of every carrier on the delivery path
           (park held empties, store held cars — inside the sim)
        2. grow big air to what the dig will need (extractions, eagerly)
        3. DIG LOOP: while the target is not on top:
               blocker = top of the dig stack
               try, in order:
                 a) requested blocker  → HOLD on a spare carrier
                 b) scored shelf placement (the disposal table, §7.2)
                 c) big blocker, no big air → HOLD, else extraction
                 d) HOLD as last resort
               (each choice = one "intent", recorded with stack
                sequence numbers)
        4. deliver the target to the room
        5. land every held blocker back onto the dig shelf
           (unrequested first; requested land last → they end on top)

        validate: oracle.check_view(sim's terminal stacks) — reject
                  the whole plan if the end state is not solvable
    return the cheapest validated plan (deterministic tie-breaks)
```

Note what is *absent*: no backtracking inside the dig loop (a dead end
fails the candidate; the caller tries the next parameterization), no
global optimization, no learned components. Boring on purpose.

## 6. The executor: from "move pallet 42 to shelf B7" to carrier motion

A **Move** is the only physical action the upper layers know:
*relocate exactly one pallet* from a stack top (or a carrier's hands) to
another shelf, a room, or a carrier (a HOLD). The executor compiles it
into per-carrier scripts:

```
Move: shelf A1 → shelf B7, chain (L1, S1)
  L1: GOTO A1 · TAKE · GOTO handoff(S1) · SEND
  S1: GOTO handoff(L1) · RECV · GOTO B7 · GIVE
```

`SEND`/`RECV` are passive: the simulator's *automatic rendezvous
transfer* fires the instant both partners are docked at matching poses,
one loaded, one empty. Three rules make concurrent moves safe
([moves.py](../oos/plan/moves.py), header):

1. **Atomic all-free claims.** A move starts only if *every* carrier on
   its chain is unclaimed, idle, and empty-handed (the source excepted).
   Nobody ever queues on a busy carrier → circular wait is impossible →
   **deadlock is impossible by construction**, not detected after the
   fact.
2. **Exclusive shelf locks.** The source shelf is locked until the pop
   happens, the destination until the push. Two moves can never disagree
   about what a TAKE will grab.
3. **Rendezvous single-authorship.** Both sides of a handoff belong to
   the same move, so the auto-transfer can't fire against a bystander.

Mid-chain carriers are released the moment they hand off — a 3-carrier
relay doesn't hold all three for its whole duration.

## 7. The planner, a level deeper

### 7.1 The virtual simulation

`PlanSim` ([planner.py](../oos/plan/planner.py)) is a tiny model of just
what planning needs: stacks as `(pallet_id, contents)` lists, an **air
ledger** per shelf (free slots minus what *other* plans have reserved),
each carrier's **hands** as the plan's sequence advances, and the growing
intent list. Every decision mutates the sim, never the world. A
`snapshot()/restore()` pair lets the planner *attempt* an emission (say,
an extraction) and roll back cleanly if it dead-ends — bounded,
local backtracking.

### 7.2 The disposal scoring table

"Where do I put this blocker?" is the most consequential choice, made by
`placement_score` against the *virtual* stacks. It is a sum of penalty
constants whose **ordering is the real design** (the values only encode
the ordering):

| Penalty | Value | Meaning — and why it ranks where it does |
|---|---|---|
| `HARD` | 10⁹ | Never bury a *protected* (soon-to-be-served) request. Forbidden outright. |
| `RESERVE_BIG` | 8×10⁶ | Placement would starve the big air the current dig still needs. Plan-fatal, so it outranks everything soft. |
| `BURY_PROTECTED` | 5×10⁶ | Last-resort mode only: bury a protected request one deeper when refusing would make planning *impossible* (mutually-protecting stacks). The buried peer just digs one extra blocker later. |
| `TOP_EMPTY_LAST` | 3×10⁶ | Burying one of the *last* stageable empties — the staging pipeline dies with it. |
| depth-k (`SOFT_VIOLATION`) | 10⁶ | Placement leaves a car deeper than `k` (=1). The tidiness preference — deliberately *below* all correctness costs, because at high fullness depth-k is unmaintainable and must never win against safety. |
| `CLASS_FLOOR` | 5×10⁵ | Spends the last slot of a size class on a car. |
| `EMPTY_FLOOR` | 10⁴ | Spends the last class slot on an empty (re-movable, so much cheaper). |
| `TOP_EMPTY` | 10³ | Buries a top empty when empties are plentiful. |
| `POLLUTE_BIG` | 300 | A sedan onto a scarce big shelf. |
| `EMPTY_ON_BIG` | 100 | An empty onto a big shelf while small air exists. |
| chain length | 200/hop | Prefer disposals inside the dig's own region — every extra hop rides a carrier other plans contend for. |
| makespan | ×0.001 | Final tie-break: analytic seconds. |

Read bottom-up it is a story: *prefer near and tidy; never sacrifice the
staging pipeline for tidiness; never sacrifice class-air correctness for
anything.*

### 7.3 Holds

A **HOLD** parks a blocker on a spare carrier instead of a shelf — the
carrier *is* storage for a while. Why this is always sound: pops are
LIFO-serialized, and digging `d` blockers frees `d+1` slots on the dig
shelf while at most `d` holds land back. Requested blockers *prefer* a
hold: they land back on top of the dig shelf (depth 0) — perfectly
positioned for their own upcoming delivery.

### 7.4 Three plan flavors

The same machinery builds three kinds of plan:

- **Retrieve** (`plan`) — the full dig described above.
- **Store** (`plan_store`) — recovery for a held car no single move can
  place: hand the car to a spare shuttle first if the holder's own full
  hands block every route, park a helper's staging empty, even run
  extractions to *create* the air it needs.
- **Stage** (`plan_stage`) — staging *is* a retrieval whose target is an
  empty pallet: dig out the cheapest buried empty and deliver it to the
  room. Used when no empty is on top of any reachable stack.

## 8. The solver, a level deeper

### 8.1 Reservations — how concurrent plans coexist

A committed plan owns, until it finishes ([solver.py](../oos/plan/solver.py),
`reserved_carriers` / `locked_shelves` / `pending_reserved_slots` /
`owned_pallets`):

- its **delivery lift** and its **dig carrier**;
- every **holder** carrying one of its blockers;
- its **dig shelf** and every extraction source still being popped
  (nobody else may push onto or pop from them);
- the **air slots** its future disposals will consume (other planners see
  that air as already occupied — "phantom fills" in the future view);
- its **pallets** (nobody else may relocate them).

Reservations are *derived from the plans* every time they're needed,
never stored separately — so they cannot drift out of sync. Everything
not reserved keeps flowing through the lower rungs: stores and staging
continue in the gaps while digs run.

### 8.2 Submit-when-ready and per-shelf sequencing

The planner emits intents in one consistent global order, but the solver
does **not** execute them strictly serially — that would waste the
parallel hardware. Instead each intent starts when its preconditions hold
*right now*:

- its pallet is the current top of its planned source shelf (LIFO
  self-serialization — you physically cannot pop out of order);
- its destination has room and is unlocked;
- a free carrier chain exists that avoids other plans' reservations;
- and its **stack-operation sequence numbers** are satisfied: the plan
  sim stamped every pop/push on a shelf with a per-shelf counter
  (`src_seq`/`dst_seq`); operation *j* may start only when every earlier
  op on that same shelf has *done its stack op*. Pops count at
  TAKE-completion — so a dig pipelines: blocker 2 can be lifted while
  blocker 1 is still riding to its destination.

Why per-shelf and not a simple "all pops before all pushes"? Because
extraction hops *return*: the same shelf is popped, pushed, popped again.
A blanket rule self-blocks those pairs; per-shelf sequence numbers taken
from one consistent global schedule order them correctly and — because
that schedule was actually simulated — can never deadlock.

### 8.3 Watchdogs — plans are disposable

The solver treats every plan as **cheap to throw away**. Three timers:

- A plan with nothing in flight and nothing startable for **120 s** is
  dropped and re-planned from the live state (the world diverged — a rung
  raced it, a manual edit happened, whatever).
- A *delivered* plan still waiting to land its holds gets **10×** that
  window (its resources are reserved; the cleanup chains free
  eventually).
- A plan whose in-flight move produces no completion for 10× the window
  is force-aborted (`MoveExecutor.abort`) — an in-flight move that never
  completes means its completion has become *impossible* (e.g. a
  delivery whose request was canceled mid-ride), and no legitimate move
  takes 20 sim-minutes.

Dropped plans orphan nothing permanently: held pallets become ordinary
"held cars" and the store rung re-shelves them. Self-healing beats
never-failing.

### 8.4 The gates — refusing work you cannot finish

The solver also answers two questions *for the world* (wired into the
engine as callbacks):

- **Admission** (`admission_ok`): may this arriving car enter at all?
  Sedans: always (a sedan store is a pure contents swap with its staging
  empty — net-zero occupancy, it can never make things worse). SUVs: only
  if ≥1 spare empty remains after it parks (else the room can never
  re-stage), the future view stays solvable with the SUV placed
  *somewhere*, and every lift that might absorb it would still have a
  storable ordering for its held cars. A refused SUV drives away — by
  design, exactly like the real facility.
- **Serve** (`store_serve_ok`): the customer physically walks in only
  when the receiving lift is not mid-plan and the resulting set of held
  cars can *all* be stored in some order of startable moves
  (`held_set_storable` — a tiny DFS that also models staging turnover and
  air-creating extractions). A store that is not serveable *right now*
  simply waits at the door; it never becomes an unstorable held car.

## 9. Worked example, step by step

A toy facility. One lift `L1` (serves room `R1`), one shuttle `S1`,
handoff between them. Stacks written bottom→top; capacity 3 everywhere.

```
        L1's shelves                 S1's shelves
  A1: [ car7, car12, e3 ]      B1: [ big20 ]          (big shelf)
  A2: [ e5 ]                   B2: [ ]                (small shelf)

  L1: holding e9, docked at R1 (room is staged)
  S1: empty-handed, idle
```

`e·` are empty pallets. **A customer requests car 7** — buried at depth 2
under empty `e3` and sedan `car12`, on lift shelf `A1`.

**Planning** (all inside the virtual sim; nothing moves yet):

1. *Candidates.* One room → one candidate `(L1, R1)`. The dig carrier is
   `L1` (A1 is its shelf); the delivery chain is just `(L1,)`.
2. *Free the mandatory hands.* `L1` is on the delivery path but holds the
   staged empty `e9`. Emit **`park_empty e9 → A2`** (scored: A2 is near,
   has air, buries no car). Sim: A2 = `[e5, e9]`, L1's hands free.
3. *Dig loop, blocker 1.* Top of A1 is `e3`. Scored disposal: A2 has one
   slot left; placing an empty on empties costs ~nothing. Emit
   **`dispose e3 → A2`**. Sim: A2 = `[e5, e9, e3]` (full), A1 = `[car7, car12]`.
4. *Dig loop, blocker 2.* Top is `car12` (sedan). A2 is full; B2 is
   reachable via chain `(L1, S1)` — the sim checks S1's hands are free at
   this point in the sequence (they are). Emit **`dispose car12 → B2`**
   (+200 for the extra hop, but it's the only option). Sim: B2 = `[car12]`.
5. *Deliver.* Top is the target. Emit **`deliver car7 → R1`**.
6. *Land holds.* None were needed.
7. *Validate.* Terminal sim stacks: A1 `[]`, A2 `[e5,e9,e3]`, B1
   `[big20]`, B2 `[car12]`, nothing held. Oracle: every car's blockers
   fit elsewhere (car12 and big20 are on top — depth 0); solvable ✓.

The committed plan, with its per-shelf sequence stamps:

```
intents:
  1. park_empty  e9    → shelf A2       (A2 push #0)
  2. dispose     e3    → shelf A2       (A1 pop #0, A2 push #1)
  3. dispose     car12 → shelf B2       (A1 pop #1, B2 push #0)
  4. deliver     car7  → room  R1       (A1 pop #2)
reserved: lift L1, dig shelf A1, one air slot on A2, one on B2
```

**Execution** (submit-when-ready against the live world):

- *t=0* — Intent 1 is ready (e9 is in L1's hands): L1 `GOTO A2 · GIVE`.
  Intent 2 is not ready yet: its A2-push (#1) must wait for push #0.
  Intent 3 *is* ready by sequence (B2 push #0, A1 pop #1 — but A1 pop #1
  needs pop #0 first) — so it waits too. Nothing else can start.
- *t≈8s* — e9 lands. Intent 2 ready: L1 `GOTO A1 · TAKE e3 · GOTO A2 ·
  GIVE`.
- The **pop of e3 completes at TAKE time** — so intent 3's A1-pop (#1)
  unblocks *while e3 is still riding to A2*. But intent 3 needs L1 in its
  chain and L1 is busy; it starts the moment L1 frees. The dig
  pipelines exactly as far as physics allows.
- Intent 3: L1 `TAKE car12 · GOTO handoff · SEND`; S1 `GOTO handoff ·
  RECV · GOTO B2 · GIVE`. **L1 is released at the handoff** (mid-chain
  release) and immediately starts intent 4: `GOTO A1 · TAKE car7 · GOTO
  R1`.
- L1 docks at R1 holding the requested car → the engine's
  arrival-triggered **serve** fires: the customer drives off, and the
  *pallet stays* — L1 is now holding a fresh empty at R1. **The delivery
  itself re-staged the room.** Conserved pallets make the rest state an
  attractor.

Total: four moves, two of them overlapped, zero wasted motion, and the
end state was proven solvable before the first carrier twitched.

## 10. The apex maneuver (extraction), by example

The single cleverest behavior. Setup: you must dig an SUV out, but **big
air is zero** — every big-shelf slot is full. Condition (2) fails…
unless air can be *grown*:

```
  X  (big, dig shelf):  [ target_SUV, SUV_a ]      ← 1 free slot
  Y  (big):             [ sedan30, SUV_b ]         ← full? cap 2: full
  Z  (small):           [ ]                        ← small air
```

`SUV_a` must go somewhere big — but no big shelf has air. The closure:

1. **Hop** `SUV_b` (the big above Y's sedan) *onto X's own free slot* —
   yes, on top of the very stack being dug; it is provably temporary.
2. **Extract** `sedan30 → Z` (small air). Y is now empty: big air +1.
3. **Return** `SUV_b` home onto Y (the hop-return; per-shelf sequencing
   orders this correctly).
4. Now `SUV_a → Y` is a normal disposal; the dig proceeds; the target
   surfaces.

Net effect of steps 1–3: one small-air slot became one big-air slot. The
oracle *counts* this maneuver (so admission and gating know it exists),
and the planner *emits* it as plain moves (so it actually happens) — the
two implementations mirror each other deliberately; states the oracle
certifies solvable are states the planner can realize.

## 11. Domain rules that took a fight to learn

Each of these is a one-liner in the code guarding against a specific
observed failure. They are the "engineering sediment" a textbook usually
hides:

- **Protect the head window, not the whole queue.** "Never bury a
  requested car" applied to *all* requests deadlocks a mass rush-out
  (when every car is requested, every placement is forbidden). Only the
  ~10 oldest requests are protected; a tail request buried one deeper
  just digs one extra blocker when its turn comes.
- **A staged room's empty is infrastructure, never a staging source.**
  Stealing lift B's staged empty to stage room A re-stages one room by
  un-staging another — an infinite ping-pong relay (observed live).
  Buried empties are reached by uncover moves or a stage plan instead.
- **Full-facility rest ("keep on lift").** When zero free empties remain,
  a just-parked car *stays on its lift*: storing it couldn't re-stage the
  room anyway, and on the lift it is instantly deliverable. A pending
  retrieval overrides the keep — at total saturation the kept car's slot
  may be exactly what the dig needs.
- **Overload quiescence is not a wedge.** A full facility with only
  stores queued is *supposed* to rest — customers wait at the door until
  a retrieval frees an empty. The liveness watchdogs must know this state
  or they false-alarm on correct behavior.
- **Grooming must provably terminate.** Idle tidying only makes moves
  that *strictly reduce* the global depth-violation count — a bounded
  non-negative potential can't descend forever, so no shuffle loops. And
  it only runs below 45% load: depth-k is unmaintainable at high
  fullness and must never fight the correctness machinery.
- **Cancel-mid-delivery must abort, not wait.** A delivery whose request
  is canceled while the car rides to the room can *never* complete (the
  serve will never fire). The solver aborts the move; the carrier frees
  holding the car; the store rung re-shelves it. Backstop: any in-flight
  move with no completion for 20 sim-minutes is presumed wedged and
  aborted — unknown future causes self-heal too.
- **Determinism everywhere.** Sorted iteration orders, explicit
  tie-breaks, seeds threaded through. The same world always produces the
  same plan — which is why every bug in this file's history was
  reproducible, and why they're fixed.

## 12. Where to read the code

| File | What you'll find |
|---|---|
| [`oos/plan/oracle.py`](../oos/plan/oracle.py) | FutureView, the counting conditions, the extraction closure, the O(1) fast path, admission. ~750 lines, start here. |
| [`oos/plan/planner.py`](../oos/plan/planner.py) | `PlanSim`, the dig loop, the scoring table, holds, extraction emission, `plan_store`/`plan_stage`. |
| [`oos/plan/solver.py`](../oos/plan/solver.py) | The tick ladder, reservations, submit-when-ready, sequencing, watchdogs, gates, the rungs. |
| [`oos/plan/moves.py`](../oos/plan/moves.py) | Move, claims/locks, scripts, `held_set_storable`, `abort`. |
| [`oos/plan/runtime.py`](../oos/plan/runtime.py) | The headless pump loop + the stuck watchdog (how to *drive* the solver without the viz). |
| [`oos/plan/battery.py`](../oos/plan/battery.py) | The acceptance battery — 7 gates from single deep digs to 7-day continuous operation. Run `python -m oos.plan.battery --gate 2` for a 30-second taste. |
| [`docs/SOLUTION_V3.md`](SOLUTION_V3.md) | The design spec this implements. |
| [`docs/AGENT_BEHAVIOR.md`](AGENT_BEHAVIOR.md) | The behavior contract (what "correct" means, operator-approved). |

### Glossary

| Term | Meaning |
|---|---|
| **air** | Free shelf slots. *Big air* = free slots on big shelves. |
| **blocker** | Any pallet above a target in its stack (empties count). |
| **dig** | Relocating all blockers so a target surfaces. |
| **staging** | Placing an empty pallet on a lift at a room so an arriving car can park onto it. |
| **HOLD** | Parking a blocker on a spare carrier temporarily. |
| **extraction** | Pulling a non-big off a big shelf into small air to grow big air. |
| **intent** | One planned relocation inside a plan. |
| **rung** | One priority level of the dispatcher's ladder. |
| **future view** | The world as it will be when all in-flight moves finish. |
| **solvable** | Every stored car retrievable and every held car storable. |
