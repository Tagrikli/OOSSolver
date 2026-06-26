# OOSSolver — a complete deterministic planner for the Parkolay ASRS

A standalone planner that services a store/retrieve stream on the automated
parking facility with **100% completeness**, a **sub-5-second** per-task compute
budget, real **carrier concurrency**, recognition of **unsolvable** layouts, and
a hard **never-strand** guarantee (it never leaves any stored item unretrievable
and keeps rooms staged when idle).

Built from scratch on top of the `oos.sim` world model only — it does **not**
depend on the prior RL (`oos.learn`) or `oos.plan` attempts. Validated on the
**campus** facility (10 carriers, 140 LIFO shelves cap-3, 5 rooms, 25 handoffs).

```
python -m oos.solver.bench          # full validation report (the proof)
python -m oos.solver.bench quick    # fast subset
pytest tests/test_oossolver.py -q   # regression subset
```

## Run it live in the visualizer

```
python -m oos.viz --facility campus
```

Then press **P** (policy picker) and choose **★ OOSSolver (complete planner)**
(it's the first entry). The solver now drives the facility in real time:

- press **m** to toggle manual mode, **space** to play/pause, **./→** to step;
- **click a pallet** to request its retrieve — watch the carriers dig it out and
  deliver it to a room, then restore the layout;
- queue stores from the side panel (**queue small / queue big**) — the solver
  stages a room, the customer loads, and it carries the item to a safe shelf;
- when the queue empties, every room re-stages with an empty pallet (idle invariant).

Selecting any other policy (or the search PLANNER) switches back to the normal
agent loop. Implemented by `oos.solver.live` (`LiveSolver` + `SolverDriver`),
which executes the planner's multi-carrier plans chunk-by-chunk so the sim
animates smoothly while still reacting to live clicks; it is wired into the viz
as a selectable driver (`oos/viz/pickers/policy.py`, `oos/viz/app.py`).

## Why classical (not RL)

The requirement is *hard completeness* plus a *wall-clock bound* plus *certified
unsolvability*. RL gives none of those guarantees. This is a classical
complete-search / constructive-planning problem, so we solve it as one.

## How it works

The only scarce resource is **big-shelf capacity** (big items live only on big
shelves; smalls/empties fit anywhere and are abundant). Every hard case — deep
big burial, the SUV "buffer-on-target / put-back" deadlock — is therefore a pure
multi-stack LIFO relocation puzzle over the ~40 big shelves.

| Module | Role |
|--------|------|
| `world.py` | static topology view: owners, sizes, room routes, handoff fabric (≤2 handoffs between any two carriers). |
| `relocate.py` | **the core.** A* over an abstract big-shelf token model to expose a buried target. Destinations include the target shelf itself, so the put-back maneuver is found automatically. **Complete**: exhausting the (closed-set, finite) frontier *proves* unsolvability. Validated against an exhaustive brute force — 0 mismatches over thousands of instances. |
| `plan.py` | compiles abstract pallet moves into concrete per-carrier primitives (GOTO/TAKE/GIVE/WAIT), routing cross-region moves over the handoff fabric. A **retrieve** = dig → deliver. Never-strand holds *without* putting blockers back: a retrieve only removes an item, which strictly adds slack (the removed item frees a slot, and each relocated blocker's origin freed a slot), so every other item stays retrievable — the un-burying is done lazily, only when a buried item is itself requested. An optional `restore=True` reverses the dig (kept for tidiness/debugging) but is off by default since it just doubles the moves. |
| `runner.py` | the executor: drives `SimEngine` with per-carrier primitive streams gated by live-state predicates (pallet-on-top, dest-has-room, carrier-holds). Independent carriers move concurrently; it never crashes on a stale precondition (degrades to WAIT). |
| `solver.py` | orchestrator: `execute_retrieve`, `execute_store` (lands items only where every item stays retrievable — checked with the exact `plan_dig` oracle), parallel staging of all rooms, `serve_retrieves` (footprint-disjoint parallel waves), `run_tasks` (mixed FIFO stream). `recover()` returns all carriers to empty between tasks. |

### Key correctness facts established here

- The project's fast `_layout_is_solvable` is **unsound** (it passes layouts that
  are provably unretrievable — the source of the prior deadlocks), so the
  never-strand guard uses the exact `plan_dig` feasibility instead.
- A retrieve is **restore-clean**: after it, the layout equals the prior layout
  minus the one retrieved item, so serving one task never strands another.
- A big store is **rejected** iff landing it would make some item unretrievable —
  correct capacity-limited behavior, not a failure.

### Concurrency

Carrier motion regions are disjoint, so the executor runs every carrier whose
next step is ready. All 5 rooms stage in parallel (one op, not five);
footprint-disjoint retrieves run in the same wave (~2.7× on a non-packed
facility). On the *fully packed* campus (417/420 pallets ⇒ 3 free slots)
concurrency is capped by physics — all digs compete for the few free slots — yet
completeness and the time bound still hold.
