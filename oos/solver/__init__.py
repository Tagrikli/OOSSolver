"""OOSSolver — a deterministic, complete planner for the Parkolay ASRS.

Standalone planner built from scratch on top of the `oos.sim` world model only
(no dependency on the prior RL/`oos.plan` attempts). See docs/APPROACH.md.

Layers:
  world.py     — read-only topology/state view (owners, sizes, routes, fabric)
  runner.py    — parallel executor: drives SimEngine with per-carrier primitive
                 streams gated by state-predicate readiness (real concurrency)
  relocate.py  — block-relocation core over big shelves (feasibility + dig plan)
  plan.py      — single-task planners (retrieve / store / stage) -> CarrierPlan
  solver.py    — orchestrator: task manager, staging, multi-task parallelism
"""

from oos.solver.runner import Step, run_plan  # noqa: F401
