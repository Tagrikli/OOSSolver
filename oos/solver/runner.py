"""Parallel executor: drive a SimEngine with per-carrier primitive streams.

A *plan* is a mapping `carrier_id -> list[Step]`. Each carrier consumes its
steps in order. A step may carry a `ready(state) -> bool` gate; the executor
submits the step only once the carrier is idle AND the gate is satisfied,
otherwise the carrier WAITs and is re-queried on the next state change. This is
how cross-carrier synchronization (handoffs, LIFO/shelf-access ordering) is
expressed without freezing the rest of the facility — every carrier whose next
step is ready runs concurrently.

The executor is intentionally dumb: all correctness (which step, which gate)
lives in the planner. Here we only translate ready steps into `engine.submit` /
`engine.wait` and pump `advance_until`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from oos.sim.actions import Give, Goto, PreconditionError, Take
from oos.sim.facility import SimEngine, TaskCompletion
from oos.sim.state import DockRef, FacilityState


@dataclass
class Step:
    """One primitive for one carrier.

    op:
      'goto'  -> Goto(carrier, target)         (target: DockRef required)
      'take'  -> Take(carrier)                 (pop docked shelf / handoff pull)
      'give'  -> Give(carrier)                 (push docked shelf / handoff push)
      'wait'  -> engine.wait(carrier)          (serve at room / park)
    ready: optional gate; the step is held (carrier WAITs) until it returns True.
    """

    carrier: str
    op: str
    target: Optional[DockRef] = None
    ready: Optional[Callable[[FacilityState], bool]] = None
    label: str = ""


@dataclass
class RunResult:
    completions: list[TaskCompletion] = field(default_factory=list)
    makespan: float = 0.0
    finished: bool = False          # every carrier consumed all its steps
    stuck: bool = False             # progress stalled with steps remaining
    instants: int = 0


def run_plan(
    engine: SimEngine,
    plan: dict[str, list[Step]],
    max_instants: int = 200_000,
    trace: Optional[list] = None,
) -> RunResult:
    """Execute `plan` against `engine` to completion (or stall).

    Returns RunResult with collected task completions and the final makespan.
    """
    idx: dict[str, int] = {cid: 0 for cid in engine.state.carriers}
    completions: list[TaskCompletion] = []
    start = engine.state.time
    instants = 0
    # Clear any stale WAIT-holds left by a prior plan so this plan's carriers are
    # re-queried from the current state (otherwise a fresh run sees no decisions
    # and falsely reports terminal).
    engine.wake_waiting_carriers()

    def remaining(cid: str) -> bool:
        return idx.get(cid, 0) < len(plan.get(cid, []))

    # Guard against pathological non-progress: if we cycle through many instants
    # without consuming a step or completing a task, declare stuck.
    no_progress = 0

    while True:
        instants += 1
        if instants > max_instants:
            return RunResult(completions, engine.state.time - start, False, True, instants)

        decisions = engine.carriers_needing_decision()

        if not decisions:
            # Nobody to query: either commands are in flight (advance to their
            # completion) or the world is idle/terminal.
            res = engine.advance_until(None)
            completions.extend(res.completions)
            if res.completions:
                no_progress = 0
            if res.terminal:
                break
            # Empty scheduler, no decisions, not terminal would loop forever.
            if len(engine.scheduler) == 0 and not engine.carriers_needing_decision():
                break
            continue

        progressed = False
        for cid in decisions:
            steps = plan.get(cid, [])
            i = idx.get(cid, 0)
            if i >= len(steps):
                engine.wait(cid)            # done -> park in place
                continue
            st = steps[i]
            if st.ready is not None and not st.ready(engine.state):
                engine.wait(cid)            # gated -> hold, re-queried on change
                continue
            try:
                if st.op == "goto":
                    assert st.target is not None
                    engine.submit(Goto(cid, st.target))
                elif st.op == "take":
                    engine.submit(Take(cid))
                elif st.op == "give":
                    engine.submit(Give(cid))
                elif st.op == "wait":
                    engine.wait(cid)
                else:
                    raise ValueError(f"unknown step op {st.op!r}")
            except PreconditionError:
                # The live state isn't ready for this step yet (or never will be).
                # Hold the carrier; the stuck-guard ends the plan if no progress.
                engine.wait(cid)
                continue
            if trace is not None:
                trace.append((round(engine.state.time, 3), cid, st.op, st.label))
            idx[cid] = i + 1
            progressed = True

        res = engine.advance_until(None)
        completions.extend(res.completions)
        if res.completions or progressed:
            no_progress = 0
        else:
            no_progress += 1
            # All current decisions were gated/parked and nothing advanced.
            if no_progress > 4 and len(engine.scheduler) == 0:
                still = any(remaining(cid) for cid in engine.state.carriers)
                return RunResult(
                    completions, engine.state.time - start, not still, still, instants
                )
        if res.terminal:
            break

    finished = all(not remaining(cid) for cid in engine.state.carriers)
    return RunResult(
        completions, engine.state.time - start, finished, not finished, instants
    )
