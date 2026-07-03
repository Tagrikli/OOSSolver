"""SolverBridge — drive the viz's primitive Environment with the V3 plan
solver (oos/plan/solver.py, SOLUTION_V3).

The Session queries the bridge once per idle carrier. The bridge keeps a
MoveExecutor + SolvabilityOracle + PlanSolver bound to the session's LIVE
engine: at each query it syncs executor scripts against the engine, ticks
the solver (which plans and starts moves itself), and answers with the
primitive action index for the querying carrier — GOTO/TAKE/GIVE translated
into the env's enumerated entries, WAIT for passive rendezvous steps and
idle holds.

Self-healing: any manual world edit that invalidates in-flight scripts
(re-roll layout, facility swap, reset) is caught — on engine identity change
or any internal inconsistency the bridge drops its claims and rebuilds from
the current state (plans are recomputed — nothing is lost but seconds).
"""

from __future__ import annotations

from typing import Optional

from oos.env.action import ActionType
from oos.plan.moves import MoveExecutor
from oos.plan.oracle import SolvabilityOracle
from oos.plan.solver import PlanSolver


class SolverBridge:
    def __init__(self, env) -> None:
        self.env = env
        self._engine = None
        self.ex: Optional[MoveExecutor] = None
        self.oracle: Optional[SolvabilityOracle] = None
        self.solver: Optional[PlanSolver] = None
        self.notes: list[str] = []   # surfaced by the session log

    # ------------------------------------------------------------------

    def _rebind(self) -> None:
        engine = self.env.engine
        if engine is self._engine:
            return
        self._engine = engine
        self.oracle = SolvabilityOracle(engine.topology, max_holds=1)
        self.ex = MoveExecutor(engine, self.oracle)
        self.solver = PlanSolver(engine, self.ex)
        self.notes.append("V3 plan solver bound to engine")

    def _drop_state(self) -> None:
        """Emergency reset after an inconsistency (manual world edit mid-move):
        forget all claims/locks; carriers finish their current primitive and
        then idle until re-dispatched."""
        self._engine = None

    # ------------------------------------------------------------------

    def admission_ok_for(self, size: str) -> bool:
        """Deployment admission gate for the session's manual store
        buttons — the solver's reservation-aware oracle check."""
        self._rebind()
        assert self.solver is not None
        return self.solver.admission_ok(size)

    def decide(self) -> int:
        """Answer the env's current query: the action index for the querying
        carrier. Ticks the solver first, so new plans/moves start the instant
        any carrier is asked."""
        # Index against the env's LIVE decoder — an index computed on a stale
        # entry list aliases to a wrong primitive.
        entries = list(self.env._ctx.decoder.entries)
        wait_idx = max(0, len(entries) - 1)
        try:
            self._rebind()
            engine = self.env.engine
            ex = self.ex
            solver = self.solver
            assert ex is not None and solver is not None
            cid = self.env.querying_carrier
            for c in list(ex.claimed.keys()):
                ex.sync_role(c)
            if solver.tick():
                # New moves started for other carriers: re-open their
                # decisions so the env queries them this instant.
                engine.wake_waiting_carriers()
            while solver.notes:
                self.notes.append(solver.notes.popleft())
            step = ex.sync_role(cid)
            if step is None:
                return wait_idx
            return self._entry_index(step, entries, wait_idx)
        except Exception as e:  # noqa: BLE001 — self-heal on any surprise
            self.notes.append(f"bridge reset: {type(e).__name__}: {e}"[:70])
            self._drop_state()
            return wait_idx

    @staticmethod
    def _entry_index(step: tuple, entries, wait_idx: int) -> int:
        kind = step[0]
        if kind == "goto":
            for i, e in enumerate(entries):
                if e.type == ActionType.GOTO and e.target == step[1]:
                    return i
            return wait_idx
        if kind == "take":
            for i, e in enumerate(entries):
                if e.type == ActionType.TAKE:
                    return i
            return wait_idx
        if kind == "give":
            for i, e in enumerate(entries):
                if e.type == ActionType.GIVE:
                    return i
            return wait_idx
        return wait_idx  # send/recv: passive rendezvous wait
