"""MovePolicyBridge — drive the viz's primitive Environment with a trained
move-level agent (SOLUTION_V2 stack).

The viz queries a `PolicyFn (obs, info) -> action_idx` once per idle carrier.
The bridge keeps a MoveExecutor + SolvabilityOracle bound to the session's
LIVE engine: at each query it syncs executor scripts against the engine,
starts new moves when the (greedy) move policy dispatches them, and answers
with the primitive action index for the querying carrier — GOTO/TAKE/GIVE
translated into the env's enumerated entries, WAIT for passive rendezvous
steps and HOLD.

The env's RL-only guards (reverse-GOTO / immediate-inverse) are disabled
while a bridge drives (Environment._policy_guards=False — the documented
planner escape), because executor scripts are physically legal by
construction and must never be masked away.

Self-healing: any manual world edit that invalidates in-flight scripts
(re-roll layout, facility swap, reset) is caught — on engine identity change
or any internal inconsistency the bridge drops its claims and rebuilds from
the current state. A dropped move costs a few idle seconds, nothing more.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from oos.env.action import ActionType
from oos.env.moves import MoveExecutor
from oos.env.observation import ObservationBuilder, ObservationConfig
from oos.learn.move_net import MoveCollator, MoveNetConfig, MovePolicyNet
from oos.plan.oracle import SolvabilityOracle
from oos.sim.tasks import Retrieve, Store


class MovePolicyBridge:
    """PolicyFn adapter for move-level checkpoints (runs/move/*.pt)."""

    def __init__(self, env, checkpoint_path: str, device: str = "cpu") -> None:
        self._init_common(env)
        self.device = torch.device(device)
        blob = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        cfg = MoveNetConfig(**blob["net_cfg"])
        n_c, n_s, n_r = (len(self.carrier_ids), len(self.shelf_ids),
                         len(self.room_ids))
        from oos.env.observation import (
            CARRIER_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            ROOM_FEATURE_NAMES,
            shelf_feature_count,
        )
        self.net = MovePolicyNet(
            carrier_dim=len(CARRIER_FEATURE_NAMES) + 1,
            shelf_dim=shelf_feature_count() + 2,
            room_dim=len(ROOM_FEATURE_NAMES),
            global_dim=len(GLOBAL_FEATURE_NAMES) + 1,
            cfg=cfg,
        ).to(self.device)
        self.net.load_state_dict(blob["state_dict"])
        self.net.eval()
        self.collator = MoveCollator(n_c, n_s, n_r)
        self.iteration = int(blob.get("meta", {}).get("iter", -1))

    def _init_common(self, env) -> None:
        self.env = env
        topo = env.topology
        self._topo = topo
        self._obs_builder = ObservationBuilder(topo, ObservationConfig())
        self.carrier_ids = self._obs_builder.carrier_ids
        self.shelf_ids = self._obs_builder.shelf_ids
        self.room_ids = self._obs_builder.room_ids
        n_c, n_s = len(self.carrier_ids), len(self.shelf_ids)
        self.n_src = n_s + n_c + 1
        self.n_dst = n_s + len(self.room_ids)
        self.hold_idx = self.n_src - 1
        self._shelf_slot = {sid: i for i, sid in enumerate(self.shelf_ids)}
        self._carrier_slot = {cid: n_s + i
                              for i, cid in enumerate(self.carrier_ids)}
        self._room_dst = {rid: n_s + i for i, rid in enumerate(self.room_ids)}

        self._engine = None
        self.ex: Optional[MoveExecutor] = None
        self.oracle: Optional[SolvabilityOracle] = None
        self._completed_stamp = 0
        self._task_sig: tuple = ()
        self._inverse_block = None
        self._arrangements: set[int] = set()
        self.notes: list[str] = []   # surfaced by the session log

        # Compatibility with the viz's action-dist panel (unused → panel idle).
        self.last_logits = None
        self.last_action_mask = None
        self.last_chosen = None

    # ------------------------------------------------------------------

    def _rebind(self) -> None:
        engine = self.env.engine
        if engine is self._engine:
            return
        self._engine = engine
        self.oracle = SolvabilityOracle(engine.topology, max_holds=1)
        self.ex = MoveExecutor(engine, self.oracle)
        self._completed_stamp = 0
        self._task_sig = ()
        self._inverse_block = None
        self._arrangements = set()
        self.notes.append("move-bridge bound to engine")

    def _drop_state(self) -> None:
        """Emergency reset after an inconsistency (manual world edit mid-move):
        forget all claims/locks; carriers finish their current primitive and
        then idle until re-dispatched."""
        self._engine = None

    # ------------------------------------------------------------------
    # Legality (mirrors MoveEnv._refresh_legal + the inverse guard)
    # ------------------------------------------------------------------

    def _queue_sig(self, engine) -> tuple:
        return tuple(sorted(
            (type(t).__name__, getattr(t, "pallet", getattr(t, "size", "")))
            for t in engine.queue.pending
        ))

    def _update_inverse_block(self, engine) -> None:
        assert self.ex is not None
        sig = self._queue_sig(engine)
        if sig != self._task_sig:
            self._arrangements.clear()
        if self.ex is not None and self.ex.n_inflight == 0:
            self._arrangements.add(self._arrangement_hash(engine))
        n_done = self.ex.completed_moves
        if n_done != self._completed_stamp:
            delta = n_done - self._completed_stamp
            self._completed_stamp = n_done
            self._inverse_block = (
                self.ex.last_completed
                if (delta == 1 and sig == self._task_sig) else None
            )
        elif sig != self._task_sig:
            self._inverse_block = None
        self._task_sig = sig

    def _room_disposition(self, engine) -> dict:
        out = {}
        for rid in self.room_ids:
            scs = engine.state.carriers[self._topo.rooms[rid].served_by]
            if (scs.docked_at is not None and scs.docked_at.kind == "room"
                    and scs.docked_at.id == rid and scs.load is not None):
                out[rid] = scs.load.id
            else:
                out[rid] = None
        return out

    def _arrangement_hash(self, engine) -> int:
        rooms = tuple(sorted(self._room_disposition(engine).items()))
        return hash((tuple(
            (sid, tuple((p.id, p.contents) for p in engine.state.shelves[sid].stack))
            for sid in self.shelf_ids
        ), rooms))

    def _predicted_arrangement(self, engine, mv) -> int:
        parts = []
        for sid in self.shelf_ids:
            col = [(p.id, p.contents) for p in engine.state.shelves[sid].stack]
            if mv.src_kind == "shelf" and mv.src_id == sid and col:
                col = col[:-1]
            if mv.dst_kind == "shelf" and mv.dst_id == sid:
                col = col + [(mv.pallet_id, mv.contents)]
            parts.append((sid, tuple(col)))
        rooms = self._room_disposition(engine)
        if mv.src_kind == "carrier":
            cs = engine.state.carriers[mv.src_id]
            if cs.docked_at is not None and cs.docked_at.kind == "room":
                rooms[cs.docked_at.id] = None
        if mv.dst_kind == "room":
            rooms[mv.dst_id] = mv.pallet_id
        return hash((tuple(parts), tuple(sorted(rooms.items()))))

    def _inverse_blocked(self, mv) -> bool:
        blk = self._inverse_block
        if blk is None:
            return False
        pid, from_key, to_key = blk
        if mv.pallet_id != pid or from_key is None:
            return False
        if mv.src_kind == "shelf":
            mv_from = ("shelf", mv.src_id)
        else:
            d = self._engine.state.carriers[mv.src_id].docked_at
            mv_from = (d.kind, d.id) if d is not None else None
        return mv_from == to_key and (mv.dst_kind, mv.dst_id) == from_key

    def _work_pending(self, engine) -> bool:
        assert self.ex is not None
        if engine.queue.pending:
            return True
        for rid in self.room_ids:
            if not self._room_staged(engine, rid):
                return True
        for cid, cs in engine.state.carriers.items():
            if self.ex.is_claimed(cid):
                continue
            if cs.load is not None and not cs.load.is_empty:
                return True
        return False

    def _room_staged(self, engine, rid: str) -> bool:
        scs = engine.state.carriers[self._topo.rooms[rid].served_by]
        return (scs.docked_at is not None and scs.docked_at.kind == "room"
                and scs.docked_at.id == rid and scs.load is not None
                and scs.load.is_empty)

    def _staging_owed(self, engine, legal) -> bool:
        assert self.ex is not None
        inbound = self.ex.inflight_dst_rooms()
        unexcused = {
            rid for rid in self.room_ids
            if not self._room_staged(engine, rid)
            and rid not in inbound
            and not self.ex.is_claimed(self._topo.rooms[rid].served_by)
        }
        if not unexcused:
            return False
        for mv in legal.values():
            if (mv.dst_kind == "room" and mv.contents == "empty"
                    and mv.dst_id in unexcused and len(mv.chain) == 1):
                return True
        return False

    def admission_ok_for(self, size: str) -> bool:
        """The deployment admission gate, exposed for the session's manual
        store buttons: admit iff SOME placement of the new car keeps the
        future view solvable AND a storable ordering exists for the held
        cars + the new one (hands-aware — see the stall-dump fix)."""
        self._rebind()
        assert self.ex is not None and self.oracle is not None
        if size != "big":
            return True   # net-zero service cycle — see oracle.admission_ok
        if not self.oracle.admission_ok(self.ex.future_view(), size):
            return False
        engine = self.env.engine
        held = []
        for cid, cs in engine.state.carriers.items():
            if self.ex.is_claimed(cid):
                continue
            if cs.load is not None and not cs.load.is_empty:
                held.append((cid, cs.load.contents))
        serving = {self._topo.rooms[rid].served_by for rid in self.room_ids}
        for scid in serving:
            if any(cid == scid for cid, _ in held):
                continue
            if not self.ex.held_set_storable(held + [(scid, size)]):
                return False
        return True

    def _guard_unfundable_stores(self, engine) -> None:
        """Manual mode can inject stores the admission gate would refuse; a
        pending unfundable store masks EVERY move (its projected car can't be
        funded), freezing the carriers. Detect that and plan without the
        store projection until the queue drains — the store simply waits."""
        assert self.ex is not None and self.oracle is not None
        ex = self.ex
        has_stores = any(isinstance(t, Store) for t in engine.queue.pending)
        if not has_stores:
            ex.project_pending_stores = True
            return
        ex.project_pending_stores = True
        if self.oracle.check_view(ex.future_view()):
            return
        ex.project_pending_stores = False
        if self.oracle.check_view(ex.future_view()):
            self.notes.append("⚠ unfundable STORE pending — planning without it")
        # else: the world itself is unsolvable (manual over-stuffing);
        # carriers will hold. Leave projection off so any solvable-again
        # transition resumes planning immediately.

    def _legal_moves(self, engine):
        assert self.ex is not None
        self._update_inverse_block(engine)
        self._guard_unfundable_stores(engine)
        legal: dict[tuple[int, int], object] = {}
        blocked = []
        for mv in self.ex.iter_startable():
            if self._inverse_blocked(mv):
                blocked.append(mv)
                continue
            if (self.ex.n_inflight == 0
                    and self._predicted_arrangement(engine, mv)
                    in self._arrangements):
                blocked.append(mv)
                continue
            si = (self._shelf_slot[mv.src_id] if mv.src_kind == "shelf"
                  else self._carrier_slot[mv.src_id])
            di = (self._shelf_slot[mv.dst_id] if mv.dst_kind == "shelf"
                  else self._room_dst[mv.dst_id])
            prev = legal.get((si, di))
            if prev is None or mv.est_makespan < prev.est_makespan:
                legal[(si, di)] = mv
        if not legal and blocked and self._work_pending(engine) \
                and self.ex.n_inflight == 0:
            for mv in blocked:
                si = (self._shelf_slot[mv.src_id] if mv.src_kind == "shelf"
                      else self._carrier_slot[mv.src_id])
                di = (self._shelf_slot[mv.dst_id] if mv.dst_kind == "shelf"
                      else self._room_dst[mv.dst_id])
                legal[(si, di)] = mv
        return legal

    # ------------------------------------------------------------------
    # Observation (mirrors MoveEnv._build_obs)
    # ------------------------------------------------------------------

    def _build_obs(self, engine, legal) -> dict:
        assert self.ex is not None
        ex = self.ex
        base = self._obs_builder.build(engine, engine.queue,
                                       self.carrier_ids[0])
        n_c, n_s = len(self.carrier_ids), len(self.shelf_ids)
        claimed = np.zeros((n_c, 1), dtype=np.float32)
        for i, cid in enumerate(self.carrier_ids):
            if ex.is_claimed(cid):
                claimed[i, 0] = 1.0
        cf = np.concatenate([base["carrier_features"], claimed], axis=1)
        locks = np.zeros((n_s, 2), dtype=np.float32)
        for i, sid in enumerate(self.shelf_ids):
            if sid in ex.src_locked:
                locks[i, 0] = 1.0
            if sid in ex.dst_locked:
                locks[i, 1] = 1.0
        sf = np.concatenate([base["shelf_features"], locks], axis=1)
        gf = np.concatenate([
            base["global_features"],
            np.array([ex.n_inflight / max(1, n_c)], dtype=np.float32),
        ])
        inflight_edges = []
        s_off, r_off = n_c, n_c + n_s
        for ms in ex.inflight:
            mv = ms.move
            src_node = (s_off + self._shelf_slot[mv.src_id]
                        if mv.src_kind == "shelf"
                        else self.carrier_ids.index(mv.src_id))
            dst_node = (s_off + self._shelf_slot[mv.dst_id]
                        if mv.dst_kind == "shelf"
                        else r_off + self.room_ids.index(mv.dst_id))
            inflight_edges.append((src_node, dst_node))
        edges_inflight = (np.array(inflight_edges, dtype=np.int64).T
                          if inflight_edges else np.zeros((2, 0), dtype=np.int64))
        src_mask = np.zeros(self.n_src, dtype=np.int8)
        dst_mask = np.zeros((self.n_src, self.n_dst), dtype=np.int8)
        for (si, di) in legal:
            src_mask[si] = 1
            dst_mask[si, di] = 1
        hold_ok = (not self._work_pending(engine)) or ex.n_inflight > 0
        if hold_ok and not self._staging_owed(engine, legal):
            src_mask[self.hold_idx] = 1
        return {
            "carrier_features": cf, "shelf_features": sf,
            "room_features": base["room_features"], "global_features": gf,
            "edges_accesses": base["edges_accesses"],
            "edges_handoff": base["edges_handoff"],
            "edges_transfer": base["edges_transfer"],
            "edges_docked": base["edges_docked"],
            "edges_inflight": edges_inflight,
            "src_mask": src_mask, "dst_mask": dst_mask,
        }

    # ------------------------------------------------------------------
    # The PolicyFn surface
    # ------------------------------------------------------------------

    def __call__(self, obs: dict, info: dict) -> int:
        # Index against the env's LIVE decoder — never the (possibly stale)
        # cached info entries: submit_action decodes against the live list,
        # and an index computed on a stale list aliases to a wrong primitive.
        try:
            entries = list(self.env._ctx.decoder.entries)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            entries = info.get("action_entries", [])
        wait_idx = max(0, len(entries) - 1)
        try:
            self._rebind()
            engine = self.env.engine
            ex = self.ex
            assert ex is not None
            cid = self.env.querying_carrier
            # Sync every claimed carrier (releases finished moves).
            for c in list(ex.claimed.keys()):
                ex.sync_role(c)
            # Dispatch new moves while the policy wants to (bounded).
            for _ in range(len(self.carrier_ids)):
                if ex.is_claimed(cid):
                    break
                legal = self._legal_moves(engine)
                if not legal:
                    break
                mv = self._select_move(engine, legal)
                if mv is None:
                    break
                ex.start(mv, serves_retrieve=self._serves_retrieve(engine, mv))
                engine.wake_waiting_carriers()
                self.notes.append(f"⚙ {mv.describe()}")
            # Answer for the querying carrier.
            step = ex.sync_role(cid)
            if step is None:
                return wait_idx
            return self._entry_index(step, entries, wait_idx)
        except Exception as e:  # noqa: BLE001 — self-heal on any surprise
            self.notes.append(f"bridge reset: {type(e).__name__}: {e}"[:70])
            self._drop_state()
            return wait_idx

    def _select_move(self, engine, legal):
        """Pick the next move to start, or None to stop dispatching. The RL
        bridge builds the move-level obs and asks the net; subclasses may
        substitute any other brain."""
        mv_obs = self._build_obs(engine, legal)
        if not mv_obs["src_mask"].any():
            return None
        from oos.learn.move_eval import greedy_action
        a = greedy_action(self.net, self.collator, mv_obs, device=self.device)
        if a[0] == self.hold_idx:
            return None
        return legal.get((a[0], a[1]))

    def _serves_retrieve(self, engine, mv) -> bool:
        requested = {t.pallet for t in engine.queue.pending
                     if isinstance(t, Retrieve)}
        if not requested:
            return False
        if mv.pallet_id in requested:
            return True
        if mv.src_kind == "shelf":
            return any(p.id in requested
                       for p in engine.state.shelves[mv.src_id].stack)
        return False

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


class ClassicalPolicyBridge(MovePolicyBridge):
    """PolicyFn adapter for the V3 plan solver (oos/plan/solver.py,
    SOLUTION_V3): the solver owns dispatch (plans + rungs) and starts
    executor moves itself; this bridge just ticks it once per query and
    answers with the querying carrier's current primitive. Self-healing:
    any engine swap or internal surprise drops all state and rebuilds from
    the live world (plans are recomputed — nothing is lost but seconds)."""

    def __init__(self, env) -> None:
        self._init_common(env)
        self.solver = None
        self.iteration = -1

    def _rebind(self) -> None:
        engine = self.env.engine
        if engine is self._engine:
            return
        self._engine = engine
        self.oracle = SolvabilityOracle(engine.topology, max_holds=1)
        self.ex = MoveExecutor(engine, self.oracle)
        from oos.plan.solver import PlanSolver
        self.solver = PlanSolver(engine, self.ex)
        self.notes.append("V3 plan solver bound to engine")

    def admission_ok_for(self, size: str) -> bool:
        """Deployment admission gate for the session's manual store
        buttons — the solver's reservation-aware oracle check."""
        self._rebind()
        assert self.solver is not None
        return self.solver.admission_ok(size)

    def __call__(self, obs: dict, info: dict) -> int:
        try:
            entries = list(self.env._ctx.decoder.entries)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            entries = info.get("action_entries", [])
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
            if solver.notes:
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
