"""Tests for the GOTO/TAKE/GIVE/WAIT primitive action model:

  * the action-index ↔ type alignment invariant (the one silent footgun),
  * the carrier→carrier handoff transfer (receiver-initiated, atomic),
  * the arrival-triggered store/retrieve serve at a room (no WAIT needed),
  * the no-immediate-inverse masking guard.
"""

from __future__ import annotations

import numpy as np

from oos.env.action import ActionType, enumerate_actions
from oos.env.env import Environment
from oos.facilities import get_facility
from oos.sim.actions import Give, Goto, Take
from oos.sim.durations import LinearDurations
from oos.sim.facility import SimEngine
from oos.sim.state import DockRef, Pallet
from oos.sim.tasks import Retrieve, Store


def _engine(name: str = "tiny_medipol") -> SimEngine:
    topo, seed = get_facility(name)()
    engine = SimEngine(topology=topo, seeding=seed, durations=LinearDurations())
    # No env decision predicate here, so `advance()` would otherwise return the
    # instant any carrier is idle (every carrier is, at t=0). For these unit
    # tests we drive specific carriers and want `advance()` to drain the
    # scheduler (process command_done events) — a never-needs-a-decision
    # predicate makes `carriers_needing_decision()` empty so it does.
    engine.decision_predicate = lambda cid: False
    return engine


def _run_to_idle(engine: SimEngine, carrier: str) -> None:
    """Advance until `carrier` is idle again (its command completed)."""
    for _ in range(100):
        engine.advance()
        if not engine.state.carriers[carrier].is_busy:
            return
    raise AssertionError(f"{carrier} never went idle")


# ---------------------------------------------------------------------------
# 1. The load-bearing invariant: enumerate order == collated type-per-slot
# ---------------------------------------------------------------------------


def test_action_index_type_alignment():
    import torch  # noqa: F401 — guard: skip silently if torch absent
    from oos.learn.batching import GraphCollator, sample_from_env_step

    for name in ("tiny_medipol", "stacker_deep", "campus"):
        env = Environment.from_name(name)
        obs, info = env.reset(seed=0)
        coll = GraphCollator(env.topology)
        rng = np.random.default_rng(0)
        for _ in range(30):
            entries = info["action_entries"]
            # Canonical order: all GOTO, then optional TAKE, then optional GIVE,
            # then optional WAIT (last). WAIT may be masked for a room carrier
            # docked at a shelf, so it is at most one and, when present, last.
            types = [e.type for e in entries]
            n_wait = types.count(ActionType.WAIT)
            assert n_wait <= 1
            if n_wait == 1:
                assert types[-1] == ActionType.WAIT
            # GOTO entries are a contiguous prefix; TAKE/GIVE (if present) come
            # after, before WAIT.
            seen_non_goto = False
            for t in types:
                if t == ActionType.GOTO:
                    assert not seen_non_goto, "GOTO after a non-GOTO entry"
                else:
                    seen_non_goto = True
            s = sample_from_env_step(obs, info, entries)
            batch = coll.collate([s], n_max=env.n_actions)
            tps = batch.type_per_slot[0].tolist()
            for i, e in enumerate(entries):
                assert int(e.type) == tps[i], (
                    f"{name}: slot {i} decodes {e.type} but collated {tps[i]}"
                )
            legal = np.flatnonzero(obs["action_mask"])
            obs, _r, term, trunc, info = env.step(int(rng.choice(legal)))
            if term or trunc:
                break


# ---------------------------------------------------------------------------
# 2. Carrier→carrier handoff transfer
# ---------------------------------------------------------------------------


def test_handoff_auto_transfers_when_empty_receiver_arrives_last():
    """A handoff is AUTOMATIC on rendezvous: when two partner carriers meet at
    the matching pose, one loaded + one empty, the pallet moves to the empty
    partner the instant the second one arrives — no GIVE/TAKE action. Here the
    loaded giver waits and the empty receiver arrives last."""
    engine = _engine("tiny_medipol")
    topo = engine.topology
    # A lift L1 and a shuttle S1 are handoff partners in tiny_medipol.
    lift, shuttle = "L1", "S1"
    assert shuttle in topo.handoff_partners[lift]

    # Put a known item on one of the shuttle's shelves; shuttle picks it up.
    shelf = sorted(topo.accessible_shelves[shuttle])[0]
    engine.state.shelves[shelf].stack.append(Pallet(id=4242, contents="small"))
    engine.submit(Goto(carrier_id=shuttle, target=DockRef("shelf", shelf)))
    _run_to_idle(engine, shuttle)
    engine.submit(Take(carrier_id=shuttle))
    _run_to_idle(engine, shuttle)
    assert engine.state.carriers[shuttle].load.id == 4242

    # Loaded shuttle: GOTO its handoff pose toward the lift, then WAIT.
    engine.submit(Goto(carrier_id=shuttle, target=DockRef("handoff", lift)))
    _run_to_idle(engine, shuttle)
    engine.wait(shuttle)

    # Empty lift: GOTO the matching pose. Its ARRIVAL auto-fires the transfer —
    # no TAKE/GIVE is submitted by anyone.
    engine.submit(Goto(carrier_id=lift, target=DockRef("handoff", shuttle)))
    _run_to_idle(engine, lift)

    # Item moved to the lift; both released and empty/loaded correctly.
    assert engine.state.carriers[lift].load is not None
    assert engine.state.carriers[lift].load.id == 4242
    assert engine.state.carriers[shuttle].load is None
    assert not engine.state.carriers[lift].is_busy
    assert not engine.state.carriers[shuttle].is_busy


def test_handoff_auto_transfers_when_loaded_carrier_arrives_last():
    """Symmetric: the empty receiver waits and the loaded giver arrives last.
    The transfer still auto-fires on rendezvous, and there is no manual
    handoff GIVE/TAKE action enumerated at a handoff pose."""
    engine = _engine("tiny_medipol")
    topo = engine.topology
    lift, shuttle = "L1", "S1"

    shelf = sorted(topo.accessible_shelves[shuttle])[0]
    engine.state.shelves[shelf].stack.append(Pallet(id=4242, contents="small"))
    engine.submit(Goto(carrier_id=shuttle, target=DockRef("shelf", shelf)))
    _run_to_idle(engine, shuttle)
    engine.submit(Take(carrier_id=shuttle))
    _run_to_idle(engine, shuttle)
    assert engine.state.carriers[shuttle].load.id == 4242

    # Empty lift (the receiver): GOTO its pose, then WAIT.
    engine.submit(Goto(carrier_id=lift, target=DockRef("handoff", shuttle)))
    _run_to_idle(engine, lift)
    engine.wait(lift)
    assert engine.state.carriers[lift].load is None

    # Loaded shuttle arrives last at the matching pose. Handoffs are automatic,
    # so even before the transfer there is NO manual GIVE action to enumerate.
    engine.submit(Goto(carrier_id=shuttle, target=DockRef("handoff", lift)))
    _run_to_idle(engine, shuttle)

    # Transfer fired on arrival.
    assert engine.state.carriers[lift].load is not None
    assert engine.state.carriers[lift].load.id == 4242
    assert engine.state.carriers[shuttle].load is None
    assert not engine.state.carriers[lift].is_busy
    assert not engine.state.carriers[shuttle].is_busy
    # No manual handoff action is offered at a handoff pose (auto only).
    entries = enumerate_actions(lift, engine.state, topo, engine.queue)
    assert all(e.type not in (ActionType.TAKE, ActionType.GIVE) for e in entries)


# ---------------------------------------------------------------------------
# 3. Arrival-triggered serve at a room (no WAIT needed)
# ---------------------------------------------------------------------------


def test_arrival_serves_store_into_held_empty():
    engine = _engine("tiny_medipol")
    room = next(iter(engine.topology.rooms))
    carrier = engine.topology.rooms[room].served_by
    cs = engine.state.carriers[carrier]
    cs.load = Pallet(id=7, contents="empty")
    cs.docked_at = DockRef("room", room)   # carrier idle at the room, empty

    # A car is queued while the carrier sits there → loaded immediately, no WAIT
    # (enqueue_store scans rooms for a ready serve).
    engine.enqueue_store("small")

    assert cs.load is not None and cs.load.contents == "small"
    assert not any(isinstance(t, Store) for t in engine.queue.pending)
    assert any(isinstance(c.task, Store) for c in engine._pending_completions)


def test_camped_retrieve_is_not_an_agent_delivery():
    """A carrier camping at a room with a stored car until it is asked for must
    NOT earn a DELIVER (the parked-car exploit). A Retrieve issued while the
    target is already at a room is tagged not-agent-delivered, and is served
    immediately on injection."""
    engine = _engine("tiny_medipol")
    room = next(iter(engine.topology.rooms))
    car = engine.topology.rooms[room].served_by
    cs = engine.state.carriers[car]
    cs.docked_at = DockRef("room", room)
    cs.load = Pallet(id=55, contents="big")   # a stored car camped at the room

    # Requesting the car while it already sits at the room serves it at once,
    # tagged not-agent-delivered (camped) → no DELIVER.
    assert engine.toggle_retrieve_for_pallet(55) is True
    comps = [c for c in engine._pending_completions if isinstance(c.task, Retrieve)]
    assert comps and comps[0].agent_delivered is False
    assert cs.load is not None and cs.load.is_empty   # car handed to the customer
    # A target NOT sitting at a room (e.g. on a shelf) is a real delivery.
    assert engine._target_already_staged(999) is False


def test_arrival_serves_retrieve_from_held_target():
    engine = _engine("tiny_medipol")
    room = next(iter(engine.topology.rooms))
    carrier = engine.topology.rooms[room].served_by
    cs = engine.state.carriers[carrier]
    # Car requested while still in transit (not yet at the room) → a real,
    # agent-credited delivery once the carrier arrives.
    cs.load = Pallet(id=99, contents="big")
    cs.docked_at = None
    engine.queue.add(Retrieve(
        arrived_at=0.0, pallet=99, initial_depth=1,
        already_staged=engine._target_already_staged(99),   # False: not at a room
    ))

    # Carrier arrives at the room → arrival-triggered serve fires.
    cs.docked_at = DockRef("room", room)
    assert engine._serve_ready_rooms(engine._pending_completions)

    # The target is consumed; an empty pallet is left on the carrier.
    assert cs.load is not None and cs.load.is_empty and cs.load.id == 99
    assert not any(isinstance(t, Retrieve) for t in engine.queue.pending)
    comps = [c for c in engine._pending_completions if isinstance(c.task, Retrieve)]
    assert comps and comps[0].agent_delivered


# ---------------------------------------------------------------------------
# 4. No-immediate-inverse masking guard
# ---------------------------------------------------------------------------


def test_no_immediate_reverse_goto():
    """After GOTOing to a shelf/handoff, going straight back to where it came
    from (no TAKE/GIVE in between) is masked — except returning to a room."""
    engine = _engine("tiny_medipol")
    carrier = "L1"
    topo = engine.topology
    room = next(iter(topo.accessible_rooms[carrier]))
    shelf_a = sorted(topo.accessible_shelves[carrier])[0]
    shelf_b = sorted(topo.accessible_shelves[carrier])[1]
    cs = engine.state.carriers[carrier]

    def goto_targets():
        return {
            e.target
            for e in enumerate_actions(carrier, engine.state, topo, engine.queue)
            if e.type == ActionType.GOTO
        }

    # --- shelf reverse is masked ---
    engine.submit(Goto(carrier_id=carrier, target=DockRef("shelf", shelf_a)))
    _run_to_idle(engine, carrier)
    engine.submit(Goto(carrier_id=carrier, target=DockRef("shelf", shelf_b)))
    _run_to_idle(engine, carrier)
    assert DockRef("shelf", shelf_a) not in goto_targets()   # reverse masked
    assert any(t.kind == "shelf" for t in goto_targets())    # others reachable

    # A TAKE at shelf_b lifts the guard → reverse to shelf_a allowed again.
    engine.state.shelves[shelf_b].stack = [Pallet(id=7, contents="small")]
    engine.submit(Take(carrier_id=carrier))
    _run_to_idle(engine, carrier)
    assert DockRef("shelf", shelf_a) in goto_targets()

    # --- reverse to a ROOM is always allowed ---
    cs.load = Pallet(id=8, contents="empty")     # so GOTO(room) passes its own gate
    cs.docked_at = DockRef("room", room)
    cs.came_from = None
    cs.last_take_give = None
    engine.submit(Goto(carrier_id=carrier, target=DockRef("shelf", shelf_a)))
    _run_to_idle(engine, carrier)                # came_from is now the room
    assert DockRef("room", room) in goto_targets()


def test_wait_is_legal_anywhere_for_every_carrier():
    """WAIT is now unconditionally legal: any carrier may rest anywhere — at a
    shelf, at a room, at a handoff pose, or undocked. The loiter mask is gone;
    the all-wait stall is handled by the env's penalty + re-query rescue."""
    engine = _engine("tiny_medipol")
    topo = engine.topology

    def has_wait(c):
        return any(
            e.type == ActionType.WAIT
            for e in enumerate_actions(c, engine.state, topo, engine.queue)
        )

    for carrier in topo.carriers:
        cs = engine.state.carriers[carrier]
        # At a shelf — previously masked for room-serving carriers.
        shelf = sorted(topo.accessible_shelves[carrier])[0]
        cs.docked_at = DockRef("shelf", shelf)
        assert has_wait(carrier), f"{carrier} should be able to WAIT at a shelf"
        # At a room (room carriers only).
        rooms = sorted(topo.accessible_rooms[carrier])
        if rooms:
            cs.docked_at = DockRef("room", rooms[0])
            assert has_wait(carrier)
        # Undocked.
        cs.docked_at = None
        assert has_wait(carrier)


def test_no_immediate_give_back_after_take():
    engine = _engine("stacker_deep")
    carrier = "L1"
    shelf = sorted(engine.topology.accessible_shelves[carrier])[0]
    # A single item on an otherwise-empty shelf (so capacity is never the
    # reason GIVE is unavailable).
    engine.state.shelves[shelf].stack = [Pallet(id=1, contents="small")]

    engine.submit(Goto(carrier_id=carrier, target=DockRef("shelf", shelf)))
    _run_to_idle(engine, carrier)
    engine.submit(Take(carrier_id=carrier))
    _run_to_idle(engine, carrier)
    # Just took from `shelf` and still docked there → GIVE back is masked.
    entries = enumerate_actions(carrier, engine.state, engine.topology, engine.queue)
    assert not any(e.type == ActionType.GIVE for e in entries), (
        "GIVE-back onto the just-taken shelf must be masked"
    )
    # A GIVE would otherwise be physically legal (shelf has room) — prove the
    # guard is what suppressed it: clearing it re-enables GIVE.
    engine.state.carriers[carrier].last_take_give = None
    entries2 = enumerate_actions(carrier, engine.state, engine.topology, engine.queue)
    assert any(e.type == ActionType.GIVE for e in entries2)
