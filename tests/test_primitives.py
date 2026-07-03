"""Tests for the GOTO/TAKE/GIVE/WAIT primitive action model:

  * the canonical enumeration order (GOTO..., TAKE, GIVE, WAIT last),
  * the carrier→carrier handoff transfer (receiver-initiated, atomic),
  * the arrival-triggered store/retrieve serve at a room (no WAIT needed).
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
# 1. Canonical enumeration order
# ---------------------------------------------------------------------------


def test_action_enumeration_order():
    """Canonical order: all GOTO, then optional TAKE, then optional GIVE, then
    WAIT (always present, always last)."""
    for name in ("tiny_medipol", "stacker_deep", "campus"):
        env = Environment.from_name(name)
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(30):
            entries = list(env._ctx.decoder.entries)
            types = [e.type for e in entries]
            assert types.count(ActionType.WAIT) == 1
            assert types[-1] == ActionType.WAIT
            # GOTO entries are a contiguous prefix; TAKE/GIVE (if present)
            # come after, before WAIT.
            seen_non_goto = False
            for t in types:
                if t == ActionType.GOTO:
                    assert not seen_non_goto, "GOTO after a non-GOTO entry"
                else:
                    seen_non_goto = True
            env.submit_action(int(rng.integers(len(entries))))
            env.advance_until(sim_time=None)
            if not env.needs_decision():
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
# 4. WAIT legality
# ---------------------------------------------------------------------------


def test_wait_is_legal_anywhere_for_every_carrier():
    """WAIT is unconditionally legal: any carrier may rest anywhere — at a
    shelf, at a room, at a handoff pose, or undocked."""
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

