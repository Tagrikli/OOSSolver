"""Tests for the GOTO/TAKE/GIVE/WAIT primitive action model:

  * the action-index ↔ type alignment invariant (the one silent footgun),
  * the carrier→carrier handoff transfer (receiver-initiated, atomic),
  * the WAIT-triggered store/retrieve serve at a room,
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
            # then exactly one WAIT (last).
            types = [e.type for e in entries]
            assert types[-1] == ActionType.WAIT
            assert types.count(ActionType.WAIT) == 1
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


def test_handoff_transfer_moves_item_and_releases_both():
    engine = _engine("tiny_medipol")
    topo = engine.topology
    # A lift L1 and a shuttle S1 are handoff partners in tiny_medipol.
    lift, shuttle = "L1", "S1"
    assert shuttle in topo.handoff_partners[lift]

    # Put a known item on one of the shuttle's shelves.
    shelf = sorted(topo.accessible_shelves[shuttle])[0]
    engine.state.shelves[shelf].stack.append(Pallet(id=4242, contents="small"))

    # Shuttle: GOTO that shelf, TAKE the item.
    engine.submit(Goto(carrier_id=shuttle, target=DockRef("shelf", shelf)))
    _run_to_idle(engine, shuttle)
    engine.submit(Take(carrier_id=shuttle))
    _run_to_idle(engine, shuttle)
    assert engine.state.carriers[shuttle].load is not None
    assert engine.state.carriers[shuttle].load.id == 4242

    # Shuttle: GOTO its handoff pose toward the lift, then WAIT (the giver).
    engine.submit(Goto(carrier_id=shuttle, target=DockRef("handoff", lift)))
    _run_to_idle(engine, shuttle)
    engine.wait(shuttle)
    assert engine.state.carriers[shuttle].waiting

    # Lift: GOTO its matching handoff pose toward the shuttle, then TAKE — this
    # is the receiver-initiated transfer; it locks the waiting shuttle.
    engine.submit(Goto(carrier_id=lift, target=DockRef("handoff", shuttle)))
    _run_to_idle(engine, lift)
    # The lift's arrival (a state-change event) woke the waiting shuttle; in the
    # env loop the shuttle is re-queried and re-chooses WAIT. Model that.
    engine.wait(shuttle)
    take = Take(carrier_id=lift)
    assert take.partner_to_receive_from(engine.state, topo) == shuttle
    engine.submit(take)
    # While the transfer is in flight both carriers are locked on the SAME cmd.
    assert engine.state.carriers[shuttle].is_busy
    _run_to_idle(engine, lift)

    # Item moved to the lift; both carriers released and empty/loaded correctly.
    assert engine.state.carriers[lift].load is not None
    assert engine.state.carriers[lift].load.id == 4242
    assert engine.state.carriers[shuttle].load is None
    assert not engine.state.carriers[lift].is_busy
    assert not engine.state.carriers[shuttle].is_busy


# ---------------------------------------------------------------------------
# 3. WAIT-triggered serve at a room
# ---------------------------------------------------------------------------


def test_wait_serves_store_into_held_empty():
    engine = _engine("tiny_medipol")
    room = next(iter(engine.topology.rooms))
    carrier = engine.topology.rooms[room].served_by
    cs = engine.state.carriers[carrier]
    cs.load = Pallet(id=7, contents="empty")
    cs.docked_at = DockRef("room", room)
    engine.queue.add(Store(arrived_at=0.0, size="small"))

    engine.wait(carrier)   # the serve fires here, not on arrival

    assert cs.load is not None and cs.load.contents == "small"
    assert not any(isinstance(t, Store) for t in engine.queue.pending)
    assert any(isinstance(c.task, Store) for c in engine._pending_completions)


def test_camped_retrieve_is_not_an_agent_delivery():
    """A carrier camping at a room with a stored car until it is asked for must
    NOT earn a DELIVER (the parked-car exploit). A Retrieve issued while the
    target is already at a room is tagged not-agent-delivered."""
    engine = _engine("tiny_medipol")
    room = next(iter(engine.topology.rooms))
    car = engine.topology.rooms[room].served_by
    cs = engine.state.carriers[car]
    cs.docked_at = DockRef("room", room)
    cs.load = Pallet(id=55, contents="big")   # a stored car camped at the room
    # Issue the retrieve now — the target is already at the room → camped.
    engine.queue.add(Retrieve(
        arrived_at=0.0, pallet=55, initial_depth=0,
        already_staged=engine._target_already_staged(55),
    ))
    assert engine._find_pending_retrieve(55).already_staged is True
    engine.wait(car)   # serves the retrieve
    comps = [c for c in engine._pending_completions if isinstance(c.task, Retrieve)]
    assert comps and comps[0].agent_delivered is False
    # A target NOT sitting at a room (e.g. on a shelf) is a real delivery.
    assert engine._target_already_staged(999) is False


def test_wait_serves_retrieve_from_held_target():
    engine = _engine("tiny_medipol")
    room = next(iter(engine.topology.rooms))
    carrier = engine.topology.rooms[room].served_by
    cs = engine.state.carriers[carrier]
    cs.load = Pallet(id=99, contents="big")
    cs.docked_at = DockRef("room", room)
    engine.queue.add(Retrieve(arrived_at=0.0, pallet=99, initial_depth=1))

    engine.wait(carrier)

    # The target is consumed; an empty pallet is left on the carrier.
    assert cs.load is not None and cs.load.is_empty and cs.load.id == 99
    assert not any(isinstance(t, Retrieve) for t in engine.queue.pending)
    comps = [c for c in engine._pending_completions if isinstance(c.task, Retrieve)]
    assert comps and comps[0].agent_delivered


# ---------------------------------------------------------------------------
# 4. No-immediate-inverse masking guard
# ---------------------------------------------------------------------------


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
