"""carrier_position_at — the viz's single motion source.

The guarantee: the animated position is the closed-form integral of the same
trapezoidal profile the sim uses to schedule the move, so the sprite is at the
start at t0, at the target *exactly* at busy_until, monotonic in between, and
clamped outside the move window. The picture cannot disagree with the sim.
"""

from __future__ import annotations

from oos.facilities import get_facility
from oos.sim.actions import Goto, carrier_position_at
from oos.sim.durations import LinearDurations
from oos.sim.facility import SimEngine
from oos.sim.state import DockRef


def _engine(name: str = "tiny_medipol") -> SimEngine:
    topo, seed = get_facility(name)()
    engine = SimEngine(topology=topo, seeding=seed, durations=LinearDurations())
    engine.decision_predicate = lambda cid: False
    return engine


def _carrier_to_far_shelf(engine: SimEngine, topo):
    """Pick (carrier, shelf_id, start_pos, target_pos) the carrier can reach at
    a track position distinct from where it currently sits."""
    for cid, c in topo.carriers.items():
        cur = engine.state.carriers[cid].position
        best = None
        for sid, sh in topo.shelves.items():
            if cid in sh.access:
                pos = sh.position_for[cid]
                d = abs(pos - cur)
                if d > 0 and (best is None or d > best[0]):
                    best = (d, sid, pos)
        if best is not None:
            return cid, best[1], cur, best[2]
    raise AssertionError("no reachable shelf at a distinct position")


def test_position_tracks_profile_and_lands_on_target():
    topo, seed = get_facility("tiny_medipol")()
    engine = SimEngine(topology=topo, seeding=seed, durations=LinearDurations())
    engine.decision_predicate = lambda cid: False

    cid, sid, start, target = _carrier_to_far_shelf(engine, topo)
    t0 = engine.state.time
    engine.submit(Goto(cid, DockRef("shelf", sid)))

    cs = engine.state.carriers[cid]
    assert cs.is_busy
    profile = topo.carriers[cid].profile
    expected_T = profile.travel_time(abs(target - start))

    # The sim schedules the move's end to the profile travel time...
    assert abs((cs.busy_until - t0) - expected_T) < 1e-9
    # ...and the animation reaches the target at exactly that instant.
    assert abs(carrier_position_at(engine.state, topo, cid, t0) - start) < 1e-6
    assert abs(carrier_position_at(engine.state, topo, cid, cs.busy_until) - target) < 1e-6
    # Clamped before/after the window.
    assert abs(carrier_position_at(engine.state, topo, cid, t0 - 5.0) - start) < 1e-6
    assert abs(carrier_position_at(engine.state, topo, cid, cs.busy_until + 100) - target) < 1e-6

    # Monotonic toward the target across the window.
    sign = 1.0 if target > start else -1.0
    prev = None
    for k in range(11):
        t = t0 + (cs.busy_until - t0) * k / 10.0
        p = carrier_position_at(engine.state, topo, cid, t)
        assert min(start, target) - 1e-6 <= p <= max(start, target) + 1e-6
        if prev is not None:
            assert sign * (p - prev) >= -1e-6
        prev = p


def test_static_when_not_moving():
    engine = _engine("tiny_medipol")
    topo, _ = get_facility("tiny_medipol")()
    cid = next(iter(engine.state.carriers))
    cs = engine.state.carriers[cid]
    assert cs.current_command is None  # idle at construction
    p = carrier_position_at(engine.state, engine.topology, cid, engine.state.time)
    assert p == float(cs.position)
