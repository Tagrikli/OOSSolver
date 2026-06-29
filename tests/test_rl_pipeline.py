"""Tests for the RL deployment pipeline: the sound SUV admission gate, the
streaming env mode, and (if present) the trained deliverable policy."""

import os

import numpy as np
import pytest

from oos.facilities import get_facility


def _engine():
    from oos.sim.durations import LinearDurations
    from oos.sim.facility import SimEngine
    topo, seeding = get_facility("tiny_medipol")()
    return SimEngine(topology=topo, seeding=seeding, durations=LinearDurations(),
                     rng=np.random.default_rng(0)), topo


def _set_big_layout(eng, stacks):
    from oos.sim.state import Pallet
    pid = 1
    for ss in eng.state.shelves.values():
        ss.stack = []
    for sid, contents in stacks.items():
        eng.state.shelves[sid].stack = [Pallet(pid + i, c) for i, c in enumerate(contents)]
        pid += len(contents)


def test_sound_suv_gate_rejects_unsolvable_admits_safe():
    """The sound SUV gate must reject a big that would bury an item unretrievably
    and admit one when there is genuine room — matching the solver oracle."""
    from oos.learn.continuous import install_sound_suv_gate
    from oos.solver.solver import Solver
    from oos.solver.world import World
    from oos.sim.state import Pallet

    bigs = ("A1", "A2", "E1", "E2", "B1", "B2", "D1", "D2")

    # Packed: every big shelf full of bigs → no SUV can be admitted.
    eng, topo = _engine()
    _set_big_layout(eng, {s: ["big", "big", "big"] for s in bigs})
    install_sound_suv_gate(eng)
    assert eng._big_admission_ok() is False

    # Roomy: mostly empties → a SUV is admissible.
    eng, topo = _engine()
    _set_big_layout(eng, {"A1": ["empty"], "A2": ["empty"], "E1": ["big"],
                          "E2": ["empty"], "B1": ["empty"], "B2": ["empty"],
                          "D1": ["empty"], "D2": ["empty"]})
    install_sound_suv_gate(eng)
    assert eng._big_admission_ok() is True

    # One free slot but landing a big there strands an item → reject, and the gate
    # agrees with the ground-truth solver oracle.
    eng, topo = _engine()
    layout = {"A1": ["big", "big"]}
    layout.update({s: ["big", "big", "big"] for s in bigs if s != "A1"})
    _set_big_layout(eng, layout)
    install_sound_suv_gate(eng)
    eng.state.shelves["A1"].stack.append(Pallet(999, "big"))
    truth = Solver(eng, World(topo))._all_retrievable()
    eng.state.shelves["A1"].stack.pop()
    assert eng._big_admission_ok() == truth


def test_stream_env_runs_continuously():
    """RetrieveEnv stream mode: auto-arrivals on, no clean-rest termination,
    a finite reward, and it steps without error."""
    from oos.config.schema import (EpisodeConfig, ExperimentConfig,
                                   TaskStreamConfig)
    from oos.env.retrieve_env import RetrieveEnv

    fac = get_facility("tiny_medipol")
    exp = ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=0.05, mean_dwell_seconds=120,
                                     std_dwell_seconds=40),
        episode=EpisodeConfig(max_sim_time=1e9, max_steps=80))
    env = RetrieveEnv(facility_factory=fac, stream=True, omni=True,
                      target_any_shelf=True, fullness=0.5, reward_serve=2.0,
                      reward_deliver=2.0, shape_room_carrier_empty_at_room=3.0,
                      experiment_config=exp)
    obs, info = env.reset(seed=1)
    assert env.engine.auto_arrivals_enabled is True
    terminated_early = False
    for _ in range(80):
        mask = obs["action_mask"].astype(bool)
        a = int(np.flatnonzero(mask)[0])  # first legal action
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r)
        if term:
            terminated_early = True
            break
    # Stream never terminates on clean-rest (only truncates at max_steps).
    assert not terminated_early


@pytest.mark.skipif(not os.path.exists("runs/medipol_policy/policy.pt"),
                    reason="trained policy not present (runs/ is gitignored)")
def test_trained_policy_solves_easy_retrieve():
    """The deliverable policy greedily reaches clean-rest on a trivial retrieve."""
    import torch

    from oos.env import hardcases as hc
    from oos.env.hardcases import CaseSpec
    from oos.learn.acceptance import load_net, make_eval_env
    from oos.learn.batching import GraphCollator, sample_from_env_step

    fac = get_facility("tiny_medipol")
    topo, _ = fac()
    coll = GraphCollator(topo)
    net, _ = load_net("runs/medipol_policy/policy.pt")
    env = make_eval_env(fac, 160)
    n_max = env.n_actions
    wins = 0
    for sd in range(5):
        env.set_forced_layout(hc.case_builder(CaseSpec(1, 1, 2, "direct"), seed=sd))
        obs, info = env.reset(seed=100 + sd)
        for _ in range(160):
            s = sample_from_env_step(obs, info, info["action_entries"])
            b = coll.collate([s], n_max=n_max, device="cpu")
            with torch.no_grad():
                a = int(net(b).logits[0].argmax().item())
            obs, r, term, trunc, info = env.step(a)
            if info.get("success"):
                wins += 1
                break
            if term or trunc:
                break
    assert wins == 5


def test_big_load_not_offered_size_incompatible_shelf():
    """A carrier holding an SUV (big) is never offered a GOTO to a small 'sedan'
    shelf it could never give to (the stranding bug)."""
    from oos.env.action import ActionType, enumerate_actions
    from oos.sim.durations import LinearDurations
    from oos.sim.facility import SimEngine
    from oos.sim.state import DockRef, Pallet

    topo, seeding = get_facility("tiny_medipol")()
    eng = SimEngine(topology=topo, seeding=seeding, durations=LinearDurations(),
                    rng=np.random.default_rng(0))
    carrier = "L1"
    cs = eng.state.carriers[carrier]
    cs.load = Pallet(id=1, contents="big")
    small = [sid for sid in topo.accessible_shelves[carrier]
             if topo.shelves[sid].size_class == "small"]
    big = [sid for sid in topo.accessible_shelves[carrier]
           if topo.shelves[sid].size_class == "big"]
    goto = {e.target.id for e in enumerate_actions(carrier, eng.state, topo, eng.queue)
            if e.type == ActionType.GOTO and e.target.kind == "shelf"}
    assert not (goto & set(small)), "SUV must not be offered a small shelf"
    assert goto & set(big), "SUV should still be offered a big shelf with room"


def test_idle_staged_room_carrier_stays_put():
    """A staged room carrier (at its room holding an empty) with no task pending is
    masked to WAIT only — it never relocates the room's empty for nothing."""
    from oos.env.action import (ActionType, enumerate_actions,
                                has_non_wait_action)
    from oos.sim.durations import LinearDurations
    from oos.sim.facility import SimEngine
    from oos.sim.state import DockRef, Pallet

    topo, seeding = get_facility("tiny_medipol")()
    eng = SimEngine(topology=topo, seeding=seeding, durations=LinearDurations(),
                    rng=np.random.default_rng(0))
    carrier = "L1"
    room = next(iter(topo.accessible_rooms[carrier]))
    cs = eng.state.carriers[carrier]
    cs.load = Pallet(id=1, contents="empty")
    cs.docked_at = DockRef("room", room)
    entries = enumerate_actions(carrier, eng.state, topo, eng.queue)
    assert [e.type for e in entries] == [ActionType.WAIT]
    assert has_non_wait_action(carrier, eng.state, topo, eng.queue) is False


def test_carrier_holding_car_is_driven_to_park():
    """Store responsiveness: a carrier holding a (non-target) car with no retrieve
    pending is driven to PARK it — offered only park moves (GIVE here / GOTO a free
    compatible shelf), never WAIT — so it can't sit on the car. While a retrieve is
    pending the guard is silent (the dig is left to the policy)."""
    from oos.env.action import (ActionType, enumerate_actions,
                                has_non_wait_action)
    from oos.sim.durations import LinearDurations
    from oos.sim.facility import SimEngine
    from oos.sim.state import DockRef, Pallet
    from oos.sim.tasks import Retrieve

    topo, seeding = get_facility("tiny_medipol")()
    eng = SimEngine(topology=topo, seeding=seeding, durations=LinearDurations(),
                    rng=np.random.default_rng(0))
    for ss in eng.state.shelves.values():
        ss.stack = []                 # all shelves free → parking always possible
    carrier = "L1"  # big shelves A1/A2 free for an SUV
    cs = eng.state.carriers[carrier]
    cs.load = Pallet(id=1, contents="big")   # holding an SUV, undocked, nothing pending

    entries = enumerate_actions(carrier, eng.state, topo, eng.queue)
    assert ActionType.WAIT not in [e.type for e in entries], "must not be allowed to sit on the car"
    big = {sid for sid in topo.accessible_shelves[carrier]
           if topo.shelves[sid].size_class == "big"}
    goto = {e.target.id for e in entries if e.type == ActionType.GOTO and e.target.kind == "shelf"}
    assert goto and goto <= big, "only free big (size-compatible) shelves offered to park the SUV"
    assert has_non_wait_action(carrier, eng.state, topo, eng.queue) is True

    # With a retrieve pending, the park guard goes silent (dig left to the policy).
    eng.queue.pending.append(Retrieve(arrived_at=0.0, pallet=999))
    entries2 = enumerate_actions(carrier, eng.state, topo, eng.queue)
    assert ActionType.WAIT in [e.type for e in entries2]


def test_idle_unstaged_room_carrier_is_driven_to_stage():
    """Proactive staging: when nothing is pending, an UNSTAGED room carrier is
    routed to stage its room (fetch a top empty → dock at the room) instead of
    being allowed to idle unstaged. The counterpart of stays-put above."""
    from oos.env.action import (ActionType, enumerate_actions,
                                has_non_wait_action)
    from oos.sim.durations import LinearDurations
    from oos.sim.facility import SimEngine
    from oos.sim.state import DockRef, Pallet

    topo, seeding = get_facility("tiny_medipol")()
    eng = SimEngine(topology=topo, seeding=seeding, durations=LinearDurations(),
                    rng=np.random.default_rng(0))
    carrier = "L1"  # owns R1, shelves A1/A2 (big) A3/A4 (small)
    room = next(iter(topo.accessible_rooms[carrier]))
    for ss in eng.state.shelves.values():
        ss.stack = []
    # A single reachable top empty, on A1.
    eng.state.shelves["A1"].stack = [Pallet(id=1, contents="empty")]
    cs = eng.state.carriers[carrier]
    assert not eng.queue.pending  # fully idle

    # (1) Empty-handed, undocked → only offered the GOTO to the shelf with the
    # top empty (fetch it); never WAIT.
    entries = enumerate_actions(carrier, eng.state, topo, eng.queue)
    assert ActionType.WAIT not in [e.type for e in entries]
    assert all(e.type == ActionType.GOTO and e.target.kind == "shelf"
               and e.target.id == "A1" for e in entries)
    assert has_non_wait_action(carrier, eng.state, topo, eng.queue) is True

    # (2) Holding an empty (not at the room) → only offered the GOTO to its room.
    cs.load = Pallet(id=1, contents="empty")
    entries = enumerate_actions(carrier, eng.state, topo, eng.queue)
    assert [(e.type, e.target.kind, e.target.id) for e in entries] == [
        (ActionType.GOTO, "room", room)]

    # (3) But while a task is pending, the guard is silent (retrieval untouched):
    # the carrier sees its normal action set, WAIT included.
    from oos.sim.tasks import Retrieve
    cs.load = None
    eng.queue.pending.append(Retrieve(arrived_at=0.0, pallet=999))
    entries = enumerate_actions(carrier, eng.state, topo, eng.queue)
    assert ActionType.WAIT in [e.type for e in entries]


@pytest.mark.skipif(not os.path.exists("runs/medipol_policy/policy.pt"),
                    reason="trained policy not present (runs/ is gitignored)")
def test_viz_session_drives_policy_to_clean_rest():
    """Regression: the viz Session's realtime playback (`tick` → `_drive_to` →
    `_record_advance`) must feed the policy FRESH observations. A stale-obs bug
    there made a correct brain wander and never finish; this drives a depth-2
    handoff-route SUV retrieve through the playback path and asserts it delivers,
    stages both rooms, and settles to idle."""
    from oos.sim.state import pallet_depth
    from oos.sim.tasks import Retrieve
    from oos.viz.session import Session

    def depth2_suv_on_s1(sess):
        st = sess.env.engine.state
        for sid in ("B1", "B2"):  # S1 (handoff) track big shelves
            ss = st.shelves[sid]
            for i, p in enumerate(ss.stack):
                if p.contents == "big" and (len(ss.stack) - 1 - i) == 2:
                    return p.id
        return None

    solved = 0
    attempts = 0
    for seed in range(300):
        sess = Session("tiny_medipol")
        ck = [e for e in sess.checkpoints() if "medipol_policy" in e.display_name][0]
        sess.load_policy(ck, deterministic=True)
        sess.reroll_layout(fullness=0.5, seed=seed)
        tgt = depth2_suv_on_s1(sess)
        if tgt is None:
            continue
        attempts += 1
        sess.request_retrieve(tgt)
        sess.playing = True
        sess.speed = 30.0
        for _ in range(600):
            sess.tick(dt_wall=0.5)
        eng = sess.env.engine
        rc = [c for c in eng.topology.carriers if eng.topology.accessible_rooms[c]]
        staged = sum(
            1 for c in rc
            if (cs := eng.state.carriers[c]).docked_at is not None
            and cs.docked_at.kind == "room" and cs.load is not None and cs.load.is_empty)
        delivered = not any(isinstance(t, Retrieve) for t in eng.queue.pending)
        # Functional clean-rest: delivered, both rooms staged, and the facility
        # settled (no carrier mid-command). We check "not busy" rather than the
        # cs.waiting flag because an uninvolved staged carrier is kept put by the
        # action guard without being queried to set the flag.
        settled = all(not cs.is_busy for cs in eng.state.carriers.values())
        if delivered and staged == len(rc) and settled:
            solved += 1
        if attempts >= 6:
            break
    assert attempts >= 1, "no depth-2 S1 SUV scenario found"
    assert solved == attempts, f"viz playback solved {solved}/{attempts} (stale-obs regression?)"

