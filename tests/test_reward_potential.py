"""Tests for the continuous reward: the 3-term PBRS potential, the flat
DELIVER / SERVE outcomes, and the pump-safety of staging."""

from __future__ import annotations

from oos.env.env import Environment
from oos.env.reward import RewardConfig
from oos.env.reward_system import PotentialTerm, RewardContext, StepEvents, base_system
from oos.sim.state import DockRef, Pallet
from oos.sim.tasks import Retrieve


def _env():
    # w_ret=1, w_ready=2, w_wrong=2, deliver=50, serve=20.
    env = Environment.from_name("tiny_medipol", reward_config=RewardConfig())
    env.reset(seed=0)
    return env


def _stage_empty(env, contents="empty"):
    sim = env.engine
    room = next(iter(sim.topology.rooms))
    car = sim.topology.rooms[room].served_by
    cs = sim.state.carriers[car]
    cs.docked_at = DockRef("room", room)
    cs.load = Pallet(id=1, contents=contents)
    return sim, room, car


# ---------------------------------------------------------------------------
# Φ term values
# ---------------------------------------------------------------------------


def test_potential_room_ready_and_wrong_car():
    env = _env()
    sim = env.engine
    assert env._potential(sim) == 0.0
    # Stage an empty at a room → +w_ready.
    sim2, room, car = _stage_empty(env, "empty")
    assert env._potential(sim) == 2.0
    # A store fills it → no longer empty, now a non-requested car → −w_wrong.
    sim.state.carriers[car].load = Pallet(id=1, contents="small")
    assert env._potential(sim) == -2.0
    # The drop across the serve is w_ready + w_wrong = 4 → serve must clear it.
    # Leaving the room (docked_at cleared) restores it to 0.
    sim.state.carriers[car].docked_at = None
    assert env._potential(sim) == 0.0


def test_potential_retrieval_progress():
    # Isolate the retrieval term (w_ret=1; others off).
    cfg = RewardConfig(
        potential_item_retrieval=1.0, potential_room_ready=0.0,
        potential_wrong_car=0.0, potential_shallowest_empty=0.0,
    )
    env = Environment.from_name("tiny_medipol", reward_config=cfg)
    env.reset(seed=0)
    sim = env.engine
    car = next(iter(sim.topology.carriers))
    room = next(iter(sim.topology.accessible_rooms[car]))
    shelf = sorted(sim.topology.accessible_shelves[car])[0]
    sim.state.shelves[shelf].stack = [
        Pallet(id=99, contents="small"),   # the target, buried at depth 2
        Pallet(id=2, contents="small"),
        Pallet(id=3, contents="small"),    # top
    ]
    sim.queue.add(Retrieve(arrived_at=0.0, pallet=99, initial_depth=2))
    assert env._potential(sim) == -5.0            # −(depth 2 + 3)
    sim.state.shelves[shelf].stack.pop()          # dig one blocker → depth 1
    assert env._potential(sim) == -4.0            # rose by w_ret
    # TAKE it (now held, not at a room) → cost 2; dense shaping even though the
    # target is now at depth 0 (no dig left).
    cs = sim.state.carriers[car]
    cs.load = Pallet(id=99, contents="small")
    sim.state.shelves[shelf].stack = []
    assert env._potential(sim) == -2.0
    # Carry it to a room (held at a room) → cost 1.
    cs.docked_at = DockRef("room", room)
    assert env._potential(sim) == -1.0
    # A non-requested layout has Φ = 0.
    env2 = Environment.from_name("tiny_medipol", reward_config=cfg)
    env2.reset(seed=0)
    assert env2._potential(env2.engine) == 0.0


# ---------------------------------------------------------------------------
# Outcome rewards
# ---------------------------------------------------------------------------


def test_base_system_flat_deliver_and_serve():
    sys = base_system(RewardConfig(reward_deliver=50.0, reward_serve=20.0))
    # Two deliveries at different depths still pay FLAT (depth lives in Φ).
    ctx = RewardContext(n_deliveries=2, delivery_depth_weight=7, n_stores_served=1)
    total, bd = sys.compute(ctx)
    assert bd["DELIVER"] == 100.0     # 2 × 50, NOT depth-scaled
    assert bd["SERVE"] == 20.0


def test_idle_while_task_wired_through_base_context():
    """End-to-end: the StepEvents decision-point flags reach the base reward
    context, so base_system's IDLE_TASK penalty fires on an all-idle step with
    work pending. Pins the env→reward wiring (not just the term in isolation)."""
    cfg = RewardConfig(
        # isolate the penalty: potential off so SHAPE stays 0.
        potential_item_retrieval=0.0, potential_room_ready=0.0,
        potential_wrong_car=0.0, potential_shallowest_empty=0.0,
        penalty_idle_while_task=2.0,
    )
    env = Environment.from_name("tiny_medipol", reward_config=cfg)
    env.reset(seed=0)
    # All carriers waiting, a retrieve still pending, no room staged → both
    # additive charges → −4.0 through the real base_system.
    events = StepEvents(
        all_carriers_waiting=True,
        retrieve_pending_at_decision=True,
        room_has_staged_empty_at_decision=False,
    )
    ctx = env._reward_context_from_events(events, env.engine)
    assert ctx.all_carriers_waiting and ctx.retrieve_pending
    assert ctx.room_has_staged_empty is False
    total, bd = env._reward_system.compute(ctx)
    assert bd["IDLE_TASK"] == -4.0
    # A serve on the same instant is productive → no idle charge.
    served = StepEvents(
        all_carriers_waiting=True, retrieve_pending_at_decision=True,
        room_has_staged_empty_at_decision=False, n_deliveries=1,
    )
    _, bd2 = env._reward_system.compute(
        env._reward_context_from_events(served, env.engine))
    assert "IDLE_TASK" not in bd2


# ---------------------------------------------------------------------------
# Pump-safety: a stage→leave→stage→leave cycle nets negative under γ<1
# ---------------------------------------------------------------------------


def test_pbrs_telescopes_through_the_env_step():
    """The shaped reward each env.step must be exactly γ·Φ(s′) − Φ(s) with Φ(s)
    snapshotted BEFORE the action mutates state. If potential_before is captured
    after submit_action (the bug), a GOTO that clears docked_at / a WAIT that
    serves a store silently drops Φ-changes and the staging pump opens up."""
    import numpy as np

    env = Environment.from_name("tiny_medipol", reward_config=RewardConfig())
    env.reward_gamma = 0.99
    obs, info = env.reset(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(80):
        phi_before = env._potential(env.engine)
        a = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, _r, term, trunc, info = env.step(a)
        phi_after = env._potential(env.engine)
        shape = info["reward_breakdown"].get("SHAPE", 0.0)
        assert abs(shape - (0.99 * phi_after - phi_before)) < 1e-6, (
            f"SHAPE {shape} != γΦ'−Φ {0.99 * phi_after - phi_before} "
            f"(potential_before not snapshotted pre-action)"
        )
        if term or trunc:
            obs, info = env.reset(seed=2)


def test_staging_pump_nets_negative():
    term = PotentialTerm()
    gamma = 0.99
    w_ready = 2.0
    # Φ sequence as the carrier stages (0→2), leaves (2→0), stages, leaves.
    phis = [0.0, w_ready, 0.0, w_ready, 0.0]
    total = 0.0
    for before, after in zip(phis[:-1], phis[1:]):
        total += term.compute(
            RewardContext(gamma=gamma, potential_before=before, potential_after=after)
        )
    # Telescopes to γ^4·Φ_T − Φ_0 = 0; each cycle bleeds (γ−1)·w_ready < 0.
    assert total < 0.0
    assert abs(total - (2 * (gamma - 1) * w_ready)) < 1e-9
