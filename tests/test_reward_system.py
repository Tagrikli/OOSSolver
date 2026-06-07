"""Unit tests for the pluggable reward suite (oos.env.reward_system).

Pins the one intentional behaviour change from the StepEvents migration —
parked-car retrieves stop paying DELIVER — plus the anti-farm symmetry
properties and the PBRS no-op default.
"""

from types import SimpleNamespace

from oos.env.reward_system import (
    DeliveryTerm,
    IdleWhileTaskTerm,
    PotentialTerm,
    ProgressTerm,
    RewardContext,
    base_system,
    delivery_system,
)


def _base_cfg(**over):
    """Duck-typed base reward config (base_system reads by attr name)."""
    base = dict(
        reward_deliver=50.0,
        reward_serve=20.0,
        potential_item_retrieval=0.0,
        potential_room_ready=0.0,
        potential_wrong_car=0.0,
        potential_shallowest_empty=0.0,
        penalty_idle_while_task=0.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── delivery: depth scaling + parked-car fix ────────────────────────────────
# `delivery_depth_weight` = Σ(initial_depth + 1) over real (agent-delivered)
# retrievals; free/parked deliveries contribute 0 to it.

def test_delivery_term_scales_with_depth():
    term = DeliveryTerm(bonus=50.0)
    # depth 0 → weight 1 → bonus·1
    assert term.compute(RewardContext(delivery_depth_weight=1)) == 50.0
    # depth 2 → weight 3 → bonus·3
    assert term.compute(RewardContext(delivery_depth_weight=3)) == 150.0
    # two deliveries at depth 0 and depth 1 → weight 1 + 2 = 3
    assert term.compute(RewardContext(delivery_depth_weight=3)) == 150.0


def test_delivery_term_pays_nothing_for_free_deliveries():
    term = DeliveryTerm(bonus=50.0)
    # A parked-car (free) retrieve contributes 0 to the weight → pays nothing.
    assert term.compute(RewardContext(delivery_depth_weight=0)) == 0.0


# ── delivery_system: the retrieval spine's one-term reward ──────────────────

def test_delivery_system_is_delivery_only():
    sys = delivery_system(_base_cfg())
    assert sys.labels() == ["DELIVER"]
    # Flat (scale_by_depth=False): a real delivery pays reward_deliver regardless
    # of depth; a parked-car free delivery pays nothing.
    total, bd = sys.compute(RewardContext(n_deliveries=1, n_free_deliveries=0))
    assert bd["DELIVER"] == 50.0 and total == 50.0
    assert sys.compute(RewardContext(n_deliveries=1, n_free_deliveries=1))[0] == 0.0
    # A no-op step (nothing delivered) pays exactly 0 — no shaping/idle drip.
    assert sys.compute(RewardContext(potential_before=-9.0, potential_after=-9.0))[0] == 0.0


# ── all-idle-while-task penalty (two additive charges) ──────────────────────

def test_idle_while_task_charges_each_condition():
    term = IdleWhileTaskTerm(penalty=2.0)
    # Not all idle → never fires, regardless of pending work.
    assert term.compute(RewardContext(
        all_carriers_waiting=False, retrieve_pending=True,
        room_has_staged_empty=False)) == 0.0
    # All idle + retrieve pending + a room IS staged → one charge.
    assert term.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=True,
        room_has_staged_empty=True)) == -2.0
    # All idle + no retrieve + no staged room → one charge.
    assert term.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=False,
        room_has_staged_empty=False)) == -2.0
    # All idle + BOTH conditions → two additive charges.
    assert term.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=True,
        room_has_staged_empty=False)) == -4.0
    # All idle but no work remains (room staged, nothing pending) → silent.
    assert term.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=False,
        room_has_staged_empty=True)) == 0.0


def test_idle_while_task_not_charged_on_a_serving_step():
    """A WAIT at a room that delivered/served this step is productive — even
    though every carrier is 'waiting' at that instant — so it is not charged."""
    term = IdleWhileTaskTerm(penalty=2.0)
    # A delivery fired this advance → no idle charge despite all-waiting.
    assert term.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=True,
        room_has_staged_empty=False, n_deliveries=1)) == 0.0
    # A store served this advance → likewise no charge.
    assert term.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=True,
        room_has_staged_empty=False, n_stores_served=1)) == 0.0


def test_idle_while_task_off_by_default_in_base_system():
    sys = base_system(_base_cfg())              # penalty 0 → term silent
    total, bd = sys.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=True,
        room_has_staged_empty=False))
    assert "IDLE_TASK" not in bd and total == 0.0
    # Enabled → the penalty shows up in the base system's breakdown.
    sys2 = base_system(_base_cfg(penalty_idle_while_task=2.0))
    total2, bd2 = sys2.compute(RewardContext(
        all_carriers_waiting=True, retrieve_pending=True,
        room_has_staged_empty=False))
    assert bd2["IDLE_TASK"] == -4.0 and total2 == -4.0


# ── PBRS shaping: F = γ·Φ(s') − Φ(s) ─────────────────────────────────────────

def test_potential_term_is_noop_when_potentials_zero():
    assert PotentialTerm().compute(RewardContext()) == 0.0


def test_potential_term_forms_discounted_difference():
    term = PotentialTerm()
    # potential rose (e.g. an empty got staged / a retrieve completed) → +reward
    assert term.compute(RewardContext(
        gamma=1.0, potential_before=-10.0, potential_after=-4.0)) == 6.0
    # potential fell (unstaged / a retrieve arrived) → −reward
    assert term.compute(RewardContext(
        gamma=1.0, potential_before=-4.0, potential_after=-10.0)) == -6.0
    # γ scales Φ(s')
    assert term.compute(RewardContext(
        gamma=0.99, potential_before=0.0, potential_after=-10.0)) == -9.9


def test_progress_term_is_undiscounted_difference():
    """ProgressTerm = Φ(s') − Φ(s), with NO γ — so a no-op step (Φ unchanged)
    pays exactly 0, killing the γ<1 idle-drip that PotentialTerm has."""
    term = ProgressTerm()
    # no-op: Φ flat and negative → PROGRESS 0 (PBRS would pay (γ−1)·Φ > 0).
    assert term.compute(RewardContext(
        gamma=0.99, potential_before=-10.0, potential_after=-10.0)) == 0.0
    # real progress: Φ rose by 1 → +1 regardless of γ.
    assert term.compute(RewardContext(
        gamma=0.99, potential_before=-10.0, potential_after=-9.0)) == 1.0
