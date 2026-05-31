"""Unit tests for the pluggable reward suite (oos.env.reward_system).

Pins the one intentional behaviour change from the StepEvents migration —
parked-car retrieves stop paying DELIVER — plus the anti-farm symmetry
properties and the PBRS no-op default.
"""

from types import SimpleNamespace

from oos.env.reward_system import (
    DeliveryTerm,
    PotentialTerm,
    RewardContext,
    StageTerm,
    UnstageTerm,
    WrongItemTerm,
    EvacTerm,
    continuous_system,
)


def _cont_cfg(**over):
    """Duck-typed continuous reward config (the factory reads by attr name)."""
    base = dict(
        delivery_bonus=50.0,
        store_serve_bonus=15.0,
        wrong_item_penalty=5.0,
        stage_bonus=2.0,
        time_weight=0.0,
        movement_weight=0.0,
        all_idle_retrieve_penalty=0.0,
        all_idle_no_room_empty_penalty=0.0,
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


def test_continuous_system_skips_parked_car_deliver():
    sys = continuous_system(_cont_cfg())
    # Free delivery only → weight 0 → DELIVER contributes 0 to the total.
    total_free, bd_free = sys.compute(RewardContext(delivery_depth_weight=0))
    assert bd_free.get("DELIVER", 0.0) == 0.0
    assert total_free == 0.0
    # Real depth-0 delivery → DELIVER pays the bonus.
    total_real, bd_real = sys.compute(RewardContext(delivery_depth_weight=1))
    assert bd_real["DELIVER"] == 50.0
    assert total_real == 50.0


# ── anti-farm symmetry (round-trips net zero) ───────────────────────────────

def test_stage_unstage_round_trip_nets_zero():
    stage, unstage = StageTerm(2.0), UnstageTerm(2.0)
    staged = stage.compute(RewardContext(n_stage=1))
    unstaged = unstage.compute(RewardContext(n_unstage=1))
    assert staged + unstaged == 0.0


def test_wrong_evac_round_trip_nets_zero():
    wrong, evac = WrongItemTerm(5.0), EvacTerm(5.0)
    placed = wrong.compute(RewardContext(n_wrong=1))
    stowed = evac.compute(RewardContext(n_evac=1))
    assert placed + stowed == 0.0


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
