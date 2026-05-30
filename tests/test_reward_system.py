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


# ── parked-car fix ─────────────────────────────────────────────────────────

def test_delivery_term_pays_only_real_deliveries():
    term = DeliveryTerm(bonus=50.0)
    # A real agent-delivered retrieve pays.
    assert term.compute(RewardContext(n_deliveries=1, n_free_deliveries=0)) == 50.0
    # A parked-car (free) retrieve pays nothing — the exploit is closed.
    assert term.compute(RewardContext(n_deliveries=1, n_free_deliveries=1)) == 0.0
    # Mixed: only the non-free one is paid.
    assert term.compute(RewardContext(n_deliveries=2, n_free_deliveries=1)) == 50.0


def test_continuous_system_skips_parked_car_deliver():
    sys = continuous_system(_cont_cfg())
    # Free delivery only → DELIVER contributes 0 to the total.
    total_free, bd_free = sys.compute(
        RewardContext(n_deliveries=1, n_free_deliveries=1))
    assert bd_free.get("DELIVER", 0.0) == 0.0
    assert total_free == 0.0
    # Real delivery → DELIVER pays the bonus.
    total_real, bd_real = sys.compute(
        RewardContext(n_deliveries=1, n_free_deliveries=0))
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


# ── PBRS default is a no-op ──────────────────────────────────────────────────

def test_potential_term_default_is_noop():
    term = PotentialTerm()  # default Φ ≡ 0
    ctx = RewardContext(
        state=object(), state_before=object(), gamma=0.99)
    assert term.compute(ctx) == 0.0
