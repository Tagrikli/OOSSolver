"""Five-lift × five-shuttle campus facility.

Topology:

    L1 (room R1) ─┐  ┌─ S1
    L2 (room R2) ─┤  ├─ S2
    L3 (room R3) ─┼──┼─ S3      (every Li ↔ Sj has its own handoff)
    L4 (room R4) ─┤  ├─ S4
    L5 (room R5) ─┘  └─ S5

  * 5 lift carriers, each owns one room.
  * 5 shuttle carriers, none own rooms — purely transfer carriers.
  * 25 handoff points: every (Li, Sj) pair has a distinct meeting position
    on both Li's track and Sj's track.

Per-lift shelves (8 total, all capacity 3):
    4 positions × {up, down} = 8 shelves. Sizes: big, big, small, small
    (pattern matches `stacker`). Positions sit AFTER the lift's 5
    shuttle-handoff poses on the track.

    Naming: `L{i}_A{k}{u|d}` for lift Li, position k=1..4, side u/d.

Per-shuttle shelves (20 total, all capacity 3):
    10 positions × {up, down} = 20 shelves. First 2 positions = big
    (4 big shelves total), remaining 8 positions = small (16 small
    shelves total). Positions sit AFTER the shuttle's 5 lift-handoff
    poses on the track.

    Naming: `S{j}_B{k}{u|d}` for shuttle Sj, position k=1..10, side u/d.

Position layout per carrier:
    Lift Li:
        pos 0           → room Ri
        pos k·SP_L      → handoff to shuttle Sk  (k=1..5)
        pos (5+k)·SP_L  → shelf pair `L{i}_A{k}{u|d}`  (k=1..4)
        max_pos         = 10·SP_L

    Shuttle Sj:
        pos k·SP_S      → handoff to lift Lk     (k=1..5)
        pos (5+k)·SP_S  → shelf pair `S{j}_B{k}{u|d}`  (k=1..10)
        max_pos         = 16·SP_S
"""

from __future__ import annotations

from oos.dsl import Carrier, Facility, Handoff, Room, Shelf
from oos.sim.facility import SeedingConfig
from oos.sim.motion import LIFT_SHELF_SPACING_MM, SHUTTLE_SHELF_SPACING_MM
from oos.sim.topology import Topology

SP_L = LIFT_SHELF_SPACING_MM     # 2300 mm — lift spacing
SP_S = SHUTTLE_SHELF_SPACING_MM  # 5500 mm — shuttle spacing

N_LIFTS = 5
N_SHUTTLES = 5
LIFT_SHELF_POSITIONS = 4        # → 8 shelves per lift (×2 sides)
SHUTTLE_SHELF_POSITIONS = 10    # → 20 shelves per shuttle (×2 sides)
SHUTTLE_BIG_POSITIONS = 2       # first 2 positions on each shuttle are big
                                # → 4 big shelves per shuttle (×2 sides)


def make_facility() -> tuple[Topology, SeedingConfig]:
    # ---- carriers --------------------------------------------------------
    lift_max = (N_SHUTTLES + LIFT_SHELF_POSITIONS + 1) * SP_L
    lifts = [
        Carrier(f"L{i}", min_pos=0, max_pos=lift_max, initial_pos=0, kind="lift")
        for i in range(1, N_LIFTS + 1)
    ]
    shuttle_max = (N_LIFTS + SHUTTLE_SHELF_POSITIONS + 1) * SP_S
    shuttles = [
        Carrier(f"S{j}", min_pos=0, max_pos=shuttle_max, initial_pos=0, kind="shuttle")
        for j in range(1, N_SHUTTLES + 1)
    ]

    # ---- rooms (one per lift, at lift pos 0) -----------------------------
    rooms = [Room(f"R{i}", position=0) for i in range(1, N_LIFTS + 1)]

    # ---- lift shelves: big/big/small/small × {up,down} -------------------
    lift_shelves: dict[int, list[Shelf]] = {i: [] for i in range(1, N_LIFTS + 1)}
    for i in range(1, N_LIFTS + 1):
        for k in range(1, LIFT_SHELF_POSITIONS + 1):
            pos = (N_SHUTTLES + k) * SP_L
            size = "big" if k <= 2 else "small"
            lift_shelves[i].append(Shelf(
                f"L{i}_A{k}u", position=pos, capacity=3, size=size,
                orientation="up",
            ))
            lift_shelves[i].append(Shelf(
                f"L{i}_A{k}d", position=pos, capacity=3, size=size,
                orientation="down",
            ))

    # ---- shuttle shelves: 2 big positions then 8 small positions ---------
    shuttle_shelves: dict[int, list[Shelf]] = {j: [] for j in range(1, N_SHUTTLES + 1)}
    for j in range(1, N_SHUTTLES + 1):
        for k in range(1, SHUTTLE_SHELF_POSITIONS + 1):
            pos = (N_LIFTS + k) * SP_S
            size = "big" if k <= SHUTTLE_BIG_POSITIONS else "small"
            shuttle_shelves[j].append(Shelf(
                f"S{j}_B{k}u", position=pos, capacity=3, size=size,
                orientation="up",
            ))
            shuttle_shelves[j].append(Shelf(
                f"S{j}_B{k}d", position=pos, capacity=3, size=size,
                orientation="down",
            ))

    # ---- handoffs: 25 distinct (Li, Sj) pairs ----------------------------
    # On lift Li, shuttle Sj's handoff sits at position j·SP_L.
    # On shuttle Sj, lift Li's handoff sits at position i·SP_S.
    # So every pair has its own meeting position on BOTH tracks.
    lift_handoffs: dict[int, list[Handoff]] = {i: [] for i in range(1, N_LIFTS + 1)}
    shuttle_handoffs: dict[int, list[Handoff]] = {j: [] for j in range(1, N_SHUTTLES + 1)}
    pairs: list[tuple[Handoff, Handoff]] = []
    for i in range(1, N_LIFTS + 1):
        for j in range(1, N_SHUTTLES + 1):
            h_l = Handoff(f"h_L{i}_S{j}/L{i}", position=j * SP_L)
            h_s = Handoff(f"h_L{i}_S{j}/S{j}", position=i * SP_S)
            lift_handoffs[i].append(h_l)
            shuttle_handoffs[j].append(h_s)
            pairs.append((h_l, h_s))

    # ---- registration ----------------------------------------------------
    for i, L in enumerate(lifts, start=1):
        L.register_shelves(*lift_shelves[i])
        L.register_rooms(rooms[i - 1])
        L.register_handoffs(*lift_handoffs[i])
    for j, S in enumerate(shuttles, start=1):
        S.register_shelves(*shuttle_shelves[j])
        S.register_handoffs(*shuttle_handoffs[j])

    fac = Facility("campus")
    fac.register_carriers(*lifts, *shuttles)
    for h_l, h_s in pairs:
        fac.pair(h_l, h_s)
    fac.seed_pool()
    return fac.build()
