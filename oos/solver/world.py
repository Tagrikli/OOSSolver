"""Read-only static analysis of a facility topology.

Precomputes everything the planner needs about *structure* (independent of the
dynamic pallet state): who owns what, sizes, room routes, and the handoff fabric.
Built only from `oos.sim.topology` — no dependency on prior attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from oos.sim.topology import Topology

SMALL = "__small_storage__"   # sentinel destination: "any small shelf with room"


@dataclass(frozen=True)
class HandoffRoute:
    """How a pallet held by `src` carrier reaches `dst` carrier.

    relays: ordered intermediate carriers (each a handoff). len 0 = same carrier
    (no handoff), 1 = single handoff (src directly hands to dst), 2 = one relay.
    The full carrier chain is [src, *relays, dst].
    """

    src: str
    dst: str
    chain: tuple[str, ...]   # full carrier path incl. endpoints

    @property
    def n_handoffs(self) -> int:
        return len(self.chain) - 1


class World:
    """Static view over a Topology."""

    def __init__(self, topo: Topology) -> None:
        self.topo = topo
        self.carriers = dict(topo.carriers)
        self.shelves = dict(topo.shelves)
        self.rooms = dict(topo.rooms)

        self.lifts = sorted(cid for cid, c in topo.carriers.items() if c.kind == "lift")
        self.shuttles = sorted(
            cid for cid, c in topo.carriers.items() if c.kind == "shuttle"
        )

        # shelf -> owner carrier (single-owner; transfer shelves would have 2 but
        # campus has none — we take the first and note transfer separately).
        self.owner: dict[str, str] = {}
        for sid, s in topo.shelves.items():
            self.owner[sid] = s.access[0]

        # carrier -> its room (lifts only)
        self.room_of: dict[str, str] = {}
        for rid, r in topo.rooms.items():
            self.room_of[r.served_by] = rid

        self.big_shelves = frozenset(
            sid for sid, s in topo.shelves.items() if s.size_class == "big"
        )
        self.small_shelves = frozenset(
            sid for sid, s in topo.shelves.items() if s.size_class == "small"
        )
        # shelves owned by each carrier, split by size
        self.shelves_of: dict[str, frozenset[str]] = {
            cid: topo.accessible_shelves[cid] for cid in topo.carriers
        }

    # ------------------------------------------------------------------
    # Sizes / kinds
    # ------------------------------------------------------------------

    def is_lift(self, cid: str) -> bool:
        return self.carriers[cid].kind == "lift"

    def shelf_size(self, sid: str) -> str:
        return self.shelves[sid].size_class

    def shelf_cap(self, sid: str) -> int:
        return self.shelves[sid].capacity

    def serves_room(self, cid: str) -> str | None:
        return self.room_of.get(cid)

    # ------------------------------------------------------------------
    # Handoff fabric — routes between carriers
    # ------------------------------------------------------------------

    @cached_property
    def _partners(self) -> dict[str, frozenset[str]]:
        return {cid: self.topo.handoff_partners[cid] for cid in self.carriers}

    def can_handoff(self, a: str, b: str) -> bool:
        return b in self._partners.get(a, frozenset())

    def route_between(self, src: str, dst: str) -> HandoffRoute | None:
        """Carrier chain to move a pallet from src to dst. <=2 handoffs on the
        bipartite lift/shuttle fabric. Returns None if unreachable."""
        if src == dst:
            return HandoffRoute(src, dst, (src,))
        if self.can_handoff(src, dst):
            return HandoffRoute(src, dst, (src, dst))
        # same-kind: need one opposite-kind relay both can reach
        relays = self._partners[src] & self._partners[dst]
        if relays:
            relay = sorted(relays)[0]
            return HandoffRoute(src, dst, (src, relay, dst))
        return None

    def route_shelf_to_shelf(self, from_sid: str, to_sid: str) -> HandoffRoute | None:
        return self.route_between(self.owner[from_sid], self.owner[to_sid])

    # ------------------------------------------------------------------
    # Delivery: a pallet on `sid` -> some room (only lifts have rooms)
    # ------------------------------------------------------------------

    def lift_for_delivery(self, sid: str, prefer: list[str] | None = None) -> str | None:
        """Pick the lift that will carry `sid`'s pallet to its room.

        If sid is on a lift -> that lift (direct, no handoff). If sid is on a
        shuttle -> a lift reachable by one handoff (any lift; `prefer` orders
        candidates, e.g. idle/already-staged lifts first)."""
        owner = self.owner[sid]
        if self.is_lift(owner):
            return owner
        cands = [lc for lc in self.lifts if self.can_handoff(owner, lc)]
        if not cands:
            return None
        if prefer:
            cands.sort(key=lambda lc: (lc not in prefer, prefer.index(lc) if lc in prefer else 0))
        return cands[0]
