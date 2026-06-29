"""Facility → 2D canvas geometry.

Carriers are horizontal lanes stacked top-to-bottom. A single shared
world→pixel mapping (`pos_to_x`) is used for every lane, so a position in mm
lands at the same x on every lane — handoff poses and shared shelves line up
vertically. Pure geometry: no DearPyGui, no sim state. The canvas adds the
dynamic bits (carrier sprites, pallet stacks) on top each frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oos.sim.topology import Topology


@dataclass(frozen=True)
class Lane:
    cid: str
    y: float       # track centre-line (px)
    x0: float      # pos_to_x(min_pos)
    x1: float      # pos_to_x(max_pos)


@dataclass(frozen=True)
class ShelfBox:
    sid: str
    cid: str       # the lane this placement lives on
    x: float       # centre x of the stack
    y: float       # the lane's track y
    up: bool       # stack grows upward (above track) vs downward
    size_class: str
    capacity: int
    is_transfer: bool


@dataclass(frozen=True)
class RoomBox:
    rid: str
    x: float
    y: float


@dataclass(frozen=True)
class HandoffMark:
    a: str
    b: str
    ax: float
    bx: float
    ay: float
    by: float


@dataclass
class Geometry:
    width: int
    height: int
    gmin: int
    px_per_mm: float
    left: float
    lanes: dict[str, Lane] = field(default_factory=dict)
    shelves: list[ShelfBox] = field(default_factory=list)
    rooms: list[RoomBox] = field(default_factory=list)
    handoffs: list[HandoffMark] = field(default_factory=list)
    lane_pitch: float = 0.0

    def pos_to_x(self, p: float) -> float:
        return self.left + (p - self.gmin) * self.px_per_mm


def build_geometry(
    topo: Topology,
    host_w: float,
    host_h: float,
    zoom: float = 1.0,
    margin: int = 30,
    label_w: int = 44,
) -> Geometry:
    """Lay the facility out for a `host_w`×`host_h` viewport at `zoom`.

    At zoom=1 the whole track span fits the host width. zoom>1 scales the system
    HORIZONTALLY — the returned `width` (the drawlist size) then exceeds host_w,
    and the host's horizontal scrollbar pans it. Lanes always fit the host
    height. One lane per carrier; a single shared world→px mapping.
    """
    carriers = sorted(topo.carriers.values(), key=lambda c: c.id)
    n = max(1, len(carriers))

    gmin = min(c.min_pos for c in carriers)
    gmax = max(c.max_pos for c in carriers)
    span = max(1, gmax - gmin)
    base_fit = max(1e-4, (host_w - 2 * margin - label_w) / span)
    px_per_mm = base_fit * max(0.05, zoom)
    left = margin + label_w

    track_w = span * px_per_mm
    content_w = int(left + track_w + margin)
    width = max(int(host_w), content_w)
    height = max(1, int(host_h))

    geo = Geometry(width=width, height=height, gmin=gmin,
                   px_per_mm=px_per_mm, left=left)
    geo.lane_pitch = (height - 2 * margin) / n

    for i, c in enumerate(carriers):
        y = margin + (i + 0.5) * geo.lane_pitch
        geo.lanes[c.id] = Lane(
            cid=c.id, y=y,
            x0=geo.pos_to_x(c.min_pos), x1=geo.pos_to_x(c.max_pos),
        )

    for s in topo.shelves.values():
        for cid in s.access:                        # transfer shelves: both lanes
            lane = geo.lanes[cid]
            geo.shelves.append(ShelfBox(
                sid=s.id, cid=cid,
                x=geo.pos_to_x(s.position_for[cid]), y=lane.y,
                up=(s.orientation_at(cid) == "up"),
                size_class=s.size_class, capacity=s.capacity,
                is_transfer=s.is_transfer,
            ))

    for r in topo.rooms.values():
        lane = geo.lanes[r.served_by]
        geo.rooms.append(RoomBox(rid=r.id, x=geo.pos_to_x(r.position), y=lane.y))

    for h in topo.handoffs:
        a, b = h.carriers
        geo.handoffs.append(HandoffMark(
            a=a, b=b,
            ax=geo.pos_to_x(h.positions[a]), bx=geo.pos_to_x(h.positions[b]),
            ay=geo.lanes[a].y, by=geo.lanes[b].y,
        ))

    return geo
