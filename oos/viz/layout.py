"""Auto-layout: assigns each carrier its own horizontal strip and computes
pixel placements for shelves, rooms, handoff markers along each strip.

Per the design: each carrier is self-contained. Transfer shelves and handoff
poses appear as 'shared with X' badges on each of the two participants'
strips, with a thin connector line drawn between the two instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oos.sim.topology import (
    CarrierId,
    Handoff,
    RoomId,
    Shelf,
    ShelfId,
    Topology,
)


# ---------------------------------------------------------------------------
# Layout result types
# ---------------------------------------------------------------------------


@dataclass
class StripGeom:
    """Pixel geometry for a single carrier strip."""

    carrier_id: CarrierId
    rect: tuple[int, int, int, int]  # x, y, w, h of the whole strip region
    track_y: int                      # y coordinate of the carrier track
    track_x_start: int                # x where position=0 lands
    track_x_end: int                  # x where position=(N-1) lands
    positions: int                    # carrier's number of positions
    label_rect: tuple[int, int, int, int]  # area reserved for the carrier label

    def pos_to_x(self, p: float) -> int:
        if self.positions <= 1:
            return self.track_x_start
        frac = p / (self.positions - 1)
        return int(self.track_x_start + frac * (self.track_x_end - self.track_x_start))


@dataclass
class ShelfPlacement:
    shelf_id: ShelfId
    carrier_id: CarrierId            # which strip this placement lives on
    x: int                            # center x along the track
    is_transfer: bool
    partner: CarrierId | None         # other carrier if transfer shelf
    partner_x: int | None             # x in the partner strip (for connector hint)
    partner_strip_track_y: int | None


@dataclass
class RoomPlacement:
    room_id: RoomId
    carrier_id: CarrierId
    x: int


@dataclass
class HandoffPlacement:
    a: CarrierId
    b: CarrierId
    # Pixel coordinates of the actual handoff positions on each strip's track.
    # The carrier moves to these positions when it does the handoff, and the
    # badge is drawn here so visual = semantic.
    a_x: int
    b_x: int
    a_strip_track_y: int
    b_strip_track_y: int


@dataclass
class Layout:
    canvas_rect: tuple[int, int, int, int]
    sidebar_rect: tuple[int, int, int, int]
    queue_strip_rect: tuple[int, int, int, int]
    strips: dict[CarrierId, StripGeom]
    shelves: list[ShelfPlacement] = field(default_factory=list)
    rooms: list[RoomPlacement] = field(default_factory=list)
    handoffs: list[HandoffPlacement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layout algorithm
# ---------------------------------------------------------------------------


@dataclass
class LayoutConfig:
    window_w: int = 1920
    window_h: int = 1080
    sidebar_w: int = 460
    strip_pad: int = 18
    strip_label_w: int = 130
    canvas_pad: int = 24
    track_right_pad: int = 80   # extra room on the right of the track for end-of-track shelf labels
    queue_strip_h: int = 86     # top-of-canvas customer queue strip height


def compute_layout(topo: Topology, cfg: LayoutConfig | None = None) -> Layout:
    cfg = cfg or LayoutConfig()

    canvas_x = 0
    canvas_y = 0
    canvas_w = cfg.window_w - cfg.sidebar_w
    canvas_h = cfg.window_h
    sidebar_x = canvas_w
    sidebar_y = 0
    sidebar_w = cfg.sidebar_w
    sidebar_h = cfg.window_h

    # Reserve top area for the global customer-queue strip.
    queue_strip_rect = (
        cfg.canvas_pad,
        cfg.canvas_pad,
        canvas_w - 2 * cfg.canvas_pad,
        cfg.queue_strip_h,
    )
    carriers_top = cfg.canvas_pad + cfg.queue_strip_h + cfg.strip_pad

    n = len(topo.carriers)
    inner_h = canvas_h - carriers_top - cfg.canvas_pad
    strip_h = inner_h // max(n, 1)

    # Order carriers by id for a stable visual layout.
    ordered = sorted(topo.carriers.values(), key=lambda c: c.id)

    strips: dict[CarrierId, StripGeom] = {}
    for i, c in enumerate(ordered):
        y0 = carriers_top + i * strip_h
        rect = (cfg.canvas_pad, y0, canvas_w - 2 * cfg.canvas_pad, strip_h - cfg.strip_pad)
        track_y = y0 + (strip_h - cfg.strip_pad) // 2 + 20
        track_x_start = cfg.canvas_pad + cfg.strip_label_w
        track_x_end = canvas_w - cfg.canvas_pad - cfg.track_right_pad
        strips[c.id] = StripGeom(
            carrier_id=c.id,
            rect=rect,
            track_y=track_y,
            track_x_start=track_x_start,
            track_x_end=track_x_end,
            positions=c.positions,
            label_rect=(cfg.canvas_pad, y0, cfg.strip_label_w, strip_h - cfg.strip_pad),
        )

    shelf_placements: list[ShelfPlacement] = []
    for s in topo.shelves.values():
        for cid in s.access:
            partner = next((other for other in s.access if other != cid), None)
            partner_x = (
                strips[partner].pos_to_x(s.position_for[partner]) if partner else None
            )
            partner_track_y = strips[partner].track_y if partner else None
            shelf_placements.append(
                ShelfPlacement(
                    shelf_id=s.id,
                    carrier_id=cid,
                    x=strips[cid].pos_to_x(s.position_for[cid]),
                    is_transfer=s.is_transfer,
                    partner=partner if s.is_transfer else None,
                    partner_x=partner_x if s.is_transfer else None,
                    partner_strip_track_y=partner_track_y if s.is_transfer else None,
                )
            )

    room_placements: list[RoomPlacement] = []
    for r in topo.rooms.values():
        room_placements.append(
            RoomPlacement(
                room_id=r.id,
                carrier_id=r.served_by,
                x=strips[r.served_by].pos_to_x(r.position),
            )
        )

    handoff_placements: list[HandoffPlacement] = []
    for h in topo.handoffs:
        a, b = h.carriers
        handoff_placements.append(
            HandoffPlacement(
                a=a,
                b=b,
                a_x=strips[a].pos_to_x(h.positions[a]),
                b_x=strips[b].pos_to_x(h.positions[b]),
                a_strip_track_y=strips[a].track_y,
                b_strip_track_y=strips[b].track_y,
            )
        )

    return Layout(
        canvas_rect=(canvas_x, canvas_y, canvas_w, canvas_h),
        sidebar_rect=(sidebar_x, sidebar_y, sidebar_w, sidebar_h),
        queue_strip_rect=queue_strip_rect,
        strips=strips,
        shelves=shelf_placements,
        rooms=room_placements,
        handoffs=handoff_placements,
    )
