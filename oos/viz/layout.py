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
    ShelfOrientation,
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
    track_x_start: int                # x where position=min_pos lands
    track_x_end: int                  # x where position=max_pos lands
    min_pos: int                      # carrier's min position (mm)
    max_pos: int                      # carrier's max position (mm)
    label_rect: tuple[int, int, int, int]  # area reserved for the carrier label

    def pos_to_x(self, p: float) -> int:
        span = self.max_pos - self.min_pos
        if span <= 0:
            return self.track_x_start
        frac = (p - self.min_pos) / span
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
    orientation: ShelfOrientation = "up"   # "up" → above track, "down" → below


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
    # Scrollable region for the stacked carrier strips. The renderer clips
    # carrier drawing to this rect and applies a scroll offset when the
    # total stacked height exceeds carriers_visible_h.
    carriers_top: int
    carriers_visible_h: int
    strip_row_h: int            # pitch per carrier row (strip_h + strip_pad)
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
    strip_label_w: int = 56     # left gutter — holds the [L1] / [S2] tag
    canvas_pad: int = 24
    track_right_pad: int = 56   # right gutter — kept equal to strip_label_w
                                # so both track endpoints sit the same
                                # distance from the strip's vertical edges

    queue_strip_h: int = 86     # top-of-canvas customer queue strip height
    # Each carrier strip is a FIXED size — they do not stretch to fill the
    # canvas. When N carriers overflow the canvas vertically the renderer
    # scrolls them; the strip itself never gets squished.
    strip_h: int = 200          # fixed strip height (incl. internal pad)

    # Track length is mm-driven: width_px = (max_pos - min_pos) * px_per_mm.
    # `base_px_per_mm` is the px-per-mm at zoom=1.0; `zoom` is the user-
    # controllable multiplier (Shift+wheel in the canvas). The LEFT
    # endpoint stays anchored to `canvas_pad + strip_label_w`; zoom only
    # extends/contracts the right end.
    base_px_per_mm: float = 0.05
    zoom: float = 1.0


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

    # Strip width is mm-driven: track_len = (max_pos - min_pos) * px_per_mm.
    # Each carrier gets its own strip width; left endpoint stays anchored at
    # canvas_pad + strip_label_w. Zoom multiplies px_per_mm. If a strip
    # extends past the canvas right edge, drawing is clipped to the canvas
    # area (already handled per-frame for vertical overflow).
    strip_h = cfg.strip_h
    strip_row_h = strip_h + cfg.strip_pad   # pitch per row
    px_per_mm = cfg.base_px_per_mm * cfg.zoom
    track_x_start = cfg.canvas_pad + cfg.strip_label_w

    # Order carriers by id for a stable visual layout.
    ordered = sorted(topo.carriers.values(), key=lambda c: c.id)

    strips: dict[CarrierId, StripGeom] = {}
    for i, c in enumerate(ordered):
        y0 = carriers_top + i * strip_row_h
        track_len_px = max(1, int((c.max_pos - c.min_pos) * px_per_mm))
        track_x_end = track_x_start + track_len_px
        strip_w = cfg.strip_label_w + track_len_px + cfg.track_right_pad
        rect = (cfg.canvas_pad, y0, strip_w, strip_h)
        track_y = y0 + strip_h // 2 + 20
        strips[c.id] = StripGeom(
            carrier_id=c.id,
            rect=rect,
            track_y=track_y,
            track_x_start=track_x_start,
            track_x_end=track_x_end,
            min_pos=c.min_pos,
            max_pos=c.max_pos,
            label_rect=(cfg.canvas_pad, y0, cfg.strip_label_w, strip_h),
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
                    orientation=s.orientation_at(cid),
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
        carriers_top=carriers_top,
        carriers_visible_h=max(0, canvas_h - carriers_top - cfg.canvas_pad),
        strip_row_h=strip_row_h,
        strips=strips,
        shelves=shelf_placements,
        rooms=room_placements,
        handoffs=handoff_placements,
    )
