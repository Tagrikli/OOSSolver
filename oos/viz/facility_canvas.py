"""FacilityCanvas — draws a facility into an arbitrary surface rect.

Lifecycle:
    canvas = FacilityCanvas(topology)
    # On window resize / rect change:
    canvas.relayout(rect, zoom=1.0)
    # Per frame:
    canvas.draw(surface, facility, queue, anim_now, wall_now)

The canvas owns its own widgets (carrier strips, queue strip, solvability
overlay) and its scroll state. Hit-test areas (pallet, shelf) are exposed
as attributes after each `draw()` call so the app can route clicks.

Everything the canvas reads about the simulation comes from `facility` —
nothing is captured at construction time. So pure inspection of a static
state works the same as live animation: just call `draw()` with the
matching `anim_now`. Interpolation uses `(command_start_position,
command_started_at, busy_until, anim_now)` from `facility.state.carriers`.
"""

from __future__ import annotations

import pygame

from oos.sim.actions import MultiRelocate, Relocate
from oos.sim.facility import Facility
from oos.sim.tasks import Retrieve, Store, TaskQueue
from oos.sim.topology import Topology
from oos.viz.animation import (
    interpolated_position,
    multi_relocate_visual_position,
    relocate_visual_state,
)
from oos.viz.components import (
    CYAN_BRIGHT,
    CarrierPanel,
    CustomerQueueWidget,
    Fonts,
    SolvabilityOverlay,
    draw_corner_brackets,
    draw_grid_background,
    short_action_label,
)
from oos.viz.layout import Layout, LayoutConfig, compute_layout


class FacilityCanvas:
    """Owns the canvas-side widgets for a topology; draws into an arbitrary rect.

    Two phases:
      1. `relayout(rect, zoom)` — call whenever the destination rect or zoom
         changes. Recomputes the internal Layout and rebuilds CarrierPanels.
      2. `draw(surface, facility, queue, anim_now, wall_now)` — per frame,
         draws the queue strip, all carrier strips with interpolated
         carrier positions and in-flight pallet overlays, the carrier
         scrollbar, the canvas corner brackets, and the solvability overlay.
    """

    SCROLLBAR_GUTTER_W = 6
    CORNER_BRACKET_SIZE = 14
    CORNER_BRACKET_INSET = 8
    SOLVABILITY_PAD = 16

    def __init__(self, topology: Topology, fonts: Fonts | None = None):
        self.topology = topology
        self.fonts = fonts or Fonts.default()

        # Initialised lazily by relayout(). draw() requires relayout() first.
        self._layout: Layout | None = None
        self._rect: pygame.Rect | None = None
        self._zoom: float = 1.0

        self._panels: dict[str, CarrierPanel] = {}
        self._panel_virtual_top: dict[str, int] = {}
        self._carriers_top: int = 0
        self._carriers_visible_h: int = 0
        self._strip_row_h: int = 0

        self._queue_strip: CustomerQueueWidget | None = None
        self._solvability_overlay = SolvabilityOverlay()

        self._carrier_scroll: int = 0

        # Populated each draw(); read by callers for click routing.
        self.pallet_hit_areas: list[tuple[pygame.Rect, int]] = []
        self.shelf_hit_areas: list[tuple[pygame.Rect, str]] = []

    # ---- layout ----------------------------------------------------------

    def relayout(self, rect: pygame.Rect, zoom: float = 1.0) -> None:
        """Rebuild the internal Layout + CarrierPanels for a given target
        rect and zoom. Cheap enough to call on every resize but not every
        frame — `draw()` does NOT call this for you."""
        self._rect = pygame.Rect(rect)
        self._zoom = zoom

        # The existing Layout API takes window_w/h. We map our rect to
        # those so the internal mm→px math is unchanged.
        cfg = LayoutConfig(window_w=rect.w, window_h=rect.h,
                           sidebar_w=0,                # we own no sidebar
                           zoom=zoom)
        # compute_layout returns absolute pixel coords assuming origin (0,0).
        # We translate them to the actual rect.left/top.
        layout = compute_layout(self.topology, cfg)
        self._layout = _translate_layout(layout, dx=rect.left, dy=rect.top)

        self._carriers_top = self._layout.carriers_top
        self._carriers_visible_h = self._layout.carriers_visible_h
        self._strip_row_h = self._layout.strip_row_h

        # Build one CarrierPanel per carrier from the Layout.
        topo = self.topology
        shelves_by_cid: dict[str, list] = {}
        for sp in self._layout.shelves:
            shelves_by_cid.setdefault(sp.carrier_id, []).append(sp)
        rooms_by_cid: dict[str, list] = {}
        for rp in self._layout.rooms:
            rooms_by_cid.setdefault(rp.carrier_id, []).append(rp)
        handoffs_by_cid: dict[str, list[int]] = {}
        for h in self._layout.handoffs:
            handoffs_by_cid.setdefault(h.a, []).append(h.a_x)
            handoffs_by_cid.setdefault(h.b, []).append(h.b_x)

        self._panels.clear()
        self._panel_virtual_top.clear()
        for cid, strip in self._layout.strips.items():
            shelf_specs = []
            for sp in shelves_by_cid.get(cid, []):
                s = topo.shelves[sp.shelf_id]
                shelf_specs.append((
                    sp.shelf_id, sp.x, s.capacity, s.size_class,
                    sp.is_transfer, sp.partner, sp.orientation,
                ))
            room_specs = [(rp.room_id, rp.x) for rp in rooms_by_cid.get(cid, [])]
            self._panels[cid] = CarrierPanel(
                carrier_id=cid,
                strip_rect=strip.rect,
                track_y=strip.track_y,
                track_x_start=strip.track_x_start,
                track_x_end=strip.track_x_end,
                label_rect=strip.label_rect,
                min_pos=strip.min_pos,
                max_pos=strip.max_pos,
                kind=topo.carriers[cid].kind,
                shelf_specs=shelf_specs,
                room_specs=room_specs,
                handoff_positions=handoffs_by_cid.get(cid, []),
            )
            self._panel_virtual_top[cid] = strip.rect[1]

        # Queue strip across the top of our rect.
        qs = self._layout.queue_strip_rect
        self._queue_strip = CustomerQueueWidget(pygame.Rect(*qs))

        # Clamp scroll to the new content height.
        self._clamp_scroll()

    # ---- carrier scroll --------------------------------------------------

    @property
    def carrier_area_rect(self) -> pygame.Rect:
        if self._rect is None:
            return pygame.Rect(0, 0, 0, 0)
        return pygame.Rect(
            self._rect.left, self._carriers_top,
            self._rect.w, self._carriers_visible_h,
        )

    def scroll_carriers(self, dy_pixels: int) -> None:
        self._carrier_scroll += dy_pixels
        self._clamp_scroll()

    def _carrier_content_h(self) -> int:
        if not self._panels:
            return 0
        last_strip_h = next(iter(self._panels.values())).background.rect.h
        n = len(self._panels)
        # All but the last row contribute strip + pad; last contributes
        # only its strip height.
        return (n - 1) * self._strip_row_h + last_strip_h if n > 0 else 0

    def _max_carrier_scroll(self) -> int:
        return max(0, self._carrier_content_h() - self._carriers_visible_h)

    def _clamp_scroll(self) -> None:
        self._carrier_scroll = max(
            0, min(self._max_carrier_scroll(), self._carrier_scroll),
        )

    # ---- draw ------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        facility: Facility,
        queue: TaskQueue,
        anim_now: float,
        wall_now: float = 0.0,
    ) -> None:
        if self._rect is None or self._layout is None:
            raise RuntimeError(
                "FacilityCanvas.draw() called before relayout(); call "
                "relayout(rect, zoom) at least once first.",
            )
        rect = self._rect

        # Grid background inside our rect.
        draw_grid_background(surface, rect, spacing=24)

        # Filter pending tasks (queue strip wants Stores only; pulse-flash
        # logic wants the set of retrieve target pallet ids).
        pending_stores = [t for t in queue.pending if isinstance(t, Store)]
        requested_pallets = frozenset(
            t.pallet for t in queue.pending if isinstance(t, Retrieve)
        )

        # Queue strip across the top.
        assert self._queue_strip is not None
        self._queue_strip.set_pending(pending_stores)
        self._queue_strip.set_now(anim_now)
        self._queue_strip.draw(surface, self.fonts)

        # In-flight overlay (commitment projection — same one the obs
        # builder uses to decide where a mid-transit pallet visually lives).
        from oos.env.observation import compute_in_flight_overlay
        in_flight_loads, pickups_in_flight = compute_in_flight_overlay(facility)

        # Reset hit-test caches each frame.
        self.pallet_hit_areas = []
        self.shelf_hit_areas = []

        # Clip carrier drawing to the visible carrier area so off-screen
        # panels don't paint into the queue strip or past our rect.
        self._clamp_scroll()
        carrier_clip = self.carrier_area_rect
        surface.set_clip(carrier_clip)

        for cid, panel in self._panels.items():
            panel.set_y(self._panel_virtual_top[cid] - self._carrier_scroll)
            panel.set_pulsing_items(requested_pallets)
            panel.set_wall_now(wall_now)

            # Shelves.
            for sid in panel.shelves:
                ss = facility.state.shelves[sid]
                panel.update_shelf(
                    sid, ss.stack, n_hidden=pickups_in_flight.get(sid, 0),
                )

            # Rooms.
            for rid in panel.rooms:
                rs_room = facility.state.rooms[rid]
                visual_load = rs_room.load if pickups_in_flight.get(rid, 0) == 0 else None
                state_name = "ready" if visual_load is None else "idle"
                panel.update_room(rid, visual_load, state_name)

            # Carrier icon — interpolated position over the current command.
            cs = facility.state.carriers[cid]
            cmd = cs.current_command
            if isinstance(cmd, Relocate) and cs.command_started_at is not None:
                pos_now, _ = relocate_visual_state(facility, cid, anim_now)
            elif isinstance(cmd, MultiRelocate) and cs.command_started_at is not None:
                pos_now = multi_relocate_visual_position(
                    facility, cid, cmd, anim_now,
                )
            else:
                pos_now = interpolated_position(facility, cid, anim_now)
            visual_load = in_flight_loads.get(cid, cs.load)

            panel.set_carrier_position(panel.pos_to_x(pos_now))
            panel.set_carrier_load(visual_load)
            panel.set_carrier_state("busy" if cmd is not None else "idle")
            panel.set_carrier_action(short_action_label(cmd))

            panel.draw(
                surface, self.fonts,
                pallet_hit_areas=self.pallet_hit_areas,
                shelf_hit_areas=self.shelf_hit_areas,
            )

        surface.set_clip(None)

        # Vertical scrollbar for the carrier column (only when overflowing).
        self._draw_carrier_scrollbar(surface)

        # Canvas corner brackets.
        draw_corner_brackets(
            surface, rect.inflate(-self.CORNER_BRACKET_INSET,
                                  -self.CORNER_BRACKET_INSET),
            CYAN_BRIGHT,
            size=self.CORNER_BRACKET_SIZE, width=2,
        )

        # Retrievability overlay — bottom-right of our rect.
        # (Uses the underscore helper from oos.sim.shuffle; same function
        # used by the live Store-gate flow and SingleTaskEnv's
        # require_solvable retry loop.)
        from oos.sim.shuffle import _layout_is_solvable
        ov = self._solvability_overlay
        ov.set_rect(pygame.Rect(
            rect.right - ov.W - self.SOLVABILITY_PAD,
            rect.bottom - ov.H - self.SOLVABILITY_PAD,
            ov.W, ov.H,
        ))
        ov.update(_layout_is_solvable(facility))
        ov.draw(surface, self.fonts)

    def _draw_carrier_scrollbar(self, surface: pygame.Surface) -> None:
        max_scroll = self._max_carrier_scroll()
        if max_scroll <= 0:
            return
        area = self.carrier_area_rect
        content_h = self._carrier_content_h()
        track_x = area.right - self.SCROLLBAR_GUTTER_W
        track_top = area.top
        track_h = area.h
        pygame.draw.rect(
            surface, (24, 16, 56),
            pygame.Rect(track_x, track_top, 3, track_h),
        )
        thumb_h = max(20, int(track_h * area.h / content_h))
        thumb_y = track_top + int(
            (track_h - thumb_h) * (self._carrier_scroll / max_scroll)
        )
        pygame.draw.rect(
            surface, CYAN_BRIGHT,
            pygame.Rect(track_x - 1, thumb_y, 5, thumb_h),
            border_radius=2,
        )


def _translate_layout(layout: Layout, dx: int, dy: int) -> Layout:
    """Return a Layout with all pixel coords translated by (dx, dy). Used
    by FacilityCanvas to map (0,0)-based compute_layout output into the
    canvas's actual position on screen."""
    if dx == 0 and dy == 0:
        return layout

    def shift_rect(r):
        x, y, w, h = r
        return (x + dx, y + dy, w, h)

    # Translate strips.
    from oos.viz.layout import (
        HandoffPlacement,
        Layout as L,
        RoomPlacement,
        ShelfPlacement,
        StripGeom,
    )
    new_strips = {}
    for cid, strip in layout.strips.items():
        new_strips[cid] = StripGeom(
            carrier_id=strip.carrier_id,
            rect=shift_rect(strip.rect),
            track_y=strip.track_y + dy,
            track_x_start=strip.track_x_start + dx,
            track_x_end=strip.track_x_end + dx,
            min_pos=strip.min_pos,
            max_pos=strip.max_pos,
            label_rect=shift_rect(strip.label_rect),
        )
    new_shelves = [
        ShelfPlacement(
            shelf_id=sp.shelf_id, carrier_id=sp.carrier_id,
            x=sp.x + dx,
            is_transfer=sp.is_transfer, partner=sp.partner,
            partner_x=(sp.partner_x + dx) if sp.partner_x is not None else None,
            partner_strip_track_y=(
                sp.partner_strip_track_y + dy
                if sp.partner_strip_track_y is not None else None
            ),
            orientation=sp.orientation,
        )
        for sp in layout.shelves
    ]
    new_rooms = [
        RoomPlacement(room_id=rp.room_id, carrier_id=rp.carrier_id, x=rp.x + dx)
        for rp in layout.rooms
    ]
    new_handoffs = [
        HandoffPlacement(
            a=h.a, b=h.b,
            a_x=h.a_x + dx, b_x=h.b_x + dx,
            a_strip_track_y=h.a_strip_track_y + dy,
            b_strip_track_y=h.b_strip_track_y + dy,
        )
        for h in layout.handoffs
    ]
    return L(
        canvas_rect=shift_rect(layout.canvas_rect),
        sidebar_rect=shift_rect(layout.sidebar_rect),
        queue_strip_rect=shift_rect(layout.queue_strip_rect),
        carriers_top=layout.carriers_top + dy,
        carriers_visible_h=layout.carriers_visible_h,
        strip_row_h=layout.strip_row_h,
        strips=new_strips,
        shelves=new_shelves,
        rooms=new_rooms,
        handoffs=new_handoffs,
    )
