"""Orchestrate drawing: per-frame, mutate cached widget state, then draw.

Each carrier owns a `CarrierPanel` (built once at __init__ from the layout)
that bundles its strip background, its shelves, its rooms, and its moving
icon. Per-frame work is just:
  1. Compute live carrier position via the animation module.
  2. Call setters on each panel to push the latest sim state.
  3. Tell each panel to draw itself.

The sidebar panels are constructed once and laid out top→bottom each frame
based on which are collapsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from oos.sim.actions import MultiRelocate, Relocate
from oos.sim.facility import Facility
from oos.sim.tasks import Store, TaskQueue
from oos.sim.topology import Topology
from oos.viz.animation import (
    interpolated_position,
    multi_relocate_visual_position,
    relocate_visual_state,
)
from oos.viz.components import (
    CYAN_BRIGHT,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    VIOLET_BRIGHT,
    YELLOW_BRIGHT,
    CarrierPanel,
    Column,
    CustomerQueueWidget,
    Fonts,
    Panel,
    PanelChrome,
    SolvabilityOverlay,
    TabStrip,
    Toast,
    draw_corner_brackets,
    draw_grid_background,
    draw_scanlines,
    draw_toasts,
    short_action_label,
)
from oos.viz.layout import Layout
from oos.viz.sidebar import (
    ControlsContent,
    DistributionContent,
    LegendContent,
    QueueContent,
    RandomizeContent,
    StatsContent,
)


@dataclass
class RenderState:
    """Everything the renderer needs that isn't already in the Facility."""

    mode: str
    wall_speed: float
    last_reward: float
    n_completed: int
    last_action: str
    querying: str
    anim_now: float
    toasts: list[Toast] = field(default_factory=list)
    wall_now: float = 0.0
    policy_label: str = "(random policy)"
    facility_name: str = ""
    policy_logits: object = None
    policy_action_mask: object = None
    policy_chosen: int | None = None
    policy_action_entries: list = field(default_factory=list)
    policy_query_log: dict = field(default_factory=dict)
    mouse_pos: tuple[int, int] = (0, 0)


def _resolve_panel_heights(
    panels: list[Panel], total_h: int, gap: int, min_expanded: int = 60,
) -> list[int]:
    """Distribute `total_h` across `panels` for vertical stacking.

    Collapsed panels get a fixed `HEADER_H + COLLAPSED_LIP`. The remaining
    space is split among expanded panels using their `preferred_h` as
    weights, with each expanded panel clamped to ≥ `min_expanded`.

    Returned list is parallel to `panels` and consumable as `Column.heights`.
    """
    collapsed_h = PanelChrome.HEADER_H + PanelChrome.COLLAPSED_LIP
    n = len(panels)
    if n == 0:
        return []
    gap_total = gap * (n - 1) if n > 1 else 0
    collapsed_total = sum(collapsed_h for p in panels if p.chrome.collapsed)
    expanded_panels = [p for p in panels if not p.chrome.collapsed]
    remaining = total_h - gap_total - collapsed_total

    weight_sum = sum(p.preferred_h for p in expanded_panels) or 1
    expanded_h: dict[int, int] = {}
    if expanded_panels:
        allotted = 0
        for p in expanded_panels[:-1]:
            h = max(min_expanded, int(remaining * p.preferred_h / weight_sum))
            expanded_h[id(p)] = h
            allotted += h
        expanded_h[id(expanded_panels[-1])] = max(
            min_expanded, remaining - allotted,
        )
    return [
        collapsed_h if p.chrome.collapsed else expanded_h[id(p)]
        for p in panels
    ]


class Renderer:
    @property
    def stats_panel(self) -> Panel: return self._stats_panel

    @property
    def queue_panel(self) -> Panel: return self._queue_panel

    @property
    def dist_panel(self) -> Panel: return self._dist_panel

    @property
    def legend_panel(self) -> Panel: return self._legend_panel

    @property
    def controls_panel(self) -> Panel: return self._controls_panel

    @property
    def queue_content(self) -> QueueContent:
        """Typed accessor for queue panel's content — exposes hit_button /
        hit_slider / fullness / set_fullness_from_x for the app event loop."""
        return self._queue_panel.content  # type: ignore[return-value]

    @property
    def randomize_panel(self) -> Panel:
        return self._randomize_panel

    @property
    def randomize_content(self) -> RandomizeContent:
        return self._randomize_panel.content  # type: ignore[return-value]

    @property
    def tab_strip(self) -> TabStrip:
        return self._tab_strip

    @property
    def active_tab(self) -> int:
        return self._tab_strip.active

    def active_panels(self) -> list[Panel]:
        """The panels currently visible in the sidebar for the active tab."""
        if self._tab_strip.active == 0:
            return [
                self._stats_panel, self._queue_panel, self._dist_panel,
                self._controls_panel, self._legend_panel,
            ]
        return [self._randomize_panel]

    def __init__(self, layout: Layout, topology: Topology, fonts: Fonts | None = None):
        self.layout = layout
        self.topology = topology
        self.fonts = fonts or Fonts.default()

        # Build one CarrierPanel per carrier from the layout. Each panel
        # owns its strip + shelves + rooms + icon and is self-contained.
        # Group placements by carrier_id first so each panel only sees its
        # own shelves and rooms.
        shelves_by_cid: dict[str, list] = {}
        for sp in layout.shelves:
            shelves_by_cid.setdefault(sp.carrier_id, []).append(sp)
        rooms_by_cid: dict[str, list] = {}
        for rp in layout.rooms:
            rooms_by_cid.setdefault(rp.carrier_id, []).append(rp)
        # Per-carrier handoff x's on that carrier's own track. The strip
        # background paints a dot at each so the human can see where a
        # handoff would meet.
        handoffs_by_cid: dict[str, list[int]] = {}
        for h in layout.handoffs:
            handoffs_by_cid.setdefault(h.a, []).append(h.a_x)
            handoffs_by_cid.setdefault(h.b, []).append(h.b_x)

        self._panels: dict[str, CarrierPanel] = {}
        for cid, strip in layout.strips.items():
            shelf_specs = []
            for sp in shelves_by_cid.get(cid, []):
                s = topology.shelves[sp.shelf_id]
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
                kind=topology.carriers[cid].kind,
                shelf_specs=shelf_specs,
                room_specs=room_specs,
                handoff_positions=handoffs_by_cid.get(cid, []),
            )

        sb = pygame.Rect(*layout.sidebar_rect)
        self._sb_pad = 10
        self._panel_w = sb.w - 2 * self._sb_pad
        self._col_x = sb.left + self._sb_pad

        # Each side panel = generic Panel wrapping a content class. The
        # content owns per-frame state + paint logic; Panel handles chrome.
        self._stats_panel = Panel(
            pygame.Rect(self._col_x, sb.top + self._sb_pad, self._panel_w, 180),
            title="STATUS", content=StatsContent(),
            accent=MAGENTA_BRIGHT, preferred_h=180,
        )
        self._queue_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 240),
            title="Pending tasks", content=QueueContent(),
            accent=CYAN_BRIGHT, preferred_h=240,
        )
        self._dist_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 140),
            title="Action dist", content=DistributionContent(),
            accent=LIME_BRIGHT, preferred_h=140,
        )
        self._controls_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 150),
            title="Controls", content=ControlsContent(),
            accent=VIOLET_BRIGHT, preferred_h=150, collapsed=True,
        )
        self._legend_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 200),
            title="Legend", content=LegendContent(),
            accent=YELLOW_BRIGHT, preferred_h=200, collapsed=True,
        )

        # Tab 2 panel: randomize/preview controls. Single non-collapsable
        # panel that fills the whole sidebar body below the tab strip.
        self._randomize_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 400),
            title="Random initial state", content=RandomizeContent(),
            accent=MAGENTA_BRIGHT, preferred_h=400, collapsable=False,
        )

        # Tab strip lives at the very top of the sidebar. Tab 0 = STATUS
        # (the existing 5 panels), Tab 1 = RANDOMIZE (preview generator).
        self._tab_strip = TabStrip(
            labels=["status", "randomize"], active=0, accent=MAGENTA_BRIGHT,
        )

        # Bottom-right canvas overlay showing whether the current layout
        # is retrievable. Position is re-pinned every frame in draw() so
        # it survives window resizes.
        self._solvability_overlay = SolvabilityOverlay()

        self._canvas_rect = pygame.Rect(*layout.canvas_rect)
        self._sidebar_rect = sb
        self._queue_strip = CustomerQueueWidget(pygame.Rect(*layout.queue_strip_rect))

        # Carrier-area scroll: panels stack at a fixed strip height; if the
        # total stacked column exceeds the visible canvas, the renderer
        # offsets each panel's y by `_carrier_scroll`. Each panel was built
        # with its "virtual" y (panel_virtual_top[cid]); per-frame we call
        # set_y(virtual_top - scroll) before drawing.
        self._carriers_top = layout.carriers_top
        self._carriers_visible_h = layout.carriers_visible_h
        self._strip_row_h = layout.strip_row_h
        self._panel_virtual_top: dict[str, int] = {
            cid: layout.strips[cid].rect[1] for cid in self._panels
        }
        self._carrier_scroll: int = 0

        # Hit-test surface, refreshed each draw().
        self.pallet_hit_areas: list[tuple[pygame.Rect, int]] = []
        self.shelf_hit_areas: list[tuple[pygame.Rect, str]] = []

    # ---- carrier-area scroll ---------------------------------------------

    @property
    def carrier_area_rect(self) -> pygame.Rect:
        """The visible region into which carrier panels are clipped."""
        return pygame.Rect(
            self._canvas_rect.left, self._carriers_top,
            self._canvas_rect.w, self._carriers_visible_h,
        )

    def _carrier_content_h(self) -> int:
        n = len(self._panels)
        if n == 0:
            return 0
        # Last panel only contributes its strip height, not strip+pad.
        # row_h = strip_h + strip_pad → content = n*row_h - strip_pad.
        return n * self._strip_row_h - max(0, self._strip_row_h - self._panels[
            next(iter(self._panels))].background.rect.h)

    def _max_carrier_scroll(self) -> int:
        return max(0, self._carrier_content_h() - self._carriers_visible_h)

    def _clamp_scroll(self) -> None:
        self._carrier_scroll = max(
            0, min(self._max_carrier_scroll(), self._carrier_scroll),
        )

    def scroll_carriers(self, dy_pixels: int) -> None:
        """Scroll the carrier column. Positive dy = content moves up (next
        carriers come into view from the bottom)."""
        self._carrier_scroll += dy_pixels
        self._clamp_scroll()

    def _draw_carrier_scrollbar(self, surface: pygame.Surface) -> None:
        max_scroll = self._max_carrier_scroll()
        if max_scroll <= 0:
            return
        area = self.carrier_area_rect
        content_h = self._carrier_content_h()
        track_x = area.right - 6
        track_top = area.top
        track_h = area.h
        pygame.draw.rect(
            surface, (24, 16, 56),  # GRID_LINE_BRIGHT-ish
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

    # ------------------------------------------------------------------

    def _layout_panels(self) -> None:
        """Re-flow side panels each frame based on collapsed state and the
        active tab. The tab strip claims the top slot of the sidebar; the
        active tab's panels fill the remainder, distributed by
        `preferred_h` weight among the expanded panels."""
        sb = self._sidebar_rect
        pad = self._sb_pad
        col_x = self._col_x
        panel_w = self._panel_w

        # Tab strip at the top of the sidebar (always fixed height).
        tab_top = sb.top + pad
        self._tab_strip.set_rect(pygame.Rect(
            col_x, tab_top, panel_w, TabStrip.H,
        ))

        panels = self.active_panels()
        if not panels:
            return

        panels_top = tab_top + TabStrip.H + pad
        panels_h = sb.bottom - pad - panels_top
        panels_rect = pygame.Rect(col_x, panels_top, panel_w, panels_h)

        heights = _resolve_panel_heights(
            panels, total_h=panels_h, gap=pad, min_expanded=60,
        )
        Column(panels, gap=pad, heights=heights).lay_out(panels_rect)

    def draw(
        self,
        surface: pygame.Surface,
        facility: Facility,
        queue: TaskQueue,
        rs: RenderState,
        manual_mode: bool = False,
    ) -> None:
        self._layout_panels()

        surface.fill((5, 3, 16))
        draw_grid_background(surface, self._canvas_rect, spacing=24)

        from oos.sim.tasks import Retrieve
        pending_stores = [t for t in queue.pending if isinstance(t, Store)]
        requested_pallets = frozenset(
            t.pallet for t in queue.pending if isinstance(t, Retrieve)
        )

        # Top customer queue strip.
        self._queue_strip.set_pending(pending_stores)
        self._queue_strip.set_now(rs.anim_now)
        self._queue_strip.draw(surface, self.fonts)

        # In-flight overlay: same commitment-state projection the observation
        # builder uses. Phase-aware: the pallet only flips from src to
        # carrier once the carrier has physically completed the take_op.
        from oos.env.observation import compute_in_flight_overlay
        in_flight_loads, pickups_in_flight = compute_in_flight_overlay(facility)

        # Reset hit-test caches; carrier panels will populate them.
        self.pallet_hit_areas = []
        self.shelf_hit_areas = []

        # Clamp scroll, then clip carrier drawing to the visible carrier
        # area so off-screen panels don't paint over the queue strip or
        # spill into the sidebar gutter.
        self._clamp_scroll()
        carrier_clip = pygame.Rect(
            self._canvas_rect.left, self._carriers_top,
            self._canvas_rect.w, self._carriers_visible_h,
        )
        surface.set_clip(carrier_clip)

        # Per-carrier update + draw. Each CarrierPanel is self-contained:
        # we just push the latest shelf stacks, room load, carrier pose into
        # its setters, then ask it to paint.
        for cid, panel in self._panels.items():
            # Apply the scroll offset to this panel's vertical position.
            panel.set_y(self._panel_virtual_top[cid] - self._carrier_scroll)
            panel.set_pulsing_items(requested_pallets)
            panel.set_wall_now(rs.wall_now)

            # Shelves on this carrier's strip.
            for sid, sw in panel.shelves.items():
                ss = facility.state.shelves[sid]
                panel.update_shelf(
                    sid, ss.stack, n_hidden=pickups_in_flight.get(sid, 0),
                )

            # Rooms served by this carrier.
            for rid in panel.rooms:
                rs_room = facility.state.rooms[rid]
                visual_load = rs_room.load if pickups_in_flight.get(rid, 0) == 0 else None
                state_name = "ready" if visual_load is None else "idle"
                panel.update_room(rid, visual_load, state_name)

            # Carrier icon: physical position (phase-aware) + visual load.
            cs = facility.state.carriers[cid]
            cmd = cs.current_command
            if isinstance(cmd, Relocate) and cs.command_started_at is not None:
                pos_now, _ = relocate_visual_state(facility, cid, rs.anim_now)
            elif isinstance(cmd, MultiRelocate) and cs.command_started_at is not None:
                pos_now = multi_relocate_visual_position(facility, cid, cmd, rs.anim_now)
            else:
                pos_now = interpolated_position(facility, cid, rs.anim_now)
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

        # Drop the carrier-area clip before drawing chrome/sidebar.
        surface.set_clip(None)

        # Vertical scroll indicator along the right edge of the carrier
        # area — only shown when the column actually overflows.
        self._draw_carrier_scrollbar(surface)

        # Canvas corner brackets.
        draw_corner_brackets(
            surface, self._canvas_rect.inflate(-8, -8), CYAN_BRIGHT, size=14, width=2,
        )

        # Retrievability overlay — pinned to the canvas bottom-right.
        # Underscore-prefixed helper, but the same one used by the live
        # Store-gate flow + SingleTaskEnv require_solvable retry loop.
        from oos.sim.shuffle import _layout_is_solvable
        solvable = _layout_is_solvable(facility)
        ov = self._solvability_overlay
        pad = 16
        ov.set_rect(pygame.Rect(
            self._canvas_rect.right - ov.W - pad,
            self._canvas_rect.bottom - ov.H - pad,
            ov.W, ov.H,
        ))
        ov.update(solvable)
        ov.draw(surface, self.fonts)

        # Sidebar — push per-frame state into each content, then draw panels.
        # Always update both tab groups (cheap), but only draw the active one.
        self._stats_panel.content.update(
            sim_time=rs.anim_now,
            wall_speed=rs.wall_speed,
            mode=rs.mode,
            last_reward=rs.last_reward,
            n_completed=rs.n_completed,
            last_action=rs.last_action,
            querying=rs.querying,
            policy_label=rs.policy_label,
            facility_name=rs.facility_name,
        )
        self._queue_panel.content.update(
            pending=queue.pending, now=rs.anim_now, manual_mode=manual_mode,
        )
        self._dist_panel.content.update(
            query_log=rs.policy_query_log,
            last_queried=rs.querying,
            mouse_pos=rs.mouse_pos,
        )
        self._randomize_panel.content.update(wall_now=rs.wall_now)  # type: ignore[attr-defined]

        # Tab strip first, then the active tab's panels.
        self._tab_strip.draw(surface, self.fonts)
        for panel in self.active_panels():
            panel.draw(surface, self.fonts)

        draw_scanlines(
            surface,
            pygame.Rect(0, 0, surface.get_width(), surface.get_height()),
            color=(255, 255, 255), alpha=8, spacing=3,
        )

        draw_toasts(
            surface,
            rs.toasts,
            anchor_topright=(self._canvas_rect.right - 14, self._canvas_rect.top + 14),
            fonts=self.fonts,
            wall_now=rs.wall_now,
        )
