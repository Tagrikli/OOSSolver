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

from oos.env import Environment
from oos.sim.tasks import TaskQueue
from oos.sim.topology import Topology
from oos.viz.components import (
    CYAN_BRIGHT,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    YELLOW_BRIGHT,
    Column,
    Fonts,
    Panel,
    PanelChrome,
    TabStrip,
    Toast,
    draw_scanlines,
    draw_toasts,
)
from oos.viz.layout import Layout
from oos.viz.sidebar import (
    AutoQueueContent,
    DistributionContent,
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
    total_actions: int
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
    reward_breakdown: dict = field(default_factory=dict)
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
    def auto_queue_panel(self) -> Panel:
        return self._auto_queue_panel

    @property
    def auto_queue_content(self) -> AutoQueueContent:
        return self._auto_queue_panel.content  # type: ignore[return-value]

    @property
    def tab_strip(self) -> TabStrip:
        return self._tab_strip

    @property
    def active_tab(self) -> int:
        return self._tab_strip.active

    def active_panels(self) -> list[Panel]:
        """The panels currently visible BELOW the tab strip for the
        active tab. The Status panel itself is persistent at the top of
        the sidebar (above the tab strip), so it's NOT included here —
        it's drawn separately by `_layout_panels` / `draw`."""
        if self._tab_strip.active == 0:        # status
            return [self._queue_panel, self._dist_panel]
        if self._tab_strip.active == 1:        # randomize
            return [self._randomize_panel]
        # tab 2 — auto-queue (task-stream config)
        return [self._auto_queue_panel]

    def __init__(
        self,
        layout: Layout,
        topology: Topology,
        fonts: Fonts | None = None,
        zoom: float = 1.0,
    ):
        self.layout = layout
        self.topology = topology
        self.fonts = fonts or Fonts.default()

        # Canvas side: FacilityCanvas owns the carrier strips, queue strip,
        # solvability overlay, and carrier-area scroll state. We hand it
        # the rect derived from the Layout AND the current zoom so its
        # internal layout matches the outer Layout's mm→px scale.
        from oos.viz.facility_canvas import FacilityCanvas
        self._canvas = FacilityCanvas(topology, fonts=self.fonts)
        self._canvas.relayout(pygame.Rect(*layout.canvas_rect), zoom=zoom)

        # Sidebar side: panel construction stays here. The sidebar is
        # UI/policy state, not sim state.
        sb = pygame.Rect(*layout.sidebar_rect)
        self._sb_pad = 10
        self._panel_w = sb.w - 2 * self._sb_pad
        self._col_x = sb.left + self._sb_pad
        self._sidebar_rect = sb

        # Each side panel = generic Panel wrapping a content class. The
        # content owns per-frame state + paint logic; Panel handles chrome.
        # Status panel is a persistent header above the tab strip; not
        # collapsable (the user expects always-visible at-a-glance data).
        self._stats_panel = Panel(
            pygame.Rect(self._col_x, sb.top + self._sb_pad, self._panel_w, 180),
            title="STATUS", content=StatsContent(),
            accent=MAGENTA_BRIGHT, preferred_h=180, collapsable=False,
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
        self._randomize_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 400),
            title="Random initial state", content=RandomizeContent(),
            accent=MAGENTA_BRIGHT, preferred_h=400, collapsable=False,
        )
        self._auto_queue_panel = Panel(
            pygame.Rect(self._col_x, 0, self._panel_w, 240),
            title="Auto-queue (task stream)", content=AutoQueueContent(),
            accent=YELLOW_BRIGHT, preferred_h=240, collapsable=False,
        )
        self._tab_strip = TabStrip(
            labels=["status", "randomize", "auto-queue"],
            active=0, accent=MAGENTA_BRIGHT,
        )

    # ---- canvas forwarders -----------------------------------------------
    # FacilityCanvas owns the carrier-area scroll + hit-test surfaces;
    # expose them on Renderer so existing app.py callers don't change.

    @property
    def carrier_area_rect(self) -> pygame.Rect:
        return self._canvas.carrier_area_rect

    def scroll_carriers(self, dy_pixels: int) -> None:
        self._canvas.scroll_carriers(dy_pixels)

    @property
    def pallet_hit_areas(self) -> list[tuple[pygame.Rect, int]]:
        return self._canvas.pallet_hit_areas

    @property
    def shelf_hit_areas(self) -> list[tuple[pygame.Rect, str]]:
        return self._canvas.shelf_hit_areas

    # ------------------------------------------------------------------

    def _layout_panels(self) -> None:
        """Re-flow side panels each frame based on collapsed state and the
        active tab.

        Sidebar vertical stack (top → bottom):
          1. Status panel (persistent — same content across all tabs)
          2. Tab strip
          3. Active tab's panels (distributed by `preferred_h` weight)
        """
        sb = self._sidebar_rect
        pad = self._sb_pad
        col_x = self._col_x
        panel_w = self._panel_w

        # 1. Status panel pinned at the top — same height regardless of
        #    tab so it doesn't jump around when the user switches.
        stats_top = sb.top + pad
        stats_h = self._stats_panel.preferred_h
        self._stats_panel.rect = pygame.Rect(
            col_x, stats_top, panel_w, stats_h,
        )

        # 2. Tab strip just below the status panel.
        tab_top = stats_top + stats_h + pad
        self._tab_strip.set_rect(pygame.Rect(
            col_x, tab_top, panel_w, TabStrip.H,
        ))

        # 3. Active tab's panels fill the rest.
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
        facility: Environment,
        queue: TaskQueue,
        rs: RenderState,
        manual_mode: bool = False,
    ) -> None:
        self._layout_panels()

        # Screen-wide background.
        surface.fill((5, 3, 16))

        # Canvas — delegated to FacilityCanvas. It paints grid, queue
        # strip, all carrier strips with interpolation + in-flight overlay,
        # scrollbar, corner brackets, and the solvability overlay.
        self._canvas.draw(
            surface, facility, queue,
            anim_now=rs.anim_now, wall_now=rs.wall_now,
        )

        # Sidebar — push per-frame state into each content, then draw panels.
        self._stats_panel.content.update(
            sim_time=rs.anim_now,
            wall_speed=rs.wall_speed,
            mode=rs.mode,
            last_reward=rs.last_reward,
            n_completed=rs.n_completed,
            total_actions=rs.total_actions,
            last_action=rs.last_action,
            querying=rs.querying,
            policy_label=rs.policy_label,
            facility_name=rs.facility_name,
            last_reward_breakdown=rs.reward_breakdown,
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
        self._auto_queue_panel.content.update(  # type: ignore[attr-defined]
            wall_now=rs.wall_now, auto_enabled=not manual_mode,
        )

        # Persistent Status panel at the top (visible on every tab).
        self._stats_panel.draw(surface, self.fonts)
        self._tab_strip.draw(surface, self.fonts)
        for panel in self.active_panels():
            panel.draw(surface, self.fonts)

        # Screen-wide overlays.
        draw_scanlines(
            surface,
            pygame.Rect(0, 0, surface.get_width(), surface.get_height()),
            color=(255, 255, 255), alpha=8, spacing=3,
        )
        canvas_rect = self._canvas._rect  # type: ignore[attr-defined]
        if canvas_rect is not None:
            draw_toasts(
                surface,
                rs.toasts,
                anchor_topright=(canvas_rect.right - 14, canvas_rect.top + 14),
                fonts=self.fonts,
                wall_now=rs.wall_now,
            )
