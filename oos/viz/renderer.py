"""Orchestrate drawing: takes a Layout + live Facility + queue, draws a frame."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pygame

from oos.sim.actions import (
    Handoff,
    MoveToPartner,
    Relocate,
)
from oos.sim.facility import Facility
from oos.sim.state import FacilityState
from oos.sim.tasks import Store, TaskQueue
from oos.sim.topology import Topology
from oos.viz.components import (
    ACCENT,
    CYAN_BRIGHT,
    YELLOW_BRIGHT,
    CarrierIconView,
    CarrierStripView,
    ControlsPanel,
    CustomerQueueStrip,
    DistributionPanel,
    Fonts,
    HandoffHint,
    LegendPanel,
    QueuePanel,
    RoomView,
    ShelfView,
    StatsPanel,
    Toast,
    TransferConnectorHint,
    draw_corner_brackets,
    draw_grid_background,
    draw_scanlines,
    draw_toasts,
    short_action_label,
)
from oos.viz.layout import Layout


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
    # Session-level provenance shown in the status panel so the user can
    # see at a glance which model is driving and which facility is loaded.
    policy_label: str = "(random policy)"
    facility_name: str = ""
    # Last forward-pass artifacts from the active LearnedPolicy. None when
    # a random policy is active or the first decision hasn't fired yet.
    policy_logits: object = None       # np.ndarray | None
    policy_action_mask: object = None  # np.ndarray | None
    policy_chosen: int | None = None
    # Legal action entries for the current querying carrier; entries[i] is the
    # ActionEntry at action-index i for i in [0, len(entries)). Used to label
    # the distribution panel's bars on hover.
    policy_action_entries: list = field(default_factory=list)
    mouse_pos: tuple[int, int] = (0, 0)


class Renderer:
    @property
    def stats_panel(self):
        return self._stats_panel

    @property
    def queue_panel(self):
        return self._queue_panel

    @property
    def dist_panel(self):
        return self._dist_panel

    @property
    def legend_panel(self):
        return self._legend_panel

    @property
    def controls_panel(self):
        return self._controls_panel

    def __init__(self, layout: Layout, topology: Topology, fonts: Fonts | None = None):
        self.layout = layout
        self.topology = topology
        self.fonts = fonts or Fonts.default()

        self._strips: dict[str, CarrierStripView] = {}
        for cid, g in layout.strips.items():
            self._strips[cid] = CarrierStripView(
                carrier_id=cid,
                rect=pygame.Rect(*g.rect),
                track_y=g.track_y,
                track_x_start=g.track_x_start,
                track_x_end=g.track_x_end,
                label_rect=pygame.Rect(*g.label_rect),
            )

        sb = pygame.Rect(*layout.sidebar_rect)
        self._sb_pad = 10
        self._panel_w = sb.w - 2 * self._sb_pad
        self._col_x = sb.left + self._sb_pad

        # Per-panel preferred (expanded) heights — used as the FLEX WEIGHTS in
        # _layout_panels. Stored separately from panel.rect because that rect
        # gets overwritten each frame by the layout pass.
        self._stats_panel = StatsPanel(pygame.Rect(self._col_x, sb.top + self._sb_pad, self._panel_w, 180))
        self._queue_panel = QueuePanel(pygame.Rect(self._col_x, 0, self._panel_w, 240))
        self._dist_panel = DistributionPanel(pygame.Rect(self._col_x, 0, self._panel_w, 180))
        self._controls_panel = ControlsPanel(pygame.Rect(self._col_x, 0, self._panel_w, 150))
        self._legend_panel = LegendPanel(pygame.Rect(self._col_x, 0, self._panel_w, 200))

        self._panel_preferred_h: dict[int, int] = {
            id(self._stats_panel):    180,
            id(self._queue_panel):    240,
            id(self._dist_panel):     180,
            id(self._controls_panel): 150,
            id(self._legend_panel):   200,
        }

        # Default-collapse the panels that the user usually doesn't need open
        # while watching the agent run. Click their header to expand.
        self._controls_panel.chrome.collapsed = True
        self._legend_panel.chrome.collapsed = True

        self._canvas_rect = pygame.Rect(*layout.canvas_rect)
        self._sidebar_rect = sb
        self._queue_strip = CustomerQueueStrip(pygame.Rect(*layout.queue_strip_rect))

        # Hit-test surface: (rect, pallet_id) for every pallet slot in the
        # current frame (including empty pallets). Refreshed each `draw()`.
        self.pallet_hit_areas: list[tuple[pygame.Rect, int]] = []
        # Bounding rect per shelf (covers all slots whether filled or not).
        # Used for shelf-level hotkeys that operate on the shelf itself, not
        # a specific pallet (e.g. push empty / pop top).
        self.shelf_hit_areas: list[tuple[pygame.Rect, str]] = []

    # ------------------------------------------------------------------

    def _layout_panels(self) -> None:
        """Re-flow the side panels each frame based on their collapsed state.

        Collapsed panels shrink to header-only height; expanded panels share
        the remaining vertical space proportionally to their preferred height
        (stored in `panel.rect.height` at construction time). Panel rects and
        chrome rects are mutated in place — both must agree for hit-tests to
        be consistent with drawing.
        """
        from oos.viz.components import PanelChrome

        sb = self._sidebar_rect
        pad = self._sb_pad
        col_x = self._col_x
        panel_w = self._panel_w

        # Fixed top→bottom order. Don't reorder; users expect spatial stability.
        panels = [
            self._stats_panel,
            self._queue_panel,
            self._dist_panel,
            self._controls_panel,
            self._legend_panel,
        ]
        collapsed_h = PanelChrome.HEADER_H + PanelChrome.COLLAPSED_LIP

        total_h = sb.h - 2 * pad
        gap_total = pad * (len(panels) - 1)
        collapsed_total = sum(collapsed_h for p in panels if p.chrome.collapsed)
        expanded_panels = [p for p in panels if not p.chrome.collapsed]
        remaining = total_h - gap_total - collapsed_total

        # Distribute `remaining` across expanded panels in proportion to each
        # panel's preferred height (stored at construction; immutable across
        # frames). At least 60 px per expanded panel to keep the body usable.
        weight_sum = sum(self._panel_preferred_h[id(p)] for p in expanded_panels) or 1
        min_expanded = 60
        expanded_heights: dict[int, int] = {}
        if expanded_panels:
            allotted = 0
            for p in expanded_panels[:-1]:
                pref = self._panel_preferred_h[id(p)]
                h = max(min_expanded, int(remaining * pref / weight_sum))
                expanded_heights[id(p)] = h
                allotted += h
            expanded_heights[id(expanded_panels[-1])] = max(min_expanded, remaining - allotted)

        # Place panels top → bottom.
        y = sb.top + pad
        for p in panels:
            h = collapsed_h if p.chrome.collapsed else expanded_heights[id(p)]
            new_rect = pygame.Rect(col_x, y, panel_w, h)
            p.rect = new_rect
            p.chrome.rect = new_rect
            y += h + pad

    def draw(
        self,
        surface: pygame.Surface,
        facility: Facility,
        queue: TaskQueue,
        rs: RenderState,
        manual_mode: bool = False,
    ) -> None:
        # Re-flow the side panels based on current collapsed state. Must
        # happen before any panel hit-test or draw so coordinates agree.
        self._layout_panels()

        # Background: dark + grid
        surface.fill((5, 3, 16))
        draw_grid_background(surface, self._canvas_rect, spacing=24)

        # Pull the set of items currently being requested via pending retrieves
        # so we can pulse them everywhere they appear (shelves + carrier loads).
        from oos.sim.tasks import Retrieve, Store
        pending_stores = [t for t in queue.pending if isinstance(t, Store)]
        requested_pallets = frozenset(
            t.pallet for t in queue.pending if isinstance(t, Retrieve)
        )

        # Global customer queue strip across the top.
        self._queue_strip.draw(surface, self.fonts, pending_stores, rs.anim_now)

        # Strips (background)
        for strip in self._strips.values():
            strip.draw(surface, self.fonts)

        # Hints (drawn before shelves so they sit visually behind)
        self._draw_handoff_hints(surface)
        self._draw_transfer_hints(surface)

        # Shelves
        self._draw_shelves(surface, facility.state, requested_pallets, rs.wall_now)

        # Rooms (use interpolated carrier position so the room dims the instant
        # the carrier starts moving away, instead of waiting for the move command
        # to complete).
        self._draw_rooms(surface, facility, rs.anim_now)

        # Carriers (top)
        self._draw_carriers(surface, facility, rs, requested_pallets)

        # Canvas corner brackets
        draw_corner_brackets(
            surface, self._canvas_rect.inflate(-8, -8), CYAN_BRIGHT, size=14, width=2
        )

        # Sidebar
        self._stats_panel.draw(
            surface,
            self.fonts,
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
        self._queue_panel.draw(
            surface, self.fonts, queue.pending, rs.anim_now, manual_mode=manual_mode
        )
        self._dist_panel.draw(
            surface, self.fonts,
            logits=rs.policy_logits,
            action_mask=rs.policy_action_mask,
            chosen=rs.policy_chosen,
            action_entries=rs.policy_action_entries,
            mouse_pos=rs.mouse_pos,
        )
        self._controls_panel.draw(surface, self.fonts)
        self._legend_panel.draw(surface, self.fonts)

        # Scanlines overlay on the whole window for atmosphere
        draw_scanlines(
            surface,
            pygame.Rect(0, 0, surface.get_width(), surface.get_height()),
            color=(255, 255, 255),
            alpha=8,
            spacing=3,
        )

        # Toasts (top-right of canvas, above scanlines)
        draw_toasts(
            surface,
            rs.toasts,
            anchor_topright=(self._canvas_rect.right - 14, self._canvas_rect.top + 14),
            fonts=self.fonts,
            wall_now=rs.wall_now,
        )

    # ------------------------------------------------------------------

    def _draw_shelves(
        self,
        surface: pygame.Surface,
        state: FacilityState,
        requested_pallets: frozenset,
        wall_now: float,
    ) -> None:
        self.pallet_hit_areas = []
        self.shelf_hit_areas = []
        for sp in self.layout.shelves:
            s = self.topology.shelves[sp.shelf_id]
            ss = state.shelves[sp.shelf_id]
            shelf_rect = ShelfView(
                shelf_id=sp.shelf_id,
                cx=sp.x,
                cy=self.layout.strips[sp.carrier_id].track_y,
                capacity=s.capacity,
                size_class=s.size_class,
                is_transfer=s.is_transfer,
                stack=ss.stack,
                partner=sp.partner,
                pulsing_items=requested_pallets,
                wall_now=wall_now,
            ).draw(surface, self.fonts, hit_areas=self.pallet_hit_areas)
            self.shelf_hit_areas.append((shelf_rect, sp.shelf_id))

    def _draw_rooms(
        self, surface: pygame.Surface, facility: Facility, anim_now: float
    ) -> None:
        state = facility.state
        # The unified-action model puts pallets *in* the room (RoomState.load)
        # rather than on the carrier during customer interaction. Room state:
        #   - "busy"  : a customer interaction is in progress (mutating room.load)
        #   - "ready" : room is empty and idle (waiting for a deposit)
        #   - "idle"  : room holds an unconsumed pallet but no pending task matches
        del anim_now  # no longer needed; visual readiness is room-load-driven
        for rp in self.layout.rooms:
            rs = state.rooms[rp.room_id]
            if rs.customer_interaction_until is not None:
                state_name = "busy"
            elif rs.load is None:
                state_name = "ready"
            else:
                state_name = "idle"
            RoomView(
                room_id=rp.room_id,
                cx=rp.x,
                cy=self.layout.strips[rp.carrier_id].track_y,
                state=state_name,
                load=rs.load,
            ).draw(surface, self.fonts)

    def _draw_carriers(
        self,
        surface: pygame.Surface,
        facility: Facility,
        rs: RenderState,
        requested_pallets: frozenset,
    ) -> None:
        pulse_phase = (rs.wall_now * 0.8) % 1.0
        for cid, cs in facility.state.carriers.items():
            strip = self.layout.strips[cid]
            pos_now = _interpolated_position(facility, cid, rs.anim_now)
            x = strip.pos_to_x(pos_now)
            y = strip.track_y
            cmd = cs.current_command
            state_name = "busy" if cmd is not None else "idle"
            label = short_action_label(cmd)
            CarrierIconView(
                carrier_id=cid,
                x=x,
                y=y,
                load=cs.load,
                state=state_name,
                action_label=label,
                is_querying=(cid == rs.querying),
                pulse_phase=pulse_phase,
                pulsing_items=requested_pallets,
                wall_now=rs.wall_now,
            ).draw(surface, self.fonts)

    def _draw_handoff_hints(self, surface: pygame.Surface) -> None:
        for h in self.layout.handoffs:
            HandoffHint(
                a_label=h.a,
                b_label=h.b,
                a_x=h.a_x,
                b_x=h.b_x,
                a_track_y=h.a_strip_track_y,
                b_track_y=h.b_strip_track_y,
            ).draw(surface, self.fonts)

    def _draw_transfer_hints(self, surface: pygame.Surface) -> None:
        by_shelf: dict[str, list] = {}
        for sp in self.layout.shelves:
            if sp.is_transfer:
                by_shelf.setdefault(sp.shelf_id, []).append(sp)
        for sid, placements in by_shelf.items():
            if len(placements) == 2:
                a, b = placements
                a_y = self.layout.strips[a.carrier_id].track_y
                b_y = self.layout.strips[b.carrier_id].track_y
                TransferConnectorHint(a.x, a_y, b.x, b_y).draw(surface, self.fonts)


# ---------------------------------------------------------------------------
# Position interpolation
# ---------------------------------------------------------------------------


def _interpolated_position(facility: Facility, carrier_id: str, anim_now: float) -> float:
    """Carrier x-position along its track at `anim_now`.

    Only the MOVE portion of a command's duration is used for interpolation;
    any trailing shelf-op time leaves the carrier stationary at the target.
    For Relocate (two moves chained by a take+place), we use the destination
    as the visual endpoint and let the chained shelf-op time hold the carrier
    there once the interpolation reaches 1.0. The source pose is implicit.
    """
    cs = facility.state.carriers[carrier_id]
    cmd = cs.current_command
    if cmd is None or cs.command_started_at is None or cs.busy_until is None:
        return float(cs.position)
    end_pos = _command_end_position(facility, cmd, carrier_id)
    if end_pos is None:
        return float(cs.position)
    start_pos = (
        cs.command_start_position if cs.command_start_position is not None else cs.position
    )
    carrier = facility.topology.carriers[carrier_id]
    move_dur = facility.durations.move(carrier, start_pos, end_pos)
    if move_dur <= 0:
        return float(end_pos)
    elapsed = max(0.0, anim_now - cs.command_started_at)
    frac = min(1.0, elapsed / move_dur)
    return start_pos + frac * (end_pos - start_pos)


def _command_end_position(facility: Facility, cmd, carrier_id: str):
    topo = facility.topology
    if isinstance(cmd, Relocate):
        # Visual end is the destination — the source is just a waypoint.
        if cmd.dst in topo.shelves:
            return topo.shelves[cmd.dst].position_for[carrier_id]
        if cmd.dst in topo.rooms:
            return topo.rooms[cmd.dst].position
        return None
    if isinstance(cmd, MoveToPartner):
        pair = (carrier_id, cmd.partner_id)
        if pair in topo.handoff_positions:
            return topo.handoff_positions[pair][0]
        for s in topo.shelves.values():
            if s.is_transfer and carrier_id in s.access and cmd.partner_id in s.access:
                return s.position_for[carrier_id]
        return None
    if isinstance(cmd, Handoff):
        return None
    return None
