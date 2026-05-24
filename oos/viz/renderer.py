"""Orchestrate drawing: takes a Layout + live Facility + queue, draws a frame."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pygame

from oos.sim.actions import (
    MultiRelocate,
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

        # In-flight overlay: same commitment-state projection the observation
        # builder uses. Phase-aware: the pallet only flips from src to
        # carrier once the carrier has physically completed the take_op.
        # Keeps the human's view of state byte-identical to the agent's view.
        from oos.env.observation import compute_in_flight_overlay
        in_flight_loads, pickups_in_flight = compute_in_flight_overlay(facility)

        # Shelves
        self._draw_shelves(
            surface, facility.state, requested_pallets, rs.wall_now,
            pickups_in_flight=pickups_in_flight,
        )

        # Rooms
        self._draw_rooms(
            surface, facility, rs.anim_now,
            pickups_in_flight=pickups_in_flight,
        )

        # Carriers (top)
        self._draw_carriers(
            surface, facility, rs, requested_pallets,
            in_flight_loads=in_flight_loads,
        )

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
        pickups_in_flight: dict[str, int] | None = None,
    ) -> None:
        self.pallet_hit_areas = []
        self.shelf_hit_areas = []
        pickups = pickups_in_flight or {}
        for sp in self.layout.shelves:
            s = self.topology.shelves[sp.shelf_id]
            ss = state.shelves[sp.shelf_id]
            # Hide the topmost N pallets if N carriers are visually mid-Relocate
            # having picked up from this shelf; the sim still has them on the
            # stack until atomic complete(), but visually they're in transit.
            n_hidden = pickups.get(sp.shelf_id, 0)
            visual_stack = ss.stack[:-n_hidden] if n_hidden > 0 else ss.stack
            shelf_rect = ShelfView(
                shelf_id=sp.shelf_id,
                cx=sp.x,
                cy=self.layout.strips[sp.carrier_id].track_y,
                capacity=s.capacity,
                size_class=s.size_class,
                is_transfer=s.is_transfer,
                stack=visual_stack,
                partner=sp.partner,
                pulsing_items=requested_pallets,
                wall_now=wall_now,
            ).draw(surface, self.fonts, hit_areas=self.pallet_hit_areas)
            self.shelf_hit_areas.append((shelf_rect, sp.shelf_id))

    def _draw_rooms(
        self,
        surface: pygame.Surface,
        facility: Facility,
        anim_now: float,
        pickups_in_flight: dict[str, int] | None = None,
    ) -> None:
        state = facility.state
        # Customer interactions are now instant: as soon as a carrier deposits
        # a pallet into a room, the matching task fires and `room.load` either
        # vanishes (retrieve) or has its contents mutated (store). There's no
        # "busy" window. Two states:
        #   - "ready" : room is empty (and possibly serving a pending task)
        #   - "idle"  : room holds an unconsumed pallet (no matching task yet)
        # When a carrier is mid-Relocate FROM a room, the room's load is
        # logically already on the carrier even though the sim still holds it
        # in room.load until atomic complete(); hide it visually.
        del anim_now  # visual readiness is room-load-driven
        pickups = pickups_in_flight or {}
        for rp in self.layout.rooms:
            rs = state.rooms[rp.room_id]
            visual_load = rs.load if pickups.get(rp.room_id, 0) == 0 else None
            state_name = "ready" if visual_load is None else "idle"
            RoomView(
                room_id=rp.room_id,
                cx=rp.x,
                cy=self.layout.strips[rp.carrier_id].track_y,
                state=state_name,
                load=visual_load,
            ).draw(surface, self.fonts)

    def _draw_carriers(
        self,
        surface: pygame.Surface,
        facility: Facility,
        rs: RenderState,
        requested_pallets: frozenset,
        in_flight_loads: dict | None = None,
    ) -> None:
        pulse_phase = (rs.wall_now * 0.8) % 1.0
        loads = in_flight_loads or {}
        for cid, cs in facility.state.carriers.items():
            strip = self.layout.strips[cid]
            cmd = cs.current_command
            # Position interpolation is phase-aware for the multi-step
            # commands (Relocate and MultiRelocate) so the carrier visually
            # follows its actual physical trajectory through each sub-phase.
            # The load shown matches the observation overlay, so what the
            # human sees mirrors what the agent sees in features.
            if (
                isinstance(cmd, Relocate)
                and cs.command_started_at is not None
            ):
                pos_now, _ = _relocate_visual_state(
                    facility, cid, rs.anim_now,
                )
            elif (
                isinstance(cmd, MultiRelocate)
                and cs.command_started_at is not None
            ):
                pos_now = _multi_relocate_visual_position(
                    facility, cid, cmd, rs.anim_now,
                )
            else:
                pos_now = _interpolated_position(facility, cid, rs.anim_now)
            visual_load = loads.get(cid, cs.load)
            x = strip.pos_to_x(pos_now)
            y = strip.track_y
            state_name = "busy" if cmd is not None else "idle"
            label = short_action_label(cmd)
            CarrierIconView(
                carrier_id=cid,
                x=x,
                y=y,
                load=visual_load,
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


def _location_visual_pos(loc: str, carrier_id: str, topo) -> int | None:
    """Carrier-track x-position (discrete slot index) of a Relocate endpoint
    — a real shelf or a room. Matches the sim's discrete `Position` type so
    duration calculations get the integer arguments they expect; the visual
    smoothing that interpolates between two such positions happens at the
    call site and is the only place that produces fractional values."""
    if loc in topo.shelves:
        return topo.shelves[loc].position_for[carrier_id]
    if loc in topo.rooms:
        return topo.rooms[loc].position
    return None


def _relocate_visual_state(facility: Facility, carrier_id: str, anim_now: float):
    """Synthesize the physical visual position of a carrier mid-Relocate.

    The sim treats Relocate as one atomic command — at `start()` the carrier
    just becomes busy; at `complete()` the pallet teleports src→dst. The viz
    interpolates the carrier through four physical sub-phases for an
    accurate motion trajectory:

      Phase 1: travel to src       — position interpolates start→src
      Phase 2: shelf-op take       — position holds at src
      Phase 3: travel to dst       — position interpolates src→dst
      Phase 4: shelf-op place      — position holds at dst

    The carrier's *visual load* is decided separately by the observation's
    in-flight overlay (see compute_in_flight_overlay) so what the human sees
    matches what the agent sees: the pallet is on the carrier from t=0 of
    the Relocate, the source loses it at t=0 too. This function therefore
    returns position only; the second tuple slot is reserved/ignored.
    """
    cs = facility.state.carriers[carrier_id]
    cmd = cs.current_command
    assert isinstance(cmd, Relocate)
    topo = facility.topology
    durs = facility.durations
    carrier = topo.carriers[carrier_id]

    start_pos = (
        cs.command_start_position
        if cs.command_start_position is not None
        else cs.position
    )
    src_pos = _location_visual_pos(cmd.src, carrier_id, topo)
    dst_pos = _location_visual_pos(cmd.dst, carrier_id, topo)
    if src_pos is None or dst_pos is None:
        return float(cs.position), None

    move1 = durs.move(carrier, start_pos, src_pos)
    take_op = (
        durs.shelf_op("take", topo.shelves[cmd.src])
        if cmd.src in topo.shelves else 0.0
    )
    move2 = durs.move(carrier, src_pos, dst_pos)
    # place_op is consumed visually within phase 4; no separate computation needed.

    elapsed = max(0.0, anim_now - (cs.command_started_at or 0.0))

    # Phase 1: travel to src.
    if elapsed < move1:
        frac = elapsed / max(move1, 1e-9)
        return start_pos + frac * (src_pos - start_pos), None
    elapsed -= move1
    # Phase 2: holding at src during take-op.
    if elapsed < take_op:
        return src_pos, None
    elapsed -= take_op
    # Phase 3: travel to dst.
    if elapsed < move2:
        frac = elapsed / max(move2, 1e-9)
        return src_pos + frac * (dst_pos - src_pos), None
    # Phase 4: holding at dst during place-op.
    return dst_pos, None


def _multi_relocate_visual_position(
    facility: Facility, carrier_id: str, cmd, anim_now: float,
) -> float:
    """Phase-aware physical position for a carrier mid-MultiRelocate.

    For the initiator A:
        Phase 1: travel from A_start to src
        Phase 2: take_op at src (stationary)
        Phase 3: travel from src to A's handoff pose
        Phase 4+: stationary at A's handoff pose (post-handoff A is "done"
                  with its motion even though it remains locked until B finishes)

    For the partner B:
        Phase 1: travel from B_start to B's handoff pose
        Phase 2: stationary at B's handoff pose (waiting for handoff)
        Phase 3: post-handoff travel from B's handoff pose to dst
        Phase 4: stationary at dst (give_op)
    """
    state = facility.state
    topo = facility.topology
    durs = facility.durations
    a_cs = state.carriers[cmd.carrier_id]
    b_cs = state.carriers[cmd.partner_id]
    a_car = topo.carriers[cmd.carrier_id]
    b_car = topo.carriers[cmd.partner_id]
    pair = (cmd.carrier_id, cmd.partner_id)
    a_pose, b_pose = topo.handoff_positions[pair]
    a_start = (
        a_cs.command_start_position
        if a_cs.command_start_position is not None
        else a_cs.position
    )
    b_start = (
        b_cs.command_start_position
        if b_cs.command_start_position is not None
        else b_cs.position
    )
    src_pos = _location_visual_pos(cmd.src, cmd.carrier_id, topo)
    dst_pos = _location_visual_pos(cmd.dst, cmd.partner_id, topo)
    if src_pos is None or dst_pos is None:
        return float(state.carriers[carrier_id].position)
    take_op = (
        durs.shelf_op("take", topo.shelves[cmd.src])
        if cmd.src in topo.shelves else 0.0
    )
    a_to_src = durs.move(a_car, a_start, src_pos)
    a_to_handoff = durs.move(a_car, src_pos, a_pose)
    a_at_handoff_t = a_to_src + take_op + a_to_handoff
    b_to_handoff = durs.move(b_car, b_start, b_pose)
    sync_done_t = max(a_at_handoff_t, b_to_handoff)
    handoff_done_t = sync_done_t + durs.handoff()
    b_to_dst = durs.move(b_car, b_pose, dst_pos)

    elapsed = max(0.0, anim_now - (a_cs.command_started_at or 0.0))

    if carrier_id == cmd.carrier_id:
        # A's trajectory.
        if elapsed < a_to_src:
            frac = elapsed / max(a_to_src, 1e-9)
            return a_start + frac * (src_pos - a_start)
        e = elapsed - a_to_src
        if e < take_op:
            return float(src_pos)
        e -= take_op
        if e < a_to_handoff:
            frac = e / max(a_to_handoff, 1e-9)
            return src_pos + frac * (a_pose - src_pos)
        return float(a_pose)

    if carrier_id == cmd.partner_id:
        # B's trajectory.
        if elapsed < b_to_handoff:
            frac = elapsed / max(b_to_handoff, 1e-9)
            return b_start + frac * (b_pose - b_start)
        # B holds at handoff pose until both arrived + handoff_op done.
        if elapsed < handoff_done_t:
            return float(b_pose)
        e = elapsed - handoff_done_t
        if e < b_to_dst:
            frac = e / max(b_to_dst, 1e-9)
            return b_pose + frac * (dst_pos - b_pose)
        return float(dst_pos)

    return float(state.carriers[carrier_id].position)


def _command_end_position(facility: Facility, cmd, carrier_id: str):
    """Visual endpoint where the carrier ends up when `cmd` completes.

    For Relocate this is dst. For MultiRelocate it depends on which carrier
    is being queried: A (initiator) ends at the handoff pose; B (partner)
    ends at dst. The full phase-aware visual position for both is computed
    by `_multi_relocate_visual_state` — this function is a coarser fallback
    used only by the unphased `_interpolated_position` path.
    """
    topo = facility.topology
    if isinstance(cmd, Relocate):
        return _location_visual_pos(cmd.dst, carrier_id, topo)
    if isinstance(cmd, MultiRelocate):
        pair = (cmd.carrier_id, cmd.partner_id)
        if carrier_id == cmd.carrier_id:
            if pair in topo.handoff_positions:
                return topo.handoff_positions[pair][0]
            return None
        if carrier_id == cmd.partner_id:
            return _location_visual_pos(cmd.dst, carrier_id, topo)
        return None
    return None
