"""CarrierPanel — composite per-carrier widget."""

from __future__ import annotations

from typing import Iterable, Optional

import pygame

from oos.sim.state import Pallet
from oos.viz.components.palette import Fonts
from oos.viz.components.widgets.carrier_icon import CarrierIconWidget
from oos.viz.components.widgets.carrier_strip import CarrierStripBackground
from oos.viz.components.widgets.room import RoomWidget
from oos.viz.components.widgets.shelf import ShelfWidget


class CarrierPanel:
    """All visuals belonging to one carrier, composed into one widget.

    Owns: the strip background, the moving carrier icon, the shelves the
    carrier can access, and any rooms it serves. With inter-carrier links
    removed (no handoff/transfer connectors drawn at the canvas level),
    this widget is fully self-contained — its draw() touches only pixels
    inside its own strip.

    Construct once at renderer init with the static topology+geometry.
    Each frame: poke setters with the latest sim state, then `.draw()`.
    """

    def __init__(
        self,
        carrier_id: str,
        strip_rect: tuple[int, int, int, int],
        track_y: int,
        track_x_start: int,
        track_x_end: int,
        label_rect: tuple[int, int, int, int],
        min_pos: int,
        max_pos: int,
        kind: str = "shuttle",
        shelf_specs: Iterable[tuple] = (),
        room_specs: Iterable[tuple] = (),
        handoff_positions: Iterable[int] = (),
    ):
        """Args:
            min_pos, max_pos: carrier's mm bounds — used by pos_to_x.
            shelf_specs: iterable of
                (shelf_id, cx, capacity, size_class, is_transfer,
                 partner_or_None, orientation)
            room_specs: iterable of (room_id, cx)
            handoff_positions: x coords on the track where this carrier
                meets a partner for a handoff.
        """
        self.carrier_id = carrier_id
        self.kind = kind
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.track_y = track_y
        self.track_x_start = track_x_start
        self.track_x_end = track_x_end

        shelf_specs_list = list(shelf_specs)
        self.background = CarrierStripBackground(
            carrier_id=carrier_id,
            rect=pygame.Rect(*strip_rect),
            track_y=track_y,
            track_x_start=track_x_start,
            track_x_end=track_x_end,
            label_rect=pygame.Rect(*label_rect),
            kind=kind,
            shelf_positions=[spec[1] for spec in shelf_specs_list],
            handoff_positions=handoff_positions,
        )
        self.icon = CarrierIconWidget(carrier_id, track_y)
        self.shelves: dict[str, ShelfWidget] = {}
        for spec in shelf_specs_list:
            (shelf_id, cx, capacity, size_class,
             is_transfer, partner, orientation) = spec
            self.shelves[shelf_id] = ShelfWidget(
                shelf_id=shelf_id, cx=cx, cy=track_y,
                capacity=capacity, size_class=size_class,
                is_transfer=is_transfer, partner=partner,
                orientation=orientation,
            )
        self.rooms: dict[str, RoomWidget] = {}
        for room_id, cx in room_specs:
            self.rooms[room_id] = RoomWidget(room_id, cx, track_y)

    # ---- geometry ----------------------------------------------------------

    def pos_to_x(self, p: float) -> int:
        """Map a mm position on the carrier's track to a pixel x coord."""
        span = self.max_pos - self.min_pos
        if span <= 0:
            return self.track_x_start
        frac = (p - self.min_pos) / span
        return int(self.track_x_start + frac * (self.track_x_end - self.track_x_start))

    def set_y(self, new_top_y: int) -> None:
        """Move the entire panel vertically so its strip's top edge sits at
        `new_top_y`. Used by the renderer to apply a scroll offset when the
        canvas can't fit all carriers."""
        dy = new_top_y - self.background.rect.top
        if dy == 0:
            return
        # Strip background: rect, track_y, label_rect.
        self.background.rect = self.background.rect.move(0, dy)
        self.background.track_y += dy
        self.background.label_rect = self.background.label_rect.move(0, dy)
        # Carrier icon track_y.
        self.icon.track_y += dy
        # Shelves: cy.
        for sw in self.shelves.values():
            sw.cy += dy
        # Rooms: cy.
        for rw in self.rooms.values():
            rw.cy += dy
        # Own cached track_y.
        self.track_y += dy

    # ---- carrier-icon setters (forwarded) ----------------------------------

    def set_carrier_position(self, x: float) -> None:
        self.icon.set_position(x)

    def set_carrier_load(self, p: Optional[Pallet]) -> None:
        self.icon.set_load(p)

    def set_carrier_state(self, s: str) -> None:
        # Icon uses state for color; strip uses it to recolour the action chip.
        self.icon.set_state(s)
        self.background.set_state(s)

    def set_carrier_action(self, label: str) -> None:
        # Action label is painted by the strip background in the top-right
        # corner of the carrier's panel — not under the icon.
        self.background.set_action_label(label)

    def set_pulsing_items(self, items: frozenset) -> None:
        """Distribute the pulsing-items set to every shelf and the icon —
        used so any pallet that's the target of a pending Retrieve glows
        wherever it appears."""
        self.icon.set_pulsing(items)
        for sw in self.shelves.values():
            sw.set_pulsing(items)

    def set_wall_now(self, t: float) -> None:
        self.icon.set_wall_now(t)
        for sw in self.shelves.values():
            sw.set_wall_now(t)

    # ---- shelf / room updates ----------------------------------------------

    def update_shelf(
        self, shelf_id: str, stack: list[Pallet], n_hidden: int = 0,
    ) -> None:
        sw = self.shelves.get(shelf_id)
        if sw is None:
            return
        sw.set_stack(stack)
        sw.set_hidden_count(n_hidden)

    def update_room(
        self, room_id: str, load: Optional[Pallet], state: str,
    ) -> None:
        rw = self.rooms.get(room_id)
        if rw is None:
            return
        rw.set_load(load)
        rw.set_state(state)

    # ---- drawing -----------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        fonts: Fonts,
        pallet_hit_areas: Optional[list[tuple[pygame.Rect, int]]] = None,
        shelf_hit_areas: Optional[list[tuple[pygame.Rect, str]]] = None,
    ) -> None:
        self.background.draw(surface, fonts)
        for sid, sw in self.shelves.items():
            r = sw.draw(surface, fonts, hit_areas=pallet_hit_areas)
            if shelf_hit_areas is not None:
                shelf_hit_areas.append((r, sid))
        for rw in self.rooms.values():
            rw.draw(surface, fonts)
        self.icon.draw(surface, fonts)
