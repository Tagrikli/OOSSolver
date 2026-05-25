"""Reusable pygame visual components for the OOS visualization.

Layout:
- palette    : colors, Fonts, blit_text
- primitives : low-level drawing helpers (glow, beveled, brackets, scanlines, ...)
- chrome     : PanelChrome — header/collapse/scroll frame (internal; used by Panel)
- panel      : Panel — single reusable widget = chrome + pluggable content
- button     : Button widget with named color variants
- layout     : HRow / VStack tiling primitives for child widgets
- widgets/   : stateful canvas widgets (ShelfWidget, RoomWidget, CarrierPanel, ...)
- toasts     : Toast dataclass + ToastManager + draw_toasts

Top-level re-exports keep `from oos.viz.components import X` working for
any X regardless of submodule.

Sidebar-panel *content* classes (the body of stats / queue / etc.) live
in `oos.viz.sidebar`, not here — Panel is generic; content is per-feature.
"""

from oos.viz.components.button import Button
from oos.viz.components.chrome import PanelChrome
from oos.viz.components.layout import HRow, VStack
from oos.viz.components.palette import (
    ACCENT,
    BASE_BLACK,
    BASE_GUTTER,
    BASE_MUTED,
    BASE_SHADOW,
    BASE_SURFACE,
    BG,
    CARRIER_BUSY,
    CARRIER_CUST,
    CARRIER_IDLE,
    CARRIER_OUTLINE,
    CYAN_BRIGHT,
    CYAN_DIM,
    CYAN_MID,
    ERROR,
    GRID_LINE,
    GRID_LINE_BRIGHT,
    HANDOFF_HINT,
    LAVENDER,
    LIME_BRIGHT,
    LIME_MID,
    MAGENTA_BLOOM,
    MAGENTA_BRIGHT,
    MAGENTA_DIM,
    MAGENTA_MID,
    PALLET_BIG,
    PALLET_EMPTY,
    PALLET_SMALL,
    PANEL_BG,
    REQUESTED_GLOW,
    ROOM_BUSY,
    ROOM_IDLE,
    ROOM_PENDING,
    ROOM_READY,
    SHELF_FILL_BIG,
    SHELF_FILL_SMALL,
    SHELF_FRAME_GLOW,
    SHELF_OUTLINE,
    SHELF_TRANSFER,
    SLOT_EMPTY,
    SOFT_WHITE,
    STRIP_BG,
    STRIP_BORDER,
    TEXT,
    TEXT_ACCENT,
    TEXT_DIM,
    TEXT_HEAD,
    TRACK,
    TRANSFER_HINT,
    VIOLET,
    VIOLET_BRIGHT,
    VIOLET_DIM,
    YELLOW_BRIGHT,
    YELLOW_DIM,
    YELLOW_MID,
    Fonts,
    blit_text,
)
from oos.viz.components.panel import Panel, PanelContent
from oos.viz.components.primitives import (
    beveled_polygon,
    dashed_line,
    draw_beveled_frame,
    draw_beveled_rect,
    draw_bracketed_title,
    draw_corner_brackets,
    draw_glow_circle,
    draw_glow_line,
    draw_glow_rect,
    draw_grid_background,
    draw_request_pulse,
    draw_scanlines,
    pulsed_color,
)
from oos.viz.components.toasts import (
    Toast,
    ToastManager,
    draw_toasts,
)
from oos.viz.components.widgets import (
    CarrierIconWidget,
    CarrierPanel,
    CarrierStripBackground,
    CustomerQueueWidget,
    RoomWidget,
    ShelfWidget,
    short_action_label,
)

__all__ = [
    # palette
    "ACCENT", "BASE_BLACK", "BASE_GUTTER", "BASE_MUTED", "BASE_SHADOW",
    "BASE_SURFACE", "BG", "CARRIER_BUSY", "CARRIER_CUST", "CARRIER_IDLE",
    "CARRIER_OUTLINE", "CYAN_BRIGHT", "CYAN_DIM", "CYAN_MID", "ERROR",
    "GRID_LINE", "GRID_LINE_BRIGHT", "HANDOFF_HINT", "LAVENDER", "LIME_BRIGHT",
    "LIME_MID", "MAGENTA_BLOOM", "MAGENTA_BRIGHT", "MAGENTA_DIM",
    "MAGENTA_MID", "PALLET_BIG", "PALLET_EMPTY", "PALLET_SMALL", "PANEL_BG",
    "REQUESTED_GLOW", "ROOM_BUSY", "ROOM_IDLE", "ROOM_PENDING", "ROOM_READY",
    "SHELF_FILL_BIG", "SHELF_FILL_SMALL", "SHELF_FRAME_GLOW", "SHELF_OUTLINE",
    "SHELF_TRANSFER", "SLOT_EMPTY", "SOFT_WHITE", "STRIP_BG", "STRIP_BORDER",
    "TEXT", "TEXT_ACCENT", "TEXT_DIM", "TEXT_HEAD", "TRACK", "TRANSFER_HINT",
    "VIOLET", "VIOLET_BRIGHT", "VIOLET_DIM", "YELLOW_BRIGHT", "YELLOW_DIM",
    "YELLOW_MID", "Fonts", "blit_text",
    # primitives
    "beveled_polygon", "dashed_line", "draw_beveled_frame", "draw_beveled_rect",
    "draw_bracketed_title", "draw_corner_brackets", "draw_glow_circle",
    "draw_glow_line", "draw_glow_rect", "draw_grid_background",
    "draw_request_pulse", "draw_scanlines", "pulsed_color",
    # chrome / panel / button / layout
    "PanelChrome", "Panel", "PanelContent",
    "Button", "HRow", "VStack",
    # widgets
    "CarrierIconWidget", "CarrierPanel", "CarrierStripBackground",
    "CustomerQueueWidget", "RoomWidget", "ShelfWidget", "short_action_label",
    # toasts
    "Toast", "ToastManager", "draw_toasts",
]
