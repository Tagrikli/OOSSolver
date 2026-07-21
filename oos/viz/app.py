"""DearPyGui app — plain renderer + controls over a Session.

The view is pure glue: every frame it ticks the Session (advancing playback)
and redraws the canvas from sim truth at the current time. It mutates the sim
ONLY through the Session's World/Playback methods. Carrier sprites are placed
via `carrier_position_at`, so what you see is exactly what the sim computed.

Layout: a scrollable canvas on the left (resizes with the window; mouse-wheel
scales the system horizontally) and a fixed control panel on the right.

Run:  python -m oos.viz [facility]
"""

from __future__ import annotations

import math
import os
import time

import dearpygui.dearpygui as dpg

from oos.facilities import FACILITIES
from oos.sim.actions import carrier_position_at
from oos.viz.geometry import build_geometry
from oos.viz.session import Session
from oos.viz.state_store import ViewState, load_view_state, save_view_state

# High-visibility palette (2026-07-13 operator request: bigger elements,
# intuitive item colors, state-colored room borders). Dark background kept;
# item shades are medium-dark but saturated so they pop against black.
BG          = (5, 3, 16, 255)        # BASE_BLACK
TRACK       = (13, 74, 94, 255)      # CYAN_DIM
TEXT        = (200, 208, 235, 255)   # brighter lavender — labels
DIM         = (90, 74, 120, 255)     # BASE_MUTED
HEAD        = (252, 238, 12, 255)    # YELLOW_BRIGHT

SHELF_BGCOL = (10, 6, 24, 255)       # dark shelf background (frame fill)
SHELF_SMALL = (80, 140, 255, 255)    # BLUE — small shelf outline (the old
#                                      cyan blended with turquoise empties)
SHELF_BIG   = (255, 42, 109, 255)    # MAGENTA   — big shelf outline
SHELF_TRANS = (224, 192, 32, 255)    # YELLOW    — transfer shelf outline
SHELF_EV    = (235, 205, 70, 255)    # yellowish SECOND border — EV charger
PALLET = {
    "empty": (64, 224, 208, 255),    # turquoise — empty pallet
    "small": (34, 155, 74, 255),     # dark green — sedan
    "big":   (205, 60, 48, 255),     # dark red — SUV
}
GLYPH = {"small": "S", "big": "X"}   # letter cue (red-green colorblind safe)
GLYPH_COL   = (255, 255, 255, 230)
REQUESTED   = (252, 238, 12, 255)    # YELLOW_BRIGHT — pending-retrieve highlight
RELOCATING  = (255, 150, 40, 255)    # ORANGE — pending Evict/Place (a service
#                                      relocation, not a customer request)
ROOM_FILL   = (30, 24, 56, 255)      # BASE_SURFACE
ROOM_STAGED   = (70, 225, 105, 255)  # green — ready (staged empty waiting)
ROOM_UNSTAGED = (255, 70, 70, 255)   # red — no empty staged
ROOM_SERVING  = (255, 150, 40, 255)  # orange — customer entering/leaving
HANDOFF_DOT = (255, 140, 66, 255)    # ORANGE — handoff-pose marker on a lane
CARRIER_IDLE   = (148, 160, 176, 255)   # steel gray — the only "metal" thing
CARRIER_WORK_LO = (170, 185, 202, 255)  # working pulse: low…
CARRIER_WORK_HI = (245, 250, 255, 255)  # …to bright
CARRIER_OUT = (5, 3, 16, 255)        # BASE_BLACK
QUERY_HL    = (252, 238, 12, 255)    # querying-carrier outline

PANEL_W = 360
LOG_LINES = 28                            # pooled, individually-colored rows

C_SEDAN = (95, 205, 130, 255)             # green tint — everything sedan
C_SUV   = (255, 120, 105, 255)            # red tint — everything SUV
C_REQ   = (252, 238, 12, 255)             # yellow — requests/retrievals
C_OK    = (110, 230, 130, 255)
C_BAD   = (255, 96, 96, 255)
C_WARN  = (240, 200, 90, 255)

# DPG's built-in font is ASCII-only — every ✓/▶/⇩ glyph renders as '?'.
# Load a system font that has them; silently keep the default if none found.
_FONT_PATHS = (
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",       # Fedora
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # Debian/Ubuntu
    "/usr/share/fonts/TTF/DejaVuSans.ttf",                     # Arch
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",    # macOS
    "C:/Windows/Fonts/segoeui.ttf",                            # Windows
)


def _ui_glyphs() -> set[int]:
    """Every non-ASCII codepoint the UI can emit: scan the sources whose
    strings end up on buttons or in the event log (self-maintaining)."""
    import oos.plan.moves, oos.plan.planner, oos.plan.solver, oos.viz.session, oos.viz.solver_bridge
    chars: set[int] = set()
    for mod in (None, oos.viz.session, oos.viz.solver_bridge,
                oos.plan.solver, oos.plan.planner, oos.plan.moves):
        path = __file__ if mod is None else (mod.__file__ or "")
        try:
            with open(path, encoding="utf-8") as f:
                chars |= {ord(c) for c in f.read() if ord(c) > 127}
        except OSError:
            pass
    return chars


def _bind_ui_font() -> None:
    path = next((p for p in _FONT_PATHS if os.path.isfile(p)), None)
    if path is None:
        return
    try:
        ver = str(dpg.get_dearpygui_version())
        with dpg.font_registry():
            with dpg.font(path, 15) as f:
                if ver.startswith(("0.", "1.")):
                    # DPG ≥ 2.x loads glyph ranges automatically; older
                    # versions need the hint + explicit codepoints.
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                    dpg.add_font_chars(sorted(_ui_glyphs()))
        dpg.bind_font(f)
    except Exception:
        pass                        # any hiccup → default font, still usable


def run_app(facility: str | None = None) -> None:
    vs = load_view_state()
    facility = facility or vs.facility
    if facility not in FACILITIES:
        facility = "tiny_medipol"
    session = Session(facility)
    session.set_speed(vs.speed)
    ui: dict = {
        "geom": None,
        "geom_key": None,
        "zoom": vs.zoom,
        "pallet_rects": [],     # (x0, y0, x1, y1, pallet_id) from last redraw
        "shelf_rects": [],      # (x0, y0, x1, y1, shelf_id) from last redraw
        "legend": True,         # on-canvas color key (checkbox in LAYOUT)
        "place_pid": None,      # pallet armed for PLACE (P key), awaiting a
        #                         shelf click
    }

    dpg.create_context()
    _bind_ui_font()
    with dpg.theme(tag="tight_theme"):     # dense text blocks (status, log)
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 1)

    def _save():
        """Persist the current picks to .viz_state.json."""
        save_view_state(ViewState(
            facility=session.facility_name,
            speed=session.speed,
            auto_arrivals=session.auto_arrivals,
            target_fullness=dpg.get_value("target_full"),
            change_rate=dpg.get_value("change_rate"),
            dynamicity=dpg.get_value("dynamicity"),
            suv_rate=dpg.get_value("suv_rate"),
            random_room=dpg.get_value("random_room"),
            fullness=dpg.get_value("fullness"),
            zoom=ui["zoom"],
            serve_dwell=dpg.get_value("serve_dwell"),
        ))

    # ---- callbacks ----------------------------------------------------
    def on_facility(_s, name, _u):
        if not name or name == session.facility_name:
            return                    # ignore a spurious / no-op startup callback
        session.swap_facility(name)
        ui["geom_key"] = None        # force a geometry rebuild for the new topo
        _sync_play_label()
        _save()

    def on_play(*_):
        session.toggle_play(); _sync_play_label()

    def on_step(*_):
        session.step_once(); _sync_play_label()

    def on_reset(*_):
        session.reset(); _sync_play_label()

    def on_speed(_s, val, _u): session.set_speed(val)

    def on_dwell(_s, val, _u):
        session.set_serve_dwell(val)
        _save()

    def on_store_small(*_): session.enqueue_store("small")
    def on_store_big(*_):   session.enqueue_store("big")
    def on_clear(*_):       session.clear_queue()

    def _world_kwargs() -> dict:
        return dict(
            target=dpg.get_value("target_full"),
            change=dpg.get_value("change_rate"),
            churn=dpg.get_value("dynamicity"),
            suv_rate=dpg.get_value("suv_rate"),
        )

    def on_auto(_s, val, _u):
        session.set_auto_arrivals(val)
        if val:
            session.configure_setpoint(**_world_kwargs())
        _save()

    def on_world(*_):
        # LIVE retune — the facility keeps running (no reset).
        session.configure_setpoint(**_world_kwargs())
        _save()

    def on_random_room(_s, val, _u):
        session.set_random_room(val)
        _save()

    def _store_n(n: int, size: str):
        def cb(*_): session.burst_stores(n, size)
        return cb

    def _req_n(n: int):
        def cb(*_): session.request_random(n)
        return cb

    def on_rush_out(*_):    session.request_all()

    def on_reroll(*_):
        session.reroll_layout(fullness=dpg.get_value("fullness")); _save()

    def on_wheel(_s, delta, _u):
        if not (dpg.is_item_hovered("canvas") or dpg.is_item_hovered("canvas_host")):
            return
        z = ui["zoom"] * (1.12 if delta > 0 else 1.0 / 1.12)
        ui["zoom"] = max(0.4, min(8.0, z))

    def _under_cursor(rects):
        mx, my = dpg.get_drawing_mouse_pos()
        for (x0, y0, x1, y1, ident) in rects:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                return ident
        return None

    def on_canvas_click(*_):
        if not dpg.is_item_hovered("canvas"):
            return
        if ui["place_pid"] is not None:
            # PLACE mode: this click chooses the destination shelf.
            sid = _under_cursor(ui["shelf_rects"])
            pid = ui["place_pid"]
            ui["place_pid"] = None
            if sid is not None:
                session.request_place(pid, sid)
            else:
                session._note(f"place canceled (no shelf clicked)")
            return
        pid = _under_cursor(ui["pallet_rects"])
        if pid is not None:
            session.request_retrieve(pid)

    def on_key_evict(*_):
        if not dpg.is_item_hovered("canvas"):
            return
        pid = _under_cursor(ui["pallet_rects"])
        if pid is not None:
            session.request_evict(pid)

    def on_key_place(*_):
        if not dpg.is_item_hovered("canvas"):
            return
        pid = _under_cursor(ui["pallet_rects"])
        if pid is None or pid == ui["place_pid"]:
            if ui["place_pid"] is not None:
                session._note(f"place canceled (pallet {ui['place_pid']})")
            ui["place_pid"] = None      # P on empty space / same car: cancel
            return
        ui["place_pid"] = pid
        session._note(f"PLACE armed: pallet {pid} — click a destination shelf")

    def _sync_play_label():
        dpg.set_item_label("playbtn", "‖ Pause" if session.playing else "▶ Play")

    # ---- window: scrollable canvas (left) + fixed panel (right) --------
    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="canvas_host", width=-(PANEL_W + 8), height=-1,
                                  horizontal_scrollbar=True):
                dpg.add_drawlist(width=900, height=600, tag="canvas")
            with dpg.child_window(width=PANEL_W, height=-1, tag="controls"):
                with dpg.collapsing_header(label="STATUS", default_open=True,
                                           tag="status_hdr"):
                    with dpg.group(tag="status_grp"):
                        dpg.add_text("", tag="st_state", wrap=PANEL_W - 20)
                        dpg.add_text("", tag="st_fac")
                        dpg.add_text("", tag="st_cap")
                        dpg.add_text("", tag="st_sedans", color=C_SEDAN)
                        dpg.add_text("", tag="st_suvs", color=C_SUV)
                        dpg.add_text("", tag="st_take_s")
                        dpg.add_text("", tag="st_take_b")
                        dpg.add_text("", tag="st_wait_s", color=C_SEDAN)
                        dpg.add_text("", tag="st_wait_b", color=C_SUV)
                        dpg.add_text("", tag="st_wait_r", color=C_REQ)
                        dpg.add_text("", tag="st_drop")
                        dpg.add_text("", tag="full_state", wrap=PANEL_W - 20)
                        dpg.add_text("RETRIEVALS (since reset, seconds)", color=HEAD)
                        dpg.add_text("", tag="st_stat_sedan", color=C_SEDAN)
                        dpg.add_text("", tag="st_stat_suv", color=C_SUV)
                        dpg.add_text("", tag="st_stat_total")
                        dpg.add_text("SERVICE (turn came → delivered, seconds)",
                                     color=HEAD)
                        dpg.add_text("", tag="st_srv_sedan", color=C_SEDAN)
                        dpg.add_text("", tag="st_srv_suv", color=C_SUV)
                        dpg.add_text("", tag="st_srv_total")
                    dpg.bind_item_theme("status_grp", "tight_theme")

                dpg.add_separator()
                dpg.add_text("FACILITY", color=HEAD)
                dpg.add_combo(sorted(FACILITIES), default_value=facility,
                              callback=on_facility, width=-1, tag="facility_combo")

                dpg.add_separator()
                dpg.add_text("PLAYBACK", color=HEAD)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="▶ Play", callback=on_play, tag="playbtn", width=92)
                    dpg.add_button(label="» Step", callback=on_step, width=80)
                    dpg.add_button(label="↻ Reset", callback=on_reset, width=80)
                dpg.add_slider_float(label="speed", default_value=vs.speed, min_value=0.0,
                                     max_value=64.0, callback=on_speed, width=-60)
                dpg.add_text("", tag="status", wrap=PANEL_W - 20, color=DIM)

                dpg.add_separator()
                with dpg.collapsing_header(label="DEMAND  (you are the customer)",
                                           default_open=True, tag="demand_hdr"):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="+1 sedan", callback=on_store_small, width=82)
                        dpg.add_button(label="+5", callback=_store_n(5, "small"), width=48)
                        dpg.add_button(label="+10", callback=_store_n(10, "small"), width=48)
                        dpg.add_button(label="+20", callback=_store_n(20, "small"), width=-1)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="+1 SUV", callback=on_store_big,
                                       width=82, tag="store_big_btn")
                        dpg.add_button(label="+2", callback=_store_n(2, "big"), width=48)
                        dpg.add_button(label="+5", callback=_store_n(5, "big"), width=48)
                        dpg.add_button(label="+10", callback=_store_n(10, "big"), width=-1)
                    with dpg.group(horizontal=True):
                        dpg.add_text("request random")
                        dpg.add_button(label="1", callback=_req_n(1), width=36)
                        dpg.add_button(label="3", callback=_req_n(3), width=36)
                        dpg.add_button(label="5", callback=_req_n(5), width=36)
                        dpg.add_button(label="10", callback=_req_n(10), width=-1)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="RUSH-OUT (all)", callback=on_rush_out, width=170)
                        dpg.add_button(label="Clear queue", callback=on_clear, width=-1)
                    dpg.add_slider_float(
                        label="customer dwell (s)", default_value=vs.serve_dwell,
                        min_value=0.0, max_value=180.0, format="%.0f s",
                        callback=on_dwell, width=-120, tag="serve_dwell")
                    dpg.add_text("click a pallet on the canvas → request it", color=DIM)
                    dpg.add_text("hover a car:  E = evict · P = arm place,\n"
                                 "then click the destination shelf", color=DIM)

                dpg.add_separator()
                with dpg.collapsing_header(label="AUTO WORLD  (set-point)",
                                           default_open=True, tag="auto_hdr"):
                    dpg.add_checkbox(label="auto-world", default_value=vs.auto_arrivals,
                                     callback=on_auto, tag="auto")
                    dpg.add_slider_float(label="target fullness", default_value=vs.target_fullness,
                                         min_value=0.0, max_value=1.0,
                                         callback=on_world, width=-120, tag="target_full")
                    dpg.add_slider_float(label="change rate", default_value=vs.change_rate,
                                         min_value=0.0, max_value=1.0,
                                         callback=on_world, width=-120, tag="change_rate")
                    dpg.add_slider_float(label="dynamicity", default_value=vs.dynamicity,
                                         min_value=0.0, max_value=1.0,
                                         callback=on_world, width=-120, tag="dynamicity")
                    dpg.add_slider_float(label="SUV rate", default_value=vs.suv_rate,
                                         min_value=0.0, max_value=1.0,
                                         callback=on_world, width=-120, tag="suv_rate")
                    dpg.add_checkbox(label="random room", default_value=vs.random_room,
                                     callback=on_random_room, tag="random_room")
                    dpg.add_text("", tag="world_hint", color=C_WARN, wrap=PANEL_W - 20)
                    dpg.add_text("fullness marches to the target at the change-rate\n"
                                 "pace; dynamicity = constant in-out exchange on top\n"
                                 "(1 = doors saturated, visits get short)",
                                 color=DIM, wrap=PANEL_W - 20)

                dpg.add_separator()
                with dpg.collapsing_header(label="LAYOUT", default_open=True,
                                           tag="layout_hdr"):
                    dpg.add_slider_float(label="fullness", default_value=vs.fullness, min_value=0.0,
                                         max_value=1.0, width=-70, tag="fullness")
                    dpg.add_button(label="Re-roll layout", callback=on_reroll, width=-1)
                    dpg.add_checkbox(label="show legend", default_value=True,
                                     callback=lambda _s, v, _u: ui.__setitem__("legend", v))
                    dpg.add_text("mouse-wheel over canvas → scale horizontally", color=DIM)

                dpg.add_separator()
                dpg.add_text("EVENT LOG", color=HEAD)
                # Stretch (-1) to fill the rest of the panel: collapse the
                # sections above to hand the log more room. It still scrolls
                # inside, so a fully-expanded panel just shows fewer rows.
                with dpg.child_window(height=-1, tag="logbox"):
                    for i in range(LOG_LINES):
                        dpg.add_text("", tag=f"log{i}", wrap=PANEL_W - 30)
                dpg.bind_item_theme("logbox", "tight_theme")

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_canvas_click)
        dpg.add_mouse_wheel_handler(callback=on_wheel)
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=on_play)
        dpg.add_key_press_handler(dpg.mvKey_S, callback=on_step)
        dpg.add_key_press_handler(dpg.mvKey_R, callback=on_reset)
        # Charger-shelf service ops (SOLUTION_V3_1 §2), cursor-targeted:
        # E = evict the hovered car; P = arm it for placement, then click
        # the destination shelf.
        dpg.add_key_press_handler(dpg.mvKey_E, callback=on_key_evict)
        dpg.add_key_press_handler(dpg.mvKey_P, callback=on_key_place)

    _sync_play_label()
    session.set_serve_dwell(vs.serve_dwell)
    if vs.random_room:
        session.set_random_room(True)
    if vs.auto_arrivals:                 # restore the saved set-point world
        session.set_auto_arrivals(True)
        session.configure_setpoint(
            target=vs.target_fullness, change=vs.change_rate,
            churn=vs.dynamicity, suv_rate=vs.suv_rate)

    dpg.create_viewport(title=f"OOSSolver — {facility}", width=1320, height=760)
    dpg.setup_dearpygui()
    dpg.set_primary_window("root", True)
    dpg.show_viewport()

    last = time.perf_counter()
    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = now - last
        last = now
        session.ensure_started()      # never render an un-reset engine
        session.tick(dt)
        _relayout(session, ui)
        _redraw(session, ui)
        dpg.set_value("status", session.status_line())
        _update_status_panel(session)
        _update_log(session)
        dpg.render_dearpygui_frame()

    _save()                              # persist final speed / zoom / world knobs
    dpg.destroy_context()


def _yes_no(tag: str, label: str, verdict) -> None:
    """Render a gate verdict line: YES (green) / NO (red) / — (no gate)."""
    if verdict is None:
        dpg.set_value(tag, f"{label}  —")
        dpg.configure_item(tag, color=DIM)
    else:
        dpg.set_value(tag, f"{label}  {'YES' if verdict else 'NO'}")
        dpg.configure_item(tag, color=C_OK if verdict else C_BAD)


def _fmt_stat(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"{label:<6} —"
    return (f"{label:<6} n={s['n']:<4d} min {s['min']:.0f}  med {s['med']:.0f}  "
            f"avg {s['avg']:.0f}  max {s['max']:.0f}")


def _update_status_panel(session: Session) -> None:
    inv = session.inventory()
    qs = session.queue_stats()
    state, tone = session.system_state()
    dpg.set_value("st_state", state)
    dpg.configure_item("st_state", color={"ok": C_OK, "warn": C_WARN,
                                          "dim": DIM}[tone])
    dpg.set_value("st_fac", f"facility   {session.facility_name}")
    dpg.set_value("st_cap", f"capacity   {inv['pallets']} pallets · "
                            f"{inv['slots']} slots · "
                            f"{inv['sedans'] + inv['suvs']} cars")
    dpg.set_value("st_sedans", f"sedans stored    {inv['sedans']}")
    dpg.set_value("st_suvs",   f"SUVs stored      {inv['suvs']}")
    _yes_no("st_take_s", "accepts sedan", session.can_take("small"))
    _yes_no("st_take_b", "accepts SUV  ", session.can_take("big"))
    dpg.configure_item("store_big_btn", enabled=session.can_take("big") is not False)
    dpg.set_value("st_wait_s", f"waiting sedans   {qs['small']}")
    dpg.set_value("st_wait_b", f"waiting SUVs     {qs['big']}")
    dpg.set_value("st_wait_r", f"waiting requests {qs['retrieves']}")
    dpg.set_value("st_drop",   f"SUVs dropped     {qs['dropped_suvs']}")
    dpg.configure_item("st_drop", color=C_BAD if qs["dropped_suvs"] else DIM)
    kept = session.kept_on_lift()
    dpg.set_value("full_state",
                  f"FULL: {kept} car(s) held on lift — no free empty to "
                  f"re-stage (delivers instantly on request)" if kept else "")
    if kept:
        dpg.configure_item("full_state", color=C_WARN)
    st = session.retrieve_stats()
    dpg.set_value("st_stat_sedan", _fmt_stat("sedan", st["sedan"]))
    dpg.set_value("st_stat_suv",   _fmt_stat("SUV",   st["suv"]))
    dpg.set_value("st_stat_total", _fmt_stat("total", st["total"]))
    sv = session.serve_stats()
    dpg.set_value("st_srv_sedan", _fmt_stat("sedan", sv["sedan"]))
    dpg.set_value("st_srv_suv",   _fmt_stat("SUV",   sv["suv"]))
    dpg.set_value("st_srv_total", _fmt_stat("total", sv["total"]))
    dpg.set_value("world_hint",
                  session.setpoint_hint() if session.auto_arrivals else "")


def _update_log(session: Session) -> None:
    lines = list(session.log)[-LOG_LINES:]
    pad = LOG_LINES - len(lines)
    for i in range(LOG_LINES):
        text = lines[i - pad] if i >= pad else ""
        dpg.set_value(f"log{i}", text)
        if "✓" in text:
            color = C_OK
        elif "✗" in text or "FAILED" in text:
            color = C_BAD
        elif "RETRIEVE" in text or "RUSH-OUT" in text:
            color = C_REQ
        elif "STORE" in text:
            color = C_SEDAN if "small" in text else C_SUV
        else:
            color = DIM
        dpg.configure_item(f"log{i}", color=color)


def _relayout(session: Session, ui: dict) -> None:
    """Rebuild geometry + resize the drawlist when the host size, zoom, or
    facility changes — keeps the canvas responsive to window resizes."""
    host = dpg.get_item_rect_size("canvas_host")
    if not host or host[0] < 60 or host[1] < 60:
        return
    host_w = host[0] - 16
    host_h = host[1] - 18                       # leave room for the h-scrollbar
    key = (int(host_w), int(host_h), round(ui["zoom"], 3), session.facility_name)
    if key == ui["geom_key"]:
        return
    geom = build_geometry(session.topology, host_w, host_h, ui["zoom"])
    ui["geom"] = geom
    ui["geom_key"] = key
    dpg.configure_item("canvas", width=geom.width, height=geom.height)


def _item_glyph(contents: str, cx: float, cy: float, h: float) -> None:
    """Letter cue centered in an item rectangle: S = sedan, X = SUV.
    Dark green vs dark red is the classic red-green confusion pair — the
    glyph disambiguates for everyone. Skipped when the box is too small."""
    letter = GLYPH.get(contents)
    if letter is None or h < 13.0:
        return
    size = min(20.0, h - 3.0)
    dpg.draw_text((cx - size * 0.30, cy - size * 0.55), letter,
                  size=size, color=GLYPH_COL, parent="canvas")


def _lerp_color(a: tuple, b: tuple, k: float) -> tuple:
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3)) + (255,)


def _pulse_color(playing: bool) -> tuple:
    """Working-carrier fill: pulses between LO and HI while playing;
    steady midpoint when paused (still reads as 'working', just frozen)."""
    if playing:
        k = 0.5 + 0.5 * math.sin(time.perf_counter() * 2.0 * math.pi * 1.3)
    else:
        k = 0.5
    return _lerp_color(CARRIER_WORK_LO, CARRIER_WORK_HI, k)


def _activity(ui: dict, cid: str, busy: bool, dt: float) -> float:
    """Per-carrier work envelope, 0..1, smoothed over WALL time (`dt` =
    frame delta, computed once per redraw): rises fast when a command
    starts, decays over ~0.6 s after it ends. At high playback speed the
    solver runs many short moves back-to-back (the groom does exactly one
    at a time) — coloring raw busy/idle makes the sprite strobe every few
    frames; the envelope fades instead."""
    acts = ui.setdefault("_act", {})
    a = acts.get(cid, 0.0)
    if busy:
        a = min(1.0, a + dt / 0.15)      # attack: ~0.15 s to full
    else:
        a = max(0.0, a - dt / 0.60)      # release: ~0.6 s to idle
    acts[cid] = a
    return a


def _redraw(session: Session, ui: dict) -> None:
    from oos.sim.facility import _ServeInteraction

    geom = ui["geom"]
    if geom is None:
        return
    dpg.delete_item("canvas", children_only=True)
    dpg.draw_rectangle((0, 0), (geom.width, geom.height), fill=BG, color=BG, parent="canvas")

    st = session.state
    topo = session.topology
    t = session.sim_time
    requested = session.pending_retrieve_ids()
    relocating = session.pending_relocation_ids()
    place_pid = ui.get("place_pid")
    querying = session.querying_carrier

    # element sizing from the lane band (V3.1 visual pass: ~1.7× the old
    # sizes on roomy layouts; DENSE layouts — many lanes in the same
    # window — fall back to the old compact metrics rather than letting
    # neighboring lane bands overlap)
    half = geom.lane_pitch / 2.0
    car_h = max(12.0, min(30.0, geom.lane_pitch * 0.30))
    car_w = max(26.0, min(54.0, car_h * 1.8))
    track_gap = max(9.0, car_h / 2 + 4.0)
    pad, gap = 3.0, 2.0
    maxcap = max((s.capacity for s in geom.shelves), default=1)
    raw = (half - track_gap - 2 * pad - (maxcap - 1) * gap) / max(1, maxcap)
    if raw >= 6.0:
        cell_h = min(22.0, raw)
        cell_w = 34.0
    else:                                   # compact fallback
        car_h = max(10.0, min(18.0, geom.lane_pitch * 0.24))
        car_w = max(22.0, min(34.0, car_h * 1.7))
        track_gap, pad, gap = 9.0, 2.0, 1.0
        cell_h = max(5.0, min(13.0, (half - track_gap - 2 * pad
                                     - (maxcap - 1) * gap) / max(1, maxcap)))
        cell_w = 24.0
    pallet_rects: list = []
    shelf_rects: list = []

    # lanes + labels
    for cid, lane in geom.lanes.items():
        dpg.draw_line((lane.x0, lane.y), (lane.x1, lane.y), color=TRACK, thickness=2, parent="canvas")
        dpg.draw_text((6, lane.y - 11), cid, size=18, color=TEXT, parent="canvas")

    # handoff poses: a dot on each partner lane marking the transfer spot
    # (drawn under the carrier sprites; the pose is otherwise invisible)
    for hm in geom.handoffs:
        for x, y in ((hm.ax, hm.ay), (hm.bx, hm.by)):
            dpg.draw_circle((x, y), 2.5, fill=HANDOFF_DOT, color=HANDOFF_DOT,
                            parent="canvas")

    # shelves: dark background frame + role-colored outline (3px) +
    # borderless pallets; EV charger shelves get a yellowish SECOND border
    for sb in geom.shelves:
        stack = st.shelves[sb.sid].stack
        depth = len(stack)
        outline = SHELF_TRANS if sb.is_transfer else (
            SHELF_BIG if sb.size_class == "big" else SHELF_SMALL)
        frame_h = sb.capacity * cell_h + (sb.capacity - 1) * gap + 2 * pad
        fw = cell_w + 2 * pad
        fx0, fx1 = sb.x - fw / 2, sb.x + fw / 2
        if sb.up:
            fy1 = sb.y - track_gap
            fy0 = fy1 - frame_h
        else:
            fy0 = sb.y + track_gap
            fy1 = fy0 + frame_h
        dpg.draw_rectangle((fx0, fy0), (fx1, fy1), fill=SHELF_BGCOL, color=outline,
                           thickness=3, parent="canvas")
        shelf_rects.append((fx0, min(fy0, fy1), fx1, max(fy0, fy1), sb.sid))
        if getattr(topo.shelves[sb.sid], "is_ev", False):
            dpg.draw_rectangle((fx0 - 4, fy0 - 4), (fx1 + 4, fy1 + 4),
                               color=SHELF_EV, thickness=2, parent="canvas")
        for slot_i in range(sb.capacity):
            if sb.up:
                py1 = fy1 - pad - slot_i * (cell_h + gap)
                py0 = py1 - cell_h
            else:
                py0 = fy0 + pad + slot_i * (cell_h + gap)
                py1 = py0 + cell_h
            px0, px1 = sb.x - cell_w / 2, sb.x + cell_w / 2
            if slot_i < depth:
                p = stack[-(slot_i + 1)]
                col = PALLET.get(p.contents, PALLET["empty"])
                dpg.draw_rectangle((px0, py0), (px1, py1), fill=col, color=col, parent="canvas")
                _item_glyph(p.contents, sb.x, (py0 + py1) / 2, cell_h)
                if p.id in requested:
                    dpg.draw_rectangle((px0, py0), (px1, py1), color=REQUESTED,
                                       thickness=3, parent="canvas")
                elif p.id in relocating:
                    # Orange, not yellow: a service RELOCATION (evict/
                    # place), not a customer request.
                    dpg.draw_rectangle((px0, py0), (px1, py1),
                                       color=RELOCATING, thickness=3,
                                       parent="canvas")
                if p.id == place_pid:
                    # Armed for PLACE (P key): pulsing-thick orange until a
                    # destination shelf is clicked.
                    dpg.draw_rectangle((px0 - 2, py0 - 2), (px1 + 2, py1 + 2),
                                       color=RELOCATING, thickness=4,
                                       parent="canvas")
                pallet_rects.append((px0, py0, px1, py1, p.id))

    # rooms: border color = live state (green ready / red unstaged /
    # orange while a customer is entering or leaving, with a countdown)
    for rb in geom.rooms:
        lift = topo.rooms[rb.rid].served_by
        cs = st.carriers[lift]
        docked_here = (cs.docked_at is not None
                       and cs.docked_at.kind == "room"
                       and cs.docked_at.id == rb.rid)
        serving = docked_here and isinstance(cs.current_command,
                                             _ServeInteraction)
        staged = (docked_here and not cs.is_busy
                  and cs.load is not None and cs.load.is_empty)
        edge = (ROOM_SERVING if serving
                else ROOM_STAGED if staged else ROOM_UNSTAGED)
        # The room box must always EXCEED the carrier sprite, or a docked
        # lift occludes the left/right borders (the state color).
        rw = max(24.0, car_w / 2 + 7.0)
        rh = max(18.0, car_h / 2 + 7.0)
        dpg.draw_rectangle((rb.x - rw, rb.y - rh), (rb.x + rw, rb.y + rh),
                           fill=ROOM_FILL, color=edge, thickness=3, parent="canvas")
        dpg.draw_text((rb.x - rw + 4, rb.y - rh + 1), rb.rid, size=16,
                      color=edge, parent="canvas")
        if serving and cs.busy_until is not None:
            remain = max(0.0, cs.busy_until - t)
            dpg.draw_text((rb.x - 13, rb.y + rh - 17), f"{remain:.0f}s",
                          size=15, color=(255, 255, 255, 255), parent="canvas")

    # carriers (placed from the sim's own motion profile): steel body,
    # pulsing while working — through a smoothed activity envelope so
    # rapid busy/idle alternation (short back-to-back moves) fades
    # instead of strobing; the load is a same-ratio rectangle centered
    # inside the carrier, with the item letter on top
    work_col = _pulse_color(session.playing)
    now = time.perf_counter()
    frame_dt = min(0.25, max(0.0, now - ui.get("_act_t", now)))
    ui["_act_t"] = now
    for cid, lane in geom.lanes.items():
        cs = st.carriers[cid]
        x = geom.pos_to_x(carrier_position_at(st, topo, cid, t))
        y = lane.y
        act = _activity(ui, cid, cs.current_command is not None, frame_dt)
        body = _lerp_color(CARRIER_IDLE, work_col, act)
        edge = QUERY_HL if cid == querying else CARRIER_OUT
        dpg.draw_rectangle((x - car_w / 2, y - car_h / 2), (x + car_w / 2, y + car_h / 2),
                           fill=body, color=edge, thickness=2, parent="canvas")
        load = cs.load
        if load is not None:
            # The item nearly fills the carrier — only a thin body rim
            # remains visible (it carries the idle/working pulse color).
            col = PALLET.get(load.contents, PALLET["empty"])
            rim = 4.0
            iw, ih = car_w - 2 * rim, car_h - 2 * rim
            edge_c = (REQUESTED if load.id in requested
                      else RELOCATING if load.id in relocating
                      else CARRIER_OUT)
            dpg.draw_rectangle((x - iw / 2, y - ih / 2), (x + iw / 2, y + ih / 2),
                               fill=col, color=edge_c,
                               thickness=2 if edge_c != CARRIER_OUT else 1,
                               parent="canvas")
            _item_glyph(load.contents, x, y, ih)

    if place_pid is not None:
        dpg.draw_text((8, 6),
                      f"PLACE armed: pallet {place_pid} — click a destination "
                      f"shelf  (P again to cancel)",
                      size=16, color=RELOCATING, parent="canvas")

    if ui.get("legend", True):
        _draw_legend(geom)

    ui["pallet_rects"] = pallet_rects
    ui["shelf_rects"] = shelf_rects


def _draw_legend(geom) -> None:
    """Fixed key in the canvas's bottom-left corner: color swatches (left
    column) and the full keyboard / mouse bindings (right column)."""
    rows = [
        ("fill", PALLET["empty"], "empty pallet"),
        ("fill", PALLET["small"], "sedan  (S)"),
        ("fill", PALLET["big"], "SUV  (X)"),
        ("edge", REQUESTED, "requested car"),
        ("edge", RELOCATING, "relocating (E/P)"),
        ("fill", CARRIER_IDLE, "carrier (pulse = working)"),
        ("edge", ROOM_STAGED, "room ready"),
        ("edge", ROOM_SERVING, "customer at door"),
        ("edge", ROOM_UNSTAGED, "room not staged"),
        ("edge", SHELF_SMALL, "small shelf"),
        ("edge", SHELF_BIG, "big (SUV) shelf"),
        ("edge", SHELF_EV, "EV charger shelf"),
    ]
    # Mirrors the handler_registry bindings + the canvas mouse actions.
    keys = [
        ("Space", "play / pause"),
        ("S", "step one command"),
        ("R", "reset"),
        ("click", "request the pallet"),
        ("E", "evict hovered car"),
        ("P", "arm place, click shelf"),
        ("wheel", "scale canvas (zoom)"),
    ]
    row_h, sw = 17.0, 13.0
    col2 = 190.0                          # right-column offset from the box left
    w = 384.0
    h = max(len(rows), len(keys) + 1) * row_h + 12.0
    x0, y0 = 8.0, geom.height - h - 8.0
    dpg.draw_rectangle((x0, y0), (x0 + w, y0 + h), fill=(10, 6, 24, 215),
                       color=DIM, thickness=1, parent="canvas")
    for i, (kind, col, label) in enumerate(rows):
        ry = y0 + 7.0 + i * row_h
        if kind == "fill":
            dpg.draw_rectangle((x0 + 7, ry + 1), (x0 + 7 + sw, ry + 1 + sw),
                               fill=col, color=col, parent="canvas")
        else:
            dpg.draw_rectangle((x0 + 7, ry + 1), (x0 + 7 + sw, ry + 1 + sw),
                               color=col, thickness=2, parent="canvas")
        dpg.draw_text((x0 + 26, ry), label, size=13, color=TEXT,
                      parent="canvas")
    # right column: keyboard / mouse bindings (key = yellow, action = text)
    kx = x0 + col2
    dpg.draw_text((kx, y0 + 7.0), "KEYS", size=13, color=HEAD, parent="canvas")
    for i, (key, action) in enumerate(keys):
        ry = y0 + 7.0 + (i + 1) * row_h
        dpg.draw_text((kx, ry), key, size=13, color=HEAD, parent="canvas")
        dpg.draw_text((kx + 48, ry), action, size=13, color=TEXT,
                      parent="canvas")
