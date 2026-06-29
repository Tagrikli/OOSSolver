"""DearPyGui app — plain renderer + controls over a Session.

The view is pure glue: every frame it ticks the Session (advancing playback)
and redraws the canvas from sim truth at the current time. It mutates the sim
ONLY through the Session's World/Playback methods. Carrier sprites are placed
via `carrier_position_at`, so what you see is exactly what the sim computed.

Layout: a scrollable canvas on the left (resizes with the window; mouse-wheel
scales the system horizontally) and a fixed control panel on the right.

Run:  python -m oos.viz [facility] [--runs DIR]
"""

from __future__ import annotations

import time

import dearpygui.dearpygui as dpg

from oos.facilities import FACILITIES
from oos.sim.actions import carrier_position_at
from oos.viz.geometry import build_geometry
from oos.viz.session import Session
from oos.viz.state_store import ViewState, load_view_state, save_view_state

# "Night City" palette, carried over from the old viz (RGBA 0-255).
BG          = (5, 3, 16, 255)        # BASE_BLACK
TRACK       = (13, 74, 94, 255)      # CYAN_DIM
TEXT        = (160, 168, 200, 255)   # LAVENDER
DIM         = (90, 74, 120, 255)     # BASE_MUTED
HEAD        = (252, 238, 12, 255)    # YELLOW_BRIGHT

SHELF_BGCOL = (10, 6, 24, 255)       # dark shelf background (frame fill)
SHELF_SMALL = (5, 169, 196, 255)     # CYAN_MID  — small/general shelf outline
SHELF_BIG   = (255, 42, 109, 255)    # MAGENTA   — big shelf outline
SHELF_TRANS = (224, 192, 32, 255)    # YELLOW    — transfer shelf outline
PALLET = {
    "empty": (46, 48, 62, 255),      # PALLET_EMPTY
    "small": (5, 217, 232, 255),     # CYAN_BRIGHT
    "big":   (255, 42, 109, 255),    # MAGENTA_BRIGHT
}
REQUESTED   = (252, 238, 12, 255)    # YELLOW_BRIGHT — pending-retrieve highlight
ROOM_FILL   = (30, 24, 56, 255)      # BASE_SURFACE
ROOM_EDGE   = (204, 255, 0, 255)     # LIME_BRIGHT
CARRIER_IDLE = (5, 217, 232, 255)    # CYAN_BRIGHT
CARRIER_BUSY = (255, 42, 109, 255)   # MAGENTA_BRIGHT
CARRIER_OUT = (5, 3, 16, 255)        # BASE_BLACK
QUERY_HL    = (252, 238, 12, 255)    # querying-carrier outline

PANEL_W = 360


def run_app(facility: str | None = None, runs_dir: str = "runs") -> None:
    vs = load_view_state(runs_dir)
    facility = facility or vs.facility
    if facility not in FACILITIES:
        facility = "tiny_medipol"
    session = Session(facility, runs_dir=runs_dir)
    session.set_speed(vs.speed)
    ui: dict = {
        "geom": None,
        "geom_key": None,
        "zoom": vs.zoom,
        "pallet_rects": [],     # (x0, y0, x1, y1, pallet_id) from last redraw
        "ckpt": {},             # display_name -> CheckpointEntry
        "policy_path": vs.policy_path,
    }

    dpg.create_context()

    def _save():
        """Persist the current picks to runs/.viz_state.json."""
        save_view_state(runs_dir, ViewState(
            facility=session.facility_name,
            policy_path=ui["policy_path"],
            deterministic=dpg.get_value("deterministic"),
            speed=session.speed,
            auto_arrivals=session.auto_arrivals,
            store_rate=dpg.get_value("store_rate"),
            big_prob=dpg.get_value("big_prob"),
            fullness=dpg.get_value("fullness"),
            zoom=ui["zoom"],
        ))

    # ---- callbacks ----------------------------------------------------
    def on_facility(_s, name, _u):
        session.swap_facility(name)
        ui["policy_path"] = ""       # swap drops to random (brain was topo-sized)
        ui["geom_key"] = None        # force a geometry rebuild for the new topo
        _refresh_policy_combo()
        _sync_play_label()
        _save()

    def on_policy(_s, display, _u):
        entry = ui["ckpt"].get(display)
        if entry is not None:
            session.load_policy(entry, dpg.get_value("deterministic"))
            ui["policy_path"] = entry.path
            _save()

    def on_deterministic(_s, _val, _u):
        entry = ui["ckpt"].get(dpg.get_value("policy_combo"))
        if entry is not None:
            session.load_policy(entry, dpg.get_value("deterministic"))
        _save()

    def on_play(*_):
        session.toggle_play(); _sync_play_label()

    def on_step(*_):
        session.step_once(); _sync_play_label()

    def on_reset(*_):
        session.reset(); _sync_play_label()

    def on_speed(_s, val, _u): session.set_speed(val)
    def on_store_small(*_): session.enqueue_store("small")
    def on_store_big(*_):   session.enqueue_store("big")
    def on_clear(*_):       session.clear_queue()

    def on_auto(_s, val, _u):
        session.set_auto_arrivals(val); _save()

    def on_store_rate(_s, val, _u):
        if dpg.get_value("auto"):
            session.set_store_rate(val, big_prob=dpg.get_value("big_prob"))
            _sync_play_label()
            _save()

    def on_reroll(*_):
        session.reroll_layout(fullness=dpg.get_value("fullness")); _save()

    def on_wheel(_s, delta, _u):
        if not (dpg.is_item_hovered("canvas") or dpg.is_item_hovered("canvas_host")):
            return
        z = ui["zoom"] * (1.12 if delta > 0 else 1.0 / 1.12)
        ui["zoom"] = max(0.4, min(8.0, z))

    def on_canvas_click(*_):
        if not dpg.is_item_hovered("canvas"):
            return
        mx, my = dpg.get_drawing_mouse_pos()
        for (x0, y0, x1, y1, pid) in ui["pallet_rects"]:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                session.request_retrieve(pid)
                return

    def _sync_play_label():
        dpg.set_item_label("playbtn", "❚❚ Pause" if session.playing else "▶ Play")

    def _refresh_policy_combo():
        ui["ckpt"] = {e.display_name: e for e in session.checkpoints()}
        names = list(ui["ckpt"].keys())
        # Restore the saved brain if its checkpoint still exists.
        sel = names[0]
        for e in ui["ckpt"].values():
            if e.path and e.path == ui["policy_path"]:
                sel = e.display_name
                break
        dpg.configure_item("policy_combo", items=names, default_value=sel)
        entry = ui["ckpt"].get(sel)
        if entry is not None and entry.path:
            session.load_policy(entry, dpg.get_value("deterministic"))
            ui["policy_path"] = entry.path
        else:
            ui["policy_path"] = ""

    # ---- window: scrollable canvas (left) + fixed panel (right) --------
    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="canvas_host", width=-(PANEL_W + 8), height=-1,
                                  horizontal_scrollbar=True):
                dpg.add_drawlist(width=900, height=600, tag="canvas")
            with dpg.child_window(width=PANEL_W, height=-1, tag="controls"):
                dpg.add_text("FACILITY", color=HEAD)
                dpg.add_combo(sorted(FACILITIES), default_value=facility,
                              callback=on_facility, width=-1, tag="facility_combo")
                dpg.add_text("BRAIN", color=HEAD)
                dpg.add_combo([], callback=on_policy, width=-1, tag="policy_combo")
                dpg.add_checkbox(label="deterministic (argmax)", default_value=vs.deterministic,
                                 callback=on_deterministic, tag="deterministic")

                dpg.add_separator()
                dpg.add_text("PLAYBACK", color=HEAD)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="▶ Play", callback=on_play, tag="playbtn", width=92)
                    dpg.add_button(label="⏭ Step", callback=on_step, width=80)
                    dpg.add_button(label="↻ Reset", callback=on_reset, width=80)
                dpg.add_slider_float(label="speed", default_value=vs.speed, min_value=0.0,
                                     max_value=8.0, callback=on_speed, width=-60)

                dpg.add_separator()
                dpg.add_text("WORLD  (you are the customer)", color=HEAD)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="+ Store small", callback=on_store_small, width=120)
                    dpg.add_button(label="+ Store big", callback=on_store_big, width=110)
                dpg.add_button(label="Clear queue", callback=on_clear, width=-1)
                dpg.add_text("click a pallet on the canvas → request it", color=DIM)
                dpg.add_checkbox(label="auto-world (Poisson stream)", default_value=vs.auto_arrivals,
                                 callback=on_auto, tag="auto")
                dpg.add_slider_float(label="store rate", default_value=vs.store_rate, min_value=0.0,
                                     max_value=0.5, callback=on_store_rate, width=-70, tag="store_rate")
                dpg.add_slider_float(label="big prob", default_value=vs.big_prob, min_value=0.0,
                                     max_value=1.0, width=-70, tag="big_prob")

                dpg.add_separator()
                dpg.add_text("LAYOUT", color=HEAD)
                dpg.add_slider_float(label="fullness", default_value=vs.fullness, min_value=0.0,
                                     max_value=1.0, width=-70, tag="fullness")
                dpg.add_button(label="Re-roll layout", callback=on_reroll, width=-1)
                dpg.add_text("mouse-wheel over canvas → scale horizontally", color=DIM)

                dpg.add_separator()
                dpg.add_text("", tag="status", wrap=PANEL_W - 20)
                dpg.add_separator()
                dpg.add_text("EVENT LOG", color=HEAD)
                with dpg.child_window(height=-1, tag="logbox"):
                    dpg.add_text("", tag="log", wrap=PANEL_W - 30)

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_canvas_click)
        dpg.add_mouse_wheel_handler(callback=on_wheel)
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=on_play)
        dpg.add_key_press_handler(dpg.mvKey_S, callback=on_step)
        dpg.add_key_press_handler(dpg.mvKey_R, callback=on_reset)

    _refresh_policy_combo()
    _sync_play_label()
    if vs.auto_arrivals:                 # restore the saved auto-world stream
        session.set_auto_arrivals(True)
        session.set_store_rate(vs.store_rate, vs.big_prob)

    dpg.create_viewport(title=f"OOSKiller — {facility}", width=1320, height=760)
    dpg.setup_dearpygui()
    dpg.set_primary_window("root", True)
    dpg.show_viewport()

    last = time.perf_counter()
    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = now - last
        last = now
        session.tick(dt)
        _relayout(session, ui)
        _redraw(session, ui)
        dpg.set_value("status", session.status_line())
        dpg.set_value("log", "\n".join(list(session.log)[-14:]))
        dpg.render_dearpygui_frame()

    _save()                              # persist final speed / zoom / world knobs
    dpg.destroy_context()


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


def _redraw(session: Session, ui: dict) -> None:
    geom = ui["geom"]
    if geom is None:
        return
    dpg.delete_item("canvas", children_only=True)
    dpg.draw_rectangle((0, 0), (geom.width, geom.height), fill=BG, color=BG, parent="canvas")

    st = session.state
    topo = session.topology
    t = session.sim_time
    requested = session.pending_retrieve_ids()
    querying = session.querying_carrier

    # element sizing from the lane band
    half = geom.lane_pitch / 2.0
    track_gap = 9.0
    pad, gap = 2.0, 1.0
    maxcap = max((s.capacity for s in geom.shelves), default=1)
    cell_h = max(5.0, min(13.0, (half - track_gap - 2 * pad - (maxcap - 1) * gap) / max(1, maxcap)))
    cell_w = 20.0
    car_h = max(10.0, min(18.0, geom.lane_pitch * 0.24))
    car_w = max(22.0, min(34.0, car_h * 1.7))
    pallet_rects: list = []

    # lanes + labels
    for cid, lane in geom.lanes.items():
        dpg.draw_line((lane.x0, lane.y), (lane.x1, lane.y), color=TRACK, thickness=2, parent="canvas")
        dpg.draw_text((6, lane.y - 9), cid, size=15, color=TEXT, parent="canvas")

    # shelves: dark background frame + role-colored outline + borderless pallets
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
                           thickness=1, parent="canvas")
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
                if p.id in requested:
                    dpg.draw_rectangle((px0, py0), (px1, py1), color=REQUESTED,
                                       thickness=2, parent="canvas")
                pallet_rects.append((px0, py0, px1, py1, p.id))

    # rooms
    for rb in geom.rooms:
        dpg.draw_rectangle((rb.x - 17, rb.y - 13), (rb.x + 17, rb.y + 13),
                           fill=ROOM_FILL, color=ROOM_EDGE, thickness=2, parent="canvas")
        dpg.draw_text((rb.x - 11, rb.y - 8), rb.rid, size=13, color=ROOM_EDGE, parent="canvas")

    # carriers (placed from the sim's own motion profile; load drawn INSIDE)
    for cid, lane in geom.lanes.items():
        cs = st.carriers[cid]
        x = geom.pos_to_x(carrier_position_at(st, topo, cid, t))
        y = lane.y
        body = CARRIER_BUSY if cs.current_command is not None else CARRIER_IDLE
        edge = QUERY_HL if cid == querying else CARRIER_OUT
        dpg.draw_rectangle((x - car_w / 2, y - car_h / 2), (x + car_w / 2, y + car_h / 2),
                           fill=body, color=edge, thickness=2, parent="canvas")
        load = cs.load
        if load is not None:
            col = PALLET.get(load.contents, PALLET["empty"])
            lw = 8.0
            lx1 = x + car_w / 2 - 3
            dpg.draw_rectangle((lx1 - lw, y - car_h / 2 + 3), (lx1, y + car_h / 2 - 3),
                               fill=col, color=CARRIER_OUT, thickness=1, parent="canvas")

    ui["pallet_rects"] = pallet_rects
