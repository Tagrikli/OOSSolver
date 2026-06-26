"""DearPyGui app — plain renderer + controls over a Session.

The view is pure glue: every frame it ticks the Session (advancing playback)
and redraws the canvas from sim truth at the current time. It mutates the sim
ONLY through the Session's World/Playback methods — it never touches a carrier
directly. Carrier sprites are placed via `carrier_position_at`, so what you see
is exactly what the sim computed.

Run:  python -m oos.viz [facility] [--runs DIR]
"""

from __future__ import annotations

import time

import dearpygui.dearpygui as dpg

from oos.facilities import FACILITIES
from oos.sim.actions import carrier_position_at
from oos.viz.geometry import build_geometry
from oos.viz.session import Session

# Plain palette (RGBA 0-255).
BG          = (32, 34, 40, 255)
TRACK       = (120, 124, 135, 255)
TEXT        = (210, 212, 220, 255)
DIM         = (140, 142, 150, 255)
CARRIER     = (96, 102, 128, 255)
CARRIER_HL  = (236, 208, 96, 255)   # querying carrier outline
ROOM        = (96, 176, 120, 255)
HANDOFF     = (168, 120, 208, 255)
REQUESTED   = (232, 92, 84, 255)    # pending-retrieve outline
OUTLINE     = (20, 22, 26, 255)
PALLET = {
    "empty": (150, 152, 160, 255),
    "small": (84, 140, 208, 255),
    "big":   (224, 156, 72, 255),
}

CW, CH = 1200, 720      # canvas size (px)
PANEL_W = 360


def run_app(facility: str = "tiny_medipol", runs_dir: str = "runs") -> None:
    session = Session(facility, runs_dir=runs_dir)
    ui: dict = {
        "geom": build_geometry(session.topology, CW, CH),
        "pallet_rects": [],     # (x0, y0, x1, y1, pallet_id) from last redraw
        "ckpt": {},             # display_name -> CheckpointEntry
    }

    dpg.create_context()

    # ---- callbacks ----------------------------------------------------
    def on_facility(_s, name, _u):
        session.swap_facility(name)
        ui["geom"] = build_geometry(session.topology, CW, CH)
        _refresh_policy_combo()
        _sync_play_label()

    def on_policy(_s, display, _u):
        entry = ui["ckpt"].get(display)
        if entry is not None:
            session.load_policy(entry, dpg.get_value("deterministic"))

    def on_deterministic(_s, _val, _u):
        display = dpg.get_value("policy_combo")
        entry = ui["ckpt"].get(display)
        if entry is not None:
            session.load_policy(entry, dpg.get_value("deterministic"))

    def on_play(*_):
        session.toggle_play()
        _sync_play_label()

    def on_step(*_):
        session.step_once()
        _sync_play_label()

    def on_reset(*_):
        session.reset()
        _sync_play_label()

    def on_speed(_s, val, _u):
        session.set_speed(val)

    def on_store_small(*_): session.enqueue_store("small")
    def on_store_big(*_):   session.enqueue_store("big")
    def on_clear(*_):       session.clear_queue()

    def on_auto(_s, val, _u):
        session.set_auto_arrivals(val)

    def on_store_rate(_s, val, _u):
        if dpg.get_value("auto"):
            session.set_store_rate(val, big_prob=dpg.get_value("big_prob"))
            _sync_play_label()

    def on_reroll(*_):
        session.reroll_layout(fullness=dpg.get_value("fullness"))

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
        entries = session.checkpoints()
        ui["ckpt"] = {e.display_name: e for e in entries}
        names = list(ui["ckpt"].keys())
        dpg.configure_item("policy_combo", items=names, default_value=names[0])

    # ---- window --------------------------------------------------------
    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            dpg.add_drawlist(width=CW, height=CH, tag="canvas")
            with dpg.child_window(width=PANEL_W, tag="controls"):
                dpg.add_text("FACILITY")
                dpg.add_combo(sorted(FACILITIES), default_value=facility,
                              callback=on_facility, width=-1, tag="facility_combo")
                dpg.add_text("BRAIN")
                dpg.add_combo([], callback=on_policy, width=-1, tag="policy_combo")
                dpg.add_checkbox(label="deterministic (argmax)",
                                 callback=on_deterministic, tag="deterministic")

                dpg.add_separator()
                dpg.add_text("PLAYBACK")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="▶ Play", callback=on_play, tag="playbtn", width=92)
                    dpg.add_button(label="⏭ Step", callback=on_step, width=80)
                    dpg.add_button(label="↻ Reset", callback=on_reset, width=80)
                dpg.add_slider_float(label="speed", default_value=1.0, min_value=0.0,
                                     max_value=8.0, callback=on_speed, width=-60)

                dpg.add_separator()
                dpg.add_text("WORLD  (you are the customer)")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="+ Store small", callback=on_store_small, width=120)
                    dpg.add_button(label="+ Store big", callback=on_store_big, width=110)
                dpg.add_button(label="Clear queue", callback=on_clear, width=-1)
                dpg.add_text("click a pallet on the canvas → request it", color=DIM)
                dpg.add_checkbox(label="auto-world (Poisson stream)",
                                 callback=on_auto, tag="auto")
                dpg.add_slider_float(label="store rate", default_value=0.05, min_value=0.0,
                                     max_value=0.5, callback=on_store_rate, width=-70, tag="store_rate")
                dpg.add_slider_float(label="big prob", default_value=0.3, min_value=0.0,
                                     max_value=1.0, width=-70, tag="big_prob")

                dpg.add_separator()
                dpg.add_text("LAYOUT")
                dpg.add_slider_float(label="fullness", default_value=0.5, min_value=0.0,
                                     max_value=1.0, width=-70, tag="fullness")
                dpg.add_button(label="Re-roll layout", callback=on_reroll, width=-1)

                dpg.add_separator()
                dpg.add_text("", tag="status", wrap=PANEL_W - 20)
                dpg.add_separator()
                dpg.add_text("EVENT LOG")
                with dpg.child_window(height=200, tag="logbox"):
                    dpg.add_text("", tag="log", wrap=PANEL_W - 30)

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_canvas_click)
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=on_play)
        dpg.add_key_press_handler(dpg.mvKey_S, callback=on_step)
        dpg.add_key_press_handler(dpg.mvKey_R, callback=on_reset)

    _refresh_policy_combo()
    _sync_play_label()

    dpg.create_viewport(title=f"OOSKiller — {facility}", width=CW + PANEL_W + 30, height=CH + 16)
    dpg.setup_dearpygui()
    dpg.set_primary_window("root", True)
    dpg.show_viewport()

    last = time.perf_counter()
    while dpg.is_dearpygui_running():
        now = time.perf_counter()
        dt = now - last
        last = now
        session.tick(dt)
        _redraw(session, ui)
        dpg.set_value("status", session.status_line())
        dpg.set_value("log", "\n".join(list(session.log)[-14:]))
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def _redraw(session: Session, ui: dict) -> None:
    geom = ui["geom"]
    dpg.delete_item("canvas", children_only=True)
    dpg.draw_rectangle((0, 0), (geom.width, geom.height), fill=BG, color=BG, parent="canvas")

    st = session.state
    topo = session.topology
    t = session.sim_time
    requested = session.pending_retrieve_ids()
    querying = session.querying_carrier

    maxcap = max((s.capacity for s in geom.shelves), default=1)
    cell_h = max(7.0, min(15.0, (geom.lane_pitch / 2 - 22) / max(1, maxcap)))
    cw2 = 11.0          # half pallet width
    gap = 8.0           # gap from track to first pallet
    pallet_rects: list = []

    # lanes + labels
    for cid, lane in geom.lanes.items():
        dpg.draw_line((lane.x0, lane.y), (lane.x1, lane.y), color=TRACK, thickness=2, parent="canvas")
        dpg.draw_text((6, lane.y - 9), cid, size=15, color=TEXT, parent="canvas")

    # handoff connectors + markers
    for h in geom.handoffs:
        dpg.draw_line((h.ax, h.ay), (h.bx, h.by), color=HANDOFF, thickness=1, parent="canvas")
        for (hx, hy) in ((h.ax, h.ay), (h.bx, h.by)):
            dpg.draw_circle((hx, hy), 4, color=HANDOFF, fill=HANDOFF, parent="canvas")

    # shelves + pallet stacks
    for sb in geom.shelves:
        ss = st.shelves[sb.sid]
        # shelf foot tick on the track
        dpg.draw_line((sb.x, sb.y - 3), (sb.x, sb.y + 3), color=DIM, thickness=1, parent="canvas")
        for d, p in enumerate(reversed(ss.stack)):     # d=0 == top (nearest track)
            if sb.up:
                y1 = sb.y - gap - d * cell_h
                y0 = y1 - cell_h
            else:
                y0 = sb.y + gap + d * cell_h
                y1 = y0 + cell_h
            x0, x1 = sb.x - cw2, sb.x + cw2
            col = PALLET.get(p.contents, PALLET["empty"])
            dpg.draw_rectangle((x0, y0), (x1, y1), fill=col, color=OUTLINE, thickness=1, parent="canvas")
            if p.id in requested:
                dpg.draw_rectangle((x0 - 1, y0 - 1), (x1 + 1, y1 + 1),
                                   color=REQUESTED, thickness=2, parent="canvas")
            pallet_rects.append((x0, y0, x1, y1, p.id))

    # rooms
    for rb in geom.rooms:
        dpg.draw_rectangle((rb.x - 16, rb.y - 13), (rb.x + 16, rb.y + 13),
                           color=ROOM, thickness=2, parent="canvas")
        dpg.draw_text((rb.x - 12, rb.y - 8), rb.rid, size=13, color=ROOM, parent="canvas")

    # carriers (placed from the sim's own motion profile)
    for cid, lane in geom.lanes.items():
        x = geom.pos_to_x(carrier_position_at(st, topo, cid, t))
        y = lane.y
        edge = CARRIER_HL if cid == querying else OUTLINE
        dpg.draw_rectangle((x - 14, y - 9), (x + 14, y + 9),
                           fill=CARRIER, color=edge, thickness=2, parent="canvas")
        load = st.carriers[cid].load
        if load is not None:
            col = PALLET.get(load.contents, PALLET["empty"])
            dpg.draw_rectangle((x - 9, y - 9 - 11), (x + 9, y - 9 - 1),
                               fill=col, color=OUTLINE, thickness=1, parent="canvas")

    ui["pallet_rects"] = pallet_rects
