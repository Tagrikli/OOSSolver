"""Pygame app: main loop, event dispatch, mode toggling.

The heavy lifting is delegated:
- `SimDriver`         — env stepping + action submission + toast emission
- `policy_swap.py`    — load/rewrap policy, swap facility, random layout
- `manual_controls.py`— queue buttons + pallet/shelf editing handlers
- `Renderer`          — per-frame drawing
- `PolicyPickerWidget`/`FacilityPickerWidget` — modal pickers
- `ToastManager`      — toast lifecycle
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import pygame

from oos.env.env import OOSEnv
from oos.viz.components import ToastManager
from oos.viz.layout import LayoutConfig, compute_layout
from oos.viz.manual_controls import (
    handle_pallet_click,
    handle_queue_button,
    pop_shelf_top,
    push_empty_pallet,
    set_pallet_contents,
)
from oos.viz.pickers import FacilityPickerWidget, PolicyPickerWidget
from oos.viz.player import Player, PolicyFn, random_policy
from oos.viz.policy_swap import (
    generate_single_task,
    load_policy,
    rewrap_with_mcts,
    swap_facility,
)
from oos.viz.renderer import RenderState, Renderer
from oos.viz.sim_driver import SimDriver
from oos.viz.state_store import load_viz_state, save_viz_state


@dataclass
class VizApp:
    env: OOSEnv
    policy: PolicyFn
    seed: int = 0
    window_w: int = 1920
    window_h: int = 1080
    target_fps: int = 60
    initial_speed: float = 1.0
    runs_dir: str = "runs"
    facility_name: str = "dev"

    def run(self) -> None:
        if os.environ.get("SDL_VIDEODRIVER") is None and not os.environ.get("DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.display.set_caption("OOSKILLER ▮ NIGHT CITY OPS")
        # Resizable so the WM (or the user dragging the edge) can rescale.
        # Tiling WMs that auto-tile this window need a floating rule on
        # their side; pygame just says "I can be resized."
        surface = pygame.display.set_mode(
            (self.window_w, self.window_h), pygame.RESIZABLE,
        )
        clock = pygame.time.Clock()

        player = Player(env=self.env, policy=self.policy, seed=self.seed)
        player.reset()
        active_policy_label = "(random policy)"

        facility = player.env._ctx.facility  # type: ignore[attr-defined]
        # Viz boots in manual mode so customer arrivals don't kick off until
        # the user opts in (press 'm' to resume auto arrivals).
        facility.set_auto_arrivals(False)
        topo = facility.topology

        # Restore last-saved canvas zoom (Shift+wheel adjusts it during play)
        # and animation speed (+/- keys).
        _persisted_state = load_viz_state(self.runs_dir)
        zoom = _persisted_state.zoom

        layout = compute_layout(
            topo,
            LayoutConfig(window_w=self.window_w, window_h=self.window_h,
                         zoom=zoom),
        )
        renderer = Renderer(layout, topo)

        picker = PolicyPickerWidget(runs_dir=self.runs_dir)
        facility_picker = FacilityPickerWidget(active=self.facility_name)

        mode = "anim"
        paused = True
        speed = _persisted_state.speed
        anim_time = facility.state.time
        wall_start = time.monotonic()

        def wall_now() -> float:
            return time.monotonic() - wall_start

        toasts = ToastManager(wall_now)
        driver = SimDriver(player, toasts)

        toasts.info(
            f"FACILITY ONLINE  {len(topo.carriers)}C/{len(topo.shelves)}S/{len(topo.rooms)}R",
            lifetime=4.0,
        )

        # Auto-load the last persisted policy if its checkpoint still exists.
        _persisted = load_viz_state(self.runs_dir)
        if _persisted.policy_path:
            _matching = next(
                (e for e in picker.entries if e.path == _persisted.policy_path),
                None,
            )
            if _matching is not None:
                _label = load_policy(
                    _matching, player, topo, picker.deterministic, toasts,
                    mcts_enabled=picker.mcts_enabled,
                    mcts_n_sims=picker.mcts_n_sims,
                )
                if _label is not None:
                    active_policy_label = _label
        save_viz_state(self.runs_dir, facility_name=self.facility_name)

        # Generate-button trigger: stash params here, the main loop applies
        # them (must run outside the event handler so the local `facility`
        # / `anim_time` bindings can be reseated after the env swap).
        pending_generate: list[dict] = []

        # Snapshot the boot env so R can revert from a SingleTaskEnv back
        # to the original OOSEnv on the current facility. Updated on
        # facility swap so R always points at the most recent "real" env
        # rather than a Generate-spawned SingleTaskEnv.
        original_env_box: list = [player.env]

        def fire_generate(params: dict) -> None:
            if "_error" in params:
                toasts.error(params["_error"])
                return
            pending_generate.append(params)

        def wire_renderer(r: Renderer) -> Renderer:
            r.randomize_content.on_generate = fire_generate
            return r

        def relayout(new_w: int, new_h: int) -> Renderer:
            """Recompute layout for a new window size and build a fresh
            Renderer. Sidebar width stays fixed; canvas absorbs the rest.
            Carries the current zoom so the track length stays put."""
            self.window_w = new_w
            self.window_h = new_h
            new_layout = compute_layout(
                facility.topology,
                LayoutConfig(window_w=new_w, window_h=new_h, zoom=zoom),
            )
            return wire_renderer(Renderer(new_layout, facility.topology))

        renderer = wire_renderer(renderer)

        running = True
        dragging_fullness = False
        while running:
            dt_wall = clock.tick(self.target_fps) / 1000.0

            # Apply any pending Generate before processing events. Done
            # here (not inside the event handler) because the env swap
            # needs to reseat the local `facility`/`anim_time` bindings.
            while pending_generate:
                params = pending_generate.pop(0)
                ok = generate_single_task(
                    params, player, facility, self.facility_name, toasts,
                )
                if ok:
                    facility = player.env._ctx.facility  # type: ignore[attr-defined]
                    anim_time = facility.state.time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue

                if event.type == pygame.VIDEORESIZE:
                    # RESIZABLE already resizes the backing surface; just
                    # rebuild the layout. Calling set_mode again here can
                    # cascade into more resize events on some WMs.
                    renderer = relayout(event.w, event.h)
                    continue

                if event.type == pygame.MOUSEWHEEL:
                    if picker.open:
                        picker.handle_wheel(event.y)
                    elif facility_picker.open:
                        facility_picker.handle_wheel(event.y)
                    else:
                        mouse_pos = pygame.mouse.get_pos()
                        shift_held = bool(
                            pygame.key.get_mods() & pygame.KMOD_SHIFT
                        )
                        # Shift+wheel over the canvas zooms the px-per-mm
                        # mapping of all carrier tracks. Left endpoint of
                        # each track stays anchored — only the right end
                        # extends/contracts. The setting persists.
                        if (
                            shift_held
                            and renderer.carrier_area_rect.collidepoint(mouse_pos)
                        ):
                            zoom_step = 1.15 ** event.y    # ~15% per notch
                            new_zoom = max(0.1, min(8.0, zoom * zoom_step))
                            if abs(new_zoom - zoom) > 1e-6:
                                zoom = new_zoom
                                renderer = relayout(self.window_w, self.window_h)
                                save_viz_state(self.runs_dir, zoom=zoom)
                                toasts.info(f"ZOOM {zoom:.2f}×", lifetime=1.5)
                            continue
                        # Pixel-granular scroll for the randomize panel,
                        # row-granular for the rest.
                        scroll_specs = (
                            [(renderer.randomize_panel, 24)]
                            if renderer.active_tab == 1
                            else [
                                # Row-granular panels first, then pixel-
                                # granular (`dist_panel` needs a much
                                # bigger wheel multiplier).
                                (renderer.queue_panel,    3),
                                (renderer.controls_panel, 1),
                                (renderer.stats_panel,    1),
                                (renderer.legend_panel,   1),
                                (renderer.dist_panel,     24),
                            ]
                        )
                        handled_by_sidebar = False
                        for panel, mult in scroll_specs:
                            if panel.hit_test(mouse_pos):
                                panel.scroll(-event.y * mult)
                                handled_by_sidebar = True
                                break
                        # Fall through to the carrier area: wheel anywhere
                        # over the canvas scrolls the stacked carriers.
                        if not handled_by_sidebar and renderer.carrier_area_rect.collidepoint(mouse_pos):
                            renderer.scroll_carriers(-event.y * 40)
                    continue

                if event.type == pygame.MOUSEMOTION:
                    if dragging_fullness:
                        renderer.queue_content.set_fullness_from_x(event.pos[0])
                        continue
                    if renderer.active_tab == 1:
                        renderer.randomize_content.handle_mouse_motion(event.pos)
                    continue

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if dragging_fullness:
                        dragging_fullness = False
                        continue
                    if renderer.active_tab == 1:
                        renderer.randomize_content.handle_mouse_up(event.pos)
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    # Tab strip click switches the sidebar's active tab.
                    new_tab = renderer.tab_strip.hit_test(pos)
                    if new_tab is not None:
                        if new_tab != renderer.tab_strip.active:
                            renderer.tab_strip.active = new_tab
                            renderer.randomize_content.blur()
                        continue
                    # Randomize tab: panel-internal clicks (numeric-field
                    # drag start, checkbox toggle, Generate button).
                    if renderer.active_tab == 1 and renderer.randomize_panel.hit_test(pos):
                        renderer.randomize_content.handle_mouse_down(pos)
                        continue
                    # Header clicks toggle panel collapse — UI-only.
                    if self._handle_panel_collapse(renderer, pos):
                        continue
                    # The rest is manual-mode only.
                    if facility.auto_arrivals_enabled:
                        continue
                    # Slider drag start.
                    if renderer.queue_content.hit_slider(pos):
                        renderer.queue_content.set_fullness_from_x(pos[0])
                        dragging_fullness = True
                        continue
                    # Queue panel buttons.
                    btn = renderer.queue_content.hit_button(pos)
                    if btn is not None:
                        handle_queue_button(
                            btn, facility, player,
                            renderer.queue_content.fullness, toasts,
                        )
                        if btn == "randomize":
                            anim_time = facility.state.time
                        continue
                    # Pallet click → toggle Retrieve.
                    for rect, pallet_id in renderer.pallet_hit_areas:
                        if rect.collidepoint(pos):
                            handle_pallet_click(pallet_id, facility, player, toasts)
                            break
                    continue

                if event.type == pygame.KEYDOWN:
                    # Focused randomize-panel text field grabs keys before
                    # app shortcuts so typing "q" doesn't quit the viz.
                    if (
                        renderer.active_tab == 1
                        and renderer.randomize_content.focused()
                    ):
                        if renderer.randomize_content.handle_key(event):
                            continue
                    # Pickers consume keys first while open.
                    if facility_picker.open:
                        action = facility_picker.handle_key(event)
                        if action == "submit":
                            name = facility_picker.selected()
                            if name is not None and name != facility_picker.active:
                                renderer = wire_renderer(swap_facility(
                                    name, player, facility, self.runs_dir,
                                    self.window_w, self.window_h, toasts,
                                ))
                                # New facility → new "original" env baseline
                                # for the R-revert path.
                                original_env_box[0] = player.env
                                facility = player.env._ctx.facility  # type: ignore[attr-defined]
                                topo = facility.topology
                                anim_time = facility.state.time
                                facility_picker.active = name
                                self.facility_name = name
                                active_policy_label = "(random policy)"
                            facility_picker.close()
                        if action is not None:
                            continue

                    if picker.open:
                        action = picker.handle_key(event)
                        if action == "mcts_toggle":
                            rewrap_with_mcts(
                                player, picker.mcts_enabled, picker.mcts_n_sims,
                                toasts,
                            )
                        elif action == "submit":
                            entry = picker.selected()
                            if entry is not None:
                                label = load_policy(
                                    entry, player, topo, picker.deterministic, toasts,
                                    mcts_enabled=picker.mcts_enabled,
                                    mcts_n_sims=picker.mcts_n_sims,
                                )
                                if label is not None:
                                    active_policy_label = label
                                    save_viz_state(
                                        self.runs_dir, policy_path=entry.path,
                                    )
                            picker.close()
                        if action is not None:
                            continue

                    # Top-level sim controls.
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_p:
                        picker.toggle()
                    elif event.key == pygame.K_f:
                        facility_picker.toggle()
                    elif event.key == pygame.K_n:
                        mode = "step" if mode == "anim" else "anim"
                        if mode == "anim":
                            anim_time = facility.state.time
                        toasts.warn(f"MODE → {mode.upper()}", lifetime=2.0)
                    elif event.key == pygame.K_m:
                        facility.set_auto_arrivals(not facility.auto_arrivals_enabled)
                        if facility.auto_arrivals_enabled:
                            toasts.success("MANUAL MODE OFF (auto arrivals resumed)", lifetime=3.0)
                        else:
                            toasts.accent("MANUAL MODE ON (auto arrivals paused)", lifetime=3.0)
                    elif event.key in (pygame.K_RIGHT, pygame.K_PERIOD):
                        if not player.done:
                            driver.step_one_decision()
                            anim_time = facility.state.time
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        speed = min(speed * 1.5, 1000.0)
                        save_viz_state(self.runs_dir, speed=speed)
                    elif event.key == pygame.K_MINUS:
                        speed = max(speed / 1.5, 0.1)
                        save_viz_state(self.runs_dir, speed=speed)
                    elif event.key == pygame.K_r:
                        preserve_auto = facility.auto_arrivals_enabled
                        # If we're sitting on a Generate-spawned SingleTaskEnv,
                        # R reverts to the original OOSEnv on this facility
                        # rather than re-rolling another single-task state.
                        from oos.learn.single_task_env import SingleTaskEnv
                        reverted = False
                        if isinstance(player.env, SingleTaskEnv):
                            player.env = original_env_box[0]
                            player.seed = self.seed
                            reverted = True
                        player.reset()
                        facility = player.env._ctx.facility  # type: ignore[attr-defined]
                        facility.set_auto_arrivals(preserve_auto)
                        anim_time = facility.state.time
                        if reverted:
                            toasts.accent(
                                "ENV RESET (reverted from generated)",
                                lifetime=3.0,
                            )
                        else:
                            toasts.accent("ENV RESET", lifetime=2.0)
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        target = {pygame.K_1: "empty", pygame.K_2: "small",
                                  pygame.K_3: "big"}[event.key]
                        pid = self._hovered_pallet_id(renderer)
                        if pid is not None:
                            set_pallet_contents(pid, target, facility, player, toasts)
                    elif event.key in (pygame.K_4, pygame.K_5):
                        sid = self._hovered_shelf_id(renderer)
                        if sid is not None:
                            if event.key == pygame.K_4:
                                pop_shelf_top(sid, facility, player, toasts)
                            else:
                                push_empty_pallet(sid, facility, player, toasts)

            # Advance the sim in anim mode.
            if mode == "anim" and not paused and not player.done:
                anim_time += dt_wall * speed
                driver.drive_anim(anim_time)

            if mode == "step":
                anim_time = facility.state.time

            toasts.tick()

            rs = self._build_render_state(
                player, mode, paused, speed, anim_time, toasts,
                active_policy_label, wall_now(),
            )
            renderer.draw(
                surface, facility, facility.queue, rs,
                manual_mode=not facility.auto_arrivals_enabled,
            )

            if player.done:
                _draw_done_banner(surface, "EPISODE TERMINATED — R: reset · Q: quit")

            picker.draw(surface, renderer.fonts, active_policy_label)
            facility_picker.draw(surface, renderer.fonts)

            pygame.display.flip()

        pygame.quit()

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _handle_panel_collapse(renderer: Renderer, pos) -> bool:
        for panel in (
            renderer.stats_panel, renderer.queue_panel, renderer.dist_panel,
            renderer.controls_panel, renderer.legend_panel,
        ):
            if panel.hit_header(pos):
                panel.toggle_collapsed()
                return True
        return False

    @staticmethod
    def _hovered_pallet_id(renderer: Renderer) -> Optional[int]:
        mouse_pos = pygame.mouse.get_pos()
        for rect, pid in renderer.pallet_hit_areas:
            if rect.collidepoint(mouse_pos):
                return pid
        return None

    @staticmethod
    def _hovered_shelf_id(renderer: Renderer) -> Optional[str]:
        mouse_pos = pygame.mouse.get_pos()
        for rect, sid in renderer.shelf_hit_areas:
            if rect.collidepoint(mouse_pos):
                return sid
        return None

    def _build_render_state(
        self, player, mode, paused, speed, anim_time, toasts,
        active_policy_label, wall_now_s,
    ) -> RenderState:
        policy_logits = getattr(player.policy, "last_logits", None)
        policy_action_mask = getattr(player.policy, "last_action_mask", None)
        policy_chosen = getattr(player.policy, "last_chosen", None)
        policy_action_entries = (
            player.info.get("action_entries", []) if player.info else []
        )
        policy_query_log = dict(player.policy_query_log)
        return RenderState(
            mode=mode + (" (paused)" if paused else ""),
            wall_speed=speed,
            last_reward=(player.last_record.reward if player.last_record else 0.0),
            n_completed=player.total_completions,
            last_action=(player.last_record.action_label if player.last_record else "—"),
            querying=(player.last_record.querying if player.last_record else "—"),
            anim_now=anim_time,
            toasts=toasts.toasts,
            wall_now=wall_now_s,
            policy_label=active_policy_label,
            facility_name=self.facility_name,
            policy_logits=policy_logits,
            policy_action_mask=policy_action_mask,
            policy_chosen=policy_chosen,
            policy_action_entries=policy_action_entries,
            policy_query_log=policy_query_log,
            mouse_pos=pygame.mouse.get_pos(),
        )


def _draw_done_banner(surface: pygame.Surface, text: str) -> None:
    from oos.viz.components import (
        BASE_BLACK,
        YELLOW_BRIGHT,
        draw_beveled_frame,
        draw_beveled_rect,
    )
    font = pygame.font.SysFont("monospace", 22, bold=True)
    s = font.render(text, True, YELLOW_BRIGHT)
    r = s.get_rect(center=(surface.get_width() // 2, 30))
    bg = r.inflate(60, 18)
    draw_beveled_rect(surface, bg, BASE_BLACK, bevel=12, alpha=235)
    draw_beveled_frame(surface, bg, YELLOW_BRIGHT, bevel=12, width=2, glow=True)
    surface.blit(s, r)


def run_app(
    env: OOSEnv,
    policy: Optional[PolicyFn] = None,
    seed: int = 0,
    facility_name: str = "dev",
) -> None:
    VizApp(env=env, policy=policy or random_policy, seed=seed,
           facility_name=facility_name).run()
