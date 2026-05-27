"""Pygame app: main loop + per-event-type dispatch.

The window's mutable state lives in `_RunState` — a single dataclass
passed to each handler. The main loop is small (poll events, dispatch,
tick sim, render); per-event handlers are methods on `VizApp`.

Layers (each method touches only its layer):
  * sim mutations           — agent.facility, anim_time
  * viz / UI state          — renderer, pickers, mode, paused, speed, zoom
  * driver / toasts         — SimDriver advances the sim with toast side-effects

The heavy lifting is delegated:
  - `SimDriver`           — env stepping + action submission + toast emission
  - `policy_swap`         — load/rewrap policy, swap facility, generate
  - `manual_controls`     — queue buttons + pallet/shelf editing
  - `Renderer`            — per-frame drawing (canvas + sidebar)
  - `PolicyPickerWidget`/`FacilityPickerWidget` — modal pickers
  - `ToastManager`        — toast lifecycle
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pygame

from oos.agent import Agent, PolicyFn, random_policy
from oos.env.env import OOSEnv
from oos.facility import Facility
from oos.viz.components import ToastManager
from oos.viz.layout import LayoutConfig, compute_layout
from oos.viz.manual_controls import (
    handle_pallet_click,
    handle_queue_button,
    pop_shelf_top,
    push_empty_pallet,
    set_pallet_contents,
)
from oos.viz.pickers import (
    FacilityPickerWidget,
    PolicyPickerWidget,
    RunConfigPickerWidget,
)
from oos.viz.policy_swap import (
    generate_single_task,
    load_policy,
    load_run_config,
    rewrap_with_mcts,
    swap_facility,
)
from oos.viz.renderer import RenderState, Renderer
from oos.viz.sim_driver import SimDriver
from oos.viz.state_store import load_viz_state, save_viz_state


# ─────────────────────────────────────────────────────────────────────────
# Per-frame mutable bag
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _RunState:
    """All the mutable state the viz loop needs to thread between events.

    Grouped so every handler can read/mutate without needing closures.
    Layer hint per field:

      sim    — `agent` (owns its Facility), `anim_time`, `original_env`
      ui     — `renderer`, `picker`, `facility_picker`, `mode`, `paused`,
               `speed`, `zoom`, `dragging_fullness`, `active_policy_label`
      runtime— `surface`, `driver`, `toasts`, `running`, `pending_generate`,
               `wall_now_fn`

    Convention: read `state.agent.facility` for the live Facility — we
    don't mirror it on _RunState (the agent is its source of truth).
    """

    # Long-lived (set at boot)
    surface: pygame.Surface
    agent: Agent
    driver: SimDriver
    toasts: ToastManager
    picker: PolicyPickerWidget
    facility_picker: FacilityPickerWidget
    run_config_picker: RunConfigPickerWidget
    wall_now_fn: callable  # type: ignore[type-arg]

    # Sim
    anim_time: float
    original_env: OOSEnv

    # Viz / window
    renderer: Renderer
    zoom: float
    window_w: int
    window_h: int

    # Loop / mode
    running: bool = True
    mode: str = "anim"
    paused: bool = True
    speed: float = 1.0
    dragging_fullness: bool = False
    active_policy_label: str = "(random policy)"

    # Queues
    pending_generate: list = field(default_factory=list)

    # Training Replay tab state. `replay_cfg_*` fields are populated when
    # the user picks a config from the RunConfigPicker; episode counters
    # tick over as the agent terminates and auto-advances.
    replay_active: bool = False
    replay_cfg_name: str = "(none)"
    replay_cfg_facility: str = "?"
    replay_cfg_kind: str = "?"
    replay_n_episodes: int = 0
    replay_n_success: int = 0
    replay_last_task: str = "—"
    replay_last_return: float = 0.0

    @property
    def facility(self) -> Facility:
        """Shortcut for `self.agent.facility` — the live user-facing
        Facility. Kept as a property (not a mirror field) so env swaps
        on Agent automatically propagate."""
        return self.agent.facility


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

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        if os.environ.get("SDL_VIDEODRIVER") is None and not os.environ.get("DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.display.set_caption("OOSKILLER ▮ NIGHT CITY OPS")

        state = self._init_state()
        clock = pygame.time.Clock()

        while state.running:
            dt_wall = clock.tick(self.target_fps) / 1000.0

            self._apply_pending_generate(state)

            for event in pygame.event.get():
                self._dispatch_event(state, event)

            self._tick_sim(state, dt_wall)
            self._render(state)
            pygame.display.flip()

        pygame.quit()

    def _init_state(self) -> _RunState:
        """Build the boot-time state bag from `self.*` config."""
        surface = pygame.display.set_mode(
            (self.window_w, self.window_h), pygame.RESIZABLE,
        )

        # Wrap the gym env in a user-facing Facility, then build an Agent
        # over it. The Agent owns the policy + step loop; viz drives it.
        facility = Facility(self.env)
        agent = Agent(facility=facility, policy=self.policy, seed=self.seed)
        agent.reset()

        # Boot manual — customer arrivals are off until the user presses 'm'.
        facility.set_auto_arrivals(False)
        topo = facility.topology

        persisted = load_viz_state(self.runs_dir)
        zoom = persisted.zoom
        speed = persisted.speed

        layout = compute_layout(
            topo, LayoutConfig(
                window_w=self.window_w, window_h=self.window_h, zoom=zoom,
            ),
        )
        renderer = self._wire_renderer(
            Renderer(layout, topo, zoom=zoom), pending_generate=[],
        )

        picker = PolicyPickerWidget(runs_dir=self.runs_dir)
        facility_picker = FacilityPickerWidget(active=self.facility_name)
        run_config_picker = RunConfigPickerWidget(runs_dir=self.runs_dir)

        wall_start = time.monotonic()
        def wall_now() -> float:
            return time.monotonic() - wall_start
        toasts = ToastManager(wall_now)
        driver = SimDriver(agent, toasts)

        toasts.info(
            f"FACILITY ONLINE  {len(topo.carriers)}C/{len(topo.shelves)}S/{len(topo.rooms)}R",
            lifetime=4.0,
        )

        # Auto-load the last persisted policy if its checkpoint still exists.
        active_label = "(random policy)"
        if persisted.policy_path:
            match = next(
                (e for e in picker.entries if e.path == persisted.policy_path),
                None,
            )
            if match is not None:
                label = load_policy(
                    match, agent, topo, picker.deterministic, toasts,
                    mcts_enabled=picker.mcts_enabled,
                    mcts_n_sims=picker.mcts_n_sims,
                )
                if label is not None:
                    active_label = label
        save_viz_state(self.runs_dir, facility_name=self.facility_name)

        state = _RunState(
            surface=surface,
            agent=agent,
            driver=driver,
            toasts=toasts,
            picker=picker,
            facility_picker=facility_picker,
            run_config_picker=run_config_picker,
            wall_now_fn=wall_now,
            anim_time=facility.sim_time,
            original_env=facility.env,
            renderer=renderer,
            zoom=zoom,
            window_w=self.window_w,
            window_h=self.window_h,
            mode="anim",
            paused=True,
            speed=speed,
            active_policy_label=active_label,
        )
        # Now that state exists, re-wire the renderer's Generate callback
        # to point at state.pending_generate (vs the throwaway list above).
        # Also wire the Replay panel's buttons.
        self._wire_renderer(renderer, pending_generate=state.pending_generate,
                            toasts=toasts)
        self._wire_replay_panel(renderer, state)
        return state

    def _wire_replay_panel(self, renderer: Renderer, state: _RunState) -> None:
        """Hook ReplayContent's buttons. LOAD opens the run-config picker;
        NEW EPISODE force-resets the current replay env to sample a fresh
        scenario."""
        rc = renderer.replay_content
        rc.on_open_picker = lambda: state.run_config_picker.toggle()
        rc.on_new_episode = lambda: self._next_replay_episode(state)

    # ─────────────────────────────────────────────────────────────────────
    # Per-frame work
    # ─────────────────────────────────────────────────────────────────────

    def _apply_pending_generate(self, s: _RunState) -> None:
        """Apply any queued Generate before processing events. Done here
        (not inside the event handler) because the env swap inside
        generate_single_task re-wires the agent's Facility and we need to
        resync anim_time."""
        while s.pending_generate:
            params = s.pending_generate.pop(0)
            ok = generate_single_task(
                params, s.agent, self.facility_name, s.toasts,
            )
            if ok:
                s.anim_time = s.agent.facility.sim_time

    def _tick_sim(self, s: _RunState, dt_wall: float) -> None:
        if s.mode == "anim" and not s.paused and not s.agent.done:
            s.anim_time += dt_wall * s.speed
            s.driver.drive_anim(s.anim_time)
        if s.mode == "step":
            s.anim_time = s.agent.facility.sim_time
        s.toasts.tick()
        # Replay mode: when an episode finishes, optionally roll into the
        # next one immediately. The agent's `done` flag is set by the
        # underlying env's terminated/truncated.
        if (
            s.replay_active
            and s.agent.done
            and s.renderer.replay_content.auto_advance
        ):
            self._next_replay_episode(s)
        s.toasts.tick()  # tick once more so any toasts from the auto-advance
                         # show up on the same frame

    def _render(self, s: _RunState) -> None:
        # Push replay-tab state into the panel before the renderer draws.
        s.renderer.replay_content.update(
            cfg_name=s.replay_cfg_name,
            cfg_facility=s.replay_cfg_facility,
            cfg_kind=s.replay_cfg_kind,
            n_episodes=s.replay_n_episodes,
            n_success=s.replay_n_success,
            last_task=s.replay_last_task,
            last_return=s.replay_last_return,
            agent_done=s.agent.done,
        )
        rs = self._build_render_state(s)
        s.renderer.draw(
            s.surface, s.agent.facility, s.agent.facility.queue, rs,
            manual_mode=not s.agent.facility.auto_arrivals_enabled,
        )
        if s.agent.done:
            _draw_done_banner(
                s.surface, "EPISODE TERMINATED — R: reset · Q: quit",
            )
        s.picker.draw(s.surface, s.renderer.fonts, s.active_policy_label)
        s.facility_picker.draw(s.surface, s.renderer.fonts)
        s.run_config_picker.draw(s.surface, s.renderer.fonts)

    # ─────────────────────────────────────────────────────────────────────
    # Replay-mode helpers
    # ─────────────────────────────────────────────────────────────────────

    def _next_replay_episode(self, s: _RunState) -> None:
        """Roll the replay env forward into the next episode.

        Bumps the episode counter, attributes the previous episode's
        success/return, gives the agent a fresh seed, and resets. Only
        meaningful when `replay_active` is True (a config has been
        loaded); otherwise emits a toast and bails."""
        if not s.replay_active:
            s.toasts.warn(
                "No replay config loaded — open with 'c' / LOAD CONFIG…",
                lifetime=3.0,
            )
            return
        # Attribute the finished episode (if any).
        if s.agent.last_step is not None:
            s.replay_n_episodes += 1
            if s.agent.last_step.terminated:
                s.replay_n_success += 1
            # SingleTaskEnv populates info["task"] only via its
            # gym-shaped .step(); the viz uses submit+advance directly
            # and bypasses that. Read the live attribute as the source
            # of truth.
            env = s.agent.facility.env
            task = getattr(env, "_task", None) or s.agent.info.get("task")
            if task:
                s.replay_last_task = str(task)
            s.replay_last_return = float(s.agent.total_reward)
        # Fresh sampling → fresh seed.
        import secrets
        s.agent.seed = secrets.randbits(31)
        s.agent.reset()
        s.anim_time = s.agent.facility.sim_time
        s.toasts.accent(
            f"episode {s.replay_n_episodes + 1} · task={s.agent.info.get('task', '?')}",
            lifetime=2.5,
        )

    def _load_replay_config(self, s: _RunState) -> None:
        """Pull the selected config from the run-config picker, build a
        fresh env from it via load_run_config, reset all replay counters,
        and mark the replay mode active."""
        entry = s.run_config_picker.selected_entry()
        cfg = s.run_config_picker.selected_config()
        if entry is None or cfg is None:
            s.toasts.error("could not load selected config")
            return
        ok = load_run_config(cfg, s.agent, s.toasts)
        if not ok:
            return
        s.replay_active = True
        s.replay_cfg_name = entry.name
        s.replay_cfg_facility = entry.facility
        s.replay_cfg_kind = entry.env_kind
        s.replay_n_episodes = 0
        s.replay_n_success = 0
        s.replay_last_task = "—"
        s.replay_last_return = 0.0
        s.anim_time = s.agent.facility.sim_time
        s.run_config_picker.active_path = entry.path
        # Auto-pause so the user sees the initial state before any
        # actions fire.
        s.paused = True

    # ─────────────────────────────────────────────────────────────────────
    # Event dispatch — one method per pygame event type
    # ─────────────────────────────────────────────────────────────────────

    def _dispatch_event(self, s: _RunState, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            s.running = False
        elif event.type == pygame.VIDEORESIZE:
            self._on_resize(s, event)
        elif event.type == pygame.MOUSEWHEEL:
            self._on_mousewheel(s, event)
        elif event.type == pygame.MOUSEMOTION:
            self._on_mousemotion(s, event)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._on_mouseup(s, event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_mousedown(s, event)
        elif event.type == pygame.KEYDOWN:
            self._on_keydown(s, event)

    def _on_resize(self, s: _RunState, event: pygame.event.Event) -> None:
        # RESIZABLE already resizes the backing surface; just rebuild the
        # layout. Calling set_mode again here can cascade into more
        # resize events on some WMs.
        s.window_w = event.w
        s.window_h = event.h
        self.window_w = event.w
        self.window_h = event.h
        s.renderer = self._relayout(s)

    def _on_mousewheel(self, s: _RunState, event: pygame.event.Event) -> None:
        if s.picker.open:
            s.picker.handle_wheel(event.y)
            return
        if s.facility_picker.open:
            s.facility_picker.handle_wheel(event.y)
            return
        if s.run_config_picker.open:
            s.run_config_picker.handle_wheel(event.y)
            return

        mouse_pos = pygame.mouse.get_pos()
        shift_held = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)

        # Shift+wheel over the canvas zooms the px-per-mm mapping. Left
        # endpoint stays anchored; only the right end moves. Persists.
        if shift_held and s.renderer.carrier_area_rect.collidepoint(mouse_pos):
            zoom_step = 1.15 ** event.y  # ~15% per notch
            new_zoom = max(0.1, min(8.0, s.zoom * zoom_step))
            if abs(new_zoom - s.zoom) > 1e-6:
                s.zoom = new_zoom
                s.renderer = self._relayout(s)
                save_viz_state(self.runs_dir, zoom=s.zoom)
                s.toasts.info(f"ZOOM {s.zoom:.2f}×", lifetime=1.5)
            return

        # Pixel-granular for randomize, row-granular for the rest.
        scroll_specs = (
            [(s.renderer.randomize_panel, 24)]
            if s.renderer.active_tab == 1
            else [
                (s.renderer.queue_panel,    3),
                (s.renderer.controls_panel, 1),
                (s.renderer.stats_panel,    1),
                (s.renderer.legend_panel,   1),
                (s.renderer.dist_panel,     24),
            ]
        )
        for panel, mult in scroll_specs:
            if panel.hit_test(mouse_pos):
                panel.scroll(-event.y * mult)
                return
        # Fall through: wheel over canvas scrolls the carrier column.
        if s.renderer.carrier_area_rect.collidepoint(mouse_pos):
            s.renderer.scroll_carriers(-event.y * 40)

    def _on_mousemotion(self, s: _RunState, event: pygame.event.Event) -> None:
        if s.dragging_fullness:
            s.renderer.queue_content.set_fullness_from_x(event.pos[0])
            return
        if s.renderer.active_tab == 1:
            s.renderer.randomize_content.handle_mouse_motion(event.pos)

    def _on_mouseup(self, s: _RunState, event: pygame.event.Event) -> None:
        if s.dragging_fullness:
            s.dragging_fullness = False
            return
        if s.renderer.active_tab == 1:
            s.renderer.randomize_content.handle_mouse_up(event.pos)

    def _on_mousedown(self, s: _RunState, event: pygame.event.Event) -> None:
        pos = event.pos
        # Tab strip click switches sidebar tab.
        new_tab = s.renderer.tab_strip.hit_test(pos)
        if new_tab is not None:
            if new_tab != s.renderer.tab_strip.active:
                s.renderer.tab_strip.active = new_tab
                s.renderer.randomize_content.blur()
            return
        # Randomize tab: field drag start, checkbox toggle, Generate button.
        if (
            s.renderer.active_tab == 1
            and s.renderer.randomize_panel.hit_test(pos)
        ):
            s.renderer.randomize_content.handle_mouse_down(pos)
            return
        # Replay tab: LOAD CONFIG… + NEW EPISODE buttons + auto-advance
        # checkbox.
        if (
            s.renderer.active_tab == 2
            and s.renderer.replay_panel.hit_test(pos)
            and s.renderer.replay_content.handle_mouse_down(pos)
        ):
            return
        # Panel header clicks toggle collapse.
        if self._handle_panel_collapse(s.renderer, pos):
            return
        # The rest is manual-mode only.
        if s.agent.facility.auto_arrivals_enabled:
            return
        # Slider drag start.
        if s.renderer.queue_content.hit_slider(pos):
            s.renderer.queue_content.set_fullness_from_x(pos[0])
            s.dragging_fullness = True
            return
        # Queue panel buttons.
        btn = s.renderer.queue_content.hit_button(pos)
        if btn is not None:
            handle_queue_button(
                btn, s.agent.facility, s.agent,
                s.renderer.queue_content.fullness, s.toasts,
            )
            if btn == "randomize":
                s.anim_time = s.agent.facility.state.time
            return
        # Pallet click → toggle Retrieve.
        for rect, pallet_id in s.renderer.pallet_hit_areas:
            if rect.collidepoint(pos):
                handle_pallet_click(pallet_id, s.agent.facility, s.agent, s.toasts)
                break

    def _on_keydown(self, s: _RunState, event: pygame.event.Event) -> None:
        # Focused randomize-panel text field grabs keys before app
        # shortcuts so typing "q" doesn't quit the viz.
        if (
            s.renderer.active_tab == 1
            and s.renderer.randomize_content.focused()
            and s.renderer.randomize_content.handle_key(event)
        ):
            return

        # Modal pickers consume keys first while open.
        if s.facility_picker.open:
            self._on_keydown_facility_picker(s, event)
            return
        if s.picker.open:
            self._on_keydown_policy_picker(s, event)
            return
        if s.run_config_picker.open:
            self._on_keydown_run_config_picker(s, event)
            return

        # Top-level sim controls.
        if event.key in (pygame.K_q, pygame.K_ESCAPE):
            s.running = False
        elif event.key == pygame.K_SPACE:
            s.paused = not s.paused
        elif event.key == pygame.K_p:
            s.picker.toggle()
        elif event.key == pygame.K_f:
            s.facility_picker.toggle()
        elif event.key == pygame.K_c:
            s.run_config_picker.toggle()
        elif event.key == pygame.K_n:
            s.mode = "step" if s.mode == "anim" else "anim"
            if s.mode == "anim":
                s.anim_time = s.agent.facility.state.time
            s.toasts.warn(f"MODE → {s.mode.upper()}", lifetime=2.0)
        elif event.key == pygame.K_m:
            s.agent.facility.set_auto_arrivals(not s.agent.facility.auto_arrivals_enabled)
            if s.agent.facility.auto_arrivals_enabled:
                s.toasts.success(
                    "MANUAL MODE OFF (auto arrivals resumed)", lifetime=3.0,
                )
            else:
                s.toasts.accent(
                    "MANUAL MODE ON (auto arrivals paused)", lifetime=3.0,
                )
        elif event.key in (pygame.K_RIGHT, pygame.K_PERIOD):
            if not s.agent.done:
                s.driver.step_one_decision()
                s.anim_time = s.agent.facility.state.time
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
            s.speed = min(s.speed * 1.5, 1000.0)
            save_viz_state(self.runs_dir, speed=s.speed)
        elif event.key == pygame.K_MINUS:
            s.speed = max(s.speed / 1.5, 0.1)
            save_viz_state(self.runs_dir, speed=s.speed)
        elif event.key == pygame.K_r:
            self._reset_env(s)
        elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
            target = {pygame.K_1: "empty", pygame.K_2: "small",
                      pygame.K_3: "big"}[event.key]
            pid = self._hovered_pallet_id(s.renderer)
            if pid is not None:
                set_pallet_contents(pid, target, s.agent.facility, s.agent, s.toasts)
        elif event.key in (pygame.K_4, pygame.K_5):
            sid = self._hovered_shelf_id(s.renderer)
            if sid is not None:
                if event.key == pygame.K_4:
                    pop_shelf_top(sid, s.agent.facility, s.agent, s.toasts)
                else:
                    push_empty_pallet(sid, s.agent.facility, s.agent, s.toasts)

    def _on_keydown_facility_picker(
        self, s: _RunState, event: pygame.event.Event,
    ) -> None:
        action = s.facility_picker.handle_key(event)
        if action == "submit":
            name = s.facility_picker.selected()
            if name is not None and name != s.facility_picker.active:
                new_renderer = swap_facility(
                    name, s.agent, self.runs_dir,
                    s.window_w, s.window_h, s.toasts,
                )
                s.renderer = self._wire_renderer(
                    new_renderer, pending_generate=s.pending_generate,
                    toasts=s.toasts,
                )
                # New facility → new R-revert baseline.
                s.original_env = s.agent.facility.env
                s.anim_time = s.agent.facility.sim_time
                s.facility_picker.active = name
                self.facility_name = name
                s.active_policy_label = "(random policy)"
            s.facility_picker.close()

    def _on_keydown_policy_picker(
        self, s: _RunState, event: pygame.event.Event,
    ) -> None:
        action = s.picker.handle_key(event)
        if action == "mcts_toggle":
            rewrap_with_mcts(
                s.agent, s.picker.mcts_enabled, s.picker.mcts_n_sims, s.toasts,
            )
        elif action == "submit":
            entry = s.picker.selected()
            if entry is not None:
                label = load_policy(
                    entry, s.agent, s.agent.facility.topology, s.picker.deterministic,
                    s.toasts,
                    mcts_enabled=s.picker.mcts_enabled,
                    mcts_n_sims=s.picker.mcts_n_sims,
                )
                if label is not None:
                    s.active_policy_label = label
                    save_viz_state(self.runs_dir, policy_path=entry.path)
            s.picker.close()

    def _on_keydown_run_config_picker(
        self, s: _RunState, event: pygame.event.Event,
    ) -> None:
        action = s.run_config_picker.handle_key(event)
        if action == "submit":
            self._load_replay_config(s)
            s.run_config_picker.close()
            # Switch to the Replay tab so the user lands on the panel
            # that shows the loaded config's stats.
            s.renderer.tab_strip.active = 2

    def _reset_env(self, s: _RunState) -> None:
        """R-key handler. If we're on a Generate-spawned SingleTaskEnv,
        revert to the original OOSEnv on this facility instead of
        re-rolling another single-task state."""
        from oos.learn.single_task_env import SingleTaskEnv
        preserve_auto = s.agent.facility.auto_arrivals_enabled
        reverted = False
        if isinstance(s.agent.facility.env, SingleTaskEnv):
            # Replace the agent's Facility with one wrapping the original
            # OOSEnv (the user-facing wrapper, not the inner sim engine).
            s.agent.facility = Facility(s.original_env)
            s.agent.seed = self.seed
            reverted = True
        s.agent.reset()
        s.agent.facility.set_auto_arrivals(preserve_auto)
        s.anim_time = s.agent.facility.sim_time
        if reverted:
            s.toasts.accent("ENV RESET (reverted from generated)", lifetime=3.0)
        else:
            s.toasts.accent("ENV RESET", lifetime=2.0)

    # ─────────────────────────────────────────────────────────────────────
    # Renderer construction / wiring
    # ─────────────────────────────────────────────────────────────────────

    def _relayout(self, s: _RunState) -> Renderer:
        new_layout = compute_layout(
            s.agent.facility.topology,
            LayoutConfig(window_w=s.window_w, window_h=s.window_h, zoom=s.zoom),
        )
        return self._wire_renderer(
            Renderer(new_layout, s.agent.facility.topology, zoom=s.zoom),
            pending_generate=s.pending_generate, toasts=s.toasts,
        )

    def _wire_renderer(
        self,
        r: Renderer,
        pending_generate: list,
        toasts: Optional[ToastManager] = None,
    ) -> Renderer:
        """Hook the Renderer's randomize panel's Generate button into our
        pending-generate queue + toast bus."""
        def fire(params: dict) -> None:
            if "_error" in params:
                if toasts is not None:
                    toasts.error(params["_error"])
                return
            pending_generate.append(params)
        r.randomize_content.on_generate = fire
        return r

    # ─────────────────────────────────────────────────────────────────────
    # RenderState build + small helpers
    # ─────────────────────────────────────────────────────────────────────

    def _build_render_state(self, s: _RunState) -> RenderState:
        p = s.agent
        policy_logits = getattr(p.policy, "last_logits", None)
        policy_action_mask = getattr(p.policy, "last_action_mask", None)
        policy_chosen = getattr(p.policy, "last_chosen", None)
        policy_action_entries = p.info.get("action_entries", []) if p.info else []
        return RenderState(
            mode=s.mode + (" (paused)" if s.paused else ""),
            wall_speed=s.speed,
            last_reward=(p.last_step.reward if p.last_step else 0.0),
            n_completed=p.total_completions,
            last_action=(p.last_step.action_label if p.last_step else "—"),
            querying=(p.last_step.querying if p.last_step else "—"),
            anim_now=s.anim_time,
            toasts=s.toasts.toasts,
            wall_now=s.wall_now_fn(),
            policy_label=s.active_policy_label,
            facility_name=self.facility_name,
            policy_logits=policy_logits,
            policy_action_mask=policy_action_mask,
            policy_chosen=policy_chosen,
            policy_action_entries=policy_action_entries,
            policy_query_log=dict(p.policy_query_log),
            mouse_pos=pygame.mouse.get_pos(),
        )

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
