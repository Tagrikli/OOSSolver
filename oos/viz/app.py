"""Pygame app: main loop, event handling, mode toggling, drives Player + Renderer."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import pygame

from oos.env.env import OOSEnv
from oos.sim.state import Pallet
from oos.sim.tasks import Retrieve, Store
from oos.viz.components import (
    CYAN_BRIGHT,
    LIME_BRIGHT,
    MAGENTA_BRIGHT,
    Toast,
    YELLOW_BRIGHT,
    short_action_label,
)
from oos.viz.layout import LayoutConfig, compute_layout
from oos.viz.picker import FacilityPickerState, PickerState, draw_facility_picker, draw_picker
from oos.viz.player import Player, PolicyFn, random_policy
from oos.viz.renderer import RenderState, Renderer


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
        surface = pygame.display.set_mode((self.window_w, self.window_h))
        clock = pygame.time.Clock()

        player = Player(env=self.env, policy=self.policy, seed=self.seed)
        player.reset()
        active_policy_label = "(random policy)"

        facility = player.env._ctx.facility  # type: ignore[attr-defined]
        # Viz boots in manual mode so customer arrivals don't kick off until
        # the user opts in (press 'm' to resume auto arrivals).
        facility.set_auto_arrivals(False)
        topo = facility.topology
        layout = compute_layout(
            topo,
            LayoutConfig(window_w=self.window_w, window_h=self.window_h),
        )
        renderer = Renderer(layout, topo)

        picker = PickerState(runs_dir=self.runs_dir)
        facility_picker = FacilityPickerState(active=self.facility_name)

        mode = "anim"
        paused = True
        speed = self.initial_speed
        anim_time = facility.state.time
        toasts: list[Toast] = []
        wall_start = time.monotonic()

        def wall_now() -> float:
            return time.monotonic() - wall_start

        # Boot toast
        toasts.append(Toast(
            text=f"FACILITY ONLINE  {len(topo.carriers)}C/{len(topo.shelves)}S/{len(topo.rooms)}R",
            color=CYAN_BRIGHT, born_wall=wall_now(), lifetime=4.0,
        ))

        running = True
        while running:
            dt_wall = clock.tick(self.target_fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    if picker.open:
                        picker.move(-event.y)
                    elif facility_picker.open:
                        facility_picker.move(-event.y)
                    else:
                        # Route scroll to whichever side panel the cursor is over.
                        # event.y > 0 = scroll up (show earlier rows).
                        for panel, mult in (
                            (renderer.queue_panel,    3),
                            (renderer.controls_panel, 1),
                            (renderer.stats_panel,    1),
                            (renderer.legend_panel,   1),
                            (renderer.dist_panel,     1),
                        ):
                            if panel.hit_test(mouse_pos):
                                panel.scroll(-event.y * mult)
                                break
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    # Header-click toggles panel collapse — works in both auto
                    # and manual modes (it's a UI thing, not a sim thing).
                    collapse_hit = False
                    for panel in (
                        renderer.stats_panel,
                        renderer.queue_panel,
                        renderer.dist_panel,
                        renderer.controls_panel,
                        renderer.legend_panel,
                    ):
                        if panel.hit_header(pos):
                            panel.chrome.toggle_collapsed()
                            collapse_hit = True
                            break
                    if collapse_hit:
                        continue  # handled, don't fall through to sim controls
                    # Left-click. Only meaningful in manual mode.
                    if facility.auto_arrivals_enabled:
                        # auto mode: ignore canvas clicks; pickers and scroll
                        # are handled elsewhere.
                        pass
                    else:
                        pos = event.pos
                        btn = renderer.queue_panel.hit_button(pos)
                        if btn == "queue small":
                            facility.enqueue_store("small")
                            toasts.append(Toast(
                                text="+ STORE small", color=MAGENTA_BRIGHT,
                                born_wall=wall_now(), lifetime=2.0,
                            ))
                        elif btn == "queue big":
                            facility.enqueue_store("big")
                            toasts.append(Toast(
                                text="+ STORE big", color=MAGENTA_BRIGHT,
                                born_wall=wall_now(), lifetime=2.0,
                            ))
                        elif btn == "queue clear":
                            facility.clear_queue()
                            toasts.append(Toast(
                                text="QUEUE CLEARED", color=YELLOW_BRIGHT,
                                born_wall=wall_now(), lifetime=2.0,
                            ))
                        elif btn == "randomize":
                            # Use a fresh entropy-seeded RNG per click so
                            # consecutive presses (and viz sessions) don't
                            # replay the same sequence from facility.rng.
                            import numpy as np
                            from oos.sim.shuffle import shuffle_state
                            shuffle_state(
                                facility,
                                fullness=0.7,
                                rng=np.random.default_rng(),
                                require_solvable=True,
                            )
                            facility.clear_queue()
                            # Shuffle rebuilt the world out from under the env;
                            # its cached decoder is stale until we re-query.
                            player.env.refresh_decision_context()  # type: ignore[attr-defined]
                            # Player.obs / .info still hold the PRE-shuffle mask
                            # and action_entries; pull fresh ones so the next
                            # policy call sees the new live decoder.
                            player.obs, player.info = (
                                player.env._observation_for_current(  # type: ignore[attr-defined]
                                    facility, dt=0.0, completions=[], arrivals=[],
                                )
                            )
                            anim_time = facility.state.time
                            toasts.append(Toast(
                                text="STATE RANDOMIZED",
                                color=MAGENTA_BRIGHT,
                                born_wall=wall_now(), lifetime=2.5,
                            ))
                        else:
                            # Pallet hit-test: toggle a Retrieve for any pallet
                            # whose slot the user clicked (any contents, incl. empty).
                            for rect, pallet_id in renderer.pallet_hit_areas:
                                if rect.collidepoint(pos):
                                    now_pending = facility.toggle_retrieve_for_pallet(pallet_id)
                                    toasts.append(Toast(
                                        text=(f"+ RETRIEVE pallet={pallet_id}" if now_pending
                                              else f"– RETRIEVE pallet={pallet_id}"),
                                        color=CYAN_BRIGHT if now_pending else YELLOW_BRIGHT,
                                        born_wall=wall_now(), lifetime=2.5,
                                    ))
                                    break
                elif event.type == pygame.KEYDOWN:
                    # Facility picker captures keys first when open.
                    if facility_picker.open:
                        if event.key == pygame.K_ESCAPE:
                            facility_picker.open = False
                        elif event.key == pygame.K_f:
                            facility_picker.open = False
                        elif event.key in (pygame.K_UP, pygame.K_k):
                            facility_picker.move(-1)
                        elif event.key in (pygame.K_DOWN, pygame.K_j):
                            facility_picker.move(1)
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            name = facility_picker.selected()
                            if name is not None and name != facility_picker.active:
                                from oos.env.env import OOSEnv
                                from oos.facilities import get_facility
                                old_env = player.env
                                # Preserve the user's manual-mode preference.
                                preserve_auto = facility.auto_arrivals_enabled
                                new_env = OOSEnv(
                                    facility_factory=get_facility(name),
                                    experiment_config=old_env._experiment_cfg,  # type: ignore[attr-defined]
                                    reward_config=old_env._reward_cfg,          # type: ignore[attr-defined]
                                )
                                player.env = new_env
                                # Old learned policy is sized to the old topology;
                                # fall back to random until the user reloads one.
                                player.policy = random_policy
                                active_policy_label = "(random policy)"
                                player.reset()
                                facility = player.env._ctx.facility  # type: ignore[attr-defined]
                                facility.set_auto_arrivals(preserve_auto)
                                topo = facility.topology
                                layout = compute_layout(
                                    topo,
                                    LayoutConfig(window_w=self.window_w, window_h=self.window_h),
                                )
                                renderer = Renderer(layout, topo)
                                anim_time = facility.state.time
                                facility_picker.active = name
                                self.facility_name = name
                                toasts.append(Toast(
                                    text=f"FACILITY → {name}",
                                    color=LIME_BRIGHT, born_wall=wall_now(), lifetime=4.0,
                                ))
                            facility_picker.open = False
                        continue  # don't fall through to other handlers

                    # Policy picker captures keys next.
                    if picker.open:
                        if event.key == pygame.K_ESCAPE:
                            picker.open = False
                        elif event.key == pygame.K_p:
                            picker.open = False
                        elif event.key in (pygame.K_UP, pygame.K_k):
                            picker.move(-1)
                        elif event.key in (pygame.K_DOWN, pygame.K_j):
                            picker.move(1)
                        elif event.key == pygame.K_d:
                            picker.deterministic = not picker.deterministic
                        elif event.key == pygame.K_s:
                            picker.mcts_enabled = not picker.mcts_enabled
                            # Re-wrap the active policy with/without MCTS
                            # without re-loading the checkpoint from disk.
                            self._rewrap_policy_with_mcts(
                                player, picker, toasts, wall_now(),
                            )
                        elif event.key == pygame.K_r:
                            picker.rescan()
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            entry = picker.selected()
                            if entry is not None:
                                label = self._load_policy(
                                    entry, player, topo, picker.deterministic, toasts, wall_now(),
                                    mcts_enabled=picker.mcts_enabled,
                                    mcts_n_sims=picker.mcts_n_sims,
                                )
                                if label is not None:
                                    # Swap the policy in place — keep the env
                                    # state untouched so the user can A/B-test
                                    # different checkpoints from the same
                                    # mid-episode state.
                                    active_policy_label = label
                            picker.open = False
                        continue  # don't fall through to sim controls

                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_p:
                        picker.toggle()
                    elif event.key == pygame.K_f:
                        facility_picker.toggle()
                    elif event.key == pygame.K_g:
                        # Generate a fresh random facility layout right now —
                        # always re-rolls even if "random" is already active,
                        # which the picker can't do (it short-circuits on
                        # same-name re-select).
                        from oos.env.env import OOSEnv
                        from oos.facilities.random_gen import make_random_facility
                        preserve_auto = facility.auto_arrivals_enabled
                        old_env = player.env
                        new_env = OOSEnv(
                            facility_factory=lambda: make_random_facility(),
                            experiment_config=old_env._experiment_cfg,  # type: ignore[attr-defined]
                            reward_config=old_env._reward_cfg,          # type: ignore[attr-defined]
                        )
                        player.env = new_env
                        # Old policy is sized to the old topology; reset.
                        player.policy = random_policy
                        active_policy_label = "(random policy)"
                        player.reset()
                        facility = player.env._ctx.facility  # type: ignore[attr-defined]
                        facility.set_auto_arrivals(preserve_auto)
                        topo = facility.topology
                        layout = compute_layout(
                            topo,
                            LayoutConfig(window_w=self.window_w, window_h=self.window_h),
                        )
                        renderer = Renderer(layout, topo)
                        anim_time = facility.state.time
                        facility_picker.active = "random"
                        self.facility_name = "random"
                        toasts.append(Toast(
                            text=(f"RANDOM LAYOUT  "
                                  f"{len(topo.carriers)}C/{len(topo.shelves)}S/{len(topo.rooms)}R"),
                            color=MAGENTA_BRIGHT,
                            born_wall=wall_now(), lifetime=4.0,
                        ))
                    elif event.key == pygame.K_n:
                        # 'n' (next): toggle anim vs single-step decision mode.
                        # (Used to be 'm'; freed up for manual task injection.)
                        mode = "step" if mode == "anim" else "anim"
                        if mode == "anim":
                            anim_time = facility.state.time
                        toasts.append(Toast(
                            text=f"MODE → {mode.upper()}",
                            color=YELLOW_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                        ))
                    elif event.key == pygame.K_m:
                        # Toggle manual task-injection mode. Agent keeps running;
                        # automatic Store/Retrieve arrivals are suspended, and
                        # the queue can be driven from the buttons / pallet clicks.
                        facility.set_auto_arrivals(not facility.auto_arrivals_enabled)
                        on = facility.auto_arrivals_enabled
                        toasts.append(Toast(
                            text="MANUAL MODE OFF (auto arrivals resumed)" if on else "MANUAL MODE ON (auto arrivals paused)",
                            color=LIME_BRIGHT if on else MAGENTA_BRIGHT,
                            born_wall=wall_now(), lifetime=3.0,
                        ))
                    elif event.key in (pygame.K_RIGHT, pygame.K_PERIOD):
                        if not player.done:
                            # Manual step: process events up to the next decision instant.
                            self._step_one_decision(player, toasts, wall_now())
                            anim_time = facility.state.time
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        speed = min(speed * 1.5, 1000.0)
                    elif event.key == pygame.K_MINUS:
                        speed = max(speed / 1.5, 0.1)
                    elif event.key == pygame.K_r:
                        preserve_auto = facility.auto_arrivals_enabled
                        player.reset()
                        facility = player.env._ctx.facility  # type: ignore[attr-defined]
                        facility.set_auto_arrivals(preserve_auto)
                        anim_time = facility.state.time
                        toasts.append(Toast(
                            text="ENV RESET",
                            color=MAGENTA_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                        ))
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        # Quick pallet-contents editor. Hover a pallet slot,
                        # press 1=empty / 2=small / 3=big. Key 3 is rejected
                        # on small shelves.
                        target_contents = {
                            pygame.K_1: "empty",
                            pygame.K_2: "small",
                            pygame.K_3: "big",
                        }[event.key]
                        mouse_pos = pygame.mouse.get_pos()
                        hit_pid: int | None = None
                        for rect, pid in renderer.pallet_hit_areas:
                            if rect.collidepoint(mouse_pos):
                                hit_pid = pid
                                break
                        if hit_pid is None:
                            pass  # nothing under the cursor
                        else:
                            owner_sid: str | None = None
                            owner_idx = -1
                            for sid, ss in facility.state.shelves.items():
                                for i, p in enumerate(ss.stack):
                                    if p.id == hit_pid:
                                        owner_sid = sid
                                        owner_idx = i
                                        break
                                if owner_sid is not None:
                                    break
                            if owner_sid is None:
                                toasts.append(Toast(
                                    text=f"pallet {hit_pid} not on a shelf",
                                    color=YELLOW_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                ))
                            else:
                                shelf_topo = facility.topology.shelves[owner_sid]
                                if target_contents == "big" and shelf_topo.size_class != "big":
                                    toasts.append(Toast(
                                        text="BIG item not allowed on small shelf",
                                        color=YELLOW_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                    ))
                                else:
                                    old = facility.state.shelves[owner_sid].stack[owner_idx]
                                    facility.state.shelves[owner_sid].stack[owner_idx] = (
                                        Pallet(id=old.id, contents=target_contents)  # type: ignore[arg-type]
                                    )
                                    player.env.refresh_decision_context()  # type: ignore[attr-defined]
                                    player.obs, player.info = (
                                        player.env._observation_for_current(  # type: ignore[attr-defined]
                                            facility, dt=0.0, completions=[], arrivals=[],
                                        )
                                    )
                                    toasts.append(Toast(
                                        text=f"pallet {hit_pid} → {target_contents}",
                                        color=MAGENTA_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                    ))
                    elif event.key in (pygame.K_4, pygame.K_5):
                        # Shelf-level edits, targeting whichever shelf the
                        # mouse is over:
                        #   4 → pop top pallet (removes it from the system)
                        #   5 → push a new empty pallet (if capacity remains)
                        mouse_pos = pygame.mouse.get_pos()
                        hit_sid: str | None = None
                        for rect, sid in renderer.shelf_hit_areas:
                            if rect.collidepoint(mouse_pos):
                                hit_sid = sid
                                break
                        if hit_sid is None:
                            pass
                        else:
                            ss = facility.state.shelves[hit_sid]
                            shelf_topo = facility.topology.shelves[hit_sid]
                            if event.key == pygame.K_4:
                                if not ss.stack:
                                    toasts.append(Toast(
                                        text=f"shelf {hit_sid} already empty",
                                        color=YELLOW_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                    ))
                                else:
                                    popped = ss.stack.pop()
                                    toasts.append(Toast(
                                        text=f"removed pallet {popped.id} from {hit_sid}",
                                        color=MAGENTA_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                    ))
                                    player.env.refresh_decision_context()  # type: ignore[attr-defined]
                                    player.obs, player.info = (
                                        player.env._observation_for_current(  # type: ignore[attr-defined]
                                            facility, dt=0.0, completions=[], arrivals=[],
                                        )
                                    )
                            else:  # K_5: push empty
                                if len(ss.stack) >= shelf_topo.capacity:
                                    toasts.append(Toast(
                                        text=f"shelf {hit_sid} at capacity ({shelf_topo.capacity})",
                                        color=YELLOW_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                    ))
                                else:
                                    new_pid = facility._next_pallet_id
                                    facility._next_pallet_id += 1
                                    ss.stack.append(Pallet(id=new_pid, contents="empty"))
                                    toasts.append(Toast(
                                        text=f"pushed empty pallet {new_pid} → {hit_sid}",
                                        color=MAGENTA_BRIGHT, born_wall=wall_now(), lifetime=2.0,
                                    ))
                                    player.env.refresh_decision_context()  # type: ignore[attr-defined]
                                    player.obs, player.info = (
                                        player.env._observation_for_current(  # type: ignore[attr-defined]
                                            facility, dt=0.0, completions=[], arrivals=[],
                                        )
                                    )

            # Advance sim in anim mode.
            # New loop: anim_time creeps forward at speed * dt_wall and never
            # gets force-bumped to state.time. The renderer interpolates the
            # carrier position between command start and end based on anim_time.
            # We only advance the scheduler past anim_time when the next event
            # is due. Decision-instant submissions happen instantly (no time).
            if mode == "anim" and not paused and not player.done:
                anim_time += dt_wall * speed
                self._drive_anim(player, anim_time, toasts, wall_now())

            if mode == "step":
                anim_time = facility.state.time

            # Garbage-collect expired toasts
            wn = wall_now()
            toasts[:] = [t for t in toasts if not t.expired(wn)]

            # Pull last forward-pass artifacts from the active learned policy
            # (if any) so the distribution panel can render them. The random
            # policy doesn't expose these, so they stay None and the panel
            # renders a "no logits yet" placeholder.
            policy_logits = getattr(player.policy, "last_logits", None)
            policy_action_mask = getattr(player.policy, "last_action_mask", None)
            policy_chosen = getattr(player.policy, "last_chosen", None)
            policy_action_entries = player.info.get("action_entries", []) if player.info else []
            mouse_pos_now = pygame.mouse.get_pos()

            rs = RenderState(
                mode=mode + (" (paused)" if paused else ""),
                wall_speed=speed,
                last_reward=(player.last_record.reward if player.last_record else 0.0),
                n_completed=player.total_completions,
                last_action=(player.last_record.action_label if player.last_record else "—"),
                querying=(player.last_record.querying if player.last_record else "—"),
                anim_now=anim_time,
                toasts=toasts,
                wall_now=wn,
                policy_label=active_policy_label,
                facility_name=self.facility_name,
                policy_logits=policy_logits,
                policy_action_mask=policy_action_mask,
                policy_chosen=policy_chosen,
                policy_action_entries=policy_action_entries,
                mouse_pos=mouse_pos_now,
            )
            renderer.draw(
                surface, facility, facility.queue, rs,
                manual_mode=not facility.auto_arrivals_enabled,
            )

            if player.done:
                _draw_done_banner(surface, "EPISODE TERMINATED — R: reset · Q: quit")

            # Picker overlays drawn last so they sit above everything else.
            draw_picker(surface, renderer.fonts, picker, active_policy_label)
            draw_facility_picker(surface, renderer.fonts, facility_picker)

            pygame.display.flip()

        pygame.quit()

    # ------------------------------------------------------------------

    def _drive_anim(
        self,
        player: Player,
        anim_time: float,
        toasts: list[Toast],
        wall_now: float,
    ) -> None:
        """Drive the env in animation mode bounded by anim_time.

        Loop:
        - While a carrier needs a decision (and we've caught up to it), query
          policy and submit. Submission does not advance time.
        - Then advance the scheduler up to anim_time (no further than the
          next decision instant). If a new decision instant is reached, loop
          back to submission.
        - If anim_time runs out before the next event, return — renderer will
          interpolate carrier position based on anim_time.
        """
        env = player.env
        max_iters = 200
        i = 0
        while not player.done and i < max_iters:
            i += 1
            facility = env._ctx.facility  # type: ignore[attr-defined]
            # Submit any decisions queued at the current instant (no time advance).
            if env.needs_decision() and facility.state.time <= anim_time:
                self._submit_one(player, toasts, wall_now)
                continue
            # Otherwise, advance time up to anim_time.
            obs, reward, term, trunc, info = env.advance(time_limit=anim_time)
            self._record_advance(player, obs, reward, term, trunc, info, toasts, wall_now)
            if term or trunc:
                player.done = True
                break
            # If we didn't reach a new decision instant, anim_time bounded us.
            if not env.needs_decision():
                break
            # else: a new decision is ready — loop and submit it.

    def _submit_one(
        self, player: Player, toasts: list[Toast], wall_now: float
    ) -> None:
        env = player.env
        action_idx = player.policy(player.obs, player.info)
        # Belt-and-suspenders: if the action came from a stale obs (e.g.
        # state was mutated externally between query and submit and the
        # cached mask is longer than the live decoder), clamp to WAIT —
        # which is always the last legal entry per `enumerate_actions`.
        live_n_legal = len(env._ctx.decoder.entries)  # type: ignore[attr-defined]
        if not (0 <= action_idx < live_n_legal):
            action_idx = live_n_legal - 1
        entries = player.info.get("action_entries", [])
        if 0 <= action_idx < len(entries):
            cmd = entries[action_idx].to_command(env._ctx.querying_carrier)  # type: ignore[attr-defined]
            label = short_action_label(cmd)
        else:
            label = f"#{action_idx}"
        from oos.viz.player import StepRecord
        querying = str(env._ctx.querying_carrier)  # type: ignore[attr-defined]
        sim_t_before = env._ctx.facility.state.time  # type: ignore[attr-defined]
        env.submit_action(action_idx)
        # Refresh obs/info for the next decision (if any) using a 0-time advance.
        # We just rebuild obs from current state without advancing the scheduler.
        # advance(time_limit=current sim time) returns immediately.
        obs, reward, term, trunc, info = env.advance(time_limit=env._ctx.facility.state.time)  # type: ignore[attr-defined]
        player.obs = obs
        player.info = info
        player.total_reward += reward
        player.last_record = StepRecord(
            sim_time_before=sim_t_before,
            sim_time_after=env._ctx.facility.state.time,  # type: ignore[attr-defined]
            action_label=label,
            querying=querying,
            reward=reward,
            n_completions=len(info.get("completions", [])),
        )
        self._emit_toasts(player, info, toasts, wall_now)

    def _record_advance(
        self,
        player: Player,
        obs: dict,
        reward: float,
        term: bool,
        trunc: bool,
        info: dict,
        toasts: list[Toast],
        wall_now: float,
    ) -> None:
        player.obs = obs
        player.info = info
        player.total_reward += reward
        # Don't overwrite last_record's action label — this was a no-action advance.
        self._emit_toasts(player, info, toasts, wall_now)

    def _step_one_decision(
        self, player: Player, toasts: list[Toast], wall_now: float
    ) -> None:
        """Manual step mode: submit current pending action then advance to next decision."""
        env = player.env
        if env.needs_decision():
            self._submit_one(player, toasts, wall_now)
        # Now advance fully (no time cap) to the next decision instant.
        obs, reward, term, trunc, info = env.advance(time_limit=None)
        self._record_advance(player, obs, reward, term, trunc, info, toasts, wall_now)
        if term or trunc:
            player.done = True

    def _load_policy(
        self,
        entry,  # picker.CheckpointEntry
        player: Player,
        topo,
        deterministic: bool,
        toasts: list[Toast],
        wall_now: float,
        mcts_enabled: bool = False,
        mcts_n_sims: int = 32,
    ) -> Optional[str]:
        """Swap player.policy to the entry's policy. Returns the display label
        on success, or None if loading failed (a toast is emitted either way)."""
        if entry.path == "":
            player.policy = random_policy
            toasts.append(Toast(
                text="POLICY → random",
                color=CYAN_BRIGHT, born_wall=wall_now, lifetime=3.0,
            ))
            return entry.display_name
        try:
            # Imported lazily so the viz still runs without torch installed when
            # no checkpoint is being loaded.
            from oos.learn.policy import LearnedPolicy, MCTSPolicy
            policy = LearnedPolicy(
                checkpoint_path=entry.path,
                topology=topo,
                device="cpu",
                deterministic=deterministic,
            )
            if mcts_enabled:
                player.policy = MCTSPolicy(
                    learned=policy, env=player.env, n_sims=mcts_n_sims,
                )
            else:
                player.policy = policy
            mode_label = "argmax" if deterministic else "sample"
            mcts_label = f"+mcts:{mcts_n_sims}" if mcts_enabled else ""
            toasts.append(Toast(
                text=f"POLICY → {entry.display_name} ({mode_label}{mcts_label})",
                color=LIME_BRIGHT, born_wall=wall_now, lifetime=4.0,
            ))
            return (
                f"{entry.display_name} [iter {policy.iteration}, "
                f"{mode_label}{mcts_label}]"
            )
        except Exception as e:
            toasts.append(Toast(
                text=f"LOAD FAILED: {type(e).__name__}: {e}"[:80],
                color=MAGENTA_BRIGHT, born_wall=wall_now, lifetime=6.0,
            ))
            return None

    def _rewrap_policy_with_mcts(
        self,
        player: Player,
        picker,  # PickerState
        toasts: list[Toast],
        wall_now: float,
    ) -> None:
        """Toggle MCTS on/off for the *currently loaded* policy without re-
        reading the checkpoint from disk. Random policy is left untouched."""
        from oos.learn.policy import LearnedPolicy, MCTSPolicy
        current = player.policy
        if isinstance(current, MCTSPolicy):
            inner = current.learned
        elif isinstance(current, LearnedPolicy):
            inner = current
        else:
            toasts.append(Toast(
                text="MCTS: no learned policy loaded",
                color=MAGENTA_BRIGHT, born_wall=wall_now, lifetime=3.0,
            ))
            return
        if picker.mcts_enabled:
            player.policy = MCTSPolicy(
                learned=inner, env=player.env, n_sims=picker.mcts_n_sims,
            )
            toasts.append(Toast(
                text=f"MCTS ON  (n_sims={picker.mcts_n_sims})",
                color=CYAN_BRIGHT, born_wall=wall_now, lifetime=3.0,
            ))
        else:
            player.policy = inner
            toasts.append(Toast(
                text="MCTS OFF  (reactive policy only)",
                color=YELLOW_BRIGHT, born_wall=wall_now, lifetime=3.0,
            ))

    def _emit_toasts(
        self, player: Player, info: dict, toasts: list[Toast], wall_now: float
    ) -> None:
        for arr in info.get("arrivals", []):
            if isinstance(arr, Store):
                toasts.append(Toast(
                    text=f"⇩ STORE {arr.size}",
                    color=MAGENTA_BRIGHT,
                    born_wall=wall_now, lifetime=3.5,
                ))
        # Dropped tasks (e.g. big stores when big shelves are exhausted) leave
        # silently — the customer queue strip simply doesn't show them.
        # No toast: this is normal capacity behavior, not an error condition.
            elif isinstance(arr, Retrieve):
                toasts.append(Toast(
                    text=f"⇧ RETRIEVE pallet={arr.pallet}",
                    color=CYAN_BRIGHT,
                    born_wall=wall_now, lifetime=3.5,
                ))
        for comp in info.get("completions", []):
            t = comp.task
            if isinstance(t, Store):
                label = f"STORE {t.size} done"
            elif isinstance(t, Retrieve):
                label = f"RETRIEVE pallet={t.pallet} done"
            else:
                label = "task done"
            toasts.append(Toast(
                text=f"✓ {label}  cost={comp.cost:.1f}",
                color=LIME_BRIGHT, born_wall=wall_now, lifetime=3.5,
            ))
        # Cap toast count
        if len(toasts) > 10:
            del toasts[: len(toasts) - 10]


def _draw_done_banner(surface: pygame.Surface, text: str) -> None:
    from oos.viz.components import (
        BASE_BLACK,
        YELLOW_BRIGHT,
        beveled_polygon,
        draw_beveled_rect,
        draw_beveled_frame,
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
    VizApp(env=env, policy=policy or random_policy, seed=seed, facility_name=facility_name).run()
