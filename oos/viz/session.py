"""Session — headless logic core for the viz.

Owns an `Agent` + `Environment` and exposes exactly two interaction surfaces,
mirroring the two exogenous inputs to the real system:

  * **World** (you play the customer/operator): `enqueue_store`,
    `request_retrieve`, `set_auto_arrivals` / `set_store_rate`, `reroll_layout`,
    `swap_facility`. These mutate the *world* (the task queue + the physical
    layout) — never a carrier.
  * **Playback** (you watch the brain): `play`/`pause`/`step_once`/`set_speed`,
    plus `load_policy` to pick which brain drives the carriers.

`tick(dt_wall)` advances a playback clock and drives the sim up to it, so
carriers animate smoothly between decision instants (the canvas reads
`carrier_position_at` at `sim_time`). No DearPyGui imports live here — the
whole thing is unit-testable headless.
"""

from __future__ import annotations

import secrets
from collections import deque
from dataclasses import dataclass

import numpy as np

from oos.agent import Agent, random_policy
from oos.env import Environment
from oos.env.action import ActionType
from oos.learn.policy import CheckpointEntry, discover_checkpoints
from oos.sim.actions import short_action_label
from oos.sim.shuffle import shuffle_state
from oos.sim.tasks import Retrieve, Store


@dataclass
class PolicyInfo:
    """What's currently driving the carriers (for the status readout)."""
    label: str = "(random policy)"
    deterministic: bool = False
    iteration: int = -1


class Session:
    """Headless owner of the live facility + agent. The DearPyGui app is glue
    on top; everything stateful and sim-touching lives here."""

    MAX_ITERS_PER_FRAME = 200   # livelock guard on instant-time decision chains

    def __init__(
        self,
        facility_name: str = "tiny_medipol",
        runs_dir: str = "runs",
        seed: int = 0,
    ) -> None:
        self.facility_name = facility_name
        self.runs_dir = runs_dir
        self._seed = seed

        self.env = Environment.from_name(facility_name)
        self.agent = Agent(facility=self.env, policy=random_policy, seed=seed)
        self.policy_info = PolicyInfo()

        # Playback state. `play_time` is the sim-time the canvas renders at; it
        # is advanced by wall-clock * speed each tick, then the sim is driven up
        # to it. Manual mode (no auto-arrivals) by default — a fresh session is
        # an empty world the user pokes.
        self.playing = False
        self.speed = 1.0
        self.auto_arrivals = False
        self.log: deque[str] = deque(maxlen=200)

        self.reset()

    # ────────────────────────────────────────────────────────────────────
    # Read-only views (the canvas / status panel consume these)
    # ────────────────────────────────────────────────────────────────────

    @property
    def topology(self):
        return self.env.topology

    @property
    def state(self):
        return self.env.state

    @property
    def sim_time(self) -> float:
        return self.env.sim_time

    @property
    def queue(self):
        return self.env.queue

    @property
    def querying_carrier(self) -> str:
        return self.env.querying_carrier

    @property
    def done(self) -> bool:
        return self.agent.done

    def pending_retrieve_ids(self) -> set[int]:
        """Pallet ids with a Retrieve currently pending — for highlighting."""
        return {t.pallet for t in self.queue.pending if isinstance(t, Retrieve)}

    def pending_store_count(self) -> int:
        return sum(1 for t in self.queue.pending if isinstance(t, Store))

    def status_line(self) -> str:
        a = self.agent
        last = a.last_step.action_label if a.last_step else "—"
        return (
            f"t={self.sim_time:7.2f}s  query={self.querying_carrier}  "
            f"last={last}  Σr={a.total_reward:+.1f}  "
            f"done={a.total_completions}  pending={len(self.queue.pending)}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Playback surface
    # ────────────────────────────────────────────────────────────────────

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.0, float(speed))

    def tick(self, dt_wall: float) -> None:
        """Advance the playback clock by `dt_wall * speed` (wall seconds) and
        drive the sim up to it. Call once per rendered frame while playing."""
        if not self.playing or self.agent.done:
            return
        self._drive_to(self.sim_time + dt_wall * self.speed)

    def step_once(self) -> None:
        """Single-step: submit one decision, then run until the next one (no
        time bound). Pauses playback so the user can inspect."""
        self.playing = False
        if self.agent.done:
            return
        if self.env.needs_decision():
            self._submit_one_at_current_time()
        obs, reward, info = self.env.advance_until(sim_time=None)
        self._record_advance(reward, info)

    def reset(self) -> None:
        """Re-roll the episode from the session seed and clear playback state."""
        self.agent.reset(seed=self._seed)
        self.env.set_auto_arrivals(self.auto_arrivals)
        self.playing = False
        self.log.clear()
        self._note(f"reset · facility={self.facility_name}")

    # ────────────────────────────────────────────────────────────────────
    # World surface (you = customer/operator; never touch a carrier)
    # ────────────────────────────────────────────────────────────────────

    def enqueue_store(self, size: str = "small") -> None:
        """A car arrives wanting to be parked."""
        self.env.engine.enqueue_store(size)   # type: ignore[arg-type]
        self.env.wake_waiting_carriers()
        self._note(f"+ STORE {size}")

    def request_retrieve(self, pallet_id: int) -> bool:
        """Customer asks for pallet `pallet_id` back (toggles the request).
        Returns True iff a Retrieve is now pending for it."""
        now_pending = self.env.engine.toggle_retrieve_for_pallet(int(pallet_id))
        self.env.wake_waiting_carriers()
        self._note(f"{'+ ' if now_pending else '− '}RETRIEVE pallet={pallet_id}")
        return now_pending

    def clear_queue(self) -> None:
        self.env.engine.clear_queue()
        self.env.wake_waiting_carriers()
        self._note("cleared queue")

    def set_auto_arrivals(self, enabled: bool) -> None:
        """Turn the simulated world (Poisson store stream + dwell retrieves) on
        or off. Off = pure manual mode (you generate all demand)."""
        self.auto_arrivals = bool(enabled)
        self.env.set_auto_arrivals(self.auto_arrivals)
        self._note(f"auto-world {'ON' if enabled else 'OFF'}")

    def set_store_rate(self, rate: float, big_prob: float = 0.3,
                       mean_dwell: float = 60.0, std_dwell: float = 20.0) -> None:
        """Reconfigure the auto-world Poisson stream. Rebuilds the experiment
        config and resets so the new stream takes effect."""
        from oos.config.schema import ExperimentConfig, TaskStreamConfig
        old = self.env._experiment_cfg  # type: ignore[attr-defined]
        ts = TaskStreamConfig(
            store_rate=float(rate),
            size_mix={"small": 1.0 - big_prob, "big": big_prob},
            mean_dwell_seconds=float(mean_dwell),
            std_dwell_seconds=float(std_dwell),
        )
        self.env._experiment_cfg = ExperimentConfig(  # type: ignore[attr-defined]
            durations=old.durations, task_stream=ts, episode=old.episode,
        )
        self.reset()
        self._note(f"store-rate → {rate:.3f}")

    def reroll_layout(self, fullness: float = 0.5, seed: int | None = None) -> int:
        """Re-roll the CURRENT facility's pallet layout in place at `fullness`
        (the physical world is reconfigured), clear the queue, keep running.
        Returns the seed so `(facility, fullness, seed)` reproduces it."""
        if seed is None:
            seed = secrets.randbits(63)
        shuffle_state(
            self.env.engine, fullness=float(fullness),
            rng=np.random.default_rng(seed), require_solvable=True,
        )
        self.env.engine.clear_queue()
        self.env.wake_waiting_carriers()
        self._note(f"re-roll layout · fullness={fullness:.2f} · seed={seed}")
        return int(seed)

    def swap_facility(self, name: str) -> None:
        """Rebuild the world on a different facility. Drops to the random policy
        (the loaded brain was sized for the old topology)."""
        self.env = Environment.from_name(name)
        self.agent = Agent(facility=self.env, policy=random_policy, seed=self._seed)
        self.facility_name = name
        self.policy_info = PolicyInfo()
        self.reset()
        self._note(f"facility → {name}")

    # ────────────────────────────────────────────────────────────────────
    # Brain selection
    # ────────────────────────────────────────────────────────────────────

    def checkpoints(self) -> list[CheckpointEntry]:
        return discover_checkpoints(self.runs_dir)

    def load_policy(self, entry: CheckpointEntry, deterministic: bool = False) -> bool:
        """Swap the brain driving the carriers. `entry.path == ""` → random.
        Returns True on success (a failed load leaves the old policy in place)."""
        if entry.path == "":
            self.agent.policy = random_policy
            self.policy_info = PolicyInfo(label="(random policy)")
            self._note("policy → random")
            return True
        try:
            from oos.learn.policy import LearnedPolicy
            policy = LearnedPolicy(
                checkpoint_path=entry.path, topology=self.env.topology,
                device="cpu", deterministic=deterministic,
            )
            self.agent.policy = policy
            self.policy_info = PolicyInfo(
                label=entry.display_name, deterministic=deterministic,
                iteration=policy.iteration,
            )
            mode = "argmax" if deterministic else "sample"
            self._note(f"policy → {entry.display_name} [iter {policy.iteration}, {mode}]")
            return True
        except Exception as e:   # noqa: BLE001 — surface any load failure to the UI
            self._note(f"LOAD FAILED: {type(e).__name__}: {e}"[:80])
            return False

    # ────────────────────────────────────────────────────────────────────
    # Driver internals (port of the old SimDriver — pure Agent+Env, no UI)
    # ────────────────────────────────────────────────────────────────────

    def _drive_to(self, anim_time: float) -> None:
        """Advance the sim until `anim_time`, submitting any decision due at or
        before it first, so animation time flows smoothly between instants."""
        env = self.env
        for _ in range(self.MAX_ITERS_PER_FRAME):
            if self.agent.done:
                return
            if env.needs_decision() and env.sim_time <= anim_time:
                self._submit_one_at_current_time()
                continue
            obs, reward, info = env.advance_until(sim_time=anim_time)
            self._record_advance(reward, info)
            if self.agent.done or not env.needs_decision():
                return

    def _submit_one_at_current_time(self) -> None:
        """Submit a single decision via the agent without letting time pass
        (other carriers may decide at the same instant), then zero-advance to
        refresh obs/info."""
        agent, env = self.agent, self.env
        querying = env.querying_carrier
        action_idx = agent.act()
        agent.total_actions += 1
        agent.record_policy_query(querying)

        live_n = len(env._ctx.decoder.entries)  # type: ignore[attr-defined]
        if not (0 <= action_idx < live_n):
            action_idx = live_n - 1            # fall back to the last legal entry
        label = self._label_for(action_idx, querying)

        env.submit_action(action_idx)
        obs, reward, info = env.advance_until(sim_time=env.sim_time)  # 0-time refresh
        agent.obs, agent.info = obs, info
        agent.total_reward += reward
        agent.total_completions += len(info.get("completions", []))
        if info.get("terminated") or info.get("truncated"):
            agent.done = True
        agent.last_step = _Step(querying, action_idx, label, reward, info)
        self._emit(info)

    def _record_advance(self, reward: float, info: dict) -> None:
        agent = self.agent
        agent.obs = agent.obs  # obs already current; keep info fresh
        agent.info = info
        agent.total_reward += reward
        agent.total_completions += len(info.get("completions", []))
        if agent.last_step is not None:
            agent.last_step.reward += reward
        if info.get("terminated") or info.get("truncated"):
            agent.done = True
        self._emit(info)

    def _label_for(self, action_idx: int, querying: str) -> str:
        entries = self.agent.info.get("action_entries", [])
        if 0 <= action_idx < len(entries):
            entry = entries[action_idx]
            if entry.type == ActionType.WAIT:
                return "wait"
            return short_action_label(entry.to_command(querying))
        return f"#{action_idx}"

    # ────────────────────────────────────────────────────────────────────
    # Event log
    # ────────────────────────────────────────────────────────────────────

    def _emit(self, info: dict) -> None:
        for arr in info.get("arrivals", []):
            if isinstance(arr, Store):
                self._note(f"⇩ arrive STORE {arr.size}")
            elif isinstance(arr, Retrieve):
                self._note(f"⇧ arrive RETRIEVE pallet={arr.pallet}")
        for comp in info.get("completions", []):
            t = comp.task
            if isinstance(t, Store):
                self._note(f"✓ STORE {t.size} served  (cost {comp.cost:.1f})")
            elif isinstance(t, Retrieve):
                tag = "" if comp.agent_delivered else " [camped]"
                self._note(f"✓ RETRIEVE p={t.pallet} delivered{tag}  (cost {comp.cost:.1f})")

    def _note(self, msg: str) -> None:
        self.log.append(f"[{self.sim_time:7.2f}] {msg}")


@dataclass
class _Step:
    """Minimal last-action record for the status readout (a trimmed AgentStep)."""
    querying: str
    action_idx: int
    action_label: str
    reward: float
    info: dict
