"""Multi-process vectorized OOS env.

One worker process per env. Main process sends actions in parallel, workers
step their env, send back the next sample + reward + done + (on episode end)
the truncated sample for value bootstrap and the finished-episode stats.

Auto-resets on episode end so the main loop never has to special-case it —
just keep calling step(actions) for n_steps and you get a full rollout per env.

Uses `spawn` start method to keep worker address spaces clean (no inherited
torch state). Worker startup is ~1s/worker on first call; reuse the vec_env
across PPO iterations to amortize.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

from oos.config.schema import ExperimentConfig
from oos.env.reward import RewardConfig
from oos.facilities import get_facility
from oos.learn.batching import Sample, sample_from_env_step
from oos.learn.episode_env import EpisodeConfig, EpisodeEnv


@dataclass
class StepResult:
    """One env step's worth of data, sent over the pipe."""

    sample: Sample                          # the obs AFTER step (post-reset if done)
    reward: float
    done: bool
    constraint_cost: float                  # c_t = dt * n_pending_retrieves (task-seconds)
    terminal_sample: Optional[Sample]       # the obs AT truncation (pre-reset); None unless done
    ep_return: Optional[float]              # finalized episode return; None unless done
    ep_length: Optional[int]
    ep_completions: Optional[int]
    ep_sim_time: Optional[float]
    ep_constraint_cost: Optional[float]     # episode total task-seconds; None unless done
    ep_retrieves_completed: Optional[int]
    ep_retrieves_total: Optional[int]
    ep_stores_completed: Optional[int]


def _build_env(
    facility_name: str,
    experiment_config: ExperimentConfig,
    reward_config: RewardConfig,
    episode_config: EpisodeConfig,
) -> EpisodeEnv:
    return EpisodeEnv(
        facility_factory=get_facility(facility_name),
        episode_scenario_config=episode_config,
        experiment_config=experiment_config,
        reward_config=reward_config,
    )


def _worker(
    remote,
    experiment_config: ExperimentConfig,
    reward_config: RewardConfig,
    seed: int,
    facility_name: str,
    episode_config: EpisodeConfig,
) -> None:
    """Per-worker loop. Owns one EpisodeEnv. Handles step/reset/close."""
    env = _build_env(facility_name, experiment_config, reward_config, episode_config)
    obs, info = env.reset(seed=seed)
    # Per-worker seed space; large stride so concurrent envs don't collide.
    next_seed = seed + 1_000_003

    ep_return = 0.0
    ep_length = 0
    ep_completions = 0
    ep_constraint_cost = 0.0

    while True:
        try:
            msg = remote.recv()
        except EOFError:
            break
        cmd = msg[0]
        if cmd == "step":
            action = int(msg[1])
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            c_t = float(info.get("dt", 0.0)) * int(info.get("n_pending_retrieves", 0))
            ep_return += float(reward)
            ep_length += 1
            ep_completions += len(info.get("completions", []))
            ep_constraint_cost += c_t
            terminal_sample = None
            er = el = ec = est = ecc = None
            erc = ert = esc = None
            if done:
                terminal_sample = sample_from_env_step(obs, info, info["action_entries"])
                er, el, ec = ep_return, ep_length, ep_completions
                est = float(info.get("sim_time", 0.0))
                ecc = ep_constraint_cost
                erc = int(info.get("retrieves_completed", 0))
                ert = int(info.get("retrieves_total", 0))
                esc = int(info.get("stores_completed", 0))
                ep_return = 0.0
                ep_length = 0
                ep_completions = 0
                ep_constraint_cost = 0.0
                obs, info = env.reset(seed=next_seed)
                next_seed += 1
            sample = sample_from_env_step(obs, info, info["action_entries"])
            remote.send(
                StepResult(
                    sample=sample,
                    reward=float(reward),
                    done=done,
                    constraint_cost=c_t,
                    terminal_sample=terminal_sample,
                    ep_return=er,
                    ep_length=el,
                    ep_completions=ec,
                    ep_sim_time=est,
                    ep_constraint_cost=ecc,
                    ep_retrieves_completed=erc,
                    ep_retrieves_total=ert,
                    ep_stores_completed=esc,
                )
            )
        elif cmd == "reset":
            seed_in = int(msg[1])
            obs, info = env.reset(seed=seed_in)
            ep_return = 0.0
            ep_length = 0
            ep_completions = 0
            ep_constraint_cost = 0.0
            next_seed = seed_in + 1_000_003
            sample = sample_from_env_step(obs, info, info["action_entries"])
            remote.send(sample)
        elif cmd == "set_episode_config":
            episode_config = msg[1]
            env._cfg = episode_config  # type: ignore[attr-defined]
        elif cmd == "close":
            try:
                remote.close()
            except Exception:
                pass
            return
        else:
            raise ValueError(f"unknown vec_env cmd: {cmd!r}")


class VecEnv:
    """N worker processes, one EpisodeEnv each."""

    def __init__(
        self,
        n_envs: int,
        experiment_config: ExperimentConfig,
        reward_config: RewardConfig | None = None,
        base_seed: int = 0,
        start_method: str = "spawn",
        facility_name: str = "dev",
        episode_config: EpisodeConfig | None = None,
    ):
        if reward_config is None:
            reward_config = RewardConfig()
        if episode_config is None:
            episode_config = EpisodeConfig()
        # Validate eagerly in the main process so a bad name fails before spawning.
        get_facility(facility_name)
        assert n_envs >= 1
        self.n_envs = n_envs
        ctx = mp.get_context(start_method)
        self.remotes: list = []
        self.processes: list = []
        for i in range(n_envs):
            parent_remote, child_remote = ctx.Pipe()
            # Space worker seeds far apart so the per-worker dwell-sampler /
            # task-stream RNGs don't collide.
            seed = base_seed * 10_000 + i * 100 + 1
            p = ctx.Process(
                target=_worker,
                args=(
                    child_remote, experiment_config, reward_config, seed,
                    facility_name, episode_config,
                ),
                daemon=True,
            )
            p.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(p)
        self._closed = False

    def reset(self, seeds: Optional[list[int]] = None) -> list[Sample]:
        if seeds is None:
            seeds = [i + 1 for i in range(self.n_envs)]
        assert len(seeds) == self.n_envs
        for r, s in zip(self.remotes, seeds):
            r.send(("reset", int(s)))
        return [r.recv() for r in self.remotes]

    def step(self, actions: list[int]) -> list[StepResult]:
        assert len(actions) == self.n_envs
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        return [r.recv() for r in self.remotes]

    def set_episode_config(self, episode_config: EpisodeConfig) -> None:
        """Push a new EpisodeConfig to every worker. Consumed on next reset."""
        for r in self.remotes:
            r.send(("set_episode_config", episode_config))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        for r in self.remotes:
            try:
                r.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
