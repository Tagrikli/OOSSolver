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
from oos.env.env import OOSEnv
from oos.env.reward import RewardConfig
from oos.facilities import get_facility, make_facility
from oos.learn.batching import Sample, sample_from_env_step


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


def _worker(
    remote,
    experiment_config: ExperimentConfig,
    reward_config: RewardConfig,
    seed: int,
    facility_name: str,
    retrieve_only_config=None,
) -> None:
    """Per-worker loop. Owns one env (OOSEnv or RetrieveOnlyEnv). Handles
    step/reset/close commands.

    If `retrieve_only_config` is provided, the worker builds a RetrieveOnlyEnv
    that randomizes pallet distribution + target retrieve per episode.
    """
    if retrieve_only_config is None:
        env = OOSEnv(
            facility_factory=get_facility(facility_name),
            experiment_config=experiment_config,
            reward_config=reward_config,
        )
    else:
        # Imported lazily so workers that don't need it don't pay the cost.
        from oos.learn.retrieve_env import RetrieveOnlyEnv
        env = RetrieveOnlyEnv(
            facility_factory=get_facility(facility_name),
            retrieve_only_config=retrieve_only_config,
            experiment_config=experiment_config,
            reward_config=reward_config,
        )
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
            if done:
                terminal_sample = sample_from_env_step(obs, info, info["action_entries"])
                er, el, ec = ep_return, ep_length, ep_completions
                est = float(info.get("sim_time", 0.0))
                ecc = ep_constraint_cost
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
        elif cmd == "swap_layout":
            # Rebuild this worker's env against a freshly-generated random
            # facility. All workers receive the same `layout_seed` so they
            # agree on the topology — the main process can then build one
            # collator that fits every worker's samples this iteration.
            # `roc_override` (if not None) replaces the stored
            # retrieve_only_config — used to randomize fullness etc. per cycle.
            from oos.facilities.random_gen import make_random_facility
            layout_seed = int(msg[1])
            reset_seed = int(msg[2])
            roc_override = msg[3] if len(msg) > 3 else None
            if roc_override is not None:
                retrieve_only_config = roc_override
            factory = (lambda s=layout_seed: make_random_facility(seed=s))
            if retrieve_only_config is None:
                env = OOSEnv(
                    facility_factory=factory,
                    experiment_config=experiment_config,
                    reward_config=reward_config,
                )
            else:
                from oos.learn.retrieve_env import RetrieveOnlyEnv
                env = RetrieveOnlyEnv(
                    facility_factory=factory,
                    retrieve_only_config=retrieve_only_config,
                    experiment_config=experiment_config,
                    reward_config=reward_config,
                )
            obs, info = env.reset(seed=reset_seed)
            ep_return = 0.0
            ep_length = 0
            ep_completions = 0
            ep_constraint_cost = 0.0
            next_seed = reset_seed + 1_000_003
            sample = sample_from_env_step(obs, info, info["action_entries"])
            remote.send(sample)
        elif cmd == "set_roc":
            # Replace the env's retrieve_only_config in place. Used to update
            # fullness/etc. per iteration without swapping the facility layout.
            roc_new = msg[1]
            retrieve_only_config = roc_new
            # RetrieveOnlyEnv reads _retrieve_cfg at reset() time.
            env._retrieve_cfg = roc_new  # type: ignore[attr-defined]
        elif cmd == "close":
            try:
                remote.close()
            except Exception:
                pass
            return
        else:
            raise ValueError(f"unknown vec_env cmd: {cmd!r}")


class VecEnv:
    """N worker processes, one OOSEnv each."""

    def __init__(
        self,
        n_envs: int,
        experiment_config: ExperimentConfig,
        reward_config: RewardConfig | None = None,
        base_seed: int = 0,
        start_method: str = "spawn",
        facility_name: str = "dev",
        retrieve_only_config=None,
    ):
        if reward_config is None:
            reward_config = RewardConfig()
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
                    facility_name, retrieve_only_config,
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

    def swap_to_random_layout(
        self,
        layout_seed: int,
        reset_seed_base: int = 0,
        retrieve_only_config=None,
    ) -> list[Sample]:
        """Tell every worker to rebuild its env around a fresh random facility
        seeded with `layout_seed`. All workers use the same `layout_seed` so
        they agree on the topology (lets the main process use a single
        collator for this cycle). Each worker then resets with a per-worker
        seed `reset_seed_base + i + 1` so episode RNG stays distinct.

        `retrieve_only_config` (if given) replaces the per-worker config —
        useful for randomizing fullness etc. per cycle.

        Returns the post-reset Samples in worker order — hand this back as
        `initial_samples` to the next `collect_rollout_vec` call.
        """
        for i, r in enumerate(self.remotes):
            r.send((
                "swap_layout",
                int(layout_seed),
                int(reset_seed_base) + i + 1,
                retrieve_only_config,
            ))
        return [r.recv() for r in self.remotes]

    def set_retrieve_config(self, retrieve_only_config) -> None:
        """Update every worker's retrieve_only_config in place. No reset, no
        layout change — the new config is consumed on the next env.reset()."""
        for r in self.remotes:
            r.send(("set_roc", retrieve_only_config))

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
