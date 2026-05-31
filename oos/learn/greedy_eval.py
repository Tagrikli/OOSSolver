"""Greedy held-out evaluation for the continuous-PLR trainer.

The per-iteration `retrieves`/`stores` in `metrics.jsonl` come from *sampled*
actions on a PLR-chosen level, so they conflate two things that aren't skill:
(a) *which* level was drawn (an empty-system d0 churns 100+ stream retrievals;
a buried d2 handoff yields ~10), and (b) the policy's *entropy* (random actions
complete extra work). Staring at that number can't reveal whether the policy is
learning.

This module measures skill **directly**: it runs the policy GREEDILY (argmax,
no sampling) on a FIXED held-out grid of seeded digs with the exogenous stream
turned OFF, and reports how many digs it actually delivers. Same levels every
call + greedy + no stream ⇒ a clean learning curve immune to both confounds:

  * level confound — killed: the held-out grid never changes, and each level is
    rebuilt from a fixed seed so the *concrete* state is identical every eval.
  * entropy confound — killed: argmax, so nothing is completed by chance.

A rising `dig-solve %` here is real learning; a flat/zero one despite busy
sampled throughput is the "hollow sharpening" failure the project has hit
before. Logged to `greedy_metrics.jsonl`, TensorBoard (`eval/*`), the terminal,
and `progress.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from oos.config.schema import ExperimentConfig
from oos.env.env import FacilityFactory
from oos.env.reward import RewardConfig
from oos.learn._style import _C, C_ANCHOR, C_DIM, _color_success
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.continuous_env import ContinuousEnv
from oos.learn.network import PolicyValueNet
from oos.learn.single_task_env import SingleTaskConfig

# Held-out grid difficulty — fixed and moderate, so a change in solve-rate
# reflects POLICY change, not difficulty drift. Tune here if the benchmark
# saturates (all 100%) or floors (all 0%).
_GRID_SYSTEM_FULLNESS = 0.5
_GRID_BIG_SHELF_FULLNESS = 0.6
_GRID_BIG_RATIO = 0.5
_GRID_DISORDER = 0.6

_DEPTHS = (0, 1, 2)
_ROUTES = ("direct", "handoff")
_FROMS = ("big", "small")


@dataclass
class _HoldoutProvider:
    """A level_provider the evaluator drives by hand: it hands back whichever
    level was last `set`, with a monotonic id (the env only needs uniqueness)."""

    level: Optional[SingleTaskConfig] = None
    _id: int = 0

    def __call__(self) -> "tuple[SingleTaskConfig, int]":
        self._id += 1
        return self.level, self._id


def build_holdout_grid() -> "list[tuple[str, SingleTaskConfig]]":
    """The fixed benchmark: every (depth × route × shelf-class) at one moderate
    difficulty. ~12 levels; impossible combos are dropped at validation time."""
    levels: list[tuple[str, SingleTaskConfig]] = []
    for depth in _DEPTHS:
        for route in _ROUTES:
            for frm in _FROMS:
                name = f"d{depth}/{route[:4]}/{frm}"
                levels.append((name, SingleTaskConfig(
                    task="retrieve",
                    retrieve_from=frm,
                    retrieve_route=route,
                    target_depth=depth,
                    big_shelf_fullness=_GRID_BIG_SHELF_FULLNESS,
                    system_fullness=_GRID_SYSTEM_FULLNESS,
                    big_ratio=_GRID_BIG_RATIO,
                    big_disorder=_GRID_DISORDER,
                    small_disorder=_GRID_DISORDER,
                    room_state="empty",
                    require_solvable=True,
                    max_solvable_retries=50,
                )))
    return levels


class GreedyEvaluator:
    """Owns a private, stream-OFF `ContinuousEnv` and a frozen held-out grid.
    `evaluate(net)` greedily rolls the policy on each level and returns a
    summary dict. Side-effect-free w.r.t. training (separate env, no_grad,
    argmax ⇒ no RNG draw)."""

    def __init__(
        self,
        facility_factory: FacilityFactory,
        reward_config: RewardConfig,
        experiment_config: ExperimentConfig,
        collator: GraphCollator,
        n_max: int,
        device: "torch.device | str",
        max_steps: int,
        seed: int,
    ) -> None:
        self._provider = _HoldoutProvider()
        self._env = ContinuousEnv(
            facility_factory=facility_factory,
            level_provider=self._provider,
            reward_config=reward_config,
            experiment_config=experiment_config,
        )
        # Pure dig-solve probe: the ONLY task is the seeded retrieve, so a
        # completed retrieve unambiguously means "this dig was solved".
        self._env.stream_enabled = False
        self._collator = collator
        self._n_max = n_max
        self._device = device
        self._max_steps = max_steps
        self._seed = seed
        # Keep only levels that actually install a target on this facility
        # (e.g. a 'big' class needs big pallets present). Deterministic, so the
        # surviving set is stable across the whole run.
        self.levels = self._validate(build_holdout_grid())

    # -- internals -----------------------------------------------------------

    def _reset_into(self, level: SingleTaskConfig, seed: int):
        self._provider.level = level
        return self._env.reset(seed=seed)

    def _validate(self, candidates):
        valid = []
        for i, (name, lvl) in enumerate(candidates):
            self._reset_into(lvl, self._seed + i)
            if self._env.current_target_id is not None:
                valid.append((name, lvl))
        return valid

    def _greedy_solve(self, net: PolicyValueNet, obs, info):
        """Step greedily until the seeded dig is delivered or the cap is hit.
        Returns (solved, steps, sim_time)."""
        for t in range(self._max_steps):
            sample = sample_from_env_step(obs, info, info["action_entries"])
            batch = self._collator.collate([sample], n_max=self._n_max, device=self._device)
            with torch.no_grad():
                out = net(batch)
            action = int(out.logits[0].argmax().item())
            obs, _r, terminated, truncated, info = self._env.step(action)
            if int(info.get("retrieves_completed", 0)) >= 1:
                return True, t + 1, float(info.get("sim_time", 0.0))
            if terminated or truncated:
                break
        return False, self._max_steps, float(info.get("sim_time", 0.0))

    @staticmethod
    def _summarize(results: list[dict]) -> dict:
        n = len(results)
        n_solved = sum(r["solved"] for r in results)

        def rate(pred) -> Optional[float]:
            sub = [r for r in results if pred(r)]
            return (sum(x["solved"] for x in sub) / len(sub)) if sub else None

        solved_steps = [r["steps"] for r in results if r["solved"]]
        return {
            "n": n,
            "n_solved": n_solved,
            "solve_pct": (n_solved / n) if n else 0.0,
            "by_depth": {d: rate(lambda r, d=d: r["depth"] == d) for d in _DEPTHS},
            "by_route": {rt: rate(lambda r, rt=rt: r["route"] == rt) for rt in _ROUTES},
            "mean_steps_solved": (sum(solved_steps) / len(solved_steps)) if solved_steps else 0.0,
            "unsolved": [r["name"] for r in results if not r["solved"]],
            "levels": results,
        }

    # -- public --------------------------------------------------------------

    def evaluate(self, net: PolicyValueNet) -> dict:
        net.eval()
        results: list[dict] = []
        for i, (name, lvl) in enumerate(self.levels):
            obs, info = self._reset_into(lvl, self._seed + i)
            solved, steps, sim_t = self._greedy_solve(net, obs, info)
            results.append({
                "name": name, "depth": lvl.target_depth,
                "route": lvl.retrieve_route, "from": lvl.retrieve_from,
                "solved": solved, "steps": steps, "sim_time": sim_t,
            })
        return self._summarize(results)


# ── terminal rendering ──────────────────────────────────────────────────────

def _fmt_rate(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def print_eval(result: dict, it: int) -> None:
    """One bold summary line + an unsolved-levels line (the actionable bit)."""
    pct = result["solve_pct"]
    col = _color_success(pct)
    bd, br = result["by_depth"], result["by_route"]
    print(
        f"{_C.BOLD}{C_ANCHOR}▓ greedy eval · iter {it:<4d}{_C.RESET}  "
        f"{col}{result['n_solved']}/{result['n']} digs ({pct * 100:.0f}%){_C.RESET}   "
        f"{C_DIM}d0{_C.RESET} {_fmt_rate(bd[0])} {C_DIM}d1{_C.RESET} {_fmt_rate(bd[1])} "
        f"{C_DIM}d2{_C.RESET} {_fmt_rate(bd[2])}   "
        f"{C_DIM}direct{_C.RESET} {_fmt_rate(br['direct'])} "
        f"{C_DIM}handoff{_C.RESET} {_fmt_rate(br['handoff'])}   "
        f"{C_DIM}steps̄{_C.RESET} {result['mean_steps_solved']:.0f}"
    )
    if result["unsolved"]:
        print(f"  {C_DIM}▎ unsolved:{_C.RESET} {' '.join(result['unsolved'])}")
