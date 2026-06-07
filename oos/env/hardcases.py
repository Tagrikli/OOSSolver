"""Difficulty-controlled hard-case construction + a held-out test battery for
`tiny_medipol` retrieves — the coverage testbed.

`tiny_medipol` is 45 pallets in 48 slots (only 3 free), so **eviction room is the
binding difficulty axis**. A single retrieve is hard exactly when a BIG target is
buried under BIG blockers with too few free BIG slots to evict them into. We build
cases on a clean canonical form so difficulty is *exact and measurable*:

    target shelf S (big, of a chosen route), filled to capacity, bottom -> top:
        [ empty x (cap-1-D) ]  [ TARGET (big) ]  [ K big blockers ]  [ (D-K) empty ]

  so the target sits at depth D with K big pallets immediately in front of it
  (K>=1 is the "buffer-on-target" pathology). Every OTHER big shelf is filled with
  empties leaving exactly `free_big` free slots across them; small shelves absorb
  the remaining pallets. The ONLY big pallets in the whole facility are the target
  + its K big blockers, so retrieve feasibility reduces to the clean inequality

        solvable  <=>  K <= free_big          (slack = free_big - K)

  every big blocker needs a free big slot to evict into. slack==0 is the hardest
  solvable case; slack<0 is unsolvable (excluded from the battery).

`measure_difficulty` recomputes (depth, bigs_in_front, free_other_bigs, slack,
routing) independently from the live state, so every construction is checked
against ground truth. The same builders drive the curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oos.env import targeting
from oos.sim.scheduler import Scheduler
from oos.sim.state import Pallet


@dataclass(frozen=True)
class CaseSpec:
    """One difficulty point. `depth` D = blockers in front of the target;
    `n_big_blockers` K = how many of those are big; `free_big` = free slots left
    on the OTHER big shelves; `routing` = the target shelf's route class."""
    depth: int            # D in 0..(cap-1) = 0..2
    n_big_blockers: int   # K in 0..D
    free_big: int         # 0..3 (only 3 free slots exist)
    routing: str          # 'direct' | 'handoff'
    big_fill: int = 0     # # of OTHER-big-shelf occupied slots that hold BIG
                          # content (global congestion). 0 = clean (empties); high
                          # = the packed medipol puzzle. Does NOT change our
                          # target's feasibility (free_other_bigs is unchanged),
                          # only the surrounding density the policy must reason over.

    @property
    def slack(self) -> int:
        return self.free_big - self.n_big_blockers

    @property
    def solvable(self) -> bool:
        return self.free_big >= self.n_big_blockers

    @property
    def label(self) -> str:
        return (f"d{self.depth}-k{self.n_big_blockers}-f{self.free_big}"
                f"-{self.routing}-slack{self.slack}-bf{self.big_fill}")


def _shelf_info(facility):
    topo = facility.topology
    rc = targeting.route_class_map(topo)
    big = [sid for sid, s in topo.shelves.items() if s.size_class == "big"]
    small = [sid for sid, s in topo.shelves.items() if s.size_class == "small"]
    cap = {sid: topo.shelves[sid].capacity for sid in topo.shelves}
    return rc, big, small, cap


def _wipe(facility) -> None:
    """Clear all shelves + carriers to the bare initial condition (mirrors
    `shuffle._place_pallets`'s wipe). Auto-arrivals are already off in the env's
    setup, so no arrival is rescheduled."""
    for ss in facility.state.shelves.values():
        ss.stack = []
    for cs in facility.state.carriers.values():
        cs.load = None
        cs.current_command = None
        cs.busy_until = None
        cs.command_started_at = None
        cs.command_start_position = None
        cs.docked_at = None
        cs.last_take_give = None
        cs.came_from = None
        cs.waiting = False
    facility.scheduler = Scheduler()
    if facility.auto_arrivals_enabled:
        facility._schedule_next_arrival()


def build_layout(facility, spec: CaseSpec, rng: np.random.Generator) -> int:
    """Build `spec`'s canonical layout in `facility` (mutated in place) and return
    the target pallet id. Conserves the pallet pool exactly. Raises on an
    infeasible spec."""
    if not (0 <= spec.n_big_blockers <= spec.depth):
        raise ValueError(f"need 0<=K<=D, got K={spec.n_big_blockers} D={spec.depth}")
    rc, big, small, cap = _shelf_info(facility)
    state = facility.state
    ids = [p.id for ss in state.shelves.values() for p in ss.stack]
    ids += [cs.load.id for cs in state.carriers.values() if cs.load is not None]
    n_pool = len(ids)
    _wipe(facility)
    rng.shuffle(ids)
    ids = list(ids)

    def take() -> int:
        return ids.pop()

    # ---- target shelf S: big, of the requested route, filled to capacity ----
    cand = [sid for sid in big if rc[sid] == spec.routing]
    if not cand:
        raise ValueError(f"no big '{spec.routing}' shelf")
    S = cand[int(rng.integers(len(cand)))]
    capS = cap[S]
    D, K = spec.depth, spec.n_big_blockers
    if D > capS - 1:
        raise ValueError(f"depth {D} exceeds shelf cap-1 ({capS - 1})")
    stackS = [Pallet(take(), "empty") for _ in range(capS - 1 - D)]   # below target
    target_id = take()
    stackS.append(Pallet(target_id, "big"))                          # the target
    stackS += [Pallet(take(), "big") for _ in range(K)]              # big buffer(s)
    stackS += [Pallet(take(), "empty") for _ in range(D - K)]        # empty blockers
    state.shelves[S].stack = stackS

    # ---- other big shelves: `big_fill` BIG (congestion) + empties, leaving
    #      exactly `free_big` free slots. The bigs sit on OTHER shelves so they
    #      don't block OUR target (free_other_bigs is unchanged) — they only pack
    #      the facility so the policy must reason over a dense big state. ----
    other_big = [sid for sid in big if sid != S]
    to_fill = sum(cap[sid] for sid in other_big) - spec.free_big
    if to_fill < 0:
        raise ValueError(f"free_big {spec.free_big} exceeds other-big capacity")
    n_big_other = min(spec.big_fill, to_fill)
    placed = 0
    placed_big = 0
    for sid in other_big:
        stk = []
        while len(stk) < cap[sid] and placed < to_fill:
            contents = "big" if placed_big < n_big_other else "empty"
            stk.append(Pallet(take(), contents))
            placed_big += 1 if contents == "big" else 0
            placed += 1
        state.shelves[sid].stack = stk

    # ---- small shelves: absorb every remaining pallet (empties) ----
    for sid in small:
        stk = []
        while len(stk) < cap[sid] and ids:
            stk.append(Pallet(take(), "empty"))
        state.shelves[sid].stack = stk

    if ids:
        raise RuntimeError(f"{len(ids)} pallets unplaced — spec infeasible for pool {n_pool}")
    return target_id


def measure_difficulty(facility, target_id: int) -> dict:
    """Recompute the ground-truth difficulty of `target_id` from the live state —
    the independent check that construction did what the spec asked."""
    state = facility.state
    rc, big, small, cap = _shelf_info(facility)
    tsid = tidx = None
    for sid, ss in state.shelves.items():
        for i, p in enumerate(ss.stack):
            if p.id == target_id:
                tsid, tidx = sid, i
    if tsid is None:
        raise ValueError(f"target {target_id} not on any shelf")
    stack = state.shelves[tsid].stack
    depth = (len(stack) - 1) - tidx
    bigs_in_front = sum(1 for j in range(tidx + 1, len(stack)) if stack[j].contents == "big")
    free_other_bigs = sum(cap[sid] - len(state.shelves[sid].stack)
                          for sid in big if sid != tsid)
    return {
        "depth": depth,
        "bigs_in_front": bigs_in_front,
        "free_other_bigs": free_other_bigs,
        "slack": free_other_bigs - bigs_in_front,
        "routing": rc[tsid],
        "target_size": stack[tidx].contents,
        "solvable": bigs_in_front <= free_other_bigs,
    }


def case_builder(spec: CaseSpec, seed: int):
    """A forced-layout builder for `spec` — install via `env.set_forced_layout`.
    Deterministic per (spec, seed)."""
    def build(env, facility):
        rng = np.random.default_rng(seed)
        tid = build_layout(facility, spec, rng)
        env._task_type = "retrieve"
        env._seed_retrieve(facility, tid, spec.depth)
    return build


def eval_on_battery(net, collator, env, specs, *, n_seeds=3, max_steps=200,
                    device="cpu") -> dict[str, float]:
    """Greedy (argmax) success rate of `net` on each spec in `specs`, averaged over
    `n_seeds` fixed seeds. Returns {spec.label: rate}. `env`'s success condition
    (loose vs strict clean-terminal) decides what 'solved' means — set it to match
    what you're measuring. Restores the env's sampler (clears the forced layout) at
    the end."""
    import torch

    from oos.learn.batching import sample_from_env_step
    net.eval()
    n_max = env.n_actions
    out: dict[str, float] = {}
    for spec in specs:
        wins = 0
        for seed in range(n_seeds):
            env.set_forced_layout(case_builder(spec, seed=10_000 + seed))
            obs, info = env.reset(seed=20_000 + seed)
            for _ in range(max_steps):
                s = sample_from_env_step(obs, info, info["action_entries"])
                b = collator.collate([s], n_max=n_max, device=device)
                with torch.no_grad():
                    a = int(net(b).logits[0].argmax().item())
                obs, _r, term, trunc, info = env.step(a)
                if info.get("success", False):
                    wins += 1
                    break
                if term or trunc:
                    break
        out[spec.label] = wins / n_seeds
    env.set_forced_layout(None)
    return out


def default_battery() -> list[CaseSpec]:
    """A held-out grid spanning the difficulty space: every (depth, K, free_big,
    routing) that is well-formed (K<=D) and SOLVABLE (free_big>=K), both routes.
    Includes the hardest solvable corners (slack==0) and buffer-on-target (K>=1)."""
    specs: list[CaseSpec] = []
    for routing in ("direct", "handoff"):
        for D in (0, 1, 2):
            for K in range(0, D + 1):
                for free_big in range(0, 4):
                    spec = CaseSpec(D, K, free_big, routing)
                    if spec.solvable:
                        specs.append(spec)
    return specs
