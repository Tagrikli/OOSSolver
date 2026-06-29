"""Run a trained omni policy CONTINUOUSLY on `tiny_medipol` — a live store/retrieve
stream — and measure deployment behaviour.

A continuous facility is just a never-ending sequence of the mini-tasks the omni
policy was trained on: stores arrive (Poisson), each parked car dwells then becomes
a retrieve, and between tasks the policy must hold every room staged and every
carrier idle. We drive the base `Environment` (auto-arrivals ON) with the greedy
policy and report:

  * store wait     — arrival → room available with an empty pallet (the serve);
  * retrieve wait  — request → item delivered to a room;
  * deadlocks / undelivered tasks (should be zero on the admitted stream);
  * idle discipline — fraction of decisions that are WAIT when nothing is pending
    (the "no movement when there is no task" requirement) and staging uptime.

SUV admission: a big (SUV) store is admitted only if, after hypothetically landing
it, EVERY item in the facility is still retrievable — checked by the SOUND solver
oracle (`plan_dig`), not the optimistic `_layout_is_solvable`. Rejected SUVs are
dropped (the policy is never asked to dig an unretrievable layout).

Run:  python -m oos.learn.continuous --ckpt runs/omni/best.pt --sim-time 6000 --store-rate 0.05
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from oos.config.schema import (DurationsConfig, EpisodeConfig, ExperimentConfig,
                               TaskStreamConfig)
from oos.env.env import Environment
from oos.facilities import get_facility
from oos.learn.acceptance import load_net
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.sim.state import Pallet
from oos.sim.tasks import Retrieve, Store


def install_sound_suv_gate(engine):
    """Gate big (SUV) stores on the SOUND retrievability oracle. Overrides the
    instance's `_big_admission_ok` with a plan_dig-based check and turns the gate
    on. Returns nothing (mutates the engine)."""
    from oos.solver.solver import Solver
    from oos.solver.world import World

    world = World(engine.topology)

    def sound_ok():
        topo, state = engine.topology, engine.state
        target = next(
            (sid for sid, sh in topo.shelves.items()
             if sh.size_class == "big" and state.shelves[sid].depth < sh.capacity),
            None,
        )
        if target is None:
            return False
        stack = state.shelves[target].stack
        stack.append(Pallet(id=-99, contents="big"))
        try:
            return Solver(engine, world)._all_retrievable()
        finally:
            stack.pop()

    engine._big_admission_ok = sound_ok
    engine.gate_big_retrievability = True


def greedy_action(net, collator, obs, info, n_max, device):
    s = sample_from_env_step(obs, info, info["action_entries"])
    b = collator.collate([s], n_max=n_max, device=device)
    with torch.no_grad():
        return int(net(b).logits[0].argmax().item())


def evaluate_continuous(net, collator, fac, n_max, *, sim_time=6000.0,
                        store_rate=0.014, big_frac=0.15, mean_dwell=250.0,
                        seed=777, suv_gate=True, device="cpu"):
    """Run the greedy policy on a live stream and return deployment metrics:
    store/retrieve wait (mean), delivered/served rates, staging uptime, and the
    idle-move rate (fraction of decisions that MOVE while no task is pending AND
    every room is already staged — the truly-wasteful 'no movement when idle'
    violation; productive staging when a room is unstaged is NOT counted)."""
    exp = ExperimentConfig(
        durations=DurationsConfig(),
        task_stream=TaskStreamConfig(
            store_rate=store_rate, size_mix={"small": 1 - big_frac, "big": big_frac},
            mean_dwell_seconds=mean_dwell, std_dwell_seconds=mean_dwell / 3.0),
        episode=EpisodeConfig(max_sim_time=sim_time, max_steps=10_000_000))
    env = Environment.from_name(fac if isinstance(fac, str) else "tiny_medipol",
                                experiment_config=exp) if isinstance(fac, str) \
        else _env_from_factory(fac, exp)
    obs, info = env.reset(seed=seed)
    if suv_gate:
        install_sound_suv_gate(env.engine)
    rc = env._room_carriers
    store_costs, ret_costs = [], []
    n_store_arr = n_ret_arr = 0
    idle_move = idle_dec = staged_hits = staged_samples = 0
    while True:
        pending = len(env.engine.queue.pending)
        staged = sum(
            1 for cid in rc
            if (cs := env.engine.state.carriers[cid]).docked_at is not None
            and cs.docked_at.kind == "room" and cs.load is not None and cs.load.is_empty)
        staged_samples += len(rc); staged_hits += staged
        a = greedy_action(net, collator, obs, info, n_max, device)
        is_wait = info["action_entries"][a].type.name == "WAIT"
        all_staged = (staged == len(rc))
        if pending == 0 and all_staged:
            idle_dec += 1
            idle_move += int(not is_wait)
        obs, _r, term, trunc, info = env.step(a)
        for c in info.get("completions", []):
            if isinstance(c.task, Store):
                store_costs.append(c.cost)
            elif isinstance(c.task, Retrieve):
                ret_costs.append(c.cost)
        for t in info.get("arrivals", []):
            if isinstance(t, Store): n_store_arr += 1
            elif isinstance(t, Retrieve): n_ret_arr += 1
        if trunc or term:
            break
    return dict(
        store_wait_mean=float(np.mean(store_costs)) if store_costs else 0.0,
        retrieve_wait_mean=float(np.mean(ret_costs)) if ret_costs else 0.0,
        store_serve_rate=len(store_costs) / max(1, n_store_arr),
        deliver_rate=len(ret_costs) / max(1, n_ret_arr),
        staging_uptime=staged_hits / max(1, staged_samples),
        idle_move_rate=idle_move / max(1, idle_dec),
        n_stores=n_store_arr, n_retrieves=n_ret_arr,
    )


def _env_from_factory(fac, exp):
    from oos.env.env import Environment as _E
    return _E(facility_factory=fac, experiment_config=exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/omni/best.pt")
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--sim-time", type=float, default=6000.0)
    ap.add_argument("--store-rate", type=float, default=0.05)
    ap.add_argument("--big-frac", type=float, default=0.15)
    ap.add_argument("--mean-dwell", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suv-gate", action="store_true", default=True)
    ap.add_argument("--no-suv-gate", dest="suv_gate", action="store_false")
    ap.add_argument("--trace", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"
    fac = get_facility(args.facility)
    topo, _ = fac()
    collator = GraphCollator(topo)
    net, ckpt = load_net(args.ckpt, device)
    n_max_ckpt = None

    exp = ExperimentConfig(
        durations=DurationsConfig(),
        task_stream=TaskStreamConfig(
            store_rate=args.store_rate,
            size_mix={"small": 1.0 - args.big_frac, "big": args.big_frac},
            mean_dwell_seconds=args.mean_dwell,
            std_dwell_seconds=args.mean_dwell / 3.0,
        ),
        episode=EpisodeConfig(max_sim_time=args.sim_time, max_steps=10_000_000),
    )
    env = Environment.from_name(args.facility, experiment_config=exp)
    obs, info = env.reset(seed=args.seed)
    if args.suv_gate:
        install_sound_suv_gate(env.engine)
    n_max = env.n_actions

    store_costs, retrieve_costs = [], []
    n_store_arr = n_ret_arr = n_dropped = 0
    idle_decisions = idle_moves = 0   # decisions taken while no task pending
    staged_samples = staged_hits = 0
    steps = 0

    print(f"[continuous] ckpt={args.ckpt} iter={ckpt.get('iteration')} "
          f"sim_time={args.sim_time} store_rate={args.store_rate} "
          f"big_frac={args.big_frac} suv_gate={args.suv_gate}")

    while True:
        # idle discipline: is any task pending right now?
        pending = len(env.engine.queue.pending)
        # staging uptime sample: how many rooms staged
        rc = env._room_carriers
        staged = sum(
            1 for cid in rc
            if (cs := env.engine.state.carriers[cid]).docked_at is not None
            and cs.docked_at.kind == "room" and cs.load is not None and cs.load.is_empty
        )
        staged_samples += len(rc)
        staged_hits += staged

        a = greedy_action(net, collator, obs, info, n_max, device)
        entry = info["action_entries"][a]
        is_wait = entry.type.name == "WAIT"
        # Truly-wasteful idle movement = moving while NO task is pending AND every
        # room is already staged. Moving to stage an unstaged room is productive and
        # is not counted.
        if pending == 0 and staged == len(rc):
            idle_decisions += 1
            if not is_wait:
                idle_moves += 1

        obs, r, term, trunc, info = env.step(a)
        steps += 1
        for c in info.get("completions", []):
            if isinstance(c.task, Store):
                store_costs.append(c.cost)
            elif isinstance(c.task, Retrieve):
                retrieve_costs.append(c.cost)
        for t in info.get("arrivals", []):
            if isinstance(t, Store):
                n_store_arr += 1
            elif isinstance(t, Retrieve):
                n_ret_arr += 1
        n_dropped += len(info.get("dropped", []))
        if args.trace and steps <= args.trace:
            print(f"  {steps:4d} t={env.sim_time:7.1f} q={env.querying_carrier} "
                  f"{entry.type.name:5s} pend={pending} compl={len(info.get('completions',[]))}")
        if trunc or term:
            break

    # final: count still-pending (undelivered) tasks
    pend = env.engine.queue.pending
    undelivered_ret = sum(1 for t in pend if isinstance(t, Retrieve))
    undelivered_sto = sum(1 for t in pend if isinstance(t, Store))

    def stats(xs):
        if not xs:
            return "n=0"
        a = np.array(xs)
        return (f"n={len(a)} mean={a.mean():.1f}s p50={np.median(a):.1f} "
                f"p95={np.percentile(a,95):.1f} max={a.max():.1f}")

    print(f"\n=== CONTINUOUS RESULTS ({steps} decisions, {env.sim_time:.0f} sim-s) ===")
    print(f"stores:    arrived≈{n_store_arr} served={len(store_costs)} "
          f"dropped(SUV-reject/cap)={n_dropped} undelivered={undelivered_sto}")
    print(f"  store wait:    {stats(store_costs)}")
    print(f"retrieves: arrived={n_ret_arr} delivered={len(retrieve_costs)} "
          f"undelivered={undelivered_ret}")
    print(f"  retrieve wait: {stats(retrieve_costs)}")
    deliv_rate = (len(retrieve_costs) / n_ret_arr) if n_ret_arr else float('nan')
    print(f"  retrieve delivered-rate: {deliv_rate:.4f}")
    print(f"idle discipline: {idle_moves}/{idle_decisions} decisions MOVED while no task "
          f"pending ({(idle_moves/idle_decisions if idle_decisions else 0):.4f}) "
          f"[want ~0]")
    print(f"staging uptime: {staged_hits}/{staged_samples} room-samples staged "
          f"({staged_hits/staged_samples:.4f})")


if __name__ == "__main__":
    main()
