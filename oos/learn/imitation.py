"""Solver-guided imitation (behaviour cloning) bootstrap for the omni policy.

Plain PPO + shaping + curriculum hits a hard exploration wall at the very first
DIG: removing even one blocker to reach a buried target is a ~5-action maneuver
(TAKE blocker → carry to another shelf → GIVE → return → TAKE target) with no
shaping until the target is grabbed, so the policy collapses onto the no-dig
cases and never discovers it. The complete A* solver, by contrast, digs (and
buffer-on-target put-backs) optimally. We use it as an expert to BEHAVIOUR-CLONE
the dig/stage/settle skill into the net, then PPO fine-tunes from there.

How a demo is produced (order-independent SNAPSHOT cloning — replaying the
solver's per-carrier command stream through the env desyncs on handoffs, since the
solver assumes a specific global interleaving; snapshots avoid that entirely):
  1. Build a difficulty-controlled forced layout and reset the env.
  2. Run the complete solver on the env's engine (execute every retrieve, then
     ensure_staged + recover). Instrument `submit`/`wait` so that, the instant
     BEFORE each primitive executes, we snapshot `(obs for the acting carrier in
     the current engine state, the action index of that primitive)`. Every pair is
     taken at the true state the solver was in, so it needs no trajectory replay.
  3. After the solve reaches clean-rest, snapshot a WAIT pair for every carrier
     (teaches the all-settle / idle discipline the solver doesn't issue commands
     for).

`bc_train` then minimises masked cross-entropy of the net's action logits against
the expert action. `python -m oos.learn.imitation` runs collection + BC and saves
a checkpoint that `train.py --resume` warm-starts from.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from oos.config.schema import EpisodeConfig, ExperimentConfig
from oos.env import hardcases as hc
from oos.env.action import ActionType, enumerate_actions
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.env.action import ActionDecoder
from oos.env.observation import ObservationBuilder, ObservationConfig
from oos.learn.batching import GraphCollator, Sample
from oos.learn.checkpoint import save_checkpoint
from oos.learn.curriculum import default_tiers
from oos.learn.net import build_net
from oos.sim.actions import Give, Goto, Take
from oos.sim.tasks import Retrieve
from oos.solver.solver import Solver
from oos.solver.world import World


def make_env(fac, max_steps):
    exp = ExperimentConfig(episode=EpisodeConfig(max_steps=max_steps, max_sim_time=3600.0))
    return RetrieveEnv(
        facility_factory=fac, omni=True, require_noroom_empty=False,
        require_all_waiting=True, target_any_shelf=True, fullness=-1.0,
        reward_deliver=0.0, reward_success=15.0, reward_gamma=1.0,
        penalty_all_wait_while_task=1.0, experiment_config=exp,
    )


def _cmd_key(cmd):
    """A hashable description of a solver primitive for matching to ActionEntries."""
    if isinstance(cmd, Goto):
        return ("GOTO", cmd.target.kind, cmd.target.id)
    if isinstance(cmd, Take):
        return ("TAKE",)
    if isinstance(cmd, Give):
        return ("GIVE",)
    return ("WAIT",)


def _entry_key(entry):
    if entry.type == ActionType.GOTO:
        return ("GOTO", entry.target.kind, entry.target.id)
    if entry.type == ActionType.TAKE:
        return ("TAKE",)
    if entry.type == ActionType.GIVE:
        return ("GIVE",)
    return ("WAIT",)


def _build_sample(obs, entries, n_max) -> Sample:
    mask = np.array(ActionDecoder(entries, n_max).mask(), dtype=np.int8)
    return Sample(
        carrier_features=obs["carrier_features"], shelf_features=obs["shelf_features"],
        room_features=obs["room_features"], global_features=obs["global_features"],
        edges_accesses=obs["edges_accesses"], edges_handoff=obs["edges_handoff"],
        edges_transfer=obs["edges_transfer"], edges_docked=obs["edges_docked"],
        action_mask=mask, action_entries=list(entries),
        querying_carrier=int(obs["querying_carrier"]),
    )


def snapshot_episode(env, obs_builder, n_max, guards):
    """Run the solver on the reset env's engine; snapshot (Sample, action_idx) the
    instant before each primitive executes. Returns (samples, actions, solved)."""
    eng = env.engine
    topo = eng.topology
    targets = list(env._target_ids)
    samples, actions, misses = [], [], [0]
    orig_submit, orig_wait = eng.submit, eng.wait

    def snap(carrier, key):
        entries = enumerate_actions(carrier, eng.state, topo, eng.queue,
                                    policy_guards=guards)
        ekeys = [_entry_key(e) for e in entries]
        if key not in ekeys:
            misses[0] += 1
            return
        obs = obs_builder.build(eng, eng.queue, carrier)
        samples.append(_build_sample(obs, entries, n_max))
        actions.append(ekeys.index(key))

    def rec_submit(cmd):
        snap(cmd.carrier, _cmd_key(cmd))
        return orig_submit(cmd)

    def rec_wait(cid):
        snap(cid, ("WAIT",))
        return orig_wait(cid)

    eng.submit, eng.wait = rec_submit, rec_wait
    try:
        solver = Solver(eng, World(topo))
        for t in targets:
            solver.execute_retrieve(t)
        solver.ensure_staged()
        solver.recover()
        solver.ensure_staged()
    finally:
        eng.submit, eng.wait = orig_submit, orig_wait

    delivered = not any(isinstance(t, Retrieve) for t in eng.queue.pending)
    solved = delivered and env._park_all_staged(eng)
    # Teach the final all-settle: every carrier should WAIT at clean-rest.
    if solved:
        for cid in eng.state.carriers:
            entries = enumerate_actions(cid, eng.state, topo, eng.queue,
                                        policy_guards=guards)
            ekeys = [_entry_key(e) for e in entries]
            if ("WAIT",) in ekeys:
                obs = obs_builder.build(eng, eng.queue, cid)
                samples.append(_build_sample(obs, entries, n_max))
                actions.append(ekeys.index(("WAIT",)))
    return samples, actions, solved, misses[0]


def collect_demos(fac, specs, n_per_spec, max_steps, seed0=0, guards=True, verbose=True):
    topo, _ = fac()
    collator = GraphCollator(topo)
    env = make_env(fac, max_steps)
    env._policy_guards = guards
    obs_builder = ObservationBuilder(topo, ObservationConfig())
    n_max = env.n_actions
    all_s, all_a = [], []
    ok = tot = miss = 0
    t0 = time.time()
    for si, spec in enumerate(specs):
        for k in range(n_per_spec):
            sd = seed0 + si * 1000 + k
            env.set_forced_layout(hc.case_builder(spec, seed=sd))
            env.reset(seed=sd)
            samples, actions, solved, misses = snapshot_episode(env, obs_builder, n_max, guards)
            tot += 1
            miss += misses
            if solved:
                ok += 1
                all_s.extend(samples)
                all_a.extend(actions)
        if verbose:
            print(f"  [{si+1}/{len(specs)}] {spec.label:34s} "
                  f"{ok}/{tot} solved, {len(all_s)} pairs, {miss} unmatched "
                  f"({time.time()-t0:.0f}s)")
    env.set_forced_layout(None)
    print(f"[demos] {ok}/{tot} solver-solved episodes; {len(all_s)} (obs,action) "
          f"pairs; {miss} commands unmatched (guard-masked/illegal)")
    return all_s, all_a, collator, n_max


def bc_train(net, collator, samples, actions, n_max, *, epochs=8, batch=256,
             lr=1e-3, device="cpu"):
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    n = len(samples)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot_loss = tot_acc = nb = 0
        net.train()
        for st in range(0, n, batch):
            mb = idx[st:st + batch]
            b = collator.collate([samples[i] for i in mb], n_max=n_max, device=device)
            tgt = actions_t[mb]
            out = net(b)
            logits = out.logits
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                acc = (logits.argmax(-1) == tgt).float().mean().item()
            tot_loss += loss.item(); tot_acc += acc; nb += 1
        print(f"  BC epoch {ep+1}/{epochs}: loss={tot_loss/nb:.4f} acc={tot_acc/nb:.3f}")
    return net


import copy

from oos.solver.relocate import plan_dig  # noqa: E402  (kept near solver imports)


def solver_oracle_action(engine, C, guards):
    """The expert action INDEX for the querying carrier C in `engine`'s current
    state: clone the engine, run the complete solver to clean-rest recording every
    carrier's ORDERED command list, and return the index of C's first command that
    is LEGAL right now (else WAIT). Re-planning from the true state each call avoids
    the global-schedule desync that breaks naive replay; skipping
    already-satisfied/no-op commands avoids the livelock where C's stale first
    command (e.g. GOTO to where it already is) maps to nothing."""
    clone = copy.deepcopy(engine)
    targets = [t.pallet for t in clone.queue.pending if isinstance(t, Retrieve)]
    per = {cid: [] for cid in clone.state.carriers}
    os_, ow_ = clone.submit, clone.wait

    def rs(cmd):
        per[cmd.carrier].append(_cmd_key(cmd)); return os_(cmd)

    def rw(cid):
        return None  # solver WAITs inferred from "no legal command" below

    clone.submit, clone.wait = rs, rw
    try:
        s = Solver(clone, World(clone.topology))
        for t in targets:
            s.execute_retrieve(t)
        s.ensure_staged()
        s.recover()
        s.ensure_staged()
    except Exception:
        pass
    ent = enumerate_actions(C, engine.state, engine.topology, engine.queue,
                            policy_guards=guards)
    ek = [_entry_key(e) for e in ent]
    for key in per.get(C, []):
        if key in ek:
            return ek.index(key)
    return ek.index(("WAIT",)) if ("WAIT",) in ek else len(ent) - 1


def bc_anchor(net, opt, collator, samples, actions_t, n_max, *, n, batch=256, device="cpu"):
    """One BC-anchor pass over a random `n`-sample subset of the demos: masked
    cross-entropy of the policy logits vs the solver action. Used INTERLEAVED with
    PPO so the dig/stage behaviour the policy can't explore on its own never decays
    (POfD/DQfD-style demonstration anchoring). Touches only the policy via CE; the
    value head is left to PPO. Returns (mean_loss, mean_acc)."""
    m = len(samples)
    sub = np.random.randint(0, m, size=min(n, m))
    net.train()
    tl = ta = nb = 0
    for st in range(0, len(sub), batch):
        mb = sub[st:st + batch]
        b = collator.collate([samples[i] for i in mb], n_max=n_max, device=device)
        tgt = actions_t[mb]
        logits = net(b).logits
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            ta += (logits.argmax(-1) == tgt).float().mean().item()
        tl += loss.item(); nb += 1
    return (tl / nb, ta / nb) if nb else (0.0, 0.0)


def all_curriculum_specs():
    seen, specs = set(), []
    for t in default_tiers():
        for sp in t.specs:
            if sp.label not in seen:
                seen.add(sp.label); specs.append(sp)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--per-spec", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--out", default="runs/bc/bc.pt")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    fac = get_facility(args.facility)
    # All curriculum specs across all tiers (incl. put-back), deduped.
    seen, specs = set(), []
    for t in default_tiers():
        for sp in t.specs:
            if sp.label not in seen:
                seen.add(sp.label); specs.append(sp)
    print(f"[imitation] {len(specs)} unique specs × {args.per_spec} = "
          f"{len(specs)*args.per_spec} demo episodes")

    samples, actions, collator, n_max = collect_demos(
        fac, specs, args.per_spec, args.max_steps)
    if not samples:
        raise SystemExit("no demos collected — solver/replay failed")

    net, net_cfg, feat_dims = build_net(hidden=args.hidden, n_heads=4,
                                        n_gat_layers=args.gat_layers)
    bc_train(net, collator, samples, actions, n_max,
             epochs=args.epochs, lr=args.lr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_checkpoint(args.out, net=net, optimizer=torch.optim.Adam(net.parameters()),
                    net_cfg=net_cfg, feat_dims=feat_dims, iteration=0,
                    extra=dict(bc=True))
    print(f"[imitation] saved BC checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
