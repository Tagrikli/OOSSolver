"""Train a PPO agent to run `tiny_medipol` to the clean resting state.

The deliverable behaviour (one policy, applied continuously):

  * retrieve any requested car from any shelf / fullness — direct OR handoff
    route, at any burial depth, including the buffer-on-target dig that needs a
    blocker parked on a carrier;
  * store arriving cars onto compatible shelves and re-stage the room;
  * keep every room staged (a room carrier docked at its room holding an empty
    pallet, ready for the next arrival);
  * when nothing is pending and every room is staged, leave every carrier idle
    (WAIT) — no redundant movement.

All of that is exactly `RetrieveEnv`'s OMNI success predicate:
    all requests delivered  ∧  every room staged  ∧  every non-room carrier
    empty  ∧  every carrier WAITing.
We anchor on it with a one-time terminal reward (+REWARD_SUCCESS) and shape the
path with a depth-agnostic binary-holds PBRS ladder (so the shaping never fights
the put-back). A continuous task stream is just a sequence of these mini-episodes
(stage → store/retrieve → settle), so a policy that reliably reaches clean-rest
from any configuration runs the facility live.

Coverage is two-pronged, mixed across the parallel envs:
  * CURRICULUM envs draw a difficulty-controlled forced layout from a slack-ladder
    `Curriculum`, escalating only on measured greedy mastery of a held-out battery
    (the hard structured digs the uniform sampler under-covers);
  * SAMPLED envs draw random omni episodes (random fullness, 0..2 preloaded cars to
    store, 0..2 requests at random depth) — the easy random middle plus the store /
    multi-task / pure-idle skills the curriculum cases don't exercise.

Run:  python -m oos.learn.train  [--iters N] [--run-dir runs/omni]  ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig
from oos.env import hardcases as hc
from oos.env.retrieve_env import RetrieveEnv
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator
from oos.learn.checkpoint import load_checkpoint, restore_into, save_checkpoint
from oos.learn.curriculum import Curriculum, default_tiers
from oos.learn.imitation import (all_curriculum_specs, bc_anchor, bc_train,
                                 collect_demos)
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector
from oos.sim.shuffle import shuffle_state
from oos.sim.state import DockRef, Pallet, pallet_depth


# --------------------------------------------------------------------------
# Reward / shaping knobs (the design; see module docstring).
# --------------------------------------------------------------------------
REWARD_SUCCESS = 15.0      # one-time terminal anchor on clean-rest (unfarmable)
# Binary-holds PBRS ladder Φ (depth-agnostic — never fights the put-back).
SHAPE = dict(
    shape_target_depth=1.5,               # DIG breadcrumb: +1.5 per blocker removed
                                          # (also sharply penalizes re-burying the
                                          #  target — the dump-back dead-end)
    shape_noroom_carrier_holds=1.0,       # shuttle holds the target
    shape_room_carrier_holds=2.0,         # lift holds the target  (> shuttle: handoff is +)
    shape_room_carrier_empty_handed=1.0,  # lift ditched its car, empty-handed
    shape_room_carrier_empty_holds=2.0,   # lift carries an empty
    shape_room_carrier_empty_at_room=3.0, # lift staged the empty at its room (goal rung)
)
PENALTY_ALL_WAIT = 1.0     # rescue trigger: wake+re-query an all-WAIT-while-work stall


def make_env(fac, max_steps, sampled: bool, stage_leave: float = 0.0,
             move_cost: float = 0.0):
    """One omni env. `sampled=True` → random omni episodes; `sampled=False` → a
    bare env whose layout is injected per-reset by a curriculum forced-layout
    builder (installed by the caller).

    `stage_leave` > 0 charges a penalty whenever a room carrier goes from staged
    (docked at its room holding an empty) to not-staged — i.e. it relocates the
    room's empty pallet away. This keeps rooms staged: a carrier with no role in
    the task is discouraged from un-staging a room, while the carrier that must
    dig/deliver still does so (the task rewards dominate the one-off leave cost).
    It is an EVENT penalty (not the staging potential, which stays gated off
    during a retrieve), so it never re-introduces the dug-empty mis-reward that
    blocked the dig. `move_cost` > 0 charges a tiny per-mm travel cost to trim
    leftover redundant movement."""
    exp = ExperimentConfig(episode=EpisodeConfig(max_steps=max_steps, max_sim_time=3600.0))
    common = dict(
        facility_factory=fac, omni=True,
        penalty_stage_leave=stage_leave, move_cost=move_cost,
        # require_all_waiting → the "settle to idle, no movement" requirement.
        # require_noroom_empty is intentionally OFF: an idle shuttle parked with a
        # leftover empty pallet still satisfies "all carriers idle" and is harmless
        # (even useful pre-staged inventory) — forcing shuttles empty-handed adds an
        # unshaped last-mile that was the sole blocker to clean-rest discovery.
        require_noroom_empty=False, require_all_waiting=True,
        target_any_shelf=True, fullness=-1.0,
        reward_deliver=0.0, reward_success=REWARD_SUCCESS, reward_gamma=1.0,
        penalty_all_wait_while_task=PENALTY_ALL_WAIT,
        experiment_config=exp, **SHAPE,
    )
    if sampled:
        return RetrieveEnv(
            request_car_amounts=(0, 1, 2),
            room_car_amounts=(0, 1, 2),
            depths=(0, 1, 2),
            **common,
        )
    # Curriculum env: the forced-layout builder fully defines the episode, so the
    # sampled-mode knobs are irrelevant (request_car_amounts left default scalar).
    return RetrieveEnv(**common)


def make_curriculum_builder(curriculum: Curriculum, rng: np.random.Generator):
    """A per-reset forced-layout builder: draw a fresh CaseSpec from the (shared,
    escalating) curriculum, build its canonical layout, and seed its retrieve."""
    def build(env, facility):
        spec = curriculum.sample_spec(rng)
        tid = hc.build_layout(facility, spec, np.random.default_rng(int(rng.integers(1 << 30))))
        env._task_type = "retrieve"
        env._seed_retrieve(facility, tid, spec.depth)
    return build


def make_coverage_builder(rng: np.random.Generator, hard: bool = False):
    """DOMAIN-RANDOMIZED solvable state for broad coverage — the antidote to the
    'coverage holes' brittleness (deadlocks on solvable cases the sampler never
    showed). Each reset randomizes, independently:
      - fullness ~U(0, 0.92)  → every density of shelf;
      - carrier start: each carrier independently empty+undocked (the viz fresh
        state that deadlocks most), OR staged at its room with an empty;
      - request set: k~{0..4} random CARS currently on shelves (k=0 ⇒ a park/idle
        episode). shuffle(require_solvable) guarantees every car is individually
        retrievable, and freeing a top empty for staging only ever loosens the
        layout, so the whole episode stays solvable (deliver sequentially, stage).
    The union over many resets covers the reachable solvable state space, so an
    unknown deployment distribution (incl. whatever the agent's own parking
    creates) falls inside training coverage."""
    def build(env, facility):
        topo = facility.topology
        sub = np.random.default_rng(int(rng.integers(1 << 31)))
        # AXIS: overall fullness + whether density concentrates on big shelves.
        # `hard` biases toward the extreme corner (high fullness, dense big shelves).
        f = float(sub.uniform(0.6, 0.95) if hard else sub.uniform(0.0, 0.95))
        shuffle_state(facility, fullness=f, rng=sub, require_solvable=True,
                      prioritize_big=bool(sub.random() < (0.7 if hard else 0.5)))
        # AXIS: request SUVs specifically (the hard, under-covered multi-SUV-at-
        # high-fullness corner the viz hits) vs any car. When SUV-focused, keep big
        # shelves SUV-heavy so there are enough SUVs + SUV blockers for hard digs.
        big_only = bool(sub.random() < (0.75 if hard else 0.5))
        # AXIS: big-shelf size mix — sample an SUV fraction and convert the rest of
        # the big-shelf cars to sedans (legal: big shelves accept small). Spans
        # "SUV shelf full of sedans" ↔ "full of SUVs" and every blocker mix in
        # between (SUVs-in-front-of-sedans and vice-versa). Only SUV→sedan, which
        # never tightens solvability (a sedan can also evict to small shelves).
        suv_frac = float(sub.uniform(0.7, 1.0)) if big_only else float(sub.uniform(0.0, 1.0))
        for sid, sh in topo.shelves.items():
            if sh.size_class != "big":
                continue
            st = facility.state.shelves[sid].stack
            for i, p in enumerate(st):
                if p.contents == "big" and sub.random() > suv_frac:
                    st[i] = Pallet(id=p.id, contents="small")
        # AXIS: how many rooms start staged (each room carrier independently).
        for cid, cs in facility.state.carriers.items():
            cs.load = None
            cs.docked_at = None
            if topo.accessible_rooms[cid] and sub.random() < 0.4:
                for sid in topo.accessible_shelves[cid]:
                    ss = facility.state.shelves[sid]
                    if ss.stack and ss.stack[-1].is_empty:
                        cs.load = ss.stack.pop()
                        cs.docked_at = DockRef("room", next(iter(topo.accessible_rooms[cid])))
                        break
        # AXIS: #retrieves + which cars (random over all shelves/types → spans
        # target type, depth, shelf, route/handoff, and put-back-necessity that
        # emerge from high fullness). require_solvable keeps each individually
        # retrievable; multiple are served sequentially.
        cars = [(p.id, p.contents) for ss in facility.state.shelves.values()
                for p in ss.stack if not p.is_empty]
        if big_only:
            cand = [pid for pid, c in cars if c == "big"] or [pid for pid, _ in cars]
        else:
            cand = [pid for pid, _ in cars]
        # AXIS: #simultaneous requests. `hard` weights toward many (the k=3/4
        # joint-coordination corner); otherwise uniform 0..4 incl. park (k=0).
        if hard:
            k = int(sub.choice([2, 3, 4, 4], p=[0.2, 0.3, 0.25, 0.25]))
        else:
            k = int(sub.integers(0, 5))
        if k == 0 or not cand:
            env._task_type = "park"
            return
        env._task_type = "retrieve"
        for pid in sub.permutation(cand)[:k]:
            env._seed_retrieve(facility, int(pid), pallet_depth(facility.state, int(pid)))
    return build


def sampled_eval(net, collator, sampled_env, n_episodes, device):
    """Greedy eval on the REAL random distribution (multi-task store+retrieve,
    random fullness). Returns (per_request_delivery, clean_rest_rate) — the
    constructed battery (single retrieve) doesn't cover multi-task, so this is the
    deployment-relevant retrieve-reliability signal."""
    import torch as _t
    from oos.learn.batching import sample_from_env_step as _sfe
    from oos.sim.tasks import Retrieve as _R
    net.eval()
    n_max = sampled_env.n_actions
    cap = sampled_env._experiment_cfg.episode.max_steps
    rd = rt = cr = 0
    for sd in range(n_episodes):
        obs, info = sampled_env.reset(seed=500_000 + sd)
        targets = set(info["target_pallet_ids"]); deliv = set()
        for _ in range(cap):
            s = _sfe(obs, info, info["action_entries"])
            b = collator.collate([s], n_max=n_max, device=device)
            with _t.no_grad():
                a = int(net(b).logits[0].argmax().item())
            obs, _r, term, trunc, info = sampled_env.step(a)
            for c in info.get("completions", []):
                if isinstance(c.task, _R) and c.task.pallet in targets:
                    deliv.add(c.task.pallet)
            if info.get("success") or term or trunc:
                break
        rd += len(deliv); rt += len(targets); cr += int(info.get("success", False))
    return (rd / max(1, rt), cr / max(1, n_episodes))


def greedy_eval(net, collator, eval_env, specs, n_seeds, device):
    """Mean greedy clean-rest success over `specs` (each averaged over n_seeds)."""
    rates = hc.eval_on_battery(net, collator, eval_env, specs,
                               n_seeds=n_seeds, max_steps=eval_env._experiment_cfg.episode.max_steps,
                               device=device)
    return rates


def buffers_success_rate(bufs):
    """(retrieve_success, park_success, n_ret, n_park) over completed episodes."""
    rc = rt = pc = pt = 0
    for b in bufs:
        rc += sum(b.ep_retrieves_completed); rt += sum(b.ep_retrieves_total)
        pc += sum(b.ep_park_completed); pt += sum(b.ep_park_total)
    return (rc / rt if rt else float("nan"), pc / pt if pt else float("nan"), rt, pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facility", default="tiny_medipol")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--curriculum-frac", type=float, default=0.625)  # 10/16 curriculum
    ap.add_argument("--coverage-frac", type=float, default=0.0,
                    help="fraction of envs using the domain-randomized coverage "
                         "builder (broad solvable-state coverage for robustness)")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="keep training past the sampled-eval mastery bar (needed "
                         "to push the hard multi-request corners the eval misses)")
    ap.add_argument("--hard-coverage", action="store_true",
                    help="bias the coverage builder toward the hard corner: more "
                         "simultaneous requests, higher fullness, SUV-focused")
    ap.add_argument("--n-steps", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.025)
    ap.add_argument("--stage-leave-penalty", type=float, default=0.0,
                    help="penalty for a room carrier un-staging its room (keeps "
                         "uninvolved carriers from relocating a room's empty)")
    ap.add_argument("--move-cost", type=float, default=0.0,
                    help="tiny per-mm travel cost to trim redundant movement")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gat-layers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--sampled-eval", type=int, default=120,
                    help="# random multi-task episodes for the real-distribution eval")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--run-dir", default="runs/omni")
    ap.add_argument("--resume", default="")
    ap.add_argument("--reset-best", action="store_true",
                    help="ignore the resumed checkpoint's best_metric (start best "
                         "tracking fresh) — use after changing the best metric or to "
                         "re-capture the best checkpoint on a focused resume.")
    ap.add_argument("--start-open", type=int, default=0,
                    help="open this many curriculum tiers at start (0 = just T0). "
                         "Set to all tiers when warm-starting from a BC policy that "
                         "already spans the difficulty range.")
    ap.add_argument("--battery", action="store_true",
                    help="robustness fine-tune: replace the tier curriculum with a "
                         "single pool of EVERY solver-solvable spec (uniform), to "
                         "close coverage gaps the tiers miss (e.g. k0-f0). Warm-start "
                         "from a curriculum-mastered checkpoint.")
    ap.add_argument("--bc", action="store_true",
                    help="solver-guided BC bootstrap + interleaved BC anchoring "
                         "(POfD-style): collect solver demos, pretrain by behaviour "
                         "cloning, then keep anchoring to the demos each PPO iter so "
                         "the dig the policy can't explore on its own never decays.")
    ap.add_argument("--bc-per-spec", type=int, default=8)
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--bc-anchor", type=int, default=4096,
                    help="demo samples to BC-anchor on after each PPO update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")

    fac = get_facility(args.facility)
    topo, _ = fac()
    from oos.env.action import max_actions_per_carrier
    n_max = max(1, max_actions_per_carrier(topo))
    collator = GraphCollator(topo)
    net, net_cfg, feat_dims = build_net(hidden=args.hidden, n_heads=4,
                                        n_gat_layers=args.gat_layers, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ppo_cfg = PPOConfig(ent_coef=args.ent_coef)

    if args.battery:
        # One pool = every solver-solvable spec, sampled uniformly. Closes the
        # coverage gaps the escalating tiers leave (the acceptance battery tests
        # specs like k0-f0 that no tier contains).
        from oos.learn.curriculum import Tier
        from oos.learn.acceptance import all_specs, spec_solvable
        solvable = [sp for sp in all_specs() if spec_solvable(fac, sp)]
        # Oversample the hard corners (slack<=0 buffer-on-target, and deep handoff)
        # so the rare-but-hardest specs (e.g. d2-k2-f0-handoff-slack-2, 2/96 of the
        # uniform pool) get enough gradient instead of being averaged away.
        pool = list(solvable)
        for sp in solvable:
            reps = 0
            if sp.slack <= 0:
                reps += 6
            if sp.routing == "handoff" and sp.depth == 2:
                reps += 3
            pool += [sp] * reps
        nhard = sum(1 for sp in solvable if sp.slack <= 0)
        print(f"[battery] {len(solvable)} solver-solvable specs ({nhard} slack<=0); "
              f"pool size {len(pool)} after hard-corner oversampling")
        curriculum = Curriculum([Tier("battery", tuple(pool))])
    else:
        curriculum = Curriculum(default_tiers())
        if args.start_open > 0:
            curriculum.n_open = max(1, min(args.start_open, len(curriculum.tiers)))

    # --- Solver-guided BC bootstrap + anchoring (POfD/DQfD-style) -------------
    bc_samples = bc_actions_t = None
    if args.bc:
        specs = all_curriculum_specs()
        print(f"[bc] collecting solver demos: {len(specs)} specs × {args.bc_per_spec}")
        bc_samples, bc_actions, bc_collator, _ = collect_demos(
            fac, specs, args.bc_per_spec, max_steps=200, verbose=False)
        bc_actions_t = torch.tensor(bc_actions, dtype=torch.long, device=device)
        print(f"[bc] {len(bc_samples)} demo pairs; behaviour-cloning {args.bc_epochs} epochs")
        bc_train(net, collator, bc_samples, bc_actions, n_max,
                 epochs=args.bc_epochs, lr=1e-3, device=device)
        # The BC policy already spans the whole difficulty range → open all tiers.
        curriculum.n_open = len(curriculum.tiers)

    # Env population: curriculum (forced hard-dig layouts, rehearsal) + coverage
    # (domain-randomized solvable states, the robustness driver) + sampled slices.
    K = args.envs
    n_cur = max(1, int(round(K * args.curriculum_frac)))
    n_cov = int(round(K * args.coverage_frac))
    n_cov = max(0, min(n_cov, K - n_cur))
    envs, states = [], []
    for i in range(K):
        is_cur = i < n_cur
        is_cov = n_cur <= i < n_cur + n_cov
        forced = is_cur or is_cov
        e = make_env(fac, args.max_steps, sampled=not forced,
                     stage_leave=args.stage_leave_penalty, move_cost=args.move_cost)
        if is_cur:
            e.set_forced_layout(make_curriculum_builder(
                curriculum, np.random.default_rng(1000 + i)))
        elif is_cov:
            e.set_forced_layout(make_coverage_builder(
                np.random.default_rng(7000 + i), hard=args.hard_coverage))
        envs.append(e)
        states.append(make_collector(e, seed=args.seed * 10_000 + i))
    print(f"[envs] {n_cur} curriculum / {n_cov} coverage / {K - n_cur - n_cov} sampled")
    assert n_max == envs[0].n_actions
    eval_env = make_env(fac, args.max_steps, sampled=False)  # eval driven by eval_on_battery
    # Real-distribution eval env (multi-task store+retrieve, random fullness).
    sampled_eval_env = make_env(fac, max(args.max_steps, 200), sampled=True)

    start_iter = 0
    total_steps = 0
    best_metric = -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device=device)
        start_iter, total_steps = restore_into(ckpt, net=net, optimizer=opt)
        if "curriculum_n_open" in ckpt:
            # clamp to this run's tier count (battery mode has a single pooled tier)
            curriculum.n_open = min(int(ckpt["curriculum_n_open"]), len(curriculum.tiers))
        if not args.reset_best:
            best_metric = float(ckpt.get("best_metric", -1.0))
        print(f"[resume] iter={start_iter} steps={total_steps} "
              f"n_open={curriculum.n_open} best={best_metric:.3f}")

    print(f"[train] facility={args.facility} params={sum(p.numel() for p in net.parameters())} "
          f"K={K} ({n_cur} curriculum / {K-n_cur} sampled) n_steps={args.n_steps} "
          f"max_steps={args.max_steps} n_max={n_max} threads={args.threads}")
    print(f"[train] tiers={[t.name for t in curriculum.tiers]}")

    def log(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t_start = time.time()
    for it in range(start_iter, args.iters):
        t0 = time.time()
        bufs = collect_rollout_vec(states, net, collator, n_max,
                                   n_steps=args.n_steps, device=device,
                                   reward_normalizer=None)
        n_new = sum(len(b) for b in bufs)
        total_steps += n_new
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        bc_acc = None
        if bc_samples is not None and args.bc_anchor > 0:
            _, bc_acc = bc_anchor(net, opt, collator, bc_samples, bc_actions_t,
                                  n_max, n=args.bc_anchor, device=device)
        t_iter = time.time() - t0

        ep_returns = [r for b in bufs for r in b.ep_returns]
        ep_lengths = [l for b in bufs for l in b.ep_lengths]
        ret_succ, park_succ, n_ret, n_park = buffers_success_rate(bufs)
        sps = n_new / t_iter

        rec = dict(
            it=it, total_steps=total_steps, sps=round(sps),
            n_open=curriculum.n_open, top_tier=curriculum.top_tier.name,
            mean_R=round(float(np.mean(ep_returns)), 2) if ep_returns else None,
            mean_len=round(float(np.mean(ep_lengths)), 1) if ep_lengths else None,
            n_eps=len(ep_returns),
            ret_succ=round(ret_succ, 3) if n_ret else None,
            park_succ=round(park_succ, 3) if n_park else None,
            policy_loss=round(m.policy_loss, 4), value_loss=round(m.value_loss, 3),
            entropy=round(m.entropy, 3), kl=round(m.approx_kl, 4),
            clipfrac=round(m.clip_fraction, 3), ev=round(m.explained_variance, 3),
            bc_acc=round(bc_acc, 3) if bc_acc is not None else None,
        )

        if it % args.eval_every == 0:
            held = curriculum.held_out_specs(per_tier=64 if args.battery else 12)
            tier_rates = {}
            for tname, specs in held.items():
                r = greedy_eval(net, collator, eval_env, specs, args.eval_seeds, device)
                tier_rates[tname] = round(float(np.mean(list(r.values()))), 3)
            top_succ = tier_rates[curriculum.top_tier.name]
            opened = curriculum.record_eval(top_succ)
            rec["eval"] = tier_rates
            rec["top_succ"] = top_succ
            rec["opened"] = opened
            rec["open_mean"] = round(float(np.mean(list(tier_rates.values()))), 3)
            rec["open_min"] = round(float(min(tier_rates.values())), 3)
            # Real-distribution (multi-task) retrieve reliability — the constructed
            # battery is single-retrieve only, so this catches the multi-task gap.
            sd_deliv, sd_clean = sampled_eval(net, collator, sampled_eval_env,
                                              args.sampled_eval, device)
            rec["sampled_deliver"] = round(sd_deliv, 3)
            rec["sampled_clean"] = round(sd_clean, 3)
            # "best" prioritises the WEAKEST of {battery worst-tier, real-dist
            # delivery} so best.pt is strong on BOTH the hard corners AND multi-task.
            metric = min(rec["open_min"], sd_deliv) + 0.001 * rec["open_mean"]
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(
                    os.path.join(args.run_dir, "best.pt"), net=net, optimizer=opt,
                    net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                    total_env_steps=total_steps,
                    extra=dict(curriculum_n_open=curriculum.n_open, best_metric=best_metric),
                )
            print(f"  iter {it:4d} | open {curriculum.n_open}/{len(curriculum.tiers)} "
                  f"{curriculum.top_tier.name:12s} bat_min={rec['open_min']:.2f} "
                  f"real_deliv={rec['sampled_deliver']:.3f} real_clean={rec['sampled_clean']:.3f}"
                  f"{' OPENED' if opened else ''} | R={rec['mean_R']} "
                  f"len={rec['mean_len']} | ent={rec['entropy']:.2f} | {sps:.0f} sps")
            print(f"        tiers: {tier_rates}")
        else:
            print(f"  iter {it:4d} | {curriculum.top_tier.name:12s} | "
                  f"R={rec['mean_R']} len={rec['mean_len']} retS={rec['ret_succ']} "
                  f"parkS={rec['park_succ']} | ploss={rec['policy_loss']:.3f} "
                  f"vloss={rec['value_loss']:.2f} ev={rec['ev']} ent={rec['entropy']:.2f} "
                  f"| {sps:.0f} sps {t_iter:.1f}s")
        log(rec)

        if it % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
                net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                total_env_steps=total_steps,
                extra=dict(curriculum_n_open=curriculum.n_open, best_metric=best_metric),
            )

        # Early stop: all tiers open AND every open tier mastered (MIN tier rate)
        # AND the real multi-task distribution delivers reliably. Using the min of
        # both keeps a strong battery from masking a weak multi-task distribution.
        if (not args.no_early_stop
                and curriculum.n_open == len(curriculum.tiers)
                and rec.get("open_min", 0) >= 0.95
                and rec.get("sampled_deliver", 0) >= 0.99):
            print(f"[train] mastered: battery_min={rec['open_min']} "
                  f"real_deliver={rec['sampled_deliver']} at iter {it} — stopping.")
            break

    save_checkpoint(
        os.path.join(args.run_dir, "latest.pt"), net=net, optimizer=opt,
        net_cfg=net_cfg, feat_dims=feat_dims, iteration=args.iters,
        total_env_steps=total_steps,
        extra=dict(curriculum_n_open=curriculum.n_open, best_metric=best_metric),
    )
    print(f"[train] done in {(time.time()-t_start)/60:.1f} min, "
          f"total_steps={total_steps}, best={best_metric:.3f}, "
          f"final n_open={curriculum.n_open}")


if __name__ == "__main__":
    main()
