"""End-to-end continuous trainer — the deployment finale, done right.

GOAL ([[goal_e2e_continuous]]): ONE brain that services the continuous stream
EFFICIENTLY (high throughput, low wait, no wander, no store-starvation) AND keeps
every isolated skill (retrieve/store/park/put-back in any situation).

Built on train_continuous.py, with the fixes the _diag_stream diagnosis found:
  * warm-start from real_v2_ROBUST (the strongest isolated base), not curric_v3.
  * BALANCED reward: serve == deliver (== 15). continuous_v4's serve=6 < deliver=15
    was BELOW the Φ-drop threshold → serving a store was net-negative → starvation.
    This is the ONE change from v4's known-good recipe (PBRS, potentials 0.5/0.5,
    freeze) — changing more at once (dense_progress, 20/20, default potentials)
    backfired: idle 32→52%, ISO 1.0→0.81. Discipline: one knob at a time.
  * RANDOMIZED demand spread across the stream collectors (v4 was one fixed rate).
  * dual-battery eval (isolated robustness gate + stream efficiency) + auto-rescue
    of the best COMBINED score (no more hand-rescuing at peak).

Two configs, set by MODE below; run both, keep whichever verifies better:
  * MODE="freeze"  — freeze the GAT representation, train heads only. Safe: isolated
                     robustness is preserved exactly; reward fixes still cure the
                     head-level starvation/wander/idle. The low-risk baseline.
  * MODE="e2e"     — unfrozen + a distillation/KL anchor to a frozen real_v2 on
                     isolated states (anti-forgetting without the freeze ceiling).

Run unbuffered:  python -u scripts/train_e2e.py [freeze|e2e]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.action import ActionType
from oos.env.env import Environment
from oos.env.hardcases import build_layout
from oos.env.retrieve_env import RetrieveEnv
from oos.env.reward import RewardConfig
from oos.facilities import get_facility
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint, save_checkpoint
from oos.learn.curriculum import Curriculum, default_tiers
from oos.learn.net import build_net
from oos.learn.ppo import PPOConfig, ppo_update
from oos.learn.rollout import collect_rollout_vec, make_collector
from oos.sim.tasks import Retrieve, Store

# ── CONFIG ──────────────────────────────────────────────────────────────────
FACILITY = "tiny_medipol"
MODE = sys.argv[1] if len(sys.argv) > 1 else "freeze"   # "freeze" | "e2e"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
INIT_WEIGHTS = "runs/_rescue/real_v2_ROBUST.pt"   # strongest isolated base

FREEZE_REPR = (MODE == "freeze")
DISTILL = (MODE == "e2e")          # KL-anchor to frozen real_v2 on isolated states
DISTILL_BETA = 1.0                 # weight of the distillation loss
DISTILL_POOL = 3072                # isolated anchor states collected once at start
DISTILL_BATCH = 1024

# Balanced outcome reward (the starvation fix). The ONLY change from continuous_v4's
# known-good recipe (deliver15/serve6, which got idle→10% but starved stores because
# serve6 < the Φ-drop → net-negative serve): raise serve to == deliver, so stores and
# retrieves are equally worth serving. Kept at 15 (the rehearsal anchor) so the stream
# gradient doesn't drown the isolated rehearsal (20 drowned it: ISO 1.0→0.81).
REWARD_DELIVER = 15.0
REWARD_SERVE = 15.0
# PBRS potential weights — match continuous_v4 (low room_ready avoids over-staging
# idle; the higher defaults pushed carriers to camp at rooms → idle/wander up).
POT_ROOM_READY = 0.5
POT_ITEM_RETRIEVAL = 0.5
# Anti-wander: −MOVE_COST · mm-travelled (existing MovementTerm). Calibrated so a
# full solve's travel costs ≪ one delivery (15) — irrelevant carriers learn to
# stay put / take shortest paths, relevant ones still move. 0 = off. Stream only.
MOVE_COST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
RUN_NAME = f"e2e_{MODE}" + (f"_mv{MOVE_COST:g}" if MOVE_COST > 0 else "")

# Randomized demand spread across the stream collectors (per-index, deterministic).
STORE_RATE_LO, STORE_RATE_HI = 0.008, 0.030   # near 2-carrier capacity → overload tail
DWELL_LO, DWELL_HI = 40.0, 150.0
BIG_LO, BIG_HI = 0.10, 0.40

EPISODIC_FRAC = 0.60          # isolated rehearsal fraction (match v4; 0.55 let ISO slip)
TOTAL_ITERS = 6000
STEPS_PER_ITER = 1024 * 16
N_ENVS = 64
MAX_EP_STEPS = 1500           # truncation horizon (continuing task; GAE bootstraps)
GAMMA, LAM, LR = 0.99, 0.95, 3e-4
CLIP, VF, ENT, MAXGRAD = 0.2, 0.5, 0.02, 0.5
N_EPOCHS, MINIBATCH = 6, 1024
EVAL_EVERY, CKPT_EVERY = 10, 25

# Eval sweep: a sustainable rate (must keep up) and a stress rate (graceful triage).
EVAL_RATES = (0.012, 0.022)


def _stream_params(j, n_stream):
    """Deterministic demand spread across the n_stream stream collectors."""
    f = j / max(1, n_stream - 1)
    rate = STORE_RATE_LO + f * (STORE_RATE_HI - STORE_RATE_LO)
    dwell = DWELL_LO + ((j * 7) % max(1, n_stream)) / max(1, n_stream) * (DWELL_HI - DWELL_LO)
    big = BIG_LO + ((j * 3) % max(1, n_stream)) / max(1, n_stream) * (BIG_HI - BIG_LO)
    return rate, dwell, big


def make_stream_env(max_steps, store_rate, mean_dwell, big_prob):
    return Environment(
        facility_factory=get_facility(FACILITY),
        experiment_config=ExperimentConfig(
            task_stream=TaskStreamConfig(
                store_rate=store_rate, mean_dwell_seconds=mean_dwell,
                size_mix={"small": 1.0 - big_prob, "big": big_prob}),
            episode=EpisodeConfig(max_steps=max_steps, max_sim_time=1e9)),
        reward_config=RewardConfig(
            reward_deliver=REWARD_DELIVER, reward_serve=REWARD_SERVE,
            potential_room_ready=POT_ROOM_READY, potential_item_retrieval=POT_ITEM_RETRIEVAL,
            dense_progress=False,           # PBRS, like continuous_v4 (10% idle); True backfired
            penalty_move=MOVE_COST,         # anti-wander (stream only)
            penalty_idle_while_task=0.0, penalty_all_wait_while_task=0.0),
    )


def _curric_builder(curric):
    def build(env, facility):
        spec = curric.sample_spec(env._rng)
        tid = build_layout(facility, spec, env._rng)
        env._task_type = "retrieve"
        env._seed_retrieve(facility, tid, spec.depth)
    return build


def _episodic_env(max_steps, *, storepark, curric=None):
    """Isolated hard task (rehearsal), LOOSE success (deliver+stage, NO settle —
    settle conflicts with the stream)."""
    kw = dict(facility_factory=get_facility(FACILITY), depths=(0, 1, 2), fullness=-1,
              omni=True, target_any_shelf=True, require_noroom_empty=False,
              require_all_waiting=False, reward_success=15.0,
              penalty_all_wait_while_task=1.0, reward_gamma=GAMMA,
              experiment_config=ExperimentConfig(
                  task_stream=TaskStreamConfig(store_rate=0.0),
                  episode=EpisodeConfig(max_steps=max_steps, max_sim_time=360000.0)))
    if storepark:
        return RetrieveEnv(room_car_amounts=(1, 2), request_car_amounts=(0,), **kw)
    env = RetrieveEnv(room_car_amounts=(0,), request_car_amounts=(1,), **kw)
    env.set_forced_layout(_curric_builder(curric))
    return env


# ── eval ──────────────────────────────────────────────────────────────────
def _greedy_act(net, collator, obs, info, n_max, device):
    s = sample_from_env_step(obs, info, info["action_entries"])
    b = collator.collate([s], n_max=n_max, device=device)
    with torch.no_grad():
        return int(net(b).logits[0].argmax().item()), info["action_entries"]


def isolated_eval(net, collator, device, n_max, n=16):
    """LOOSE greedy success on isolated store + retrieve (the robustness gate)."""
    curric = Curriculum(default_tiers(), n_open=len(default_tiers()))
    out = {}
    for name, env in (("store", _episodic_env(200, storepark=True)),
                      ("retrieve", _episodic_env(200, storepark=False, curric=curric))):
        if name == "store":
            env.set_forced_task_type("park")
        w, steps = 0, []
        for i in range(n):
            obs, info = env.reset(seed=61000 + i)
            for t in range(200):
                a, _ = _greedy_act(net, collator, obs, info, n_max, device)
                obs, _r, term, trunc, info = env.step(a)
                if info.get("success", False):
                    w += 1; steps.append(t + 1); break
                if term or trunc:
                    break
        out[name] = round(w / n, 2)
    return out


def stream_eval(net, collator, device, n_max, store_rate, dwell=80.0, steps=1800):
    """Efficiency on the stream: throughput ratio, wait, and the TWO metrics that
    capture what's actually unproductive (biased proxies counted serving-WAITs and
    pallet-fetch-GOTOs as bad):
      * idle_pct   = WAIT that completes NOTHING while work pends (true idling, not
                     a productive serve-WAIT).
      * wander_pct = GOTO right after the same carrier's previous GOTO with no
                     TAKE/GIVE between (repositioning without acting — the
                     shelf1→shelf2-without-doing-anything the user sees)."""
    env = make_stream_env(4000, store_rate, dwell, 0.2)
    obs, info = env.reset(seed=987654)
    waits, served_s, served_r = [], 0, 0
    n_dec = idle = wander = 0
    arrived = 0
    last_goto = {}     # carrier -> True if its last action was a GOTO (no act since)
    qtrace = []
    for step in range(steps):
        if env.needs_decision():
            qc = env.querying_carrier
            a, entries = _greedy_act(net, collator, obs, info, n_max, device)
            n_dec += 1
            atype = entries[a].type if 0 <= a < len(entries) else None
            qlen = len(env.engine.queue.pending)
            obs, _r, term, trunc, info = env.step(a)
            comp_now = len(info.get("completions", []))
            if atype == ActionType.WAIT and qlen > 0 and comp_now == 0:
                idle += 1
            if atype == ActionType.GOTO:
                if last_goto.get(qc):
                    wander += 1            # consecutive GOTO, no act between
                last_goto[qc] = True
            elif atype in (ActionType.TAKE, ActionType.GIVE):
                last_goto[qc] = False      # acted → next GOTO is purposeful
        else:
            obs, _r, term, trunc, info = env.advance(time_limit=None)
        now = env.engine.state.time
        for arr in info.get("arrivals", []):
            arrived += 1
        for c in info.get("completions", []):
            waits.append(now - c.task.arrived_at)
            if isinstance(c.task, Store): served_s += 1
            elif isinstance(c.task, Retrieve): served_r += 1
        if step % 200 == 0:
            qtrace.append(len(env.engine.queue.pending))
        if term or trunc:
            break
    total_served = served_s + served_r
    return {
        "rate": store_rate,
        "served_s": served_s, "served_r": served_r,
        "throughput": round(total_served / max(1, arrived), 2),   # served/arrived
        "mean_wait": round(float(np.mean(waits)), 0) if waits else None,
        "idle_pct": round(100 * idle / max(1, n_dec)),            # unproductive WAIT
        "wander_pct": round(100 * wander / max(1, n_dec)),        # GOTO→GOTO no-act
        "final_q": len(env.engine.queue.pending),
        "qtrace": qtrace,
    }


def combined_score(iso, streams):
    """Robustness gate × stream efficiency. Returns (score, gated)."""
    iso_floor = min(iso["store"], iso["retrieve"])
    # efficiency: throughput up, wait/idle/wander down, balanced service.
    eff = 0.0
    for s in streams:
        bal = 1.0 - abs(s["served_s"] - s["served_r"]) / max(1, s["served_s"] + s["served_r"])
        wait_term = 1.0 - min((s["mean_wait"] or 9999) / 600.0, 1.0)
        eff += (s["throughput"] + bal + wait_term
                - s["idle_pct"] / 100.0 - s["wander_pct"] / 100.0)
    eff /= max(1, len(streams))
    gated = iso_floor >= 0.85                     # reject forgetful checkpoints
    return (iso_floor + eff, gated)


# ── distillation anchor (e2e mode) ─────────────────────────────────────────
def collect_distill_pool(ref_net, collator, device, n_max, n_states):
    """Roll ref_net (frozen real_v2) on isolated MID-TASK states; cache (sample,
    ref_logits). Mid-task only — anchor the dig/deliver skill, not the settle."""
    curric = Curriculum(default_tiers(), n_open=len(default_tiers()))
    samples, ref_logits = [], []
    envs = [_episodic_env(200, storepark=False, curric=curric),
            _episodic_env(200, storepark=True)]
    envs[1].set_forced_task_type("park")
    i = 0
    while len(samples) < n_states:
        env = envs[i % 2]; i += 1
        obs, info = env.reset(seed=4242 + i)
        for _ in range(60):
            if info.get("success", False):
                break
            s = sample_from_env_step(obs, info, info["action_entries"])
            b = collator.collate([s], n_max=n_max, device=device)
            with torch.no_grad():
                lg = ref_net(b).logits[0].detach().clone()
            samples.append(s); ref_logits.append(lg)
            if len(samples) >= n_states:
                break
            a = int(lg.argmax().item())
            obs, _r, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return samples, torch.stack(ref_logits)


def distill_step(net, opt, collator, device, n_max, pool_samples, pool_logits, rng):
    """One KL(ref || net) gradient step on a minibatch of anchor states."""
    idx = rng.choice(len(pool_samples), size=min(DISTILL_BATCH, len(pool_samples)),
                     replace=False)
    batch = [pool_samples[k] for k in idx]
    b = collator.collate(batch, n_max=n_max, device=device)
    ref = pool_logits[idx].to(device)
    out = net(b).logits
    mask = b.action_mask if hasattr(b, "action_mask") else None
    if mask is not None:
        neg = torch.finfo(out.dtype).min
        out = out.masked_fill(~mask, neg)
        ref = ref.masked_fill(~mask, neg)
    logp = torch.log_softmax(out, dim=-1)
    p_ref = torch.softmax(ref, dim=-1)
    kl = (p_ref * (torch.log(p_ref + 1e-9) - logp)).sum(-1).mean()
    opt.zero_grad(); (DISTILL_BETA * kl).backward()
    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], MAXGRAD)
    opt.step()
    return float(kl.detach().item())


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    device = torch.device(DEVICE)
    run_dir = Path("runs") / RUN_NAME; run_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(run_dir / "metrics.jsonl", "w")

    collator = GraphCollator(get_facility(FACILITY)()[0])
    n_max = make_stream_env(4000, 0.015, 80.0, 0.2).n_actions
    net, net_cfg, feat_dims = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=device)
    net.load_state_dict(load_checkpoint(INIT_WEIGHTS, device)["net_state_dict"])
    print(f"warm-started from {INIT_WEIGHTS}")

    ref_net = None
    pool_samples = pool_logits = None
    if FREEZE_REPR:
        frozen = 0
        for nm, p in net.named_parameters():
            if nm.split(".")[0] in ("proj", "gat_layers"):
                p.requires_grad = False; frozen += p.numel()
        print(f"FROZE representation ({frozen:,} params); training heads only")
    if DISTILL:
        ref_net, _, _ = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=device)
        ref_net.load_state_dict(load_checkpoint(INIT_WEIGHTS, device)["net_state_dict"])
        ref_net.eval()
        for p in ref_net.parameters():
            p.requires_grad = False
        print(f"collecting {DISTILL_POOL} distillation anchor states from real_v2...")
        pool_samples, pool_logits = collect_distill_pool(ref_net, collator, device, n_max, DISTILL_POOL)
        print(f"  pool ready: {len(pool_samples)} states")

    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=LR)
    ppo_cfg = PPOConfig(gamma=GAMMA, gae_lambda=LAM, clip_range=CLIP, vf_coef=VF,
                        ent_coef=ENT, max_grad_norm=MAXGRAD, n_epochs=N_EPOCHS,
                        minibatch_size=MINIBATCH)

    curric = Curriculum(default_tiers()[2:], n_open=len(default_tiers()) - 2)  # hard tiers
    n_epi = int(round(N_ENVS * EPISODIC_FRAC))
    n_retr = n_epi * 2 // 3
    n_stream = N_ENVS - n_epi

    def _mk(i):
        if i < n_retr:
            return _episodic_env(MAX_EP_STEPS, storepark=False, curric=curric)
        if i < n_epi:
            return _episodic_env(MAX_EP_STEPS, storepark=True)
        rate, dwell, big = _stream_params(i - n_epi, n_stream)
        return make_stream_env(MAX_EP_STEPS, rate, dwell, big)

    collectors = [make_collector(_mk(i), seed=SEED + i * 1_000_000) for i in range(N_ENVS)]
    print(f"[{RUN_NAME}] MODE={MODE} device={DEVICE} rehearsal={n_epi}/{N_ENVS} "
          f"stream={n_stream} rate∈[{STORE_RATE_LO},{STORE_RATE_HI}] "
          f"reward=D{REWARD_DELIVER}/S{REWARD_SERVE} dense_progress=ON")

    best_score = -1e9
    best_path = run_dir / "ckpt_best.pt"
    t0 = time.time()
    for it in range(1, TOTAL_ITERS + 1):
        bufs = collect_rollout_vec(states=collectors, net=net, collator=collator,
                                   n_max=n_max, n_steps=STEPS_PER_ITER, device=device,
                                   reward_normalizer=None)
        m = ppo_update(net, opt, collator, n_max, bufs, ppo_cfg, device=device)
        kl_d = None
        if DISTILL:
            kl_d = distill_step(net, opt, collator, device, n_max, pool_samples, pool_logits, rng)
        ep_ret = [x for b in bufs for x in b.ep_returns]
        row = {"iter": it, "ret": round(float(np.mean(ep_ret)), 1) if ep_ret else None,
               "entropy": round(m.entropy, 3), "kl": round(m.approx_kl, 4), "distill_kl": kl_d}
        if it % EVAL_EVERY == 0 or it == 1:
            iso = isolated_eval(net, collator, device, n_max)
            streams = [stream_eval(net, collator, device, n_max, r) for r in EVAL_RATES]
            score, gated = combined_score(iso, streams)
            row["isolated"] = iso; row["streams"] = streams; row["score"] = round(score, 3)
            star = ""
            if it == 1:
                best_score = score if gated else best_score
            elif gated and score > best_score:
                best_score = score
                save_checkpoint(best_path, net=net, optimizer=opt, net_cfg=net_cfg,
                                feat_dims=feat_dims, iteration=it, total_env_steps=it * STEPS_PER_ITER)
                star = f"  *BEST {score:.3f}"
            s0, s1 = streams[0], streams[1]
            print(f"it {it:4d} | ISO s{iso['store']}/r{iso['retrieve']} | "
                  f"r{s0['rate']}: thru{s0['throughput']} wait{s0['mean_wait']} "
                  f"idle{s0['idle_pct']}% wand{s0['wander_pct']}% S{s0['served_s']}/R{s0['served_r']} | "
                  f"r{s1['rate']}: thru{s1['throughput']} q{s1['final_q']} | "
                  f"score {score:.2f}{' ' if gated else '✗'}{star} | {time.time()-t0:.0f}s")
        else:
            print(f"it {it:4d} | ret {row['ret']} ent {m.entropy:.2f}"
                  + (f" distill_kl {kl_d:.3f}" if kl_d is not None else ""))
        metrics_f.write(json.dumps(row) + "\n"); metrics_f.flush()
        if it % CKPT_EVERY == 0 or it == TOTAL_ITERS:
            save_checkpoint(run_dir / "ckpt_latest.pt", net=net, optimizer=opt,
                            net_cfg=net_cfg, feat_dims=feat_dims, iteration=it,
                            total_env_steps=it * STEPS_PER_ITER)


if __name__ == "__main__":
    main()
