"""Diagnose the live-stream behavior the user sees in the viz: idle carriers
moving, stores ignored, getting stuck despite a solution. Drive the continuous
Environment with the loaded brain and instrument every decision.

Run at several store_rates to separate OOD-queue (high rate) from a fundamental
store/retrieve bug (fails even at low rate / isolated)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import scripts._failhunt as FH
from oos.env.action import ActionType
from oos.sim.tasks import Retrieve, Store

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/_rescue/real_v2_ROBUST.pt"


def act_and_label(net, col, env, device="cpu"):
    """Pick the greedy action; return (action_idx, ActionType, querying_carrier)."""
    from oos.learn.batching import sample_from_env_step
    obs, info = env._last_obs, env._last_info  # set by caller
    s = sample_from_env_step(obs, info, info["action_entries"])
    b = col.collate([s], n_max=env.n_actions, device=device)
    with torch.no_grad():
        idx = int(net(b).logits[0].argmax().item())
    entries = info["action_entries"]
    if 0 <= idx < len(entries):
        atype = entries[idx].type
    else:
        atype = None
    return idx, atype


def run(store_rate, dwell, steps=2500, device="cpu"):
    net, col = FH.load(CKPT)
    env = FH.cont_env(store_rate=store_rate, dwell=dwell)
    obs, info = env.reset(seed=987)
    env._last_obs, env._last_info = obs, info

    act_hist = Counter()
    n_decisions = 0
    wait_with_serveable = 0     # chose WAIT while a carrier holds a serveable load OR queue non-empty
    served_s = served_r = 0
    q_samples = []
    no_completion_streak = 0
    max_stuck_streak = 0
    moves_while_empty = 0       # GOTO by an empty-handed carrier (the "idle move")

    for t in range(steps):
        if env.needs_decision():
            qc = env.querying_carrier
            cs = env.engine.state.carriers.get(qc) if qc is not None else None
            idx, atype = act_and_label(net, col, env)
            n_decisions += 1
            if atype is not None:
                act_hist[atype.name] += 1
            qlen = len(env.engine.queue.pending)
            if atype == ActionType.WAIT and qlen > 0:
                wait_with_serveable += 1
            if atype == ActionType.GOTO and cs is not None and cs.load is None:
                moves_while_empty += 1
            obs, _r, term, trunc, info = env.step(idx)
        else:
            obs, _r, term, trunc, info = env.advance(time_limit=None)
        env._last_obs, env._last_info = obs, info

        comps = info.get("completions", [])
        for c in comps:
            if isinstance(c.task, Store): served_s += 1
            elif isinstance(c.task, Retrieve): served_r += 1
        if comps:
            no_completion_streak = 0
        else:
            no_completion_streak += 1
            max_stuck_streak = max(max_stuck_streak, no_completion_streak)
        if t % 50 == 0:
            q_samples.append(len(env.engine.queue.pending))
        if term or trunc:
            break

    qlen = len(env.engine.queue.pending)
    pend_kinds = Counter(type(x).__name__ for x in env.engine.queue.pending)
    print(f"\n=== store_rate={store_rate}  dwell={dwell}  (sim_time={env.sim_time:.0f}) ===")
    print(f"  served: store={served_s} retrieve={served_r}  | final_queue={qlen} {dict(pend_kinds)}")
    print(f"  decisions={n_decisions}  actions={dict(act_hist)}")
    print(f"  WAIT-while-queue-nonempty = {wait_with_serveable} "
          f"({100*wait_with_serveable/max(1,n_decisions):.0f}% of decisions)")
    print(f"  GOTO-while-empty-handed   = {moves_while_empty} "
          f"({100*moves_while_empty/max(1,n_decisions):.0f}% of decisions)")
    print(f"  longest no-completion streak = {max_stuck_streak} env-steps")
    print(f"  queue trajectory (every 50 steps): {q_samples[:24]}")


if __name__ == "__main__":
    print(f"BRAIN: {CKPT}")
    for sr in (0.03, 0.10):
        run(sr, dwell=80.0)
