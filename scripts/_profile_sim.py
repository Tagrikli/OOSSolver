"""Throwaway profiler: where does collection time go — sim, net, or collate?

Mirrors the omni2 training setup at small scale and cProfiles collect_rollout_vec
on CPU (so PyTorch overhead doesn't hide the sim cost), then reports steps/sec on
both CPU and the configured device. Delete when done.
"""
import importlib.util
import sys
import time
from pathlib import Path

import cProfile
import pstats

import torch

ROOT = Path("/home/lavender/Desktop/Codes/OOSKiller")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("train_omni2", ROOT / "scripts" / "train_omni2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def build(device, K):
    collator = m.GraphCollator(m.get_facility(m.FACILITY)()[0])
    env0 = m._make_env(m.MAX_EPISODE_STEPS)
    n_max = env0.n_actions
    net, _, _ = m.build_net(hidden=m.HIDDEN, n_heads=m.N_HEADS,
                            n_gat_layers=m.N_GAT_LAYERS, device=device)
    collectors = [m.make_collector(m._make_env(m.MAX_EPISODE_STEPS), seed=i * 1000)
                  for i in range(K)]
    return collator, net, n_max, collectors


def run(collator, net, n_max, collectors, n_steps, device):
    m.collect_rollout_vec(states=collectors, net=net, collator=collator,
                          n_max=n_max, n_steps=n_steps, device=device,
                          reward_normalizer=None)


K = 8
N = 2048

# ---- CPU timing + profile (isolates the Python sim cost) ----
collator, net, n_max, collectors = build(torch.device("cpu"), K)
run(collator, net, n_max, collectors, 256, "cpu")        # warmup
t0 = time.time(); run(collator, net, n_max, collectors, N, "cpu"); dt = time.time() - t0
print(f"\n=== CPU: {N} steps ({K} envs) in {dt:.2f}s = {N/dt:.0f} steps/s ===\n")

pr = cProfile.Profile(); pr.enable()
run(collator, net, n_max, collectors, N, "cpu")
pr.disable()
st = pstats.Stats(pr)
print("---- top 30 by CUMULATIVE time ----")
st.sort_stats("cumulative").print_stats(30)
print("---- top 20 by TOTAL (self) time ----")
st.sort_stats("tottime").print_stats(20)

# ---- device timing (what training actually uses) ----
if torch.cuda.is_available():
    dev = torch.device("cuda")
    collator, net, n_max, collectors = build(dev, K)
    run(collator, net, n_max, collectors, 256, dev)
    torch.cuda.synchronize()
    t0 = time.time(); run(collator, net, n_max, collectors, N, dev); torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\n=== CUDA: {N} steps ({K} envs) in {dt:.2f}s = {N/dt:.0f} steps/s ===")
