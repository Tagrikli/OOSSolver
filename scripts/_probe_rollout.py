"""Throwaway: instrumented greedy rollout to see the 'take a car and wait'
behavior. Loads omni2_d0_st/ckpt_latest, forces a park task at fullness 0.8,
argmax-rolls, and prints per-step carrier loads/docks + the chosen action +
shelf tops (to see whether an empty is reachable / whether shelves are full)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from oos.facilities import get_facility
from oos.env.retrieve_env import RetrieveEnv
from oos.learn.batching import GraphCollator, sample_from_env_step
from oos.learn.checkpoint import load_checkpoint
from oos.learn.net import build_net

FAC = "tiny_medipol"
CKPT = "runs/omni2_d0_st/ckpt_latest.pt"
DEV = "cpu"


def make_env(fullness):
    return RetrieveEnv(
        facility_factory=get_facility(FAC),
        room_car_amounts=(1, 2), request_car_amounts=(0,), depths=(0,),
        fullness=fullness, reward_deliver=0.0, require_solvable=True,
        target_any_shelf=True, omni=True, reward_gamma=0.99,
        shape_room_carrier_holds=1.0, shape_noroom_carrier_holds=0.5,
        shape_room_carrier_empty_handed=0.5, shape_room_carrier_empty_holds=1.0,
        shape_room_carrier_empty_at_room=0.5, reward_store_car=0.5,
        r_car2pallet=5.0, r_pallet2car=5.0, r_car2car=5.0, p_pallet2pallet=0.0,
        penalty_all_wait_while_task=1.0, move_cost=1e-7,
    )


def load(cs):
    if cs.load is None:
        return "----"
    return "MTpal" if cs.load.is_empty else f"CAR{cs.load.id}"


def dock(cs):
    d = cs.docked_at
    return "--" if d is None else f"{d.kind[0].upper()}:{d.id}"


def shelf_summary(eng):
    bits = []
    for sid, ss in eng.state.shelves.items():
        stk = ss.stack
        top = "_" if not stk else ("e" if stk[-1].is_empty else "C")
        bits.append(f"{sid}[{len(stk)}{top}]")
    return " ".join(bits)


def main():
    collator = GraphCollator(get_facility(FAC)()[0])
    env = make_env(0.8)
    env.set_forced_task_type("park")
    n_max = env.n_actions
    net, _, _ = build_net(hidden=64, n_heads=4, n_gat_layers=2, device=DEV)
    net.load_state_dict(load_checkpoint(CKPT, DEV)["net_state_dict"])
    net.eval()

    for seed in range(6):
        obs, info = env.reset(seed=seed)
        rc = env._room_carriers
        print(f"\n===== seed {seed}  fullness={info.get('fullness'):.2f}  "
              f"room_carriers={rc} =====")
        print(f"  shelves: {shelf_summary(env.engine)}")
        with torch.no_grad():
            for t in range(30):
                sample = sample_from_env_step(obs, info, info["action_entries"])
                batch = collator.collate([sample], n_max=n_max, device=DEV)
                action = int(net(batch).logits[0].argmax().item())
                entry = info["action_entries"][action]
                tgt = getattr(entry, "target", None)
                tname = "WAIT" if tgt is None else f"{entry.type.name}->{tgt.kind[0].upper()}:{tgt.id}"
                st = env.engine.state
                cstr = "  ".join(
                    f"{cid}:{load(st.carriers[cid])}@{dock(st.carriers[cid])}"
                    f"{'(w)' if st.carriers[cid].waiting else ''}"
                    for cid in rc
                )
                phi = env._phi_staging_ladder(env.engine)
                print(f"  t{t:02d} act={tname:14s} Φ={phi:.2f} | {cstr}")
                obs, _r, term, trunc, info = env.step(action)
                if info.get("success"):
                    print(f"  -> SOLVED at t{t}")
                    break
                if term or trunc:
                    print(f"  -> end (term={term} trunc={trunc}) at t{t}")
                    break


if __name__ == "__main__":
    main()
