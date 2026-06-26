"""Prove the PlannerPolicy end-to-end: solve() builds a frozen plan, then the
ordinary play loop (env.apply_action, the same path the viz uses) executes it.

Scenarios:
  A. TWO simultaneous direct retrieves (different lifts) -> both delivered.
  B. A handoff retrieve (target on a shuttle shelf) -> delivered via a lift.
  C. Pure staging: no retrieves, unstaged rooms -> empties carried to rooms.

Run:  python scripts/_spike_planner_policy.py
"""

from __future__ import annotations

from oos.env import Environment
from oos.facilities import get_facility
from oos.plan.policy import PlannerPolicy
from oos.sim.state import Pallet
from oos.sim.tasks import Retrieve


def fresh_env():
    env = Environment(facility_factory=get_facility("tiny_medipol"))
    env.reset(seed=0)
    # Wipe to a known-empty layout we control.
    for ss in env.engine.state.shelves.values():
        ss.stack = []
    return env


def place(env, sid, pallets):
    env.engine.state.shelves[sid].stack = [
        Pallet(id=i, contents=c) for i, c in pallets
    ]


def add_retrieve(env, pallet):
    from oos.sim.state import pallet_depth
    env.engine.queue.add(Retrieve(
        arrived_at=env.sim_time, pallet=pallet,
        initial_depth=pallet_depth(env.engine.state, pallet),
        already_staged=False,
    ))


def obs_info(env):
    env.wake_waiting_carriers()
    return env._observation_for_current(env.engine, dt=0.0, completions=[], arrivals=[])


def play(env, planner, max_steps=800):
    obs, info = obs_info(env)
    delivered = 0
    idle = 0
    for _ in range(max_steps):
        remaining = planner.steps_left
        pending = [t for t in env.engine.queue.pending if isinstance(t, Retrieve)]
        busy = any(cs.is_busy for cs in env.engine.state.carriers.values())
        if remaining == 0 and not pending and not busy:
            idle += 1
            if idle > 3:
                break
        idx = planner(obs, info)
        obs, reward, info = env.apply_action(idx)
        delivered = len(env.engine.queue.completed_costs)
        if info.get("terminated"):
            break
    return delivered


def staged_rooms(env):
    out = []
    for rid, room in env.topology.rooms.items():
        cs = env.engine.state.carriers[room.served_by]
        d = cs.docked_at
        if d and d.kind == "room" and d.id == rid and cs.load and cs.load.is_empty:
            out.append(rid)
    return out


def scenario_A():
    print("=== A: two simultaneous direct retrieves ===")
    env = fresh_env()
    place(env, "A1", [(100, "small"), (101, "empty")])   # L1 / R1
    place(env, "E1", [(200, "big"), (201, "empty")])     # L2 / R2
    add_retrieve(env, 100)
    add_retrieve(env, 200)
    planner = PlannerPolicy(env)
    secs = planner.solve()
    print(f"  solve: {secs*1000:.1f} ms   plan steps: {planner.last_plan_steps}   "
          f"unsolved: {planner.unsolved}")
    delivered = play(env, planner)
    pend = [t.pallet for t in env.engine.queue.pending if isinstance(t, Retrieve)]
    print(f"  delivered: {delivered}/2   still pending: {pend}")
    print(f"  RESULT: {'PASS ✅' if delivered == 2 and not pend else 'FAIL ❌'}\n")


def scenario_B():
    print("=== B: handoff retrieve (target on a shuttle shelf) ===")
    env = fresh_env()
    place(env, "B1", [(300, "small"), (301, "empty")])   # S1 (shuttle) -> needs a lift
    add_retrieve(env, 300)
    planner = PlannerPolicy(env)
    secs = planner.solve()
    print(f"  solve: {secs*1000:.1f} ms   plan steps: {planner.last_plan_steps}   "
          f"unsolved: {planner.unsolved}")
    delivered = play(env, planner)
    pend = [t.pallet for t in env.engine.queue.pending if isinstance(t, Retrieve)]
    print(f"  delivered: {delivered}/1   still pending: {pend}")
    print(f"  RESULT: {'PASS ✅' if delivered == 1 and not pend else 'FAIL ❌'}\n")


def scenario_C():
    print("=== C: pure staging (no retrieves, rooms unstaged) ===")
    env = fresh_env()
    place(env, "A3", [(400, "empty")])   # an empty L1 can fetch for R1
    place(env, "E3", [(500, "empty")])   # an empty L2 can fetch for R2
    planner = PlannerPolicy(env)
    secs = planner.solve()
    print(f"  solve: {secs*1000:.1f} ms   plan steps: {planner.last_plan_steps}")
    play(env, planner)
    rooms = staged_rooms(env)
    print(f"  staged rooms: {rooms}")
    print(f"  RESULT: {'PASS ✅' if set(rooms) == {'R1', 'R2'} else 'FAIL ❌'}\n")


if __name__ == "__main__":
    scenario_A()
    scenario_B()
    scenario_C()
