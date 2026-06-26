"""Drive the real VizApp loop headlessly: select PLANNER, add buried retrieves,
press-S (solve thread), poll to completion, then play (SimDriver.drive_anim)
and confirm delivery. Exercises the exact code paths the GUI uses.

Run:  SDL_VIDEODRIVER=dummy python scripts/_spike_viz_planner.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from oos.env import Environment
from oos.facilities import get_facility
from oos.sim.state import Pallet
from oos.sim.tasks import Retrieve
from oos.viz.app import VizApp
from oos.viz.pickers.policy import PLANNER_ENTRY
from oos.viz.policy_swap import load_policy


def main():
    import pygame
    pygame.init()
    env = Environment(facility_factory=get_facility("tiny_medipol"))
    appcfg = VizApp(env=env, policy=lambda o, i: 0, facility_name="tiny_medipol")
    s = appcfg._init_state()

    # Select the planner (same call the policy picker makes on Enter).
    load_policy(PLANNER_ENTRY, s.agent, env.topology, True, s.toasts)

    # Two buried retrieves on different lifts + one on a shuttle (handoff).
    for ss in env.engine.state.shelves.values():
        ss.stack = []
    env.engine.state.shelves["A1"].stack = [Pallet(100, "small"), Pallet(101, "empty")]
    env.engine.state.shelves["E1"].stack = [Pallet(200, "big"), Pallet(201, "empty")]
    env.engine.state.shelves["B1"].stack = [Pallet(300, "small"), Pallet(301, "empty")]
    for pid, d in [(100, 1), (200, 1), (300, 1)]:
        env.engine.queue.add(Retrieve(arrived_at=0.0, pallet=pid, initial_depth=d, already_staged=False))
    env.wake_waiting_carriers()

    # Press S -> background solve. Poll like the run loop does.
    appcfg._start_solve(s)
    t0 = time.perf_counter()
    while appcfg._solving and time.perf_counter() - t0 < 10.0:
        appcfg._poll_solve(s)
        time.sleep(0.005)
    appcfg._poll_solve(s)
    p = s.agent.policy
    print(f"SOLVE: {p.last_solve_seconds*1000:.1f} ms | steps={p.last_plan_steps} | "
          f"has_plan={p.has_plan} | unsolved={p.unsolved}")

    # Press SPACE (play) and tick frames until the plan drains.
    s.paused = False
    for _ in range(4000):
        appcfg._tick_sim(s, dt_wall=0.5)   # speed default 1.0 => +0.5s sim/frame
        remaining = p.steps_left
        pending = [t for t in env.engine.queue.pending if isinstance(t, Retrieve)]
        busy = any(c.is_busy for c in env.engine.state.carriers.values())
        if remaining == 0 and not pending and not busy:
            break

    delivered = len(env.engine.queue.completed_costs)
    pend = [t.pallet for t in env.engine.queue.pending if isinstance(t, Retrieve)]
    print(f"PLAY:  delivered={delivered}/3 | still pending={pend}")
    print(f"RESULT: {'PASS ✅  GUI path solves + plays' if delivered == 3 and not pend else 'FAIL ❌'}")


if __name__ == "__main__":
    main()
