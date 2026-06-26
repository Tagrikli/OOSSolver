"""`python -m oos.solver` — watch the complete planner run a facility live.

No policy picker, no RL: the OOSSolver drives the facility directly. A continuous
store stream arrives on its own; click a pallet to request its retrieve, or use
the side-panel buttons to queue stores. Rooms re-stage when idle.

    python -m oos.solver --facility campus            # manual: you drive it
    python -m oos.solver --facility campus --auto     # + continuous auto store stream

Manual (default): rooms auto-stage when idle; CLICK a pallet to request its
retrieve, or use the side-panel "queue small/big" buttons to add stores, and
watch the solver dig/deliver. --auto adds a self-arriving car stream on top.
"""

import argparse

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env import Environment
from oos.facilities import FACILITIES, get_facility
from oos.viz.app import run_app
from oos.viz.state_store import load_viz_state

if __name__ == "__main__":
    persisted = load_viz_state()
    default_facility = persisted.facility_name if persisted.facility_name in FACILITIES else "campus"

    p = argparse.ArgumentParser(description="Live OOSSolver visualizer (no RL).")
    p.add_argument("--facility", type=str, default=default_facility,
                   choices=sorted(FACILITIES.keys()),
                   help="Facility to run (default: last opened, else campus).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto", action="store_true",
                   help="Run a continuous self-arriving store stream (cars come, "
                        "park, and after a dwell are retrieved). Default: manual.")
    p.add_argument("--store-rate", type=float, default=0.04,
                   help="With --auto: mean store arrivals per sim-second.")
    args = p.parse_args()

    cfg = ExperimentConfig(
        task_stream=TaskStreamConfig(store_rate=args.store_rate if args.auto else 0.0),
        episode=EpisodeConfig(max_sim_time=float("inf"), max_steps=float("inf")),
    )
    env = Environment(facility_factory=get_facility(args.facility), experiment_config=cfg)
    run_app(env, seed=args.seed, facility_name=args.facility,
            use_solver=True, solver_auto=args.auto)
