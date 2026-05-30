"""Entry point: `python -m oos.viz` runs a facility with a random policy.

Pass --facility to pick which hand-authored facility to visualize.
"""

import argparse

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env import Environment
from oos.facilities import FACILITIES, get_facility
from oos.agent import random_policy
from oos.viz.app import run_app
from oos.viz.state_store import load_viz_state

if __name__ == "__main__":
    # Defaults pull from the persisted viz state so re-launching picks up
    # whatever you were last looking at. Explicit --facility on the CLI
    # always wins over the persisted value.
    persisted = load_viz_state()
    default_facility = persisted.facility_name or "tiny_medipol"
    if default_facility not in FACILITIES:
        default_facility = "tiny_medipol"

    p = argparse.ArgumentParser()
    p.add_argument("--facility", type=str, default=default_facility,
                   choices=sorted(FACILITIES.keys()),
                   help="Which hand-authored facility to visualize. "
                        "Defaults to the last opened facility if persisted.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Fallback boot task-stream config. Overridden in-app by any persisted
    # AUTO-QUEUE tab values (VizApp._init_state) and editable live from that
    # tab. Schema dwell defaults are 30 s mean / 10 s std.
    cfg = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=0.30,        # ~one store arrival every 3 sim-seconds
        ),
        episode=EpisodeConfig(max_sim_time=2000.0, max_steps=20_000),
    )
    env = Environment(facility_factory=get_facility(args.facility), experiment_config=cfg)
    run_app(env, policy=random_policy, seed=args.seed, facility_name=args.facility)
