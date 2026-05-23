"""Entry point: `python -m oos.viz` runs a facility with a random policy.

Pass --facility to pick which hand-authored facility to visualize.
"""

import argparse

from oos.config.schema import EpisodeConfig, ExperimentConfig, TaskStreamConfig
from oos.env.env import OOSEnv
from oos.facilities import FACILITIES, get_facility
from oos.viz.app import run_app
from oos.viz.player import random_policy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--facility", type=str, default="dev",
                   choices=sorted(FACILITIES.keys()),
                   help="Which hand-authored facility to visualize.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = ExperimentConfig(
        task_stream=TaskStreamConfig(
            store_rate=0.30,        # one store arrival every ~3 sim seconds
            # mean/std dwell use schema defaults (5 min mean, ~2 min std)
        ),
        episode=EpisodeConfig(max_sim_time=2000.0, max_steps=20_000),
    )
    env = OOSEnv(facility_factory=get_facility(args.facility), experiment_config=cfg)
    run_app(env, policy=random_policy, seed=args.seed, facility_name=args.facility)
