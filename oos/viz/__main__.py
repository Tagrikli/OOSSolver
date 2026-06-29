"""Entry point:  python -m oos.viz [facility] [--runs DIR]"""

from __future__ import annotations

import argparse

from oos.facilities import FACILITIES
from oos.viz.app import run_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="oos.viz", description="Facility viz / RL inspector")
    ap.add_argument("facility", nargs="?", default=None,
                    choices=sorted(FACILITIES), help="facility to open (default: last used)")
    ap.add_argument("--runs", default="runs", help="directory scanned for *.pt checkpoints")
    args = ap.parse_args()
    run_app(args.facility, runs_dir=args.runs)


if __name__ == "__main__":
    main()
