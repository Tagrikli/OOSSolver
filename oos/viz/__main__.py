"""Entry point:  python -m oos.viz [facility]"""

from __future__ import annotations

import argparse

from oos.facilities import FACILITIES
from oos.viz.app import run_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="oos.viz", description="Facility viz")
    ap.add_argument("facility", nargs="?", default=None,
                    choices=sorted(FACILITIES), help="facility to open (default: last used)")
    args = ap.parse_args()
    run_app(args.facility)


if __name__ == "__main__":
    main()
