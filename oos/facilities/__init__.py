"""Hand-authored facilities."""

from typing import Callable

from oos.facilities.dev import make_facility as make_dev_facility
from oos.facilities.dibaji import make_facility as make_dibaji_facility
from oos.facilities.medipol import make_facility as make_medipol_facility
from oos.facilities.random_gen import make_random_facility
from oos.facilities.tiny import make_facility as make_tiny_facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology

FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]

FACILITIES: dict[str, FacilityFactory] = {
    "dev": make_dev_facility,
    "tiny": make_tiny_facility,
    "dibaji": make_dibaji_facility,
    "medipol": make_medipol_facility,
    # Each call returns a fresh layout (wall-clock-seeded). The viz can
    # bypass the picker's "active" cache via the 'g' hotkey to re-roll
    # without having to switch facilities first.
    "random": lambda: make_random_facility(seed=None),
}

# Back-compat default: anything still importing `make_facility` keeps getting dev.
make_facility = make_dev_facility


def get_facility(name: str) -> FacilityFactory:
    """Look up a facility factory by name. Raises with a useful message on miss."""
    try:
        return FACILITIES[name]
    except KeyError as e:
        raise ValueError(
            f"unknown facility {name!r}; available: {sorted(FACILITIES)}"
        ) from e


__all__ = ["FACILITIES", "FacilityFactory", "get_facility", "make_facility"]
