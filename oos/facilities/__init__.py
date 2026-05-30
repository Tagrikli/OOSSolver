"""Hand-authored facilities."""

from typing import Callable

from oos.facilities.campus import make_facility as make_campus_facility
from oos.facilities.dibaji import make_facility as make_dibaji_facility
from oos.facilities.mini import make_facility as make_mini_facility
from oos.facilities.stacker import make_facility as make_stacker_facility
from oos.facilities.stacker_deep import make_facility as make_stacker_deep_facility
from oos.facilities.stacker_wide import make_facility as make_stacker_wide_facility
from oos.facilities.tiny import make_facility as make_tiny_facility
from oos.facilities.tiny_medipol import make_facility as make_tiny_medipol_facility
from oos.facilities.tiny_tall import make_facility as make_tiny_tall_facility
from oos.facilities.tiny_wide import make_facility as make_tiny_wide_facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology

FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]

FACILITIES: dict[str, FacilityFactory] = {
    "mini":         make_mini_facility,
    "tiny":         make_tiny_facility,
    "tiny_tall":    make_tiny_tall_facility,
    "tiny_wide":    make_tiny_wide_facility,
    "tiny_medipol": make_tiny_medipol_facility,
    "stacker":      make_stacker_facility,
    "stacker_deep": make_stacker_deep_facility,
    "stacker_wide": make_stacker_wide_facility,
    "dibaji":       make_dibaji_facility,
    "campus":       make_campus_facility,
}


def get_facility(name: str) -> FacilityFactory:
    """Look up a facility factory by name. Raises with a useful message on miss."""
    try:
        return FACILITIES[name]
    except KeyError as e:
        raise ValueError(
            f"unknown facility {name!r}; available: {sorted(FACILITIES)}"
        ) from e


__all__ = ["FACILITIES", "FacilityFactory", "get_facility"]
