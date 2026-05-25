"""Hand-authored facilities."""

from typing import Callable

from oos.facilities.tiny import make_facility as make_tiny_facility
from oos.facilities.tiny_tall import make_facility as make_tiny_tall_facility
from oos.facilities.tiny_wide import make_facility as make_tiny_wide_facility
from oos.sim.facility import SeedingConfig
from oos.sim.topology import Topology

FacilityFactory = Callable[[], tuple[Topology, SeedingConfig]]

FACILITIES: dict[str, FacilityFactory] = {
    "tiny":      make_tiny_facility,
    "tiny_tall": make_tiny_tall_facility,
    "tiny_wide": make_tiny_wide_facility,
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
