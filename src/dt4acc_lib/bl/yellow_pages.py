from enum import Enum
from typing import Sequence, Union, Dict

from dt4acc_lib.interfaces.utils.yellow_pages import YellowPagesBase


class FamilyName(Enum):
    beam_position_monitors = "beam_position_monitors"
    quadrupoles = "quadrupoles"
    sextupoles = "sextupoles"
    steerers = "steerers"
    horizontal_steerers = "horizontal_steerers"
    vertical_steerers = "vertical_steerers"
    tune_correction_quadrupoles = "tune_correction_quadrupoles"
    master_clock = "master_clock"


class YellowPages(YellowPagesBase):
    """
    or use:
    get(family_name: str)
    separate yellow pages for lattice elements and devices
    """

    def __init__(self, d: Dict[str, Sequence[str]]):
        self._d = d

    def get_families(self) -> Sequence[str]:
        return tuple(self._d.keys())

    def get(self, family_name: Union[str, FamilyName]) -> Sequence[str]:
        # check for valid key?
        # key = str(FamilyName(family_name))
        return self._d[family_name]

    def quadrupole_names(self) -> Sequence[str]:
        """
        Todo:
            remove me: favour to only provide the .get interface
        """
        return self.get("quadrupoles")

    def tune_correction_quadrupole_names(self) -> Sequence[str]:
        """
        Todo:
            remove me: favour to only provide the .get interface
        """
        return self.get("tune_correction_quadrupoles")


__all__ = ["FamilyName", "YellowPages"]
