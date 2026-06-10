import logging
from typing import Sequence

from dt4acc_lib.interfaces.utils.liaison_manager import LiaisonManagerBase
from dt4acc_lib.model.utils.identifiers import (
    LatticeElementPropertyID,
    DevicePropertyID,
)
from dt4acc_lib.model.utils.liaison_manager_lookup_table import (
    LiaisonManagerInverseLookupTable,
    LiaisonManagerForwardLookupTable
)

logger = logging.getLogger("dt4acc_lib")


class LiaisonManager(LiaisonManagerBase):
    """
    Todo:
        consider internally to represent classes of devices with a certain functionallity

        So internally have
            * classes of devices providing similar properties
            * this can be used when searching for suggesting alternatives to the
              user when searching for it

        Logging of internal knowledge? Should that be provided

        Provide _repr_html_ or similar for better output on notebooks
    """

    def __init__(
        self,
        forward_lut: LiaisonManagerForwardLookupTable,
        inverse_lut: LiaisonManagerInverseLookupTable,
    ):
        if forward_lut:
            forward_lut.verify()
        if inverse_lut:
            inverse_lut.verify()

        self.forward_lut = forward_lut
        self.inverse_lut = inverse_lut

    def __str__(self):
        if self.forward_lut:
            fwd_txt = f"fwd lut with {len(self.forward_lut.lut)} entries"
        else:
            fwd_txt = "no fwd lut"
        if self.inverse_lut:
            inv_txt = f"inv lut with {len(self.inverse_lut.lut)} entries"
        else:
            inv_txt = "no inv lut"
        return f"{self.__class__.__name__}({fwd_txt}, {inv_txt})"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"fwd_lut={repr(self.forward_lut)}"
            f", inv_lut={repr(self.inverse_lut)}"
            ")"
        )

    def forward(self, id_: LatticeElementPropertyID) -> DevicePropertyID:
        devs_p = self.forward_lut.get(id_)
        if devs_p is not None:
            # assert len(devs_p) == 1, f"lattice id {id_} translated to {devs_p}. expected only 1"
            return devs_p

        logger.error(
            f"{self.__class__.__name__} id {id_} not found in lookup table"
        )

        em = self.objects_for_lattice_element(id_.element_name)
        logger.warning(f"{self.__class__.__name__}: For the element I know {em}")
        raise KeyError(f"forward lut does not contain entry {id_}")

    def inverse(self, id_: DevicePropertyID) -> Sequence[LatticeElementPropertyID]:
        lp = self.inverse_lut.get(id_)
        if lp is not None:
            return lp

        logger.error(
            f"{self.__class__.__name__} id {id_} not found in lookup table"
        )

        od = self.objects_for_device(id_.device_name)
        logger.warning(f"{self.__class__.__name__}: For the device I know {od}")

        # Todo: give the user a hint what we know and what is close to what we know
        raise KeyError(f"inverse lut does not contain entry {id_}")

    def known_lattice_elements(self) -> Sequence[str]:
        return [key.element_name for key in self.forward_lut.keys()]

    def known_device_elements(self) -> Sequence[str]:
        return [key.device_name for key in self.inverse_lut.keys()]

    def objects_for_lattice_element(self, elem_name: str):
        return {
            key : self.forward_lut.get(key)
            for key in self.forward_lut.keys()
            if elem_name == key.element_name
        }

    def objects_for_device(self, dev_name: str):
        return {
            key : self.inverse_lut.get(key)
            for key in self.inverse_lut.keys()
            if dev_name == key.device_name
        }


__all__ = ["LiaisonManager"]
