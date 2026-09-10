import logging
from typing import Dict, Callable

from ..element_properties.element_property_interface import ElementPropertyInterface
from ..element_properties.main_strength import MainStrengthForDipole
from ...interfaces.simulator.element import ElementInterface

logger = logging.getLogger("dt4acc-lib")


class PropertiesProxy(ElementInterface):
    """Delegates modification of an element property to a dedicated handler"""

    def __init__(
        self,
        obj,
        properties_lut: Dict[str, ElementPropertyInterface],
        get_reference_energy: Callable[[], float],
        element_id: str,
        name: str = None,
        log=logger,
    ):
        self.obj = obj
        self.lut = properties_lut
        self.get_reference_energy = get_reference_energy
        self.element_id = element_id
        self.name = name
        self.log = log

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.get_name()},"
            f" obj={self.obj},"
            f" properties={list(self.lut)}"
            ")"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.get_name()},"
            f" obj={repr(self.obj)},"
            f" properties={self.lut}"
            ")"
        )

    def get_name(self):
        if self.name:
            return self.name
        return self.element_id

    def get_property_proxy(self, property_id) -> ElementPropertyInterface:
        try:
            r = self.lut[property_id]
        except KeyError as ke:
            self.log.warning(
                f"{self.__class__.__class__}(name={self.name})"
                f" can't resolve {property_id}."
                f" known properties: {list(self.lut)}"
            )
            raise ke
        return r

    async def update(self, property_id: str, value: object):
        if property_id.startswith("set_"):
            property_id = property_id[4:]
        else:
            self.log.debug(
                f"{self.__class__.__name__}("
                "name={self.name},"
                ")"
                f" updating object {self.obj} with {property_id} which does not start with set_"
            )
        # assert property_id.startswith(
        #     "set_"
        # ), f"Expected property starts with 'set_' but property_id was  {property_id}"
        pp = self.get_property_proxy(property_id)
        if isinstance(pp, MainStrengthForDipole):
            assert self.get_reference_energy is not None
            pp.set_reference_energy_cb(self.get_reference_energy)
            try:
                await pp.update(self.obj, value)
            finally:
                pp.set_reference_energy_cb(None)
        else:
            await pp.update(self.obj, value)

    def peek(self, property_id: str) -> object:
        # assert property_id.startswith(
        #     "get_"
        # ), f"Expected property starts with 'get_' but property_id was  {property_id}"
        p = property_id
        pp = self.get_property_proxy(p)
        # Todo: Fix me:
        # I add info that should not be here
        if isinstance(pp, MainStrengthForDipole):
            # If so that should be within a context manager ...
            assert self.get_reference_energy is not None
            pp.set_reference_energy_cb(self.get_reference_energy)
            try:
                return pp.peek(self.obj)
            finally:
                pp.set_reference_energy_cb(None)
        else:
            return pp.peek(self.obj)


__all__ = ["PropertiesProxy"]
