from typing import Sequence, Dict

from ..properties.interface import ElementPropertyInterface
from ...interfaces.simulator.element import ElementInterface


class PropertiesProxy(ElementInterface):
    def __init__(
        self,
        obj,
        properties_lut: Dict[str, ElementPropertyInterface],
        element_id: str,
        name: str = None,
    ):
        self.obj = obj
        self.lut = properties_lut
        self.element_id = element_id
        self.name = name

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

    async def update(self, property_id: str, value: object):
        assert property_id.startswith(
            "set_"
        ), f"Expected property starts with 'set_' but property_id was  {property_id}"
        p = property_id[4:]
        await self.lut[p].update(self.obj, value)

    def peek(self, property_id: str) -> object:
        assert property_id.startswith(
            "get_"
        ), f"Expected property starts with 'set_' but property_id was  {property_id}"
        p = property_id[4:]
        return self.lut[p].peek(self.obj)
