from typing import Sequence

from ..properties.interface import ElementPropertyInterface
from ...interfaces.simulator.element import ElementInterface


class PropertiesProxy(ElementInterface):
    def __init__(
        self,
        properties: Sequence[ElementPropertyInterface],
        element_id: str,
        name: str = None,
    ):
        self.properties = properties
        self.lut = {p.handles_property(): p for p in properties}
        self.element_id = element_id
        self.name = name

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.get_name()},"
            f" handles properties= {[p.handles_property() for p in self.properties]}"
            ")"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.get_name()},"
            f" properties = {self.properties}"
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
        await self.lut[p].update(value)

    def peek(self, property_id: str) -> object:
        assert property_id.startswith(
            "get_"
        ), f"Expected property starts with 'set_' but property_id was  {property_id}"
        p = property_id[4:]
        return self.lut[p].peek()
