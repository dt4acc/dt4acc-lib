from dt4acc_lib.pyat_simulator.element_properties.element_property_interface import ElementPropertyInterface


class Frequency(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "frequency"

    async def update(self, obj, value: float):
        obj.update(Frequency=value)

    def peek(self, obj) -> float:
        return float(obj.Frequency)


class Voltage(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "voltage"

    async def update(self, obj, value: float):
        obj.update(Voltage=value)

    def peek(self, obj) -> float:
        return float(obj.Voltage)


__all__ = ["Frequency", "Voltage"]
