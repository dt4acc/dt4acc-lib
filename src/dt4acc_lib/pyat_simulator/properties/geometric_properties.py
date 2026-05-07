from .interface import ElementPropertyInterface
from .utils import estimate_shift, update_shift


class Roll(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "roll"

    async def update(self, obj, value: float):
        obj.set_roll(value)

    def peek(self, obj) -> float:
        """
        Todo:
            need to check that it works!
        """
        raise NotImplementedError("Check this functionallity!")
        obj.get_roll()


class Dx(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "dx"

    async def update(self, obj, value: float):
        return update_shift(obj, dx=value, dy=None)

    def peek(self, obj) -> float:
        return float(estimate_shift(obj)[0])


class Dy(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "dy"

    async def update(self, obj, value: float):
        return update_shift(obj, dx=None, dy=value)

    def peek(self, obj) -> float:
        return float(estimate_shift(obj)[1])


__all__ = ["Roll", "Dx", "Dy"]
