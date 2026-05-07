from .interface import ElementPropertyInterface
from .utils import estimate_shift, update_shift


class Roll(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "roll"

    async def update(self, value: float):
        self.obj.set_roll(value)

    def peek(self) -> float:
        """
        Todo:
            need to check that it works!
        """
        raise NotImplementedError("Check this functionallity!")
        return self.obj.get_roll()


class Dx(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "dx"

    async def update(self, value: float):
        return update_shift(self.obj, dx=value, dy=None)

    def peek(self) -> float:
        return float(estimate_shift(self.obj)[0])


class Dy(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "dy"

    async def update(self, value: float):
        return update_shift(self.obj, dx=None, dy=value)

    def peek(self) -> float:
        return float(estimate_shift(self.obj)[1])


__all__ = ["Roll", "Dx", "Dy"]
