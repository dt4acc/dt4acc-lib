"""
Todo:
    The underlying object (e.g. a Quadrupole) does not
    need to have a KickAngle attribute

    What would be the appropriate treatment then?
    Fail or go ahead

"""
from .interface import ElementPropertyInterface
from .utils import manipulate_kick, peek_kick


class XKick(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "x_kick"

    async def update(self, obj, value: float):
        obj.update(KickAngle=manipulate_kick(obj.KickAngle, kick_x=value))

    def peek(self, obj) -> float:
        return peek_kick(obj, "x_kick")


class YKick(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "y_kick"

    async def update(self, obj, value: float):
        obj.update(KickAngle=manipulate_kick(obj.KickAngle, kick_y=value))

    def peek(self, obj) -> float:
        return peek_kick(obj, "y_kick")


__all__ = ["XKick", "YKick"]
