from .interface import ElementPropertyInterface
from .utils import manipulate_kick, peek_kick


class KickX(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "kick_x"

    async def update(self, obj, value: float):
        obj.update(KickAngle=manipulate_kick(obj.KickAngle, kick_x=value))

    def peek(self, obj) -> float:
        return peek_kick(obj, "kick_x")


class KickY(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "kick_y"

    async def update(self, obj, value: float):
        obj.update(KickAngle=manipulate_kick(obj.KickAngle, kick_y=value))

    def peek(self, obj) -> float:
        return peek_kick(obj, "kick_y")
