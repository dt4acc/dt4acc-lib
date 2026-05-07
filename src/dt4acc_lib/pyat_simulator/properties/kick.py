from .interface import ElementPropertyInterface
from .utils import manipulate_kick


class KickX(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "kick_x"

    async def update(self, value: float):
        self.obj.update(KickAngle=manipulate_kick(self.obj.KickAngle, kick_x=value))

    def peek(self) -> float:
        return peek_kick(self.obj, "kick_x")


class KickY(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "kick_y"

    async def update(self, value: float):
        self.obj.update(KickAngle=manipulate_kick(self.obj.KickAngle, kick_y=value))

    def peek(self) -> float:
        return peek_kick(self.obj, "kick_y")
