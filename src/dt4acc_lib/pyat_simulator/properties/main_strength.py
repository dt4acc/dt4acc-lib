"""
Todo:
    consider if main strength for octopule should be provided.
    As the octupole is provided as multipole the liasion manager
    should map to "B4" instead of anysthing else

"""
import numpy as np

from .interface import ElementPropertyInterface


class MainStrengthForQuadrupole(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "main_strength"

    async def update(self, value: float):
        self.obj.update(K=value)
        #: Todo put this into a test
        assert np.isclose(value, self.obj.PolynomB[1], rtol=1e-12, atol=1e-12)

    def peek(self) -> float:
        r = float(self.obj.K)
        #: Todo put this into a test
        assert np.isclose(r, self.obj.PolynomB[1], rtol=1e-12, atol=1e-12)
        return r


class MainStrengthForSextupole(ElementPropertyInterface):
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return f"{self.__class__.__name__}(obj={self.obj})"

    def handles_property(self) -> str:
        return "main_strength"

    async def update(self, value: float):
        self.obj.update(H=value)
        #: Todo put this into a test
        assert np.isclose(value, self.obj.PolynomB[2], rtol=1e-12, atol=1e-12)

    def peek(self) -> float:
        r = float(self.obj.H)
        #: Todo put this into a test
        assert np.isclose(r, self.obj.PolynomB[2], rtol=1e-12, atol=1e-12)
        return r


__all__ = ["MainStrengthForQuadrupole", "MainStrengthForSextupole"]
