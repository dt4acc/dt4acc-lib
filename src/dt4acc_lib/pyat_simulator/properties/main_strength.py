"""
Todo:
    consider if main strength for octopule should be provided.
    As the octupole is provided as multipole the liasion manager
    should map to "B4" instead of anysthing else

"""
import numpy as np

from .interface import ElementPropertyInterface


class MainStrengthForQuadrupole(ElementPropertyInterface):
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def handles_property(self) -> str:
        return "main_strength"

    async def update(self, obj, value: float):
        obj.update(K=value)
        #: Todo put this into a test
        assert np.isclose(value, obj.PolynomB[1], rtol=1e-12, atol=1e-12)

    def peek(self, obj) -> float:
        r = float(obj.K)
        #: Todo put this into a test
        assert np.isclose(r, obj.PolynomB[1], rtol=1e-12, atol=1e-12)
        return r


class MainStrengthForSextupole(ElementPropertyInterface):
    def handles_property(self) -> str:
        return "main_strength"

    async def update(self, obj, value: float):
        obj.update(H=value)
        #: Todo put this into a test
        assert np.isclose(value, obj.PolynomB[2], rtol=1e-12, atol=1e-12)

    def peek(self, obj) -> float:
        r = float(obj.H)
        #: Todo put this into a test
        assert np.isclose(r, obj.PolynomB[2], rtol=1e-12, atol=1e-12)
        return r


__all__ = ["MainStrengthForQuadrupole", "MainStrengthForSextupole"]
