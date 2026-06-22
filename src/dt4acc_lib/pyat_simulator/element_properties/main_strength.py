"""
Todo:
    consider if main strength for octopule should be provided.
    As the octupole is provided as multipole the liasion manager
    should map to "B4" directly

"""
from typing import Callable

import numpy as np

from .element_property_interface import ElementPropertyInterface
from .utils import estimate_dipole_main_field, update_magnetic_polynom_coefficients


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


class MainStrengthForOctupole(ElementPropertyInterface):
    """
    Todo:
        find out if there is also some parameter like K or H
    """
    def handles_property(self) -> str:
        return "main_strength"

    async def update(self, obj, value: float):
        #: Todo put this into a test
        new_poly = update_magnetic_polynom_coefficients(obj.PolynomB, {4: value})
        obj.PolynomB[:] = new_poly

        assert np.isclose(value, obj.PolynomB[3], rtol=1e-12, atol=1e-12)

    def peek(self, obj) -> float:
        r = float(obj.PolynomB[3])
        return r


class MainStrengthForDipole(ElementPropertyInterface):
    def __init__(self):
        super().__init__()
        self.get_reference_energy : Callable[[], float] = None

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def set_reference_energy_cb(self, cb: Callable[[], float]):
        self.get_reference_energy = cb

    def handles_property(self) -> str:
        return "main_strength"

    async def update(self, obj, value: object):
        raise NotImplementedError("Main strength for dipole not handled")

    def peek(self, obj) -> object:
        """estimate dipole field from energy and stored polynom B value"""
        assert self.get_reference_energy is not None
        beam_energy = self.get_reference_energy()
        r = estimate_dipole_main_field(
            beam_energy=beam_energy,
            dipole_angle=obj.BendingAngle,
            path_length=obj.Length,
        )
        # Todo: should the field PolynomB be added
        r = r + obj.PolynomB[0]
        return r



__all__ = ["MainStrengthForQuadrupole", "MainStrengthForSextupole"]
