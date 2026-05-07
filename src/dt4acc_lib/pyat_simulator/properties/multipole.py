from enum import Enum

from .interface import ElementPropertyInterface
from .utils import check_multipole_index, update_magnetic_polynom


class NormalSkew(Enum):
    """
    Todo:
        Find a good name for it
        check where this is already defined and reuse it
    """

    normal= "normal"
    skew="skew"


class Multipole(ElementPropertyInterface):
    """
    Revisit name

    Coefficient indexing follows currently the European Convention
    Thus: dipole = 1, ....
    """
    def __init__(self, normal_skew: NormalSkew, n_multipole: int):
        """
        Todo:
            See how multipoles should be used
        """
        check_multipole_index(n_multipole)
        self.n_multipole = n_multipole
        self.normal_skew = normal_skew

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f" normal_skew={self.normal_skew.value},"
            f" multipole_index={self.n_multipole}"
            ")"
        )

    def handles_property(self) -> str:
        if self.normal_skew.value == "normal":
            return f"B{self.n_multipole:d}"
        elif self.normal_skew.value == "skew":
            return f"A{self.n_multipole:d}"
        else:
            raise AssertionError("Should not end up here!")

    async def update(self, obj, value: float):
        if self.normal_skew.value == "normal":
            update_magnetic_polynom(obj.PolynomB, {self.n_multipole: value})
        elif self.normal_skew.value == "skew":
            update_magnetic_polynom(obj.PolynomA, {self.n_multipole: value})
        else:
            raise AssertionError("Should not end up here!")

    def peek(self, obj) -> float:
        """
        Todo:
            what to return if the polynom is not available?
            Could return 0 as the underlying code does the same
            calculation most probably
        """
        if self.normal_skew.value == "normal":
            return float(obj.PolynomB[self.n_multipole - 1])
        elif self.normal_skew.value == "skew":
            return float(obj.PolynomA[self.n_multipole - 1])
        else:
            raise AssertionError("Should not end up here!")
