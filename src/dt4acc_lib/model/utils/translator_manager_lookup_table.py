from dataclasses import dataclass
from functools import cached_property
from typing import Union, Sequence

from .identifiers import ConversionID


@dataclass
class IdentityMapper:
    """Nothing to do ... just feed the data through

    Needed to mark it in the data
    """

    pass


@dataclass
class PolynomCoefficients:
    coeffs: Sequence[float]
    #: does the conversion depend on the beam energy
    energy_dependent: bool


@dataclass
class CurvePoint:
    """An interpolation point

    Use a sequence of these to construct a curve
    """

    indep: float
    dep: float


@dataclass
class Range:
    min: float
    max: float

    def scale(self, value: float) -> float:
        return (value - self.min) / (self.max - self.min)


@dataclass
class MultiplyerScaledByEnergy:
    """Data whose independent variable is the beam energy

    The way how ALS does it

    These data should be presumably calculatable to
    only depend on the magnet currents

    Calculation:

    .. math::
        s = \\frac{v - r_{min}} {r_{max} - r_{min}}

    """

    reference_multiplyer: float
    "The multiplier to use for the reference energy"
    reference_energy: float
    "The energy the multiplier is valid without scale"
    range: Range
    scale_by_energy: Sequence[CurvePoint]


# just marking it as being special
@dataclass
class TuneConversionCoefficients:
    conversion: PolynomCoefficients


@dataclass
class TranslatorLookupTableElement:
    conversion_id: ConversionID
    # Todo: this needs to be much more flexible
    # But for the time being
    conversion_info: Union[
        IdentityMapper,
        PolynomCoefficients,
        TuneConversionCoefficients,
        MultiplyerScaledByEnergy,
    ]


@dataclass
class TranslatorLookupTable:
    lut: Sequence[TranslatorLookupTableElement]

    def verify(self):
        # Construct the dictionary
        # will fail e.g. if lists are used as conversion identifiers
        _ = self._dict

    def get(self, item: ConversionID):
        return self._dict.get(item)

    def keys(self):
        return self._dict.keys()

    @cached_property
    def _dict(self):
        return {entry.conversion_id: entry.conversion_info for entry in self.lut}
