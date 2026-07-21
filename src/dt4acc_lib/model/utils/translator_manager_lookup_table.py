from dataclasses import dataclass
from functools import cached_property
from typing import Union, Sequence, Dict

from .command import ReadCommand
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

    def interpolate(self, lambda_: float):
        """A value that is 0 at min and 1 at max"""
        return self.min * (1 - lambda_) + self.max * lambda_


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


@dataclass
class NeedsAReference:
    """Use it with respect to an up-to-date reference

    Warning:
        It is used as a wrapper around other objects
        It just signals translation object that it needs
        to take incoming data with respect to some reference

    """
    # How to obtain the reference data
    # todo: activate me: should be there, user can set one of them to None
    #       and see what happens
    design_view_read_commnd: ReadCommand
    device_view_read_command: ReadCommand
    translation_object: Union[MultiplyerScaledByEnergy]


# just marking it as being special
@dataclass
class TuneConversionCoefficients:
    conversion : PolynomCoefficients


@dataclass
class RemapIdentifiersAndConvertDM:
    """Map turn by turn data names and coefficients
    """
    name_mapping: Dict[str, str]
    conversion: PolynomCoefficients


@dataclass
class TranslatorLookupTableElement:
    conversion_id: ConversionID
    # Todo: this needs to be much more flexible
    # But for the time being
    # Warning: IdentityMapper must go last!
    # otherwise it will get **always** instantiated
    conversion_info: Union[
        PolynomCoefficients,
        TuneConversionCoefficients,
        MultiplyerScaledByEnergy,
        NeedsAReference,
        RemapIdentifiersAndConvertDM,
        IdentityMapper,
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
