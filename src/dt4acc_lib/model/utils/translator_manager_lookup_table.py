from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Literal, Optional, Sequence, Union

from .command import ReadCommand
# CurvePoint is defined in identifiers; importable from both modules
from .identifiers import ConversionID, CurvePoint  # noqa: F401


@dataclass
class IdentityMapper:
    """Nothing to do ... just feed the data through

    Needed to mark it in the data
    """

    pass


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
class PolynomCoefficients:
    """A calibration given as a polynomial, ``coeffs`` lowest order first

    ``polynomial_of`` says what the polynomial is a function of:

    * ``"lattice"``: device = brho * (c0 + c1 * lattice), linear only.
      ``energy_dependent``: the coefficients are per brho, else brho = 1.
    * ``"device"``: lattice = (c0 + c1 * x + c2 * x**2 + ...) / brho, with
      x = device / ``scale``, of any order, as MML's ``amp2k`` (e.g. the MLS
      sextupoles: the polynomial gives ``K * Brho``). lattice -> device
      solves for the root, as MML's ``k2amp``: the real one closest to the
      linear solution, the one in ``range`` if any is. ``energy_dependent``:
      divided by the beam's brho, else brho = 1.

    Both are the same thing for a linear calibration, only written the other
    way round; "device" is needed for higher orders, where the inverse is no
    polynomial any more.

    ``reference_brho``: the facility gives the calibration at that brho (e.g.
    a steerer's kick per current at the design energy); the conversion
    scales it with the beam's brho. Needs ``energy_dependent``.
    """

    coeffs: Sequence[float]
    #: does the conversion depend on the beam energy
    energy_dependent: bool
    #: brho the coefficients were given at (None: they are per brho)
    reference_brho: Optional[float] = None
    #: the polynomial is a function of the lattice value or the device value
    polynomial_of: Literal["lattice", "device"] = "lattice"
    #: "device" only: the device value is divided by it before the polynomial is evaluated
    scale: float = 1.0
    #: "device" only: setpoints the polynomial is valid for
    range: Optional[Range] = None


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
