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


# just marking it as being special
@dataclass
class TuneConversionCoefficients:
    conversion: PolynomCoefficients


@dataclass
class TranslatorLookupTableElement:
    conversion_id: ConversionID
    # Warning: IdentityMapper must go last!
    # otherwise it will get **always** instantiated
    conversion_info: Union[
        TuneConversionCoefficients, PolynomCoefficients, IdentityMapper
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
