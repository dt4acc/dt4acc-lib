import functools
import logging
from typing import Union

from .remap_and_convert import RemapIdentifiersAndConvert
from .tune_translator import TuneConversion
from dt4acc_lib.bl.identity_conversion import identity_conversion
from dt4acc_lib.bl.unit_conversion import (
    LinearUnitConversion,
    EnergyDependentLinearUnitConversion,
    MultiplierScaledByEnergyUnitConversion,
    UnitConversionAroundReferenceValue,
)
from dt4acc_lib.interfaces.utils.state_conversion import StateConversion
from dt4acc_lib.interfaces.utils.translator_service import TranslatorServiceBase
from dt4acc_lib.model.utils.identifiers import ConversionID
from dt4acc_lib.model.utils.translator_manager_lookup_table import (
    TranslatorLookupTable,
    PolynomCoefficients,
    TuneConversionCoefficients,
    IdentityMapper,
    MultiplyerScaledByEnergy,
    NeedsAReference,
    RemapIdentifiersAndConvertDM
)


logger = logging.getLogger("dt4acc_lib")


class TranslatorService(TranslatorServiceBase):
    """
    Todo:
        review if translation objects should only be instaniated
        when needed or during startup

        Proper handling of brho!

        Support _repr_html_ of similar for better output on jupyter notebooks
    """

    def __init__(self, *, lut: TranslatorLookupTable, brho: float):
        self.lut = lut
        self.brho = brho
        self.reference_cache = None
        self.to_needs_reference_cache = []

    def __str__(self):
        return f"{self.__class__.__name__}(brho=self.brho, lut with {len(self.lut.lut)} entries)"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            "("
            f"brho={repr(self.brho)}"
            f", lut={repr(self.lut)}"
        )

    def register_reference_cache(self, cache):
        self.reference_cache = cache
        for to in self.to_needs_reference_cache:
            to.register_reference_cache(self.reference_cache)

    def needs_reference_cache(self, to):
        assert callable(to.register_reference_cache)
        if to not in self.to_needs_reference_cache:
            self.to_needs_reference_cache.append(to)
        if self.reference_cache:
            to.register_reference_cache(self.reference_cache)

    @functools.lru_cache(maxsize=None)
    def get(self, id_: ConversionID) -> Union[StateConversion, None]:
        to = self._lookup_config(id_)
        if to is None:
            return None

        # Should one use a factory down here ?
        # Then it could be handled over and configured by the user
        r = create_translation_object_for_data_model(to, self.brho)
        # Todo: need to improve which objects need a reference
        #       furthermore, perhaps it should only be a week
        #       reference here, so it gets out of this list
        #       when not needed anymore else ....
        if isinstance(to, NeedsAReference) or isinstance(r, UnitConversionAroundReferenceValue):
            assert isinstance(to, NeedsAReference)
            assert isinstance(r, UnitConversionAroundReferenceValue)
            self.needs_reference_cache(r)
        return r

    def _lookup_config(self, id_: ConversionID):
        to = self.lut.get(id_)
        if to is not None:
            return to

        logger.error(
            f"{self.__class__.__name__}: I did not find id {id_} in lookup table"
        )
        od = self.objects_for_device(id_.device_property_id.device_name)
        logger.warning(f"{self.__class__.__name__}: For the device I know {od}")
        em = self.objects_for_lat_elem(id_.lattice_property_id.element_name)
        logger.warning(
            f"{self.__class__.__name__}: For the lattice element I know {em}"
        )
        raise KeyError(f"No translation object for {id_}")

    def objects_for_lat_elem(self, elm_name: str):
        return {
            key: self.lut.get(key)
            for key in self.lut.keys()
            if elm_name == key.lattice_property_id.element_name
        }

    def objects_for_device(self, dev_name: str):
        return {
            key: self.lut.get(key)
            for key in self.lut.keys()
            if dev_name == key.device_property_id.device_name
        }


def create_translation_object_for_data_model(
    data_model: Union[
        PolynomCoefficients | TuneConversionCoefficients, MultiplyerScaledByEnergy, NeedsAReference
    ],
    brho: float,
):
    if isinstance(data_model, IdentityMapper):
        return identity_conversion
    if isinstance(data_model, PolynomCoefficients):
        return create_translation_object_for_polynom_coefficients(data_model, brho)
    elif isinstance(data_model, TuneConversionCoefficients):
        return create_translation_object_for_tune(data_model)
    elif isinstance(data_model, MultiplyerScaledByEnergy):
        return create_translation_object_for_multiplyer_scaled_by_energy(
            data_model, brho
        )
    elif isinstance(data_model, NeedsAReference):
        return create_translation_object_for_needs_a_reference(data_model, brho)
    elif isinstance(data_model, RemapIdentifiersAndConvertDM):
        return create_translation_object_for_remap_and_convert(data_model, brho)
    else:
        # Todo: fix the error that is raised
        raise AssertionError(
            f"Don't know how to instantiate data_model of {type(data_model)} for {data_model}"
        )

def create_translation_object_for_remap_and_convert(data_model: RemapIdentifiersAndConvertDM, brho: float):
    conv = create_translation_object_for_polynom_coefficients(data_model.conversion, brho)
    r = RemapIdentifiersAndConvert(data_model.name_mapping, conv)
    return r


def create_translation_object_for_tune(
    data_model: TuneConversionCoefficients,
) -> TuneConversion:
    assert (
        not data_model.conversion.energy_dependent
    ), "Expect the tune conversion to be energy independent"
    return TuneConversion(
        create_translation_object_for_polynom_coefficients(
            data_model.conversion, brho=None
        )
    )


def create_translation_object_for_polynom_coefficients(
    data_model: PolynomCoefficients,
    brho: float,
) -> Union[LinearUnitConversion, EnergyDependentLinearUnitConversion]:
    intercept, slope = data_model.coeffs
    if data_model.energy_dependent:
        return EnergyDependentLinearUnitConversion(
            intercept=intercept, slope=slope, brho=brho
        )
    else:
        return LinearUnitConversion(slope=slope, intercept=intercept)


def create_translation_object_for_multiplyer_scaled_by_energy(
    data_model: MultiplyerScaledByEnergy, brho: float
) -> MultiplierScaledByEnergyUnitConversion:
    return MultiplierScaledByEnergyUnitConversion(conv_data=data_model, brho=brho)


def create_translation_object_for_needs_a_reference(
        data_model: NeedsAReference, brho: float
) -> UnitConversionAroundReferenceValue:
    """
    Todo:
        should it be able to look up the conversion for the given
        data model.translation_object?
        It could exist already
    """
    to = create_translation_object_for_data_model(data_model.translation_object, brho)
    return UnitConversionAroundReferenceValue(
        sub_obj=to,
        # forward_rcmd=data_model.forward_read_command,
        design_view_rcmd=data_model.design_view_read_commnd,
        # Todo: should be a specific read command
        device_view_rcmd=data_model.device_view_read_command,
    )

__all__ = ["TranslatorService"]
