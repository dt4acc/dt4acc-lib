import io
import json

import pytest
import jsons

from dt4acc_lib.model.utils.identifiers import (
    ConversionID,
    LatticeElementPropertyID,
    DevicePropertyID,
)
from dt4acc_lib.model.utils.translator_manager_lookup_table import (
    IdentityMapper,
    PolynomCoefficients,
    TuneConversionCoefficients,
    TranslatorLookupTableElement,
)


@pytest.mark.parametrize(
    "payload, expected_conversion_info_type, expected_conversion_info",
    [
        (
            {
                "conversion_id": {
                    "lattice_property_id": {
                        "element_name": "Q1",
                        "property": "k1",
                    },
                    "device_property_id": {
                        "device_name": "PS1",
                        "property": "voltage",
                    },
                },
                "conversion_info": {},
            },
            IdentityMapper,
            IdentityMapper(),
        ),
        (
            {
                "conversion_id": {
                    "lattice_property_id": {
                        "element_name": "Q1",
                        "property": "k1",
                    },
                    "device_property_id": {
                        "device_name": "PS1",
                        "property": "voltage",
                    },
                },
                "conversion_info": {
                    "coeffs": [1.0, 2.0, 3.0],
                    "energy_dependent": True,
                },
            },
            PolynomCoefficients,
            PolynomCoefficients(coeffs=[1.0, 2.0, 3.0], energy_dependent=True),
        ),
        (
            {
                "conversion_id": {
                    "lattice_property_id": {
                        "element_name": "Q1",
                        "property": "k1",
                    },
                    "device_property_id": {
                        "device_name": "PS1",
                        "property": "voltage",
                    },
                },
                "conversion_info": {
                    "conversion": {
                        "coeffs": [0.5, 1.5],
                        "energy_dependent": False,
                    }
                },
            },
            TuneConversionCoefficients,
            TuneConversionCoefficients(
                conversion=PolynomCoefficients(
                    coeffs=[0.5, 1.5],
                    energy_dependent=False,
                )
            ),
        ),
    ],
)
def test_translator_lookup_table_element_is_deserialized_correctly(
    payload,
    expected_conversion_info_type,
    expected_conversion_info,
):
    stream = io.StringIO(json.dumps(payload))

    obj = jsons.loads(stream.getvalue(), TranslatorLookupTableElement)

    assert isinstance(obj, TranslatorLookupTableElement)
    assert isinstance(obj.conversion_id, ConversionID)
    assert isinstance(obj.conversion_id.lattice_property_id, LatticeElementPropertyID)
    assert isinstance(obj.conversion_id.device_property_id, DevicePropertyID)

    assert isinstance(obj.conversion_info, expected_conversion_info_type)
    assert obj.conversion_info == expected_conversion_info
    assert obj.conversion_id == ConversionID(
        lattice_property_id=LatticeElementPropertyID(
            element_name="Q1",
            property="k1",
        ),
        device_property_id=DevicePropertyID(
            device_name="PS1",
            property="voltage",
        ),
    )
