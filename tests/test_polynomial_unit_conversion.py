import numpy as np
import pytest
from dt4acc_lib.bl.translator_service import TranslatorService, create_translation_object_for_polynom_coefficients
from dt4acc_lib.bl.unit_conversion import PolynomialUnitConversion
from dt4acc_lib.model.utils.identifiers import ConversionID, DevicePropertyID, LatticeElementPropertyID
from dt4acc_lib.model.utils.translator_manager_lookup_table import (
    PolynomCoefficients,
    Range,
    TranslatorLookupTable,
    TranslatorLookupTableElement,
)

BRHO = 629e6 / 299792458.0  # MLS, 629 MeV

# MLS S1 (S1M2K1RP), MML's amp2k (ao_mls.json): K * Brho = polyval(C, I), highest order first
S1_AMP2K = [-1.7066e-08, 0.0, 0.0, 0.0, 4.23273, 1.22127]
#: PolynomCoefficients: lowest order first
S1 = S1_AMP2K[::-1]


def _conv(coeffs=S1, *, scale=1.0, rng=Range(min=0.0, max=25.0), energy_dependent=True, brho=BRHO):
    cal = PolynomCoefficients(
        coeffs=coeffs, energy_dependent=energy_dependent, polynomial_of="device", scale=scale, range=rng
    )
    return PolynomialUnitConversion(calibration=cal, brho=brho)


@pytest.mark.parametrize("current", [0.0, 5.0, 12.5, 25.0, 60.0])
def test_inverse_is_amp2k(current):
    assert _conv().inverse(current) == pytest.approx(np.polyval(S1_AMP2K, current) / BRHO, rel=1e-14)


@pytest.mark.parametrize("current", [0.0, 0.3, 5.0, 12.5, 25.0])
def test_forward_inverts_inverse(current):
    conv = _conv()
    assert conv.forward(conv.inverse(current)) == pytest.approx(current, abs=1e-9)


def test_forward_beyond_range_takes_the_root_closest_to_linear():
    # 60 A: the 5th-order term is ~5 %; the other real roots are far off
    conv = _conv()
    assert conv.forward(conv.inverse(60.0)) == pytest.approx(60.0, abs=1e-9)


def test_scale():
    conv = _conv(scale=2.0, rng=Range(min=0.0, max=50.0))
    assert conv.inverse(10.0) == pytest.approx(np.polyval(S1_AMP2K, 5.0) / BRHO)
    assert conv.forward(conv.inverse(10.0)) == pytest.approx(10.0)


def test_energy_independent():
    conv = _conv(energy_dependent=False)
    assert conv.inverse(5.0) == pytest.approx(np.polyval(S1_AMP2K, 5.0))
    assert conv.forward(np.polyval(S1_AMP2K, 5.0)) == pytest.approx(5.0)


def test_linear_polynomial():
    conv = _conv([1.0, 2.0], rng=None, brho=3.0)
    assert conv.inverse(4.0) == pytest.approx(3.0)
    assert conv.forward(3.0) == pytest.approx(4.0)


def test_no_real_solution():
    # I^2 + 1 = k: no current gives k = 0
    with pytest.raises(ValueError, match="no real solution"):
        _conv([1.0, 1e-3, 1.0], rng=None, brho=1.0).forward(0.0)


@pytest.mark.parametrize("coeffs", [[1.0, 0.0], [0.0, 0.0, 1.0]])
def test_rejects_polynomial_without_linear_term(coeffs):
    with pytest.raises(ValueError):
        _conv(coeffs)


def test_translator_service_instantiates_it():
    id_ = ConversionID(
        LatticeElementPropertyID(element_name="S1M2K1RP", property="main_strength"),
        DevicePropertyID(device_name="S1P1RP", property="set_current"),
    )
    cal = PolynomCoefficients(coeffs=S1, energy_dependent=True, polynomial_of="device", range=Range(min=0.0, max=25.0))
    ts = TranslatorService(lut=TranslatorLookupTable(lut=[TranslatorLookupTableElement(id_, cal)]), brho=BRHO)
    conv = ts.get(id_)
    assert isinstance(conv, PolynomialUnitConversion)
    assert conv.inverse(10.0) == pytest.approx(np.polyval(S1_AMP2K, 10.0) / BRHO)


def test_reference_brho():
    # given at 2 brho: k = p(I) * 2 / brho
    cal = PolynomCoefficients(coeffs=S1, energy_dependent=True, reference_brho=2.0, polynomial_of="device")
    conv = PolynomialUnitConversion(calibration=cal, brho=BRHO)
    assert conv.inverse(5.0) == pytest.approx(np.polyval(S1_AMP2K, 5.0) * 2.0 / BRHO)
    assert conv.forward(conv.inverse(5.0)) == pytest.approx(5.0)


@pytest.mark.parametrize("energy_dependent", [False, True])
def test_linear_same_both_ways(energy_dependent):
    # device = b * (c0 + c1 * k)  <=>  k * b = device / c1 - c0 * b / c1
    c0, c1 = 0.1, 4.0
    b = BRHO if energy_dependent else 1.0
    by_lattice = create_translation_object_for_polynom_coefficients(
        PolynomCoefficients([c0, c1], energy_dependent=energy_dependent), brho=BRHO
    )
    by_device = create_translation_object_for_polynom_coefficients(
        PolynomCoefficients([-c0 * b / c1, 1.0 / c1], energy_dependent=energy_dependent, polynomial_of="device"),
        brho=BRHO,
    )
    assert isinstance(by_device, PolynomialUnitConversion)
    for current in (0.0, 3.0, 17.0):
        assert by_device.inverse(current) == pytest.approx(by_lattice.inverse(current))
    for k in (-1.0, 0.0, 2.5):
        assert by_device.forward(k) == pytest.approx(by_lattice.forward(k))


def test_polynomial_of_lattice_is_linear_only():
    with pytest.raises(ValueError, match="linear"):
        create_translation_object_for_polynom_coefficients(PolynomCoefficients(S1, energy_dependent=True), brho=BRHO)
