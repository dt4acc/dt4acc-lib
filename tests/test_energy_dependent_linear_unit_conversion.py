"""
Todo:
    add inverse test
"""
import pytest

from dt4acc_lib.bl.unit_conversion import EnergyDependentLinearUnitConversion


@pytest.fixture
def unit_conversion():
    return EnergyDependentLinearUnitConversion(
        brho=5, slope=2, intercept=7
    )


def test_linear_conversion_forward(unit_conversion):
    assert unit_conversion.forward(0) == pytest.approx(7 * 5, 1e-12)
    assert unit_conversion.forward(-3) == pytest.approx(1 * 5, 1e-12)
    assert unit_conversion.forward(11) == pytest.approx((7 + 11 * 2) * 5, 1e-12)
