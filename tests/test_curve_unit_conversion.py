import math

import numpy as np
import pytest

from dt4acc_lib.bl.unit_conversion import EnergyIndependentCurveUnitConversion
from dt4acc_lib.model.utils.translator_manager_lookup_table import CurvePoint

BRHO = 5.0034611376381655  # MAX IV R1, 1.5 GeV

# MAX IV R1 SQFI excitation table (amp2k: currents [A], B' [T/m], MAX IV's sign)
SQFI_I = [513.696725, 467.0124916667, 350.2990416667, 233.5168916667, 116.7971333333, 0.0]
SQFI_G = [-27.630869913499996, -25.343432835, -19.0493868975, -12.7075229897, -6.3765458393, 0.0]


def _conv(indep, dep, *, flip=True, length=0.4, brho=BRHO):
    pts = [CurvePoint(indep=i, dep=d) for i, d in zip(indep, dep)]
    return EnergyIndependentCurveUnitConversion(
        fwd_points=pts, bwd_points=pts, brho=brho, flip_dep_sign=flip, length=length
    )


def amp2k(currents, fields, amps, brho):
    """MAX IV R1 amp2k.m: interp1(currents, -fields, amps, 'linear', 'extrap') / brho"""
    order = np.argsort(currents)
    x, y = np.asarray(currents)[order], -np.asarray(fields)[order]
    if amps < x[0]:
        v = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (amps - x[0])
    elif amps > x[-1]:
        v = y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (amps - x[-1])
    else:
        v = np.interp(amps, x, y)
    return v / brho


@pytest.mark.parametrize("current", [0.0, 50.0, 116.8, 300.0, 461.2, 513.7, 540.0])
def test_inverse_matches_maxiv_amp2k(current):
    # curve integrated over the 0.4 m magnet, as the R1 twin stores it
    conv = _conv(SQFI_I, [g * 0.4 for g in SQFI_G], length=0.4)
    assert conv.inverse(current) == pytest.approx(amp2k(SQFI_I, SQFI_G, current, BRHO), rel=1e-12, abs=1e-15)


def test_inverse_is_not_evaluated_at_current_over_brho():
    # a curve with an offset: the old inverse() read it at I / brho
    # (curve(50) = 6 -> 3; the old code gave curve(25) = 3.5)
    conv = _conv([0.0, 10.0, 100.0], [1.0, 2.0, 11.0], flip=False, length=1.0, brho=2.0)
    assert conv.inverse(50.0) == pytest.approx(3.0)


@pytest.mark.parametrize("flip", [True, False])
@pytest.mark.parametrize("length", [0.4, 0.0, None])
def test_forward_inverts_inverse(flip, length):
    conv = _conv(SQFI_I, [g * 0.4 for g in SQFI_G], flip=flip, length=length)
    for current in [0.0, 12.5, 233.5, 400.0, 513.6, 530.0]:
        assert conv.forward(conv.inverse(current)) == pytest.approx(current, abs=1e-9)


def test_flip_changes_sign_only():
    a = _conv(SQFI_I, SQFI_G, flip=True)
    b = _conv(SQFI_I, SQFI_G, flip=False)
    assert a.inverse(300.0) == pytest.approx(-b.inverse(300.0))


def test_thin_element_curve_is_the_integrated_strength():
    # MAX IV R1 HCM: B*L table [T m], length 0 -> kick = -B*L / brho
    currents, bl = [-11.57333, 0.0, 11.57333], [0.0012907763, 0.0, -0.0012907763]
    conv = _conv(currents, bl, flip=True, length=0.0)
    assert conv.inverse(1.0) == pytest.approx(0.0012907763 / 11.57333 / BRHO)
    assert conv.inverse(1.0) == pytest.approx(2.229e-5, rel=1e-3)


def test_unsorted_points_are_accepted():
    conv = _conv(list(reversed(SQFI_I)), list(reversed(SQFI_G)))
    ref = _conv(SQFI_I, SQFI_G)
    assert conv.inverse(250.0) == pytest.approx(ref.inverse(250.0))


def test_non_monotonic_curve_returns_first_segment():
    conv = _conv([0.0, 10.0, 20.0], [0.0, 10.0, 0.0], flip=False, length=1.0, brho=1.0)
    assert conv.forward(5.0) == pytest.approx(5.0)


def test_degenerate_curves_are_rejected():
    with pytest.raises(ValueError):
        _conv([1.0], [1.0])
    with pytest.raises(ValueError):
        _conv([1.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        _conv(SQFI_I, SQFI_G, brho=0.0)


def test_unreachable_target_raises():
    conv = _conv([0.0, 10.0, 20.0], [0.0, 10.0, 10.0], flip=False, length=1.0, brho=1.0)
    assert math.isclose(conv.forward(10.0), 10.0)
    with pytest.raises(ValueError):
        conv.forward(50.0)
