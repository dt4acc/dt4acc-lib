import math

import at
import pytest
from scipy.constants import speed_of_light

from dt4acc_lib.pyat_simulator.element_properties.utils import (
    estimate_dipole_main_field,
)
from dt4acc_lib.pyat_simulator.proxies.proxy_factory import (
    create_at_properties_lut_per_element_cls,
    ElementProxyFactory,
)
from dt4acc_lib.pyat_simulator.element_properties.multipole import (
    Multipole,
    NormalSkew,
)


def test_at_proxy_instantiation():
    """Just see that instaniation works"""
    lut = create_at_properties_lut_per_element_cls()
    for at_class_name, prop_lut in lut.items():
        str(at_class_name)
        for prop_id, handler in prop_lut.items():
            str(prop_id)
            str(handler)
            repr(handler)


def test_sextupole_properties(element_proxy_factory):
    s = at.Sextupole(family_name="sext_tst", length=0.25)
    proxy = element_proxy_factory.get_proxy(s, element_id="test_id")

    H = 123
    val = proxy.peek("main_strength")
    assert val == pytest.approx(0, abs=1e-12, rel=1e-12)
    s.update(H=H)
    val = proxy.peek("main_strength")
    assert val == pytest.approx(H, abs=1e-12, rel=1e-12)

    val = proxy.peek("B3")
    assert val == pytest.approx(H, abs=1e-12, rel=1e-12)

    val = proxy.peek("A3")
    assert val == pytest.approx(0, abs=1e-12, rel=1e-12)

    val = proxy.peek("B2")
    assert val == pytest.approx(0, abs=1e-12, rel=1e-12)

    val = proxy.peek("A2")
    assert val == pytest.approx(0, abs=1e-12, rel=1e-12)


def test_quadrupole_properties(element_proxy_factory):
    K = 123
    s = at.Quadrupole(family_name="quad_tst", length=0.335)
    s.update(K=K)
    f = element_proxy_factory
    proxy = f.get_proxy(s, element_id="test_id")
    val = proxy.peek("main_strength")
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)


@pytest.mark.asyncio
async def test_sextupole_update(element_proxy_factory):
    H = 272
    f = element_proxy_factory
    s = at.Sextupole(family_name="sext_tst", length=0.25)
    s.update(H=272)
    proxy = f.get_proxy(s, element_id="test_id")

    val = proxy.peek("B3")
    assert val == pytest.approx(H, abs=1e-12, rel=1e-12)

    val = proxy.peek("main_strength")
    assert val == pytest.approx(H, abs=1e-12, rel=1e-12)

    # reenable it after attribute convention has been revisted
    # with pytest.raises(AssertionError):
    await proxy.update("main_strength", H / 3)

    await proxy.update("set_main_strength", H / 3)

    val = proxy.peek("B3")
    assert val == pytest.approx(H / 3, abs=1e-12, rel=1e-12)
    val = proxy.peek("main_strength")
    assert val == pytest.approx(H / 3, abs=1e-12, rel=1e-12)


@pytest.mark.asyncio
async def test_quadrupole_update(element_proxy_factory):
    K = 31.2
    f = element_proxy_factory
    s = at.Quadrupole(family_name="quad_tst", length=0.133, k=K)
    proxy = f.get_proxy(s, element_id="test_id")

    # needs to get consistent with what multipole returns
    with pytest.raises(IndexError):
        val = proxy.peek("B3")

    val = proxy.peek("B2")
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)

    val = proxy.peek("main_strength")
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)

    # reenable test when attribute definition is revisited
    # with pytest.raises(AssertionError):
    await proxy.update("main_strength", K / 4)

    await proxy.update("set_main_strength", K / 4)

    val = proxy.peek("B2")
    assert val == pytest.approx(K / 4, abs=1e-12, rel=1e-12)
    val = proxy.peek("main_strength")
    assert val == pytest.approx(K / 4, abs=1e-12, rel=1e-12)


@pytest.mark.asyncio
async def test_multipole_update():
    K = 31.2
    A = -0.75
    q = at.Quadrupole(family_name="multi_tst", length=0.133, k=K)

    normal = Multipole(NormalSkew.normal, 2)
    skew = Multipole(NormalSkew.skew, 2)

    assert normal.handles_property() == "B2"
    assert skew.handles_property() == "A2"

    val = normal.peek(q)
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)

    val = skew.peek(q)
    assert val == pytest.approx(0, abs=1e-12, rel=1e-12)

    await normal.update(q, K / 4)
    val = normal.peek(q)
    assert val == pytest.approx(K / 4, abs=1e-12, rel=1e-12)
    assert q.PolynomB[1] == pytest.approx(K / 4, abs=1e-12, rel=1e-12)

    await skew.update(q, A)
    val = skew.peek(q)
    assert val == pytest.approx(A, abs=1e-12, rel=1e-12)
    assert q.PolynomA[1] == pytest.approx(A, abs=1e-12, rel=1e-12)


@pytest.mark.asyncio
async def test_kick_update(element_proxy_factory):
    """
    Need to understand it


    """
    dx = 3.13e-4
    q = at.Quadrupole("quad_kick", 0.42, 27.2, KickAngle=[0, 0])
    element_proxy_factory = element_proxy_factory
    proxy = element_proxy_factory.get_proxy(q, element_id="quad_kick")

    val = proxy.peek("x_kick")
    assert val == pytest.approx(0, abs=1e-12, rel=1e-12)

    await proxy.update("x_kick", dx)
    val = proxy.peek("x_kick")
    assert val == pytest.approx(dx, abs=1e-12, rel=1e-12)


def test_dipole_field_from_electron_beam_energy():
    radius = 1.0
    angle = 2 * math.pi * (1 / 32)
    energy = 1e9  # in EV
    r = estimate_dipole_main_field(
        beam_energy=1e9, dipole_angle=angle, path_length=radius * angle
    )
    ref = energy / (speed_of_light * radius)
    assert r == pytest.approx(ref, abs=1e-6, rel=1e-6)


@pytest.mark.asyncio
async def test_dipole_update():
    """Check it for BESSY II parameters

    todo:
        update it for a correct check
    """

    dip = at.Dipole("test_dipole", length=1.0, bending_angle=math.pi / 16.0)
    lat = at.Lattice([dip], energy=1.72e9)

    def get_reference_energy():
        return lat.energy

    element_proxy_factory = ElementProxyFactory(get_reference_energy=get_reference_energy)
    # necessary so that the energy is stored in the dipole
    lat.enable_6d()
    proxy = element_proxy_factory.get_proxy(dip, element_id="test_dipole")
    main_field = proxy.peek("main_strength")
    assert main_field == pytest.approx(1.2, abs=0.1, rel=0)