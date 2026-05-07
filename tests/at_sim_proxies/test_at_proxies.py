import at
import pytest

from dt4acc_lib.pyat_simulator.proxies.proxy_factory import (
    create_at_properties_lut_per_element_cls,
    ElementProxyFactory,
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


def test_sextupole_properties():
    f = ElementProxyFactory()
    s = at.Sextupole(family_name="sext_tst", length=0.25)
    proxy = f.get_proxy(s, element_id="test_id")

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


def test_quadrupole_properties():
    K = 123
    s = at.Quadrupole(family_name="quad_tst", length=0.335)
    s.update(K=K)
    f = ElementProxyFactory()
    proxy = f.get_proxy(s, element_id="test_id")
    val = proxy.peek("main_strength")
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)


@pytest.mark.asyncio
async def test_sextupole_update():
    H = 272
    f = ElementProxyFactory()
    s = at.Sextupole(family_name="sext_tst", length=0.25)
    s.update(H=272)
    proxy = f.get_proxy(s, element_id="test_id")

    val = proxy.peek("B3")
    assert val == pytest.approx(H, abs=1e-12, rel=1e-12)

    val = proxy.peek("main_strength")
    assert val == pytest.approx(H, abs=1e-12, rel=1e-12)

    with pytest.raises(AssertionError):
        await proxy.update("main_strength", H/3)

    await proxy.update("set_main_strength", H/3)

    val = proxy.peek("B3")
    assert val == pytest.approx(H/3, abs=1e-12, rel=1e-12)
    val = proxy.peek("main_strength")
    assert val == pytest.approx(H/3, abs=1e-12, rel=1e-12)



@pytest.mark.asyncio
async def test_quadrupole_update():
    K = 31.2
    f = ElementProxyFactory()
    s = at.Quadrupole(family_name="quad_tst", length=0.133)
    s.update(K=K)
    proxy = f.get_proxy(s, element_id="test_id")

    # needs to get consistent with what multipole returns
    with pytest.raises(IndexError):
        val = proxy.peek("B3")

    val = proxy.peek("B2")
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)

    val = proxy.peek("main_strength")
    assert val == pytest.approx(K, abs=1e-12, rel=1e-12)

    with pytest.raises(AssertionError):
        await proxy.update("main_strength", K/4)

    await proxy.update("set_main_strength", K/4)

    val = proxy.peek("B2")
    assert val == pytest.approx(K/4, abs=1e-12, rel=1e-12)
    val = proxy.peek("main_strength")
    assert val == pytest.approx(K/4, abs=1e-12, rel=1e-12)
