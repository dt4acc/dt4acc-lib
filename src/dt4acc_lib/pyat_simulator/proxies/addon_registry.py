"""
addon_registry.py
=================

Registry for compound-element proxies — elements whose AT object is found
via a compound id like "CQLN:<host_uuid>" rather than a direct UUID/FamName.

The ADDON_PROXY_REGISTRY maps a type-prefix string to a factory callable:

    factory(element, element_id, host_element_id) -> AddOnElementProxy

Facilities register their own proxy types in their liaison setup.
accelerator_simulator.get() uses this registry and stays generic.

Example (SOLEIL liaison setup):

    from dt4acc_lib.pyat_simulator.proxies.addon_registry import (
        ADDON_PROXY_REGISTRY, SkewQuadCorrectorProxy
    )
    ADDON_PROXY_REGISTRY["CQLN"] = lambda el, eid, hid: SkewQuadCorrectorProxy(
        el, element_id=eid, host_element_id=hid, corrector_type="normal"
    )
    ADDON_PROXY_REGISTRY["CQLT"] = lambda el, eid, hid: SkewQuadCorrectorProxy(
        el, element_id=eid, host_element_id=hid, corrector_type="skew"
    )
"""

import logging

import numpy as np

from dt4acc_lib.interfaces.simulator.element import ElementInterface

logger = logging.getLogger("dt4acc-lib")


class AddOnElementProxy(ElementInterface):
    """Base class for compound-element proxies.

    Used when the AT element is found via a compound id like
    "CQLN:<host_uuid>" — the type prefix selects the proxy type,
    the host_uuid identifies the AT element.
    """

    def __init__(self, obj, *, element_id: str, host_element_id: str):
        self._obj = obj
        self.element_id = element_id
        self.host_element_id = host_element_id

    def get_name(self) -> str:
        return self.element_id

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self._obj}, "
            f"element_id={self.element_id}, "
            f"host_element_id={self.host_element_id})"
        )

    async def update(self, property_id: str, value, element_data=None):
        raise NotImplementedError("use derived class instead")

    def peek(self, property_id: str):
        raise NotImplementedError("use derived class instead")


class SkewQuadCorrectorProxy(AddOnElementProxy):
    """Proxy for a secondary quadrupolar corrector coil on a host element.

    In pyAT the host element holds both its main field and the corrector
    coil field in the same PolynomA/PolynomB arrays:

        corrector_type="skew"   →  PolynomA[1]  (skew quadrupole)  → property "A2"
        corrector_type="normal" →  PolynomB[1]  (normal quadrupole) → property "B2"

    Property names match Multipole(skew/normal, 2).handles_property()
    from the new proxy architecture.

    Facility mapping (done in liaison setup):
        SOLEIL CQLN  →  corrector_type="normal"  (PolynomB[1]) → "B2"
        SOLEIL CQLT  →  corrector_type="skew"    (PolynomA[1]) → "A2"
    """

    def __init__(self, obj, *, corrector_type: str = "skew", **kwargs):
        super().__init__(obj, **kwargs)
        if corrector_type not in ("skew", "normal"):
            raise ValueError(
                f"Unknown corrector_type {corrector_type!r}, "
                f"expected 'skew' or 'normal'"
            )
        self.corrector_type = corrector_type

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self._obj}, "
            f"element_id={self.element_id}, "
            f"host_element_id={self.host_element_id}, "
            f"corrector_type={self.corrector_type})"
        )

    def _get_element(self):
        if isinstance(self._obj, (list, tuple)) and len(self._obj) == 1:
            return self._obj[0]
        return self._obj

    def _expected_property(self) -> str:
        return "A2" if self.corrector_type == "skew" else "B2"

    async def update(self, property_id: str, value):
        expected = self._expected_property()
        if property_id != expected:
            raise ValueError(
                f"SkewQuadCorrectorProxy (corrector_type={self.corrector_type!r}) "
                f"handles {expected!r}, got {property_id!r}"
            )
        if value is not None:
            assert np.isfinite(value), "Value must be finite"

        element = self._get_element()
        if self.corrector_type == "skew":
            polynom_a = np.zeros(max(len(element.PolynomA), 2), dtype=float)
            polynom_a[:len(element.PolynomA)] = element.PolynomA
            polynom_a[1] = float(value)
            element.update(PolynomA=polynom_a)
        else:
            polynom_b = np.zeros(max(len(element.PolynomB), 2), dtype=float)
            polynom_b[:len(element.PolynomB)] = element.PolynomB
            polynom_b[1] = float(value)
            element.update(PolynomB=polynom_b)

        logger.debug(
            "SkewQuadCorrectorProxy.update: %s[%s].%s[1] = %s",
            self.host_element_id,
            self.corrector_type,
            "PolynomA" if self.corrector_type == "skew" else "PolynomB",
            value,
        )

    def peek(self, property_id: str) -> float:
        expected = self._expected_property()
        if property_id != expected:
            raise ValueError(
                f"SkewQuadCorrectorProxy (corrector_type={self.corrector_type!r}) "
                f"handles {expected!r}, got {property_id!r}"
            )
        element = self._get_element()
        if self.corrector_type == "skew":
            return float(element.PolynomA[1])
        else:
            return float(element.PolynomB[1])


ADDON_PROXY_REGISTRY: dict = {}


__all__ = [
    "ADDON_PROXY_REGISTRY",
    "AddOnElementProxy",
    "SkewQuadCorrectorProxy",
]