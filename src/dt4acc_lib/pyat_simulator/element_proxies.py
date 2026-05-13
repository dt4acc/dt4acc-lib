import logging

import numpy as np

from dt4acc_lib.interfaces.simulator.element import ElementInterface
from dt4acc_lib.pyat_simulator.element_properties.utils import manipulate_kick, update_shift

logger = logging.getLogger("dt4acc_lib")


def guess_multipole_main_strength_index(element, property_id: str):
    """Guessing main strength for multipole by finding the coefficient with the largest value at rref.
    Falls back to _subtype attribute (set at load time) when PolynomB is all zeros.

    TANGO_2ND: AN01-AR/EM-COR/OH.01-CQLN.01 // slow normal
    TANGO_3RD: AN01-AR/EM-COR/OH.01-CQLT.01 //slow turned
    """
    rref = 10e-3
    mul = np.absolute(element.PolynomB) * rref ** np.arange(len(element.PolynomB))
    idx = mul.argmax()
    if idx == 0 and np.all(element.PolynomB == 0):
        # PolynomB is all zeros — use _subtype annotation set at lattice load time
        subtype = getattr(element, '_subtype', None)
        if subtype == 'Quad':
            return 1   # PolynomB[1] = K
        elif subtype in ('Sext', 'SkewSext'):
            return 2   # PolynomB[2] = H
        elif subtype == 'Bend':
            return 0   # PolynomB[0] = dipole field
        # Last resort: use MaxOrder - 1
        max_order = getattr(element, 'MaxOrder', None)
        if max_order is not None and max_order > 0:
            return int(max_order) - 1
    return idx


class ElementProxy(ElementInterface):
    """
    Proxy class for an accelerator element to handle interactions and updates.

    Attributes:
        _obj: The underlying accelerator element.
        element_id: Identifier for the element.
        on_update_finished: Event triggered upon update completion.
        on_changed_value: Event triggered upon value change.

    Todo:
        Review if only a single element should be passed to the proxy
    """

    def __init__(self, obj, *, element_id: str, name: str = None):
        self._obj = obj
        self.element_id = element_id
        self.name = name

    def get_name(self) -> str:
        if self.name:
            return self.name
        return self.element_id

    def __repr__(self):
        return f"{self.__class__.__name__}({self._obj}, element_id={self.element_id})"

    def update_roll(self, *, roll):
        """
        Todo: implement setting roll
        Set the roll of the element
        """
        self._obj.set_tilt(roll)

    async def update_shift(self, *, dx=None, dy=None):
        """
        Update the element shift.

        Args:
            dx: Shift in x-direction.
            dy: Shift in y-direction.

        Raises:
            AssertionError: If both dx and dy are None.
        """
        (element,) = self._obj
        return update_shift(element, dx, dy)

    async def _delta_update(self, property_id: str, value: object):
        """
        Todo:
            This implementation is incorrect! Remove me!

            The delta is relative to some state of object ...
            Needs to be reviewed if it should be handled within this
            proxy

            Should that be rather handled at the backend layer?
        """
        raise AssertionError("delta update should not be handled by simulation backend or mexec!")
        ref = self.peek(property_id)
        t_val = value + ref
        logger.info("delta update %s.%s: ref %s, val %s -> %s",
                    self.element_id, property_id, ref, value, t_val)
        return await self._update(property_id, t_val)

    async def update(self, property_id: str, value: object):
        """
        Update element properties dynamically based on property_id.

        Args:
            property_id: The property to update.
            value: The value to set.
            element_data: Element-specific data required for update.

        Raises:
            ValueError: If an unknown property is specified.

        Todo: is that Liaison management?
              the inverse way
        """
        assert property_id[:6] != "delta_", (
            f"properties like {property_id}"
            " starting with delta should not end up here"
        )

        if value is not None:
            assert np.isfinite(value), "Value must be finite"

        return await self._update(property_id, value)

    async def _update(self, property_id: str, value: object):

        (element,) = self._obj
        method_name = f"set_{property_id}"

        if method_name == "set_x":
            await self.update_shift(dx=value)
        elif method_name == "set_y":
            await self.update_shift(dy=value)
        elif method_name == "set_roll":
            await self.update_roll(roll=value)
        elif method_name == "set_im":
            raise AssertionError("should not end up here")
            # val = value * element_data.hw2phys
            # element_type = str(element).split('\n')[0]
        elif method_name == "set_main_strength":
            element_type = element.__class__.__name__
            if "Sextupole" in element_type:
                element.update(H=value)
            elif "Quadrupole" in element_type:
                element.update(K=value)
            elif "Octupole" in element_type:
                # Octupole main strength is PolynomB[3] = K3
                # Must copy array — AT arrays are not always writable in-place
                polynom_b = element.PolynomB.copy()
                polynom_b[3] = float(value)
                element.PolynomB = polynom_b
            elif "Multipole" in element_type:
                # todo review how to handle multipoles with several components
                idx = guess_multipole_main_strength_index(element, property_id)
                if idx == 1:
                    element.update(K=value)
                elif idx == 2:
                    element.update(H=value)
                else:
                    raise NotImplementedError(
                        f"setting main strength for multipole index {idx} not yet implemented"
                    )
            else:
                raise NotImplementedError(
                    f"Don't know how to set main strength for element {element_type}"
                )
        elif method_name == "set_main_strength_k":
            # Explicit quadrupole component (PolynomB[1] = K) — used by device-view
            # facilities (e.g. MAX IV) where element class is generic 'Multipole'
            element.update(K=value)
        elif method_name == "set_main_strength_h":
            # Explicit sextupole component (PolynomB[2] = H) — used by device-view
            # facilities (e.g. MAX IV) where element class is generic 'Multipole'
            element.update(H=value)
        elif method_name == "set_main_strength_b0":
            # Explicit dipole component (PolynomB[0]) — used for bending magnets
            # in device-view facilities where element class is generic 'Multipole'
            polynom_b = element.PolynomB.copy()
            polynom_b[0] = float(value)
            element.PolynomB = polynom_b
        elif method_name == "set_freq":
            # Todo: the frequency scale here must go!
            element.update(Frequency=value * 1000)
        elif method_name in ["set_rdbk", "set_K"]:
            pass
        elif method_name == "set_x_kick":
            element.update(KickAngle=manipulate_kick(element.KickAngle, kick_x=value))
        elif method_name == "set_y_kick":
            element.update(KickAngle=manipulate_kick(element.KickAngle, kick_y=value))
        elif method_name == "set_frequency":
            # should be ok for AT
            element.update(Frequency=value)
            # raise AssertionError("Cavity control not yet declared as functional, have a look to the line below")
        else:
            method = getattr(element, method_name)
            await method(value)

    def peek(self, property_id: str) -> float:
        assert property_id[:6] != "delta_", (
            f"properties like {property_id}"
            " starting with delta should not end up here"
        )

        if property_id in ["K", "H", "main_strength"]:
            return self.peek_main_strength(property_id)
        elif property_id in ["main_strength_k", "main_strength_h", "main_strength_b0"]:
            return self.peek_main_strength(property_id)
        elif property_id in ["x_kick", "y_kick"]:
            return self.peek_kick(property_id)
        elif property_id in ["frequency"]:
            return self.peek_frequency()
        else:
            raise NotImplementedError(
                f"handling property {property_id} not (yet) implemented"
            )

    def peek_frequency(self):
        (element,) = self._obj
        return element.Frequency

    def peek_main_strength(self, property_id: str):
        (element,) = self._obj
        element_type = element.__class__.__name__
        # Explicit property names for device-view facilities (e.g. MAX IV)
        if property_id == "main_strength_k":
            return float(element.PolynomB[1])
        elif property_id == "main_strength_h":
            return float(element.PolynomB[2])
        elif property_id == "main_strength_b0":
            return float(element.PolynomB[0])
        elif element_type == "Quadrupole":
            assert property_id in ["K", "main_strength"]
            return element.K
        elif element_type == "Sextupole":
            if property_id not in ["H", "main_strength"]:
                raise AssertionError(
                    f"Not handling {property_id} for element {element_type}"
                )
            return element.H
        elif element_type == "Octupole":
            # Main strength is PolynomB[3] = K3
            assert property_id in ["main_strength"]
            return float(element.PolynomB[3])
        elif element_type == "Multipole":
            idx = guess_multipole_main_strength_index(element, property_id)
            return float(element.PolynomB[idx])
        else:
            raise NotImplementedError(
                f"main strength not implemented for element {element_type}"
            )

    def peek_kick(self, property_id: str):
        (element,) = self._obj
        lut = dict(x_kick=0, y_kick=1)
        try:
            idx = lut[property_id]
        except KeyError as ke:
            raise AssertionError(f"Did not expect kick {property_id}")
        return element.KickAngle[idx]



class AddOnElementProxy(ElementProxy):
    """
    Proxy for an element whose updates are relayed to another element.

    Attributes:
        host_element_id: ID of the host element.
    """

    def __init__(self, obj, *, element_id, host_element_id):
        super().__init__(obj, element_id=element_id)
        self.host_element_id = host_element_id

    def __str__(self):
        return f"{self.__class__.__name__}({self._obj}, element_id={self.element_id}, host_element_id={self.host_element_id})"

    def update(self, property_id: str, value, element_data):
        raise NotImplementedError("Needs to be implemented for specific case")


class KickAngleCorrectorProxy(AddOnElementProxy):
    """
    Proxy for handling kick angle corrections in a specific plane.

    Attributes:
        correction_plane: Specifies whether correction is horizontal or vertical.
    """

    def __init__(self, obj, **kwargs):
        super().__init__(*obj, **kwargs)

    async def update_kick(self, *, kick_x=None, kick_y=None):
        """
        Update kick angles for the corrector element.

        Args:
            kick_x: Horizontal kick angle.
            kick_y: Vertical kick angle.
            element_data: Element-specific conversion data.

        Todo: review if this code is still neede
        """
        element = self._obj
        if kick_x is not None:
            kick_x = kick_x
        if kick_y is not None:
            kick_y = kick_y
        element.update(
            KickAngle=manipulate_kick(self._obj.KickAngle, kick_x=kick_x, kick_y=kick_y)
        )

    async def update(self, property_id: str, value):
        """
        Handle updates for the corrector element.

        Args:
            property_id: The property to update (should be 'im').
            value: The value to set.
            element_data: Element-specific conversion data.

        Raises:
            ValueError: If an unknown property is specified.
        """
        assert property_id[:6] != "delta_", (
            f"properties like {property_id}"
            " starting with delta should not end up here"
        )

        if property_id == "x_kick":
            await self.update_kick(kick_x=value)
        elif property_id == "y_kick":
            await self.update_kick(kick_y=value)
        else:
            raise ValueError(f"Unexpected property {property_id} for kick corrector")

    def peek(self, property_id: str) -> float:
        assert property_id[:6] != "delta_", f"properties like {property_id} starting with delta should not end up here"

        element = self._obj

        if property_id == "x_kick":
            return element.KickAngle[0]
        elif property_id == "y_kick":
            return element.KickAngle[1]
        else:
            raise ValueError(f"Unexpected property {property_id} for kick corrector")


class SkewQuadCorrectorProxy(AddOnElementProxy):
    """
    Proxy for a secondary quadrupolar corrector coil mounted on a host element
    (typically an octupole in SOLEIL II, but generic to any facility).

    In pyAT the host element holds both its main field and the corrector
    coil field in the same PolynomA/PolynomB arrays:

        corrector_type="skew"   →  PolynomA[1]  (skew quadrupole, coupling)
        corrector_type="normal" →  PolynomB[1]  (normal quadrupole component)

    The property exposed to the Tango layer is "skew_quad_strength".

    The mapping from facility nomenclature to corrector_type is done in
    the facility-specific setup (e.g. ADDON_PROXY_REGISTRY in
    liasion_translator_setup.py):
        SOLEIL CQLN  →  corrector_type="normal"  (PolynomB[1], normal quad)
        SOLEIL CQLT  →  corrector_type="skew"    (PolynomA[1], skew quad)

    Parameters
    ----------
    corrector_type : "skew" | "normal"
        "skew"   → sets/reads PolynomA[1]  (skew quadrupole)
        "normal" → sets/reads PolynomB[1]  (normal quadrupole)
    """

    def __init__(self, obj, *, corrector_type: str = "skew", **kwargs):
        super().__init__(obj, **kwargs)
        if corrector_type not in ("skew", "normal"):
            raise ValueError(
                f"Unknown corrector_type {corrector_type!r}, expected 'skew' or 'normal'"
            )
        self.corrector_type = corrector_type

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self._obj}, element_id={self.element_id}, "
            f"host_element_id={self.host_element_id}, corrector_type={self.corrector_type})"
        )

    def _get_element(self):
        if isinstance(self._obj, (list, tuple)) and len(self._obj) == 1:
            return self._obj[0]
        return self._obj

    async def update(self, property_id: str, value):
        """
        Set the quadrupole corrector strength on the host element.

        property_id must be "skew_quad_strength".

        corrector_type="skew"   → PolynomA[1] = value  (skew quad, coupling)
        corrector_type="normal" → PolynomB[1] = value  (normal quad component)
        """
        assert property_id[:6] != "delta_", (
            f"properties like {property_id} starting with delta should not end up here"
        )
        if property_id != "skew_quad_strength":
            raise ValueError(
                f"SkewQuadCorrectorProxy only handles 'skew_quad_strength', got {property_id!r}"
            )
        if value is not None:
            assert np.isfinite(value), "Value must be finite"

        element = self._get_element()

        if self.corrector_type == "skew":
            # Ensure PolynomA is long enough to hold index 1
            polynom_a = np.zeros(max(len(element.PolynomA), 2), dtype=float)
            polynom_a[:len(element.PolynomA)] = element.PolynomA
            polynom_a[1] = float(value)
            element.update(PolynomA=polynom_a)
        else:
            # Ensure PolynomB is long enough to hold index 1
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
        """Read the current corrector strength from the host element."""
        assert property_id[:6] != "delta_", (
            f"properties like {property_id} starting with delta should not end up here"
        )
        if property_id != "skew_quad_strength":
            raise ValueError(
                f"SkewQuadCorrectorProxy only handles 'skew_quad_strength', got {property_id!r}"
            )

        element = self._get_element()

        if self.corrector_type == "skew":
            return float(element.PolynomA[1])
        else:
            return float(element.PolynomB[1])


# ---------------------------------------------------------------------------
# Addon proxy registry
# ---------------------------------------------------------------------------
# Maps a type-prefix string (the part before ":" in a compound element_id
# like "CQLN:<host_uuid>") to a factory callable:
#
#   factory(element, element_id, host_element_id) -> AddOnElementProxy
#
# Facilities register their own proxy types here. The core
# accelerator_simulator.get() uses this registry and stays generic.
#
# Example registration (done by SOLEIL setup code, not by the core):
#
#   from dt4acc_lib.pyat_simulator.element_proxies import ADDON_PROXY_REGISTRY, SkewQuadCorrectorProxy
#   ADDON_PROXY_REGISTRY["CQLN"] = lambda el, eid, hid: SkewQuadCorrectorProxy(
#       el, element_id=eid, host_element_id=hid, corrector_type="CQLN"
#   )
#   ADDON_PROXY_REGISTRY["CQLT"] = lambda el, eid, hid: SkewQuadCorrectorProxy(
#       el, element_id=eid, host_element_id=hid, corrector_type="CQLT"
#   )

ADDON_PROXY_REGISTRY: dict = {}