from typing import Dict, Sequence

import at

from .properties_proxy import PropertiesProxy
from ..properties.cavity_properties import Frequency, Voltage
from ..properties.geometric_properties import Dx, Dy, Roll
from ..properties.interface import ElementPropertyInterface
from ..properties.kick import XKick, YKick
from ..properties.main_strength import MainStrengthForQuadrupole, MainStrengthForSextupole
from ..properties.multipole import Multipole, NormalSkew
from ...interfaces.simulator.element import ElementInterface


class ElementProxyFactory:
    def __init__(self, elem_props_lut=None):
        if elem_props_lut is None:
            elem_props_lut = create_at_properties_lut_per_element_cls()
        self.elem_props_lut = elem_props_lut

    def __str__(self):
        return f"{self.__class__.__name__}(" f"elem_props_lut={self.elem_props_lut}" ")"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"elem_props_lut={repr(self.elem_props_lut)}"
            ")"
        )

    def get_proxy(self, obj, *, element_id: str, name: str = None) -> ElementInterface:
        """

        Todo:
            consider to cache instantiation e.g. by lattice index element
            Are elements cachable?
        """
        prop_lut = self.elem_props_lut[obj.__class__.__name__]
        r = PropertiesProxy(
            obj, properties_lut=prop_lut, element_id=element_id, name=name
        )
        return r


def create_at_properties_lut_per_element_cls() -> Dict[
    str, Dict[str, ElementPropertyInterface]
]:
    """
    precompute them as it will often be needed to instantiate them
    """
    lut = at_element_properties_lut()

    # All further processing is based on that the properties are unique
    for cls_name, props in lut.items():
        names = [p.handles_property() for p in props]
        assert len(names) == len(set(names)), f"{cls_name}: property names {names} are not unique"
    r = {
        cls_name: {p.handles_property(): p for p in props}
        for cls_name, props in lut.items()
    }
    return r


def at_element_properties_lut() -> Dict[str, Sequence[ElementPropertyInterface]]:
    geometric_properties = [Roll(), Dx(), Dy()]
    # Todo: need to  check what modyfing dipole element means
    #        How does it correspond to kick angles ?
    multipoles = [
        Multipole(normal_skew=NormalSkew.normal, n_multipole=n_mul)
        for n_mul in range(2, 20)
    ] + [
        Multipole(normal_skew=NormalSkew.skew, n_multipole=n_mul)
        for n_mul in range(2, 20)
    ]
    all_magnets_properties = multipoles + [XKick(), YKick()] + geometric_properties
    r = {
        at.Quadrupole.__name__: [MainStrengthForQuadrupole()] + all_magnets_properties,
        at.Sextupole.__name__: [MainStrengthForSextupole()] + all_magnets_properties,
        at.Multipole.__name__: all_magnets_properties,
        at.RFCavity.__name__: [Frequency(), Voltage()] + geometric_properties,
    }
    return r
