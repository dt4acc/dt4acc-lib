"""
accelerator_simulator.py
========================

PyATAcceleratorSimulator — replaces the old element_proxies-based version.

Uses ElementProxyFactory (new property proxy architecture) for all standard
AT element classes. Compound element IDs (e.g. "CQLN:<host_uuid>") are still
resolved via ADDON_PROXY_REGISTRY for correctors living on a host element.
"""

import at
import copy
from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface
from dt4acc_lib.interfaces.simulator.element import ElementInterface
from dt4acc_lib.pyat_simulator.proxies.proxy_factory import ElementProxyFactory

from ..model.output.survey import SurveyDataForElement


class PyATAcceleratorSimulator(AcceleratorSimulatorInterface):
    """
    Accelerator simulator using ElementProxyFactory.

    Element lookup strategy:
      1. UUID attribute match: element.UUID == element_id  (SOLEIL .m lattice)
      2. FamName match: element.FamName == element_id      (MAX IV JSON lattice)
    """

    def __init__(self, *, at_lattice, proxy_factory: ElementProxyFactory = None):
        self.acc_orig_store = copy.deepcopy(at_lattice)
        self.acc = None
        self.reinit()
        if proxy_factory is None:
            proxy_factory = ElementProxyFactory()
        self.proxy_factory = proxy_factory

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(at_lattice={self.acc})"

    def reinit(self):
        self.acc = copy.deepcopy(self.acc_orig_store)

    def get_optics_parameters(self):
        assert self.acc is not None, \
            f"{self.__class__.__name__} is not properly initialised"
        x0, ring_pars, elem_data = self.acc.get_optics(at.All)
        return x0, ring_pars, elem_data

    def _find_by_uuid(self, uuid: str) -> list:
        """Find element(s) by UUID (SOLEIL) or FamName (MAX IV). Returns [] if not found."""
        matches = [e for e in self.acc if getattr(e, "UUID", None) == uuid]
        if matches:
            return [matches[0]]
        matches = [e for e in self.acc if getattr(e, "FamName", None) == uuid]
        if matches:
            return [matches[0]]
        return []

    def _find_by_uuids(self, uuids: list) -> list:
        """Find multiple elements by a list of UUID/FamName strings."""
        result = []
        for uuid in uuids:
            matches = self._find_by_uuid(uuid)
            if matches:
                result.extend(matches)
        return result

    def get(self, element_id) -> ElementInterface:
        matches = self._find_by_uuid(element_id)
        if not matches:
            raise ValueError(
                f"Element with UUID/FamName {element_id!r} not found in lattice. "
                f"Check that the uuid field in the JSON setup matches either "
                f"element.UUID (SOLEIL) or element.FamName (MAX IV) in the lattice."
            )
        assert len(matches) == 1, \
            f"Expected exactly 1 element for {element_id!r}, got {len(matches)}"
        (element,) = matches
        return self.proxy_factory.get_proxy(element, element_id=element_id)


__all__ = ["PyATAcceleratorSimulator"]