import at

from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface
from dt4acc_lib.interfaces.simulator.element import ElementInterface
from dt4acc_lib.pyat_simulator.proxies.proxy_factory import ElementProxyFactory
from dt4acc_lib.pyat_simulator.proxies.addon_registry import ADDON_PROXY_REGISTRY


class PyATAcceleratorSimulator(AcceleratorSimulatorInterface):
    """
    Accelerator simulator using the new ElementProxyFactory.

    Element lookup strategy (in order):
      1. Compound id "type_prefix:host_uuid" → ADDON_PROXY_REGISTRY
         (correctors living on a host element, e.g. CQLN/CQLT)
      2. UUID attribute match: element.UUID == element_id  (SOLEIL .m lattice)
      3. FamName match: element.FamName == element_id      (MAX IV JSON lattice)
    """

    def __init__(self, *, at_lattice, proxy_factory: ElementProxyFactory = None):
        self.acc_orig_store = at_lattice.copy()
        self.acc = None
        self.reinit()
        if proxy_factory is None:
            proxy_factory = ElementProxyFactory()
        self.proxy_factory = proxy_factory

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(at_lattice={self.acc})"

    def reinit(self):
        self.acc = self.acc_orig_store.copy()

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
        # 1. Compound id: "type_prefix:host_uuid" → ADDON_PROXY_REGISTRY
        if isinstance(element_id, str) and ":" in element_id:
            type_prefix, host_uuid = element_id.split(":", 1)
            factory = ADDON_PROXY_REGISTRY.get(type_prefix)
            if factory is not None:
                matches = self._find_by_uuid(host_uuid)
                if not matches:
                    raise ValueError(
                        f"Host element {host_uuid!r} not found in lattice "
                        f"for compound id {element_id!r}."
                    )
                (element,) = matches
                return factory(element, element_id, host_uuid)

        # 2. Single UUID/FamName → ElementProxyFactory
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