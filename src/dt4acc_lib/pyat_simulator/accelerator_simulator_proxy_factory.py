from dt4acc_lib.pyat_simulator.accelerator_simulator import PyATAcceleratorSimulator
from dt4acc_lib.pyat_simulator.element_proxies import (
    KickAngleCorrectorProxy,
    ElementProxy,
)
from dt4acc_lib.interfaces.simulator.element import ElementInterface
from dt4acc_lib.pyat_simulator.proxies.proxy_factory import ElementProxyFactory


class PyATAcceleratorSimulator(PyATAcceleratorSimulator):
    """Find element by FamName

    Works only if each element has its single name
    """
    def __init__(self, *, at_lattice, proxy_factory: ElementProxyFactory = None):
        super().__init__(at_lattice=at_lattice)
        if proxy_factory is None:
            proxy_factory = ElementProxyFactory()
        self.proxy_factory = proxy_factory

    def get(self, element_id: str) -> ElementInterface:
        elements = self._find_by_uuid(element_id)
        # should be unique
        assert len(elements) == 1,  f"Expected exactly 1 element for {element_id}, but got {len(elements)}"
        (element,) = elements
        r = self.proxy_factory.get_proxy(element, element_id=element_id)
        return r


__all__ = [PyATAcceleratorSimulator]
