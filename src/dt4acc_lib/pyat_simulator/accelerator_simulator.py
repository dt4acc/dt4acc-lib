"""
accelerator_simulator.py
========================

PyATAcceleratorSimulator — replaces the old element_proxies-based version.

Uses ElementProxyFactory (new property proxy architecture) for all standard
AT element classes. Compound element IDs (e.g. "CQLN:<host_uuid>") are still
resolved via ADDON_PROXY_REGISTRY for correctors living on a host element.
"""
import logging
from typing import Optional, Sequence

import at
import copy

from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface, OpticsCalculationError
from dt4acc_lib.interfaces.simulator.element import ElementInterface
from dt4acc_lib.pyat_simulator.proxies.proxy_factory import ElementProxyFactory

from ..model.input.twiss_in import TwissIn
from ..model.output.survey import SurveyDataForElement
from ..model.output.track import StatePerTurn

logger = logging.getLogger("dt4acc_lib")


class PyATAcceleratorSimulator(AcceleratorSimulatorInterface):
    """
    Accelerator simulator using ElementProxyFactory.

    Element lookup strategy:
      1. UUID attribute match: element.UUID == element_id  (SOLEIL .m lattice)
      2. FamName match: element.FamName == element_id      (MAX IV JSON lattice)

    Ring or line:
      * ring (``twiss_in`` None): periodic optics, closed orbit at the monitors
      * line (``twiss_in`` given): optics propagated from the beam entering
        the line, trajectory of its centroid at the monitors; no tune, no
        chromaticity
    """

    def __init__(
        self,
        *,
        at_lattice,
        proxy_factory: ElementProxyFactory = None,
        use_mp: bool = False,
        twiss_in: Optional[TwissIn] = None,
    ):
        """
        use_mp: track with pyAT's multiprocessing pool. Off by default: the
            pool forks, and forking a process that already runs threads (an
            IOC, a Tango server, the shared event loop) can deadlock the
            workers, so the first tracking request never returns.
        twiss_in: beam entering a transfer line; None: a ring
        """
        self.use_mp = use_mp
        self.acc_orig_store = copy.deepcopy(at_lattice)
        self.acc = None
        self.twiss_in = twiss_in
        self.reinit()

        # Todo: address proper Energy handling
        #       need to find out how energy is handled within pyat
        #       * is it the reference energy the lattice was designed for?
        #       * the reference energy of the reference particle
        #       * how is it changed and how is that traced?
        def retrieve_energy() -> float:
            return self.acc.energy

        if proxy_factory is None:
            proxy_factory = ElementProxyFactory(get_reference_energy=retrieve_energy)
        self.proxy_factory = proxy_factory

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(at_lattice={self.acc})"

    @property
    def is_ring(self) -> bool:
        return self.twiss_in is None

    def reinit(self):
        self.acc = copy.deepcopy(self.acc_orig_store)

    def get_optics_parameters(self):
        assert self.acc is not None, \
            f"{self.__class__.__name__} is not properly initialised"
        kwargs = {} if self.is_ring else dict(twiss_in=self.twiss_in.as_at())
        try:
            x0, ring_pars, elem_data = self.acc.get_optics(at.All, **kwargs)
        except ValueError as ve:
            # Assuming that ValueError means that optics calculation failed
            #
            # log only on info: optics calculation failure is not so uncommon
            logger.info(
                f"{self.__class__.__name__}.get_optics_parameters"
                f" {self.acc} get_optics returned: {ve}"
                " raising OpticsCalculationError"
            )
            raise OpticsCalculationError("Failed to calculate optics parameters") from ve
        return x0, ring_pars, elem_data

    def get_survey(self) -> Sequence[SurveyDataForElement]:
        r = at.get_s_pos(self.acc)
        return r

    def track(self, p0, n_turns, data_needed_at_element_index: Sequence[int]):
        r = at.tracking.lattice_track(
            self.acc, p0, nturns=n_turns, refpts=data_needed_at_element_index, use_mp=self.use_mp,
        )
        return r

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