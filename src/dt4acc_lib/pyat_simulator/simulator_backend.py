"""

Todo:
    Share of responsibility: need to review it together with the twin controller
    The twin controller has all requests available.
    On the other hand the accelerator simulator knows e.g.: typically if twiss
    is calculated, orbit is calculated anyway.
"""

import logging
import threading

from transitions import Machine

from dt4acc_lib.interfaces.backend.backend import SimulatorBackendRW
from dt4acc_lib.interfaces.simulator.result_element import ResultElement
from dt4acc_lib.model.output.tune import Tune
from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface

from .model.calculation_states import CalculationStates as States
from dt4acc.core.model.output.calculated_track import CalculatedTrack, CalculatedPosition
from dt4acc.core.model.output.twiss import Twiss, TwissParameters, TwissAtPosition

logger = logging.getLogger()

class OrbitElement(ResultElement):
    """Orbit as represented by beam position monitors

    Todo:
        is it required given that track element exists?
    """
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str):
        raise NotImplementedError("Need to define orbit object?")


class TrackElement(ResultElement):
    """Orbit as represented by beam position monitors
    """
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> CalculatedTrack:
        names, optics_parameters = self.backend.get_optics()
        _, ring_pars, elem_data =  optics_parameters
        r =  CalculatedTrack(
            track=[
                CalculatedPosition(name=name, x=state[0], y=state[2])
                for name, state in zip(names, elem_data["closed_orbit"])
            ]
        )
        return r
        raise NotImplementedError("Need to define track object ?")


class TwissElement(ResultElement):
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Twiss:
        elem_names, optics_parameters = self.backend.get_optics()
        _, ring_pars, elem_data =  optics_parameters
        r = Twiss(
            twiss=[
                TwissAtPosition(
                    name=elm_name,
                    x=TwissParameters(
                        beta=ed["beta"][0],
                        alpha=ed["alpha"][0],
                        nu=ed["mu"][0]
                    ),
                    y=TwissParameters(
                        beta=ed["beta"][1],
                        alpha=ed["alpha"][1],
                        nu=ed["mu"][1]
                    )
                )
                for elm_name, ed in zip(elem_names, elem_data)
            ])
        return r

class TuneElement(ResultElement):
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Tune:
        assert prop_id == "transversal", f"Only prepared to handle transversal tune but got {prop_id}"
        names, optics_parameters = self.backend.get_optics()
        _, ring_pars, __ =  optics_parameters
        tune = ring_pars["tune"]
        return Tune(x=tune[0], y=tune[1])


class ChromaticityElement(ResultElement):
    """Returns chromaticity (xi_x, xi_y) from AT.
    Uses at.get_optics with get_chrom=True which is available in pyAT >= 0.8.
    Falls back gracefully if not supported (e.g. 4D lattice without RF).
    """
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Tune:
        assert prop_id == "transversal", f"Only prepared to handle transversal chromaticity but got {prop_id}"
        try:
            import at
            ring = self.backend.acc.acc
            _, ring_pars, _ = ring.get_optics(at.All, get_chrom=True)
            chroma = ring_pars["chromaticity"]
            if chroma is not None and not any(
                __import__("math").isnan(c) for c in chroma
            ):
                return Tune(x=float(chroma[0]), y=float(chroma[1]))
        except Exception:
            pass
        return Tune(x=0.0, y=0.0)

class SimulationStateModel:
    """all methods added by class::`transitions.Machine`

    transitions used as bluesky seems not to use
    superstate machine anymore
    """


class SimulatorBackend(SimulatorBackendRW):
    """Simulation backend based on pyAT

    I assume today that the calculation engine works the following way:

    1. set to a state
    2. then calculations are triggered

    So the calculations shall only happen after the state was
    set (completely).This is not (and can not) directly observed
    here. Here the calculation is only conducted when its results
    are requested. Calculation, setting, and  reading back
    calculation results are protected by a lock, so no more sets
    are made while calculation is running nor calculation results
    are delivered ahead of time.

    Todo:
        * where to break async / sync or threaded approach?
        * calculation lock: defaults to a threading.lock
    """

    def __init__(
        self, *, acc: AcceleratorSimulatorInterface, name: str, logger=logger, calculation_lock=None
    ):
        self.acc = acc
        self.logger = logger
        self.name = name

        self.optics = None
        self.elem_names = None

        # While calculation is running
        # * don't allow setting data
        # * don't provide calculation results:  Twiss, tune, orbit
        #
        # Todo: should reads also be protected (by a Read / Write Lock)
        #       should the lock be an asyncio lock?
        if calculation_lock is None:
            calculation_lock = threading.Lock()
        self.calculation_lock = calculation_lock
        self.model = SimulationStateModel()
        self.state = Machine(
            model=self.model,
            # fmt:off
            transitions=[
                dict( trigger = "calculate" , source = States.pending   , dest = States.executing , before=self._clear_stored_results ),
                dict( trigger = "finished"  , source = States.executing , dest = States.finished                                      ),
                dict( trigger = "changed"   , source = States.finished  , dest = States.pending   , after=self._clear_stored_results  ),
                dict( trigger = "changed"   , source = States.pending   , dest = States.pending   , after=self._clear_stored_results  ),
                dict( trigger = "clear"     , source = States.error     , dest = States.pending                                       ),
                dict( trigger = "error"     , source = "*"              , dest = States.error                                         ),
            ],
            # fmt:on
            states=[st for st in States],
            initial=States.pending,
        )

        self.result_elements = dict(
            orbit=OrbitElement(backend=self),
            track=TrackElement(backend=self),
            tune=TuneElement(backend=self),
            chromaticity=ChromaticityElement(backend=self),
            twiss=TwissElement(backend=self),
        )

    def _clear_stored_results(self):
        self.optics = None

    def get_natural_view_name(self):
        return "design"

    async def reset(self):
        with self.calculation_lock:
            self._clear_stored_results()
            if self.model.is_error():
                self.model.clear()
            elif not self.model.is_pending():
                self.model.changed()
            # Todo: find out where element names are added
            self.elem_names = None
            self.acc.reinit()

    async def trigger(self, dev_id: str, prop_id: str):
        self.logger.info(
            "%s(name=%s) no trigger needed", self.__class__.__name__, self.name
        )

    async def read(self, dev_id: str, prop_id: str) -> object:
        """

        Todo:
            acquire lock for read too? So that no inconsistent
            state will be read?
        """
        result_element = self.result_elements.get(dev_id, None)
        if result_element:
            return result_element.get(prop_id)

        elem = self.acc.get(dev_id)
        return elem.peek(prop_id)

    async def set(self, dev_id: str, prop_id: str, value: object):
        with self.calculation_lock:
            # Guard against error state — changed() is only valid from
            # finished or pending. If in error, reject the set.
            if self.model.is_error():
                raise ValueError(
                    f"SimulatorBackend is in error state — "
                    f"call Reset before writing ({dev_id}.{prop_id})"
                )
            self.model.changed()
            elem = self.acc.get(dev_id)
            r = await elem.update(property_id=prop_id, value=value)
        return r

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, acc={self.acc})"

    def get_optics(self):
        self._calculate_optics_if_required()
        assert self.optics is not None, "expected some optics stored, but only found None"
        return self.elem_names, self.optics

    def _calculate_optics_if_required(self):
        with self.calculation_lock:
            if self.model.is_pending():
                self._calculate_optics()
            assert (
                self.model.is_finished()
            ), f"expected to be in finished state, but I am in {self.model.state}"

    def _calculate_optics(self):
        """
        This method is only  a helper method for _calculate_tune_if_required,
        It is not to be called when already running
        """
        logger.debug("Calculating optics")
        assert (
            self.model.is_pending()
        ), f"expected to be in pending state, but I am in {self.model.state}"
        self.model.calculate()
        try:
            optics = self.acc.get_optics_parameters()
            self.model.finished()
        except Exception as exc:
            self.model.error()
            raise exc
        self.optics = optics
        elem_names = [elem.FamName for elem in self.acc.acc]
        # optics repeats data for the first element
        self.elem_names = elem_names + [elem_names[0]]
        logger.info("Calculated optics x0 = %s", optics[0])
        # logger.info("Calculated optics (twiss) ?to x=%.4f y=%.4f", self.tune.x, self.tune.y)


_all__ = ["SimulationBackend"]