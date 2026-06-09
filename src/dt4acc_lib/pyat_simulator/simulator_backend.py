"""

Todo:
    Share of responsibility: need to review it together with the twin controller
    The twin controller has all requests available.
    On the other hand the accelerator simulator knows e.g.: typically if twiss
    is calculated, orbit is calculated anyway.
"""

import logging
import threading
from typing import Sequence

from transitions import Machine

from dt4acc_lib.interfaces.backend.backend import SimulatorBackendRW
from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface
from dt4acc_lib.interfaces.simulator.result_element import ResultElement
from dt4acc_lib.model.output.calculated_track import CalculatedTrack, CalculatedPosition
from dt4acc_lib.model.output.survey import SurveyDataForElement
from dt4acc_lib.model.output.tune import Tune, Chromaticity
from dt4acc_lib.model.output.twiss import Twiss, TwissAtPosition, TwissParameters

from .model.calculation_states import CalculationStates as States

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
        names, uuids, optics_parameters = self.backend.get_optics()
        _, ring_pars, elem_data =  optics_parameters
        r =  CalculatedTrack(
            track=[
                CalculatedPosition(fam_name=name, uid=uid, x=state[0], y=state[2])
                for name, uid, state in zip(names, uuids, elem_data["closed_orbit"])
            ]
        )
        return r
        raise NotImplementedError("Need to define track object ?")


class TwissElement(ResultElement):
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Twiss:
        fam_names, elem_uids, optics_parameters = self.backend.get_optics()
        _, ring_pars, elem_data =  optics_parameters
        r = Twiss(
            twiss=[
                TwissAtPosition(
                    fam_name=fam_name,
                    uid=elm_uid,
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
                for fam_name, elm_uid, ed in zip(fam_names, elem_uids, elem_data)
            ])
        return r

class TuneElement(ResultElement):
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Tune:
        assert prop_id == "transversal", f"Only prepared to handle transversal tune but got {prop_id}"
        _, __, optics_parameters = self.backend.get_optics()
        _, ring_pars, __ = optics_parameters
        tune = ring_pars["tune"]
        return Tune(x=tune[0], y=tune[1])


class ChromaticityElement(ResultElement):
    """Returns chromaticity (xi_x, xi_y) from AT.

    Uses at.get_optics with get_chrom=True which is available in pyAT >= 0.8.
    Falls back gracefully if not supported (e.g. 4D lattice without RF).

    Todo:
        * rework it that is uses backend
        * return proper object
    """
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Chromaticity:
        assert prop_id == "transversal", f"Only prepared to handle transversal chromaticity but got {prop_id}"
        try:
            # Todo: fix this translation
            import at
            ring = self.backend.acc.acc
            _, ring_pars, _ = ring.get_optics(at.All, get_chrom=True)
            chroma_hor, chroma_vert = ring_pars["chromaticity"]
            assert not math.isnan(chroma_hor)
            assert not math.isnan(chroma_vert)
            return Chromaticity(x=float(chroma_hor), y=float(chroma_vert))
        except Exception:
            logger.info("Failed to retrieve chromaticity data from %s", self.backend)
        return Chromaticity(x=math.nan, y=math.nan)


class SurveyElement(ResultElement):
    """
    """
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Sequence[SurveyDataForElement]:
        assert prop_id == "s", f"Only ready for s position but got {prop_id} "
        r = self.backend.get_survey()
        return r


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
        # These "must" be unique
        self.elem_uids = None
        # These are allowed to repeat
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
            survey=SurveyElement(backend=self),
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
            self.elem_uids = None
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
        """
        Todo:
            split it up in explicit functions
            Review when an other backend is needed
        """
        self._calculate_optics_if_required()
        assert self.optics is not None, "expected some optics stored, but only found None"
        return self.elem_names, self.elem_uids, self.optics

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

    def _create_element_names_and_uids(self):
        def extract_element_id(elem):
            if hasattr(elem, "UUID"):
                return elem.UUID
            return elem.FamName

        elem_names = [elem.FamName for elem in self.acc.acc]
        elem_uids = [extract_element_id(elem) for elem in self.acc.acc]
        # optics repeats data for the first element
        self.elem_names = elem_names + [elem_names[0]]
        self.elem_uids = elem_uids + [elem_uids[0]]

    def get_element_names(self) -> Sequence[str]:
        if self.elem_names is None:
            self._create_element_names_and_uids()
        return self.elem_names

    def get_element_uids(self) -> Sequence[str]:
        if self.elem_uids is None:
            self._create_element_names_and_uids()
        return self.elem_uids

    def get_survey(self) -> Sequence[SurveyDataForElement]:
        # a sign of one layer too much ?
        elm_names = self.get_element_names()
        elm_uids = self.get_element_uids()
        s_pos = self.acc.get_survey()
        r = [
            SurveyDataForElement(s=float(s), name=name, uid=uid)
            for name, uid, s  in zip(elm_names, elm_uids, s_pos)
        ]
        return r


_all__ = ["SimulationBackend"]