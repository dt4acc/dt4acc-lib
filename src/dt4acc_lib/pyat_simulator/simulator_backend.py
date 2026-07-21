"""

Todo:
    Share of responsibility: need to review it together with the twin controller
    The twin controller has all requests available.
    On the other hand the accelerator simulator knows e.g.: typically if twiss
    is calculated, orbit is calculated anyway.
"""
import time
from copy import copy as _copy
import logging
import math
import threading
from typing import Any, Dict, Sequence, Union, Tuple, List

import numpy as np
import numpy.typing as npt
from transitions import Machine

from dt4acc_lib.interfaces.backend.backend import SimulatorBackendRW
from dt4acc_lib.interfaces.simulator.accelerator_simulator import AcceleratorSimulatorInterface, OpticsCalculationError, \
    OpticsCalculationProhibitedError
from dt4acc_lib.interfaces.simulator.result_element import ResultElement
from dt4acc_lib.model.output.calculated_track import CalculatedTrack, CalculatedPosition
from dt4acc_lib.model.output.survey import SurveyDataForElement
from dt4acc_lib.model.output.tune import Tune, Chromaticity
from dt4acc_lib.model.output.twiss import Twiss, TwissAtPosition, TwissParameters

from dt4acc_lib.interfaces.backend.calculation_states import CalculationStates as States, CalculationStates
from .model.calculation_states import CalculationStates as States
from ..model.output.track import ParticleState, StatePerTurn, StatePerElement, ParticleStateCollection, StatesForTurns, \
    StatePerElementPerTurn
from ..model.output.track_as_np_wrapper import NPStatePerElementPerTurn, NPStatesForTurns

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


class TurnByTurnStart(VirtualElementInterface):
    def __init__(self):
        self._n_turns = 1
        self._p0: List[ParticleState] = []
        self._data_needed_at: List[str] = []

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_turns={self.get_n_turns()}"
            f", p0={self.get_p0()}"
            f", data_needed_at={self.get_data_needed_at()}"
            ")"
        )

    def get_n_turns(self) -> int:
        return self._n_turns

    def set_n_turns(self, n_turns):
        assert n_turns >= 1
        self._n_turns = n_turns

    def get_data_needed_at(self) -> Sequence[str]:
        return self._data_needed_at

    def set_data_needed_at(self, data_needed_at):
        self._data_needed_at = data_needed_at

    def get_p0(self) -> Sequence[ParticleState]:
        return self._p0

    def set_p0(self, p0):
        # Todo: which checks to apply
        # at is picky on this shape
        self._p0 = p0

    def get(self, prop_id: str) -> object:
        return {
            "n_turns": self.get_n_turns(),
            "data_needed_at": self.get_data_needed_at(),
            "p0": self.get_p0(),
        }[prop_id]

    def set(self, prop_id: str, value: Union[int, Sequence[ParticleState], Sequence[str]]) -> object:
        if prop_id == "n_turns":
            self.set_n_turns(n_turns=int(value))
        elif prop_id == "p0":
            self.set_p0(value)
        elif prop_id == "data_needed_at":
            self.set_data_needed_at(data_needed_at=value)
        else:
            raise AttributeError(f"Unknown property id {prop_id}")


class TurnByTurnElement(ResultElement):
    """Turn by turn data along the lattice
    """
    def __init__(self, backend):
        self.backend: SimulatorBackend = backend

    def get(self, prop_id: str) -> NPStatesForTurns:
        assert prop_id == "pos"
        data = self.backend.compute_track_using_track_start()
        return data

class TrackElement(ResultElement):
    """Orbit as represented by beam position monitors

    Todo:
        rename to OrbitElement or ClosedOrbit / ReferenceOrbit?
    """
    def __init__(self, backend):
        self.backend = backend

    def get(self, prop_id: str) -> Union[CalculatedTrack, None]:
        # Todo: check that prop_id matches to what is expected ..
        assert prop_id == "pos"
        names, uuids, optics_parameters = self.backend.get_optics()
        if optics_parameters is None:
            return None
        _, ring_pars, elem_data = optics_parameters
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

    def get(self, prop_id: str) -> Union[Twiss, None]:
        # Todo: check that prop_id matches to what is expected ..
        fam_names, elem_uids, optics_parameters = self.backend.get_optics()
        if optics_parameters is None:
            return None
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

    def get(self, prop_id: str) -> Union[Tune, None]:
        assert prop_id == "transversal", f"Only prepared to handle transversal tune but got {prop_id}"
        _, __, optics_parameters = self.backend.get_optics()
        if optics_parameters is None:
            return None
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
            chroma_hor, chroma_vert, _ = ring_pars["chromaticity"]
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
        * review that state transitions are all delegated to the
          state engine as appropriate
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
                dict( trigger = "calculate"   , source = States.pending      , dest = States.executing    , before=self._clear_stored_results ),
                dict( trigger = "finished"    , source = States.executing    , dest = States.finished                                         ),
                dict( trigger = "changed"     , source = States.finished     , dest = States.pending      , after=self._clear_stored_results  ),
                dict( trigger = "changed"     , source = States.pending      , dest = States.pending      , after=self._clear_stored_results  ),
                dict( trigger = "acknowledge" , source = States.error        , dest = States.acknowledged , after=self._clear_stored_results  ),
                dict( trigger = "clear"       , source = States.error        , dest = States.pending                                          ),
                dict( trigger = "clear"       , source = States.acknowledged , dest = States.pending                                          ),
                dict( trigger = "error"       , source = "*"                 , dest = States.error                                            ),
            ],
            # fmt:on
            states=[st for st in States],
            initial=States.pending,
        )

        # Todo: rename to virtual elements?
        # these elements are read only
        # data filled by simulation engine
        self.result_elements = dict(
            orbit=OrbitElement(backend=self),
            track=TrackElement(backend=self),
            turn_by_turn=TurnByTurnElement(backend=self),
            tune=TuneElement(backend=self),
            chromaticity=ChromaticityElement(backend=self),
            twiss=TwissElement(backend=self),
            survey=SurveyElement(backend=self),
        )
        # These elments provide info to be filled form outside
        # so that it is available in the calculation when needed
        self.virtual_element = dict(
            turn_by_turn_start=TurnByTurnStart(),
        )

    def get_state(self) -> CalculationStates:
        return CalculationStates(self.state.model.state)

    def _clear_stored_results(self):
        self.optics = None

    def get_natural_view_name(self):
        return "design"

    async def reset(self):
        with self.calculation_lock:
            self._clear_stored_results()
            if self.model.is_error():
                self.model.clear()
            elif self.model.is_acknowledged():
                self.model.clear()
            elif not self.model.is_pending():
                self.model.changed()
            else:
                # Todo: what to do in this case?
                pass
            # Todo: find out where element names are added
            self.elem_uids = None
            self.elem_names = None

    async def reinit(self):
        """

        Todo:
            Should it automatically call reset?
            Most probably yes
        """
        self.acc.reinit()
        await self.reset()

    async def acknowledge(self):
        self.model.acknowledge()

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
        virtual_element = self.virtual_element.get(dev_id, None)
        if virtual_element:
            return virtual_element.get(prop_id)
        elem = self.acc.get(dev_id)
        return elem.peek(prop_id)

    async def set(self, dev_id: str, prop_id: str, value: object):
        with self.calculation_lock:
            # Guard against error state — changed() is only valid from
            # finished or pending. If in error, reject the set.
            if self.model.is_error():
                raise OpticsCalculationProhibitedError(
                    f"SimulatorBackend is in error state — "
                    f"call Reset before writing ({dev_id}.{prop_id})"
                )
            if not self.model.is_acknowledged():
                # if in acknowledged mode: user / higher layer needs to reset
                # it actively
                self.model.changed()
            virtual_element = self.virtual_element.get(dev_id, None)
            if virtual_element:
                return virtual_element.set(prop_id, value)
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
        elem_names = self.get_element_names()
        uids = self.get_element_uids()

        if self.model.is_error() or self.model.is_acknowledged():
            # No optics data expected in this mode
            # Todo: should one raise an exception if in error?
            return elem_names, uids, None
        assert self.optics is not None, "expected some optics stored, but only found None"
        return elem_names, uids, self.optics

    def _calculate_optics_if_required(self):
        with self.calculation_lock:
            if self.model.is_acknowledged():
                logger.warning(
                    "Calculation of optics required, but in acknowledge mode,"
                    " need to be reset before it will calculate again")
                return

            if self.model.is_pending():
                self._calculate_optics()
            assert (
                self.model.is_finished() or self.model.is_error()
            ), f"expected to be in finished or error state, but I am in {self.model.state}"

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
        except OpticsCalculationError as oe:
            # Unfortunately a not more precise error
            # for the time being I have to assume that
            # it means no closed orbit was found
            self.model.error()
            logger.error(f"{self.__class__.__name__}: optics calculation failed {oe}")
            return None

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

    def compute_track(self, p0: Sequence[ParticleState], n_turns: int, data_needed_at: Sequence[str]) -> StatesForTurns:
        # That should be really fast ... no need to go further if that
        # can not be achieved

        elem_uids = self.get_element_uids()
        indices = [idx for idx, uid in enumerate(elem_uids) if uid in data_needed_at]
        p0 = np.array([p.as_array() for p in p0]).transpose()
        start = time.time()
        track_data, info, loss_map = self.acc.track(p0, n_turns=n_turns, data_needed_at_element_index=indices)
        end = time.time()
        dt = end - start
        logger.warning(
            "Computing %d turns took %s", n_turns, dt
        )
        # return track_data
        track_data_model = fill_state_per_element(track_data, data_needed_at)
        return StatesForTurns(turns=track_data_model)


def fill_state_per_element(track_data, elm_uids: Sequence[str]) -> Sequence[StatePerTurn]:
    # check the state, in a manner that documents the assumption
    n_state_elms, n_particles, per_n_elems, n_turns = track_data.shape

    # That is the assumption for now: for each element there is data
    assert len(elm_uids) == per_n_elems

    return [
        StatePerTurn(fill_state_per_element_per_track(track, elm_uids))
        for track in track_data.transpose(3, 0, 1, 2)
    ]


def rectify_uid_for_last_element_if_needed(uids: Sequence[str], copy=True) -> Sequence[str]:
    """
    """
    if uids[0] == uids[-1]:
        if copy:
            uids = _copy(uids)
        uids[-1] = uids[-1] + "_same_pos_as_start"
    return uids


def fill_state_per_element_per_track(one_track_data, elm_uids: Sequence[str]) -> Sequence[StatePerElement]:
    return [
       StatePerElement(ParticleStateCollection([ParticleState.from_sequence(p) for p in p_for_particles]), uid)
        for p_for_particles, uid in zip(one_track_data.transpose(2, 1, 0), elm_uids)
    ]

    def compute_track_using_track_start(self) -> NPStatesForTurns:
        start = self.virtual_element["turn_by_turn_start"]
        return self.compute_track(
            start.get_p0(), start.get_n_turns(), start.get_data_needed_at()
        )

    def compute_track(self, p0: Sequence[ParticleState], n_turns: int, data_needed_at: Sequence[str]) -> NPStatesForTurns:
        # That should be really fast ... no need to go further if that
        # can not be achieved

        elem_uids = self.get_element_uids()
        indices = [idx for idx, uid in enumerate(elem_uids) if uid in data_needed_at]
        sel_elem_uids = [elem_uids[idx] for idx in indices]
        p0 = np.array([p.as_array() for p in p0]).transpose()
        if len(p0) == 0:
            raise AssertionError("No start vector given for tracking!")
        start = time.time()
        track_data, info, loss_map = self.acc.track(p0, n_turns=n_turns, data_needed_at_element_index=indices)
        end = time.time()
        dt = end - start
        logger.warning(
            "Computing %d turns took %s", n_turns, dt
        )

        return NPStatesForTurns(fill_turns_to_element(track_data, sel_elem_uids))

        # return track_data
        track_data_model = fill_state_per_element(track_data, data_needed_at)
        return StatesForTurns(turns=track_data_model)


def fill_turns_to_element(track_data: npt.NDArray[np.floating], elem_uids: Sequence[str]) -> Sequence[NPStatePerElementPerTurn]:
    return [
        NPStatePerElementPerTurn(observed_at_element, uid=uid)
        for observed_at_element, uid in zip(track_data.transpose(2, 0, 1, 3), elem_uids)
    ]


def fill_state_per_element(track_data, elm_uids: Sequence[str]) -> Sequence[StatePerTurn]:
    # check the state, in a manner that documents the assumption
    n_state_elms, n_particles, per_n_elems, n_turns = track_data.shape

    # That is the assumption for now: for each element there is data
    assert len(elm_uids) == per_n_elems

    return [
        StatePerTurn(fill_state_per_element_per_track(track, elm_uids))
        for track in track_data.transpose(3, 0, 1, 2)
    ]


def fill_state_per_element_per_track(one_track_data, elm_uids: Sequence[str]) -> Sequence[StatePerElement]:
    return [
       StatePerElement(ParticleStateCollection([ParticleState.from_sequence(p) for p in p_for_particles]), uid)
        for p_for_particles, uid in zip(one_track_data.transpose(2, 1, 0), elm_uids)
    ]


def rectify_uid_for_last_element_if_needed(uids: Sequence[str], copy=True) -> Sequence[str]:
    """
    """
    if uids[0] == uids[-1]:
        if copy:
            uids = _copy(uids)
        uids[-1] = uids[-1] + "_same_pos_as_start"
    return uids


__all__ = ["SimulatorBackend"]