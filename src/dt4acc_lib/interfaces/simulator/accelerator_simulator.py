"""

Todo:
    review if required, or if it should rather follow
    the backend interface

"""
from abc import ABCMeta, abstractmethod
from typing import Sequence

from .element import ElementInterface
from ...model.output.survey import SurveyDataForElement
from ...model.output.track import StatePerTurn


class OpticsCalculationError(Exception):
   """Failed to calculate optics parameters.
   """

class OpticsCalculationProhibitedError(Exception):
    """Calculation of optics failed already

    state engine need to be reset (and properly some actor values)
    before starting calculation again

    Todo:
        necessary to distinquish between the two errors?
    """

class AcceleratorSimulatorInterface(metaclass=ABCMeta):
    """

    Todo:
        Derive from a list interface
    """

    @abstractmethod
    def get(self, element_id) -> ElementInterface:
        """
        Review if derived classes use async implementations
        """
        pass

    @abstractmethod
    def track(self, p0: Sequence[float], n_turns: int, data_needed_at_element_index: Sequence[int]):
        """track along the ring

        Args:
            p0: start vector
            n_turns: number of turns


        Returns:
            state for each particle for each requested device for each turn

        The design follows currently at.tracking.lattice_track
        Needs to be revisited as soon as an other calculation engine is
        to be integrated

        Todo:
            do we need to be able to start tracking at any point?
            Work on tracking output
        """

    @abstractmethod
    def get_survey(self) -> Sequence[SurveyDataForElement]:
        """Get survey data information
        """

    @abstractmethod
    def get_optics_parameters(self):
        """
        Todo: fix return type
        """

    @abstractmethod
    def reinit(self):
        """Reinitialise simulator from lattice

        Todo:
            better reset?
            Or should one already give a key or an
            other lattice it should get the parameters from
        """