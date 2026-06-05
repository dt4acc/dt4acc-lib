"""

Todo:
    review if required, or if it should rather follow
    the backend interface

"""
from abc import ABCMeta, abstractmethod
from typing import Sequence

from .element import ElementInterface
from ...model.output.survey import SurveyDataForElement


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