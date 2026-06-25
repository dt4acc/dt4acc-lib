"""Probe backend: similar to view, but addressed entity can change

It needs rather to address the whole accelerator

Why: there is not always a direct mapping from one
     entity in the "design" view to the "device" view
"""
from abc import ABCMeta, abstractmethod

from dt4acc_lib.interfaces.backend.calculation_states import CalculationStates


class BackendR(metaclass=ABCMeta):
    """ """

    @abstractmethod
    def get_natural_view_name(self):
        raise NotImplementedError("use base class instead")

    @abstractmethod
    async def trigger(self, dev_id: str, prop_id: str):
        raise NotImplementedError("use base class instead")

    @abstractmethod
    async def read(self, dev_id: str, prop_id: str) -> object:
        raise NotImplementedError("use base class instead")


class BackendRW(BackendR, metaclass=ABCMeta):
    @abstractmethod
    async def set(self, dev_id: str, prop_id: str, value: object):
        raise NotImplementedError("use base class instead")


class SimulatorBackendRW(BackendRW, metaclass=ABCMeta):
    """Methods that a real world backend would not provide

    Todo:
        revisit if that is not the case ....

        Are these method better implemented by dedicated
        messages?

        Think in the way like SCPI interacts with a device?
        Set something that peek for its status ....

    """
    @abstractmethod
    def get_state(self) -> CalculationStates:
        raise NotImplementedError("use base class instead")

    @abstractmethod
    async def reset(self):
        """Reset the simulation back end """
        raise NotImplementedError("use base class instead")

    @abstractmethod
    async def acknowledge(self):
        """Flag that error state hsa been received

        Idea: user / calling side has safety that error will not pass unchecked

        In error state the model will not accept changes. In this state it will
        accept changes to its elements but not start recomputation

        The state has to be reset before calculations can start again

        Todo:
            need to check if that setup is too fine grained
        """

        raise NotImplementedError("use base class instead")
