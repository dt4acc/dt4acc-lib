"""Probe backend: similar to view, but addressed entity can change

It needs rather to address the whole accelerator

Why: there is not always a direct mapping from one
     entity in the "design" view to the "device" view
"""
from abc import ABCMeta, abstractmethod


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
    """
    @abstractmethod
    async def reset(self):
        """Reset the simulation back end """
        raise NotImplementedError("use base class instead")
