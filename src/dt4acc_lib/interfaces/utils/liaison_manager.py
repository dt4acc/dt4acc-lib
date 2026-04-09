from abc import ABCMeta, abstractmethod
from typing import Sequence

from dt4acc_lib.model.utils.identifiers import DevicePropertyID, LatticeElementPropertyID


class LiaisonManagerBase(metaclass=ABCMeta):
    """transforms pairs of (id, property)

    Warning:
        it returns a sequence of device / properties
        More than one device can be necessary to be updated

    """

    @abstractmethod
    def forward(self, id_: LatticeElementPropertyID) -> Sequence[DevicePropertyID]:
        """
        Return sequence here too: be symmetric to inverse

        Todo:
            for symmetry: should return a sequence too!
        """
        raise NotImplementedError("use derived class instead")

    @abstractmethod
    def inverse(self, id_: DevicePropertyID) -> Sequence[LatticeElementPropertyID]:
        """needs to return a sequence: e.g. power converters often power more than one magnet
        """
        raise NotImplementedError("use derived class instead")
