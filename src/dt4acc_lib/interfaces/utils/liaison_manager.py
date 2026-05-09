"""Liaison manager: bring the ones together that need to know from each other

"""
from abc import ABCMeta, abstractmethod
from typing import Sequence

from dt4acc_lib.model.utils.identifiers import DevicePropertyID, LatticeElementPropertyID


class LiaisonManagerBase(metaclass=ABCMeta):
    """transforms pairs of (id, property)

    Actor says:
        * when coming from design

             I know that I need to change the *main strength*
             of quadrupole *bar*. Whom do I need to talk to?

             The actor needs to call the *forward* method.

        * when coming from device

            I know that the current of the quadrupole power supply
            *foo* has been changed. Whom do I need to talk to

            The actor needs to call the *inverse* method

    Please note: in both cases a sequence is returned as one change
    can affect more than one.

    Warning:
        it returns a sequence of device / properties
        More than one device can be necessary to be updated

    """

    @abstractmethod
    def forward(self, id_: LatticeElementPropertyID) -> Sequence[DevicePropertyID]:
        """From design to device

        Or from physics to engineering
        """
        raise NotImplementedError("use derived class instead")

    @abstractmethod
    def inverse(self, id_: DevicePropertyID) -> Sequence[LatticeElementPropertyID]:
        """From device to design

        Or from engineering to physics

        needs to return a sequence: e.g. power converters often power more than one magnet
        """
        raise NotImplementedError("use derived class instead")
