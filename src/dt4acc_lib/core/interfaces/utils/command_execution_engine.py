from abc import ABCMeta, abstractmethod
from typing import Sequence

from ...model.utils.command import ReadCommand, Command
from ...model.output.result import ReadTogether


class CommandExecutionEngine(metaclass=ABCMeta):
    @abstractmethod
    async def trigger_read(self, cmds: Sequence[ReadCommand]) -> ReadTogether:
        """Following ophyd-async / ophyd design

            Todo:
                Is this a good idea?
        """
        raise NotImplementedError("use derived class instead")

    @abstractmethod
    async def set(self, cmds: Sequence[Command]):
        """execute these commands together"""
        raise NotImplementedError("use derived class instead")
