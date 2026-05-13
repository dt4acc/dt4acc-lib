from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence, Union

from .conv import deserialse_value, serialize_value


class BehaviourOnError(IntEnum):
    """
    Todo: use a standard enum, int enums are a bit special

    See reasons given in the book "Robust Python"
    """
    stop = 1
    ignore = 2
    roll_back = 3


@dataclass(frozen=True)
class ReadCommand:
    """Use for retrieving data"""

    id: str
    property: str


@dataclass
class Command:
    #: can be the identifier of a lattice element or a device
    id: str
    property: str
    #: the object can not be deserialized need to be more precise here
    value: Union[int, float]
    behaviour_on_error: BehaviourOnError

    @classmethod
    def from_jsons(cls, d: dict, **kwargs):
        """
        Todo:
            check if kwargs needs to be passed to some argument
        """
        v = deserialse_value(d["value"])
        return cls(
            id=d["id"],
            property=d["property"],
            value=v,
            behaviour_on_error=d["behaviour_on_error"],
        )

    def to_jsons(self, **kwargs):
        d = dict(
            id=self.id,
            property=self.property,
            value=serialize_value(self.value, **kwargs),
            behaviour_on_error=self.behaviour_on_error,
        )
        return d


@dataclass
class TransactionCommand:
    """Commands that should be executed (more or less) at the same time

    In particular: state of the machine is only relevant *after* all
    commands have been executed
    """

    transaction: Sequence[Command]


@dataclass
class CommandSequence:
    """These commands are expected to be executed on by one"""

    commands: Sequence[TransactionCommand]


def command_deserializer(obj: dict, cls: Command, **kwargs):
    assert cls == Command, "only prepared to handle single reading"
    r = cls.from_jsons(obj, **kwargs)
    return r


def register_deserializer_for_command(t_fork):
    import jsons

    jsons.set_deserializer(
        command_deserializer, Command, high_prio=True, fork_inst=t_fork
    )


def register_serializer_for_command(t_fork):
    import jsons

    def conv(ins: Command, **kwargs):
        return ins.to_jsons(**kwargs)

    jsons.set_serializer(conv, Command, high_prio=True, fork_inst=t_fork)


__all__ = [
    "BehaviourOnError",
    "Command",
    "CommandSequence",
    "ReadCommand",
    "TransactionCommand",
]
