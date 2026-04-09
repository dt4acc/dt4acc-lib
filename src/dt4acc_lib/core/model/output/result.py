import datetime
import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import jsons

from ..utils.conv import deserialse_value, serialize_value
from ..utils.command import ReadCommand, Command


@dataclass
class SingleFloat:
    value: float


@dataclass
class SingleReading:
    """

    e.g. reading from one device or tune of the machine
    """

    name: str
    payload: object
    cmd: ReadCommand

    @classmethod
    def from_jsons(cls, obj, **kwargs):
        cmd = jsons.load(obj["cmd"], ReadCommand, **kwargs)
        return cls(name=obj["name"], cmd=cmd, payload=deserialse_value(obj["payload"]))

    def to_jsons(self, **kwargs):
        import jsons

        return dict(
            name=self.name,
            cmd=jsons.dump(self.cmd, **kwargs),
            payload=serialize_value(self.payload, **kwargs),
        )


@dataclass
class TranslatedReading:
    cmd: ReadCommand
    readings: Sequence[SingleReading]


@dataclass
class ReadTogether:
    """
    data taken together
    """

    data: Sequence[SingleReading]
    start: datetime.datetime
    end: datetime.datetime

    def get(self, key):
        return self._dict[key]

    @cached_property
    def _dict(self):
        return {d.name: d for d in self.data}

    def to_jsons(self, **kwargs):
        import jsons

        def conv(obj):
            r = jsons.dump(obj, **kwargs)
            return r

        return dict(data=conv(self.data), start=conv(self.start), end=conv(self.end))


@dataclass
class ReadTogetherAndTranslated:
    """
    data taken together
    """

    data: Sequence[TranslatedReading]
    start: datetime.datetime
    end: datetime.datetime

    def all_readings(self) -> Sequence[SingleReading]:
        return tuple(itertools.chain(*[d.readings for d in self.data]))


@dataclass
class ResultOfExecutionStep:
    #: the relevant commands reflecting state changes
    #: the idea is that with data one can reconstruct
    #: in which state the machine or accelerator was
    cmds: Sequence[Command]
    data: Sequence[ReadTogether]


@dataclass
class Result:
    #: in the expected view / context of the caller
    data: Sequence[ResultOfExecutionStep]
    #: in the expected view / context as produced by the backend
    orig_data: Sequence[ResultOfExecutionStep]


def register_deserializer_for_single_reading(t_fork):
    import jsons

    def convert(obj: dict, cls: SingleReading, **kwargs):
        assert cls == SingleReading, "only prepared for SingleReading here"
        return cls.from_jsons(obj, **kwargs)

    jsons.set_deserializer(convert, SingleReading, high_prio=True, fork_inst=t_fork)


def register_serializer_for_single_reading(t_fork):
    import jsons

    def convert(obj, cls: SingleReading, **kwargs):
        if cls is not None:
            assert cls == SingleReading, "only prepared to handle single reading"
        return obj.to_jsons(**kwargs)

    jsons.set_serializer(convert, SingleReading, high_prio=True, fork_inst=t_fork)


def register_serializer_for_read_together(t_fork):
    import jsons

    def convert(obj, cls: ReadTogether, **kwargs):
        if cls is not None:
            assert cls == ReadTogether, "only prepared to handle read to gether"
        return obj.to_jsons(**kwargs)

    jsons.set_serializer(convert, ReadTogether, high_prio=True, fork_inst=t_fork)
