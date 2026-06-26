from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

import numpy as np


@dataclass
class ParticleState:
    """
    Todo:
        find out if dp/dt are swapped
        same question: AT still adheres to Tracy standards
    """
    x: float
    px: float
    y: float
    py: float
    delta: float
    ct: float

    @classmethod
    def n_elements(cls) -> int:
        return 6

    @classmethod
    def from_sequence(cls, input: Sequence[float]):
        assert len(input) == cls.n_elements()
        return cls(*input)

    def as_array(self):
        return np.array(
            [self.x, self.px, self.y, self.py, self.delta, self.ct],
            dtype=float
        )


@dataclass
class ParticleStateCollection:
    """

    Useful to combine data per device
    """
    particles: Sequence[ParticleState]


@dataclass
class StatePerElement:
    particles: ParticleStateCollection
    #: element uid
    uid: str


@dataclass
class StatePerElementPerTurn:
    """a particle collection as seen each time the track passed"""
    particles: Sequence[ParticleStateCollection]
    #: element uid
    uid: str


@dataclass
class StatePerTurn:
    """
    Assumption:
        the uids are in the same order for all state elements
    """
    data: Sequence[StatePerElement]

    def index_for_uid(self, uid: str) -> int:
        return self._index_for_uid[uid]

    @cached_property
    def _index_for_uid(self):
        """
        Todo:
            address the double name for start and end
        """
        return {elem.uid: idx  for idx, elem in enumerate(self.data)}


@dataclass
class StatesForTurns:
    turns: Sequence[StatePerTurn]

    def for_element(self, uid: str) -> StatePerElementPerTurn:
        """From each turn take the data for this particular element
        """
        idx = self.turns[0].index_for_uid(uid)
        turn_data_for_element = [turn.data[idx] for turn in self.turns]
        for datum in turn_data_for_element:
            assert datum.uid == uid
        r =  StatePerElementPerTurn([pc.particles for pc in turn_data_for_element], uid)
        return r