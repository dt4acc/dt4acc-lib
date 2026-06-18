"""Data class like view to the turn-by-turn track data

"""
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Sequence

import numpy as np
import numpy.typing as npt

from .track import ParticleState

class StateComponentIndex(Enum):
    """
    Todo:
        check if it is defined somewhere else

        This definition could be back engine specific
    """
    x = 0
    dx = 1
    y = 2
    dy = 3
    dp = 4
    dt = 5


class NPParticleState:
    def __init__(self, state: npt.NDArray[np.float64]):
        self.state = state

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"x={self.x}"
            f", dx={self.dx}"
            f", y={self.y}"
            f", dy={self.dy}"
            f", dp={self.dp}"
            f", dt={self.dt}"
            ")"
        )

    @property
    def n_elements(self) -> int:
        return ParticleState.n_elements()

    @property
    def x(self) -> float:
        return self.state[StateComponentIndex.x.value]

    @property
    def dx(self) -> float:
        return self.state[StateComponentIndex.dx.value]

    @property
    def y(self) -> float:
        return self.state[StateComponentIndex.y.value]

    @property
    def dy(self) -> float:
        return self.state[StateComponentIndex.dy.value]

    @property
    def dp(self) -> float:
        return self.state[StateComponentIndex.dp.value]

    @property
    def dt(self) -> float:
        return self.state[StateComponentIndex.dt.value]


class NPParticleCollection:
    def __init__(self, col: npt.NDArray[np.float64]):
        self.col = col

    def n_particles(self) -> int:
        n_elements, n_particles = self.col.shape
        return n_particles

    def particle_per_id(self, idx) -> NPParticleState:
        return NPParticleState(self.col[:, idx])

    def __repr__(self):
        particles = self.col.transpose(1, 0)
        if len(particles) < 6:
            particle_description = ", ".join(
                [repr(NPParticleState(p)) for p in particles]
            )
        else:
            start = ", ".join([repr(NPParticleState(p)) for p in particles[:3]])
            end = ", ".join([repr(NPParticleState(p)) for p in particles[-3:]])
            particle_description = start  + " , ... ,  " + end
        return f"{self.__class__.__name__}({particle_description})"


class NPViewForStateComponent:
    def __init__(self, track_data: npt.NDArray[np.float64], uid: str, view:str):
        # Check shape
        n_particles, n_turns = track_data.shape
        self.track_data = track_data
        self.uid = uid
        self.view = view

    @property
    def n_particles(self) -> int:
        return self.track_data.shape[0]

    @property
    def n_turns(self) -> int:
        return self.track_data.shape[1]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"element_uid={self.uid}"
            f", view={self.view}"
            f", n_turns={self.n_turns}"
            f", n_particles={self.n_particles}"
            ")"
        )

    def mean_per_turn(self):
        return self.track_data.mean(axis=0)

    def std_per_turn(self):
        return self.track_data.std(axis=0)

    def variance_per_turn(self):
        return self.track_data.var(axis=0)


class NPStatePerElementPerTurn:
    """

    Some optimisation: original data model too slow
    """

    def __init__(self, track_data: npt.NDArray[np.float64], uid: str):
        n_state_elms, n_particles, n_turns = track_data.shape
        assert n_state_elms == ParticleState.n_elements()
        assert n_particles >= 1
        assert n_turns >= 1
        self.track_data = track_data
        self.uid = uid

    def __repr__(self):
        n_state_elms, n_particles, n_turns = self.track_data.shape
        return (
            f"{self.__class__.__name__}("
            f"uid={self.uid}"
            f", n_turns={n_turns}"
            f", n_particles={n_particles}"
            ")"
        )

    def for_turn(self, turn: int) -> NPParticleCollection:
        return NPParticleCollection(col=self.track_data[:, :, turn])

    def get_x(self) -> NPViewForStateComponent:
        return NPViewForStateComponent(
            track_data=self.track_data[StateComponentIndex.x.value,:,:], uid=self.uid, view="x"
        )

    def get_y(self):
        return NPViewForStateComponent(
            track_data=self.track_data[StateComponentIndex.y.value,:,:], uid=self.uid, view="y"
        )


@dataclass
class NPStatesForTurns:
    per_element: Sequence[NPStatePerElementPerTurn]

    def for_element(self, uid: str):
        idx = self.index_for_uid(uid)
        return self.per_element[idx]

    def index_for_uid(self, uid: str) -> int:
        return self._index_for_uid[uid]

    @cached_property
    def _index_for_uid(self):
        """
        Todo:
            address the double name for start and end
        """
        return {elem.uid: idx for idx, elem in enumerate(self.per_element)}