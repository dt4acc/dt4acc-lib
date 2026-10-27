"""Beam entering a transfer line

A ring has a periodic solution: optics and closed orbit follow from
the lattice alone. A transfer line has none: optics and trajectory
are propagated from the beam that enters it.
"""
import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class TwissIn:
    """Optics and centroid of the beam at the entrance of a line

    Planes are (x, y). Units as pyAT uses them.
    """

    #: beta functions [m]
    beta: Sequence[float]
    #: alpha functions [1]
    alpha: Sequence[float] = (0.0, 0.0)
    #: (eta_x, eta_x', eta_y, eta_y') [m, 1, m, 1]
    dispersion: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    #: centroid (x, x', y, y', delta, ct) [m, rad, m, rad, 1, m]
    closed_orbit: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        for name, n in (("beta", 2), ("alpha", 2), ("dispersion", 4), ("closed_orbit", 6)):
            value = tuple(float(v) for v in getattr(self, name))
            if len(value) != n:
                raise ValueError(f"{self.__class__.__name__}.{name}: expected {n} values, got {len(value)}")
            if not all(math.isfinite(v) for v in value):
                raise ValueError(f"{self.__class__.__name__}.{name}: values must be finite, got {value}")
            object.__setattr__(self, name, value)
        if not all(b > 0 for b in self.beta):
            raise ValueError(f"{self.__class__.__name__}.beta: must be positive, got {self.beta}")

    def as_at(self) -> Dict[str, np.ndarray]:
        """As `twiss_in` of :func:`at.get_optics`"""
        return dict(
            beta=np.array(self.beta),
            alpha=np.array(self.alpha),
            dispersion=np.array(self.dispersion),
            closed_orbit=np.array(self.closed_orbit),
        )
