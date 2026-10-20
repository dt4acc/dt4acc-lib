import logging

import numpy as np
from typing import Sequence
from dt4acc_lib.interfaces.utils.state_conversion import StateConversion
from dt4acc_lib.model.utils.identifiers import CurvePoint

logger = logging.getLogger("dt4acc_lib")


class EnergyDependentLinearUnitConversion(StateConversion):
    """Typical example: magnet parameters"""

    def __init__(self, *, intercept: float, slope: float, brho: float):
        self.intercept = intercept
        self.slope = slope
        self.brho = brho

    def forward(self, state: float) -> float:
        logger.info(
            "%s.forward: brho %s, intercept %s slope %s, state %s",
            self.__class__.__name__,
            self.brho,
            self.intercept,
            self.slope,
            state,
        )
        intercept = self.intercept * self.brho
        slope = self.slope * self.brho
        return intercept + slope * state

    def inverse(self, state: float) -> float:
        logger.info(
            "%s.inverse: brho %s, intercept %s slope %s, state %s",
            self.__class__.__name__,
            self.brho,
            self.intercept,
            self.slope,
            state,
        )
        intercept = self.intercept * self.brho
        slope = self.slope * self.brho
        return (state - intercept) / slope


class LinearUnitConversion(StateConversion):
    """Typical example: tune"""

    def __init__(self, *, intercept: float, slope: float):
        self.intercept = intercept
        self.slope = slope

    def forward(self, state: float) -> float:
        logger.info(
            "%s.forward: intercept %s slope %s, state %s",
            self.__class__.__name__,
            self.intercept,
            self.slope,
            state,
        )
        return self.intercept + self.slope * state

    def inverse(self, state: float) -> float:
        logger.info(
            "%s.inverse: intercept %s slope %s, state %s",
            self.__class__.__name__,
            self.intercept,
            self.slope,
            state,
        )
        return (state - self.intercept) / self.slope



class EnergyIndependentCurveUnitConversion(StateConversion):
    """
    Excitation curve conversion: current <-> magnet strength.

    The curve maps the independent variable (current) to the dependent one,
    the field integrated over the magnet (e.g. B'L for a quadrupole, B*L for
    a corrector), as measured at the facility (e.g. MAX IV's amp2k tables).
    With s = -1 if `flip_dep_sign` else 1 and L = `length` (0 or None: 1,
    the curve is then already the strength's integral, e.g. thin correctors):

    - inverse(I) -> k = s * curve(I) / (brho * L)       (device -> lattice)
    - forward(k) -> I such that s * curve(I) = k * brho * L  (lattice -> device)

    Outside the curve, it is extended linearly from its end segments, as
    MAX IV's amp2k does (interp1(..., 'linear', 'extrap')). forward() searches
    the segments in order of increasing current and returns the first match,
    so it also works for non-monotonic curves.

    - points: sequence of objects with `indep` and `dep`.
    - inverse() uses `bwd_points`, forward() `fwd_points` (callers usually
      pass the same curve for both).
    """

    def __init__(
        self,
        *,
        fwd_points: Sequence[CurvePoint],
        bwd_points: Sequence[CurvePoint],
        brho: float,
        # todo: see how it was named at max iv
        #       iimplement it in a filter delegating implementation to this object
        flip_dep_sign: False,
        length : float
    ):
        self._fwd_indep, self._fwd_dep = _sorted_curve(fwd_points)
        self._bwd_indep, self._bwd_dep = _sorted_curve(bwd_points)
        self.brho = float(brho)
        if self.brho == 0:
            raise ValueError("brho must be non-zero")
        self.length = float(length) if length else 1.0
        self.fwd_points = fwd_points
        self.bwd_points = bwd_points
        self.flip_dep_sign = flip_dep_sign

    def _sign(self) -> float:
        return -1.0 if self.flip_dep_sign else 1.0

    def forward(self, state: float) -> float:
        logger.info(
            "%s.forward: brho %s state %s", self.__class__.__name__, self.brho, state
        )
        target = self._sign() * float(state) * self.brho * self.length
        return _solve_curve(self._fwd_indep, self._fwd_dep, target)

    def inverse(self, state: float) -> float:
        logger.info(
            "%s.inverse: brho %s state %s", self.__class__.__name__, self.brho, state
        )
        dep = _eval_curve(self._bwd_indep, self._bwd_dep, float(state))
        return self._sign() * dep / (self.brho * self.length)


def _sorted_curve(points: Sequence[CurvePoint]):
    pts = sorted((float(p.indep), float(p.dep)) for p in points)
    if len(pts) < 2:
        raise ValueError(f"an excitation curve needs at least 2 points, got {len(pts)}")
    indep = np.array([p[0] for p in pts])
    if np.any(np.diff(indep) == 0):
        raise ValueError("excitation curve has repeated independent values")
    return indep, np.array([p[1] for p in pts])


def _eval_curve(indep: np.ndarray, dep: np.ndarray, x: float) -> float:
    """Linear interpolation, extended linearly from the end segments."""
    if x < indep[0]:
        i = 0
    elif x > indep[-1]:
        i = len(indep) - 2
    else:
        return float(np.interp(x, indep, dep))
    slope = (dep[i + 1] - dep[i]) / (indep[i + 1] - indep[i])
    return float(dep[i] + slope * (x - indep[i]))


def _solve_curve(indep: np.ndarray, dep: np.ndarray, y: float) -> float:
    """x with curve(x) == y: the first segment (increasing x) containing y,
    else the end segment the extended curve reaches y on."""
    for i in range(len(indep) - 1):
        lo, hi = sorted((dep[i], dep[i + 1]))
        if lo <= y <= hi:
            if dep[i + 1] == dep[i]:
                return float(indep[i])
            return float(indep[i] + (y - dep[i]) * (indep[i + 1] - indep[i]) / (dep[i + 1] - dep[i]))
    for i, (x0, sign) in ((0, (indep[0], -1)), (len(indep) - 2, (indep[-1], 1))):
        slope = (dep[i + 1] - dep[i]) / (indep[i + 1] - indep[i])
        if slope == 0:
            continue
        x = indep[i] + (y - dep[i]) / slope
        if (x - x0) * sign >= 0:
            return float(x)
    raise ValueError(f"excitation curve never reaches {y}")

__all__ = ["EnergyDependentLinearUnitConversion", "LinearUnitConversion", "EnergyIndependentCurveUnitConversion"]