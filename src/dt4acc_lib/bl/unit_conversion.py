import logging

import numpy as np
from typing import Sequence
from dt4acc_lib.interfaces.utils.state_conversion import StateConversion
from dt4acc_lib.model.utils.translator_manager_lookup_table import CurvePoint
from scipy.interpolate import interp1d

logger = logging.getLogger("dt4acc_lib")


def calculate_brho(energy: float, rest_mass: float=511e3) -> float:
    """
    Todo:
        Energy is assumed here to be in eV not GeV
    """
    energy_without_rest_mass = np.sqrt((energy + rest_mass)**2 - rest_mass**2)
    # todo: get it from constants
    speed_of_light = 299792458.2
    brho = energy_without_rest_mass / speed_of_light
    return brho


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
    Interpolate a curve (independent -> dependent) using scipy.interpolate.interp1d
    and scale by `brho`.

    - points: sequence of 2-tuples (indep, dep) or objects with `indep` and `dep`.
    - forward(x) -> interpolated_dep(x) * brho
    - inverse(y) -> x such that interpolated_dep(x) == y / brho (searches segments;
      works for non-monotonic curves by returning the first matching segment)
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
        # forward interpolator: will raise if x out of bounds
        # TODO: clean up it is a mess at the moment
        self._fwd = interp1d(
            [t.indep for t in fwd_points],
            [t.dep for t in fwd_points],
            kind="linear",
            # Todo: change later to true ... or make user configurable
            bounds_error=False,
        )
        self._bwd = interp1d(
            [t.indep for t in bwd_points],
            [t.dep for t in bwd_points],
            kind="linear",
            # Todo: change later to true ... or make user configurable
            bounds_error=False,
        )
        self.brho = float(brho)
        self.length = float(length) if length else 1.0
        self.fwd_points = fwd_points
        self.bwd_points = bwd_points
        self.flip_dep_sign = flip_dep_sign

    def forward(self, state: float) -> float:
        logger.info(
            "%s.forward: brho %s state %s", self.__class__.__name__, self.brho, state
        )
        x = float(state)
        y = float(self._fwd(x))  # interp1d returns an array-like
        if self.flip_dep_sign:
            y = -y
        return  y * self.brho

    def inverse(self, state: float) -> float:
        # logger.info("%s.inverse: brho %s points %d state %s", self.__class__.__name__, self.brho, len(self._indep), state)
        if self.brho == 0:
            raise ValueError("brho must be non-zero for inversion")
        target = float(state) / self.brho
        y = float(self._bwd(target)) # interp1d returns an array-like
        assert np.isfinite(y), "failed to inverse {state=} ({brho=}, {target=})"
        if self.flip_dep_sign:
            y = y * -1
        return y / self.length

__all__ = ["EnergyDependentLinearUnitConversion", "LinearUnitConversion", "EnergyIndependentCurveUnitConversion"]