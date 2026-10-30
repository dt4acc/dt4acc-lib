import logging
from typing import Sequence

import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light

from dt4acc_lib.interfaces.utils.state_conversion import StateConversion
from dt4acc_lib.model.utils.command import ReadCommand
from dt4acc_lib.model.utils.translator_manager_lookup_table import (
    CurvePoint,
    MultiplyerScaledByEnergy,
    PolynomCoefficients,
)

logger = logging.getLogger("dt4acc_lib")


def calculate_brho(energy: float, rest_mass: float = 511e3) -> float:
    """
    Todo:
        Energy is assumed here to be in eV not GeV
    """
    energy_without_rest_mass = np.sqrt((energy + rest_mass) ** 2 - rest_mass ** 2)
    brho = energy_without_rest_mass / speed_of_light
    return brho


def calculate_brho_inverse(brho: float, rest_mass: float = 511e3) -> float:
    """Kinetic energy [eV] for ``brho``: the inverse of :func:`calculate_brho`"""
    momentum = brho * speed_of_light
    energy = np.sqrt(momentum ** 2 + rest_mass ** 2) - rest_mass
    return energy


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
    MAX IV's amp2k does (interp1(..., 'linear', 'extrap')). forward() inverts
    the curve: it has to be strictly monotonic.

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
        length: float
    ):
        # forward(): current for a field, the inverted curve
        self._current_for_field = LinearCurve.from_points(fwd_points).inverted()
        # inverse(): field for a current
        self._field_for_current = LinearCurve.from_points(bwd_points)
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
        return self._current_for_field(target)

    def inverse(self, state: float) -> float:
        logger.info(
            "%s.inverse: brho %s state %s", self.__class__.__name__, self.brho, state
        )
        dep = self._field_for_current(float(state))
        return self._sign() * dep / (self.brho * self.length)


class PolynomialUnitConversion(StateConversion):
    """Calibration given as a polynomial of the device value: current <-> magnet strength

    For ``PolynomCoefficients`` with ``polynomial_of="device"``. With p =
    ``calibration.coeffs`` (lowest order first), s = ``calibration.scale`` and
    b = brho (divided by ``reference_brho`` if given) if
    ``calibration.energy_dependent`` else 1:

    - inverse(I) -> k = p(I / s) / b            (device -> lattice)
    - forward(k) -> I with p(I / s) = k * b      (lattice -> device)

    as MML's ``amp2k`` / ``k2amp``. forward() solves for the roots of the
    polynomial and, as MML's MLS ``k2amp`` does, takes the real root closest
    to the linear solution (the polynomial's linear and constant term), the
    one in ``calibration.range`` if any is.
    """

    def __init__(self, *, calibration: PolynomCoefficients, brho: float):
        # numpy's polyval / roots: highest order first
        coeffs = np.trim_zeros(np.asarray(calibration.coeffs, dtype=float)[::-1], "f")
        if len(coeffs) < 2:
            raise ValueError(f"{calibration}: a polynomial of order 0 can not be inverted")
        if coeffs[-2] == 0:
            raise ValueError(f"{calibration}: no linear term, no linear solution to pick a root by")
        if calibration.scale == 0:
            raise ValueError(f"{calibration}: scale must be non-zero")
        if calibration.reference_brho is not None and not calibration.energy_dependent:
            raise ValueError(f"{calibration}: reference_brho needs energy_dependent")
        self.calibration = calibration
        self.coeffs = coeffs
        self.scale = float(calibration.scale)
        self.brho = float(brho) if calibration.energy_dependent else 1.0
        if self.brho == 0:
            raise ValueError("brho must be non-zero")
        if calibration.reference_brho is not None:
            self.brho /= float(calibration.reference_brho)

    def forward(self, state: float) -> float:
        logger.info(
            "%s.forward: brho %s state %s", self.__class__.__name__, self.brho, state
        )
        target = float(state) * self.brho
        intercept, slope = self.coeffs[-1], self.coeffs[-2]
        linear = (target - intercept) / slope
        p = self.coeffs.copy()
        p[-1] -= target
        roots = np.roots(p)
        real = roots.real[np.abs(roots.imag) <= 1e-9 * np.maximum(1.0, np.abs(roots.real))]
        if len(real) == 0:
            raise ValueError(f"{self.__class__.__name__}: no real solution for {state} ({self.calibration})")
        rng = self.calibration.range
        if rng is not None:
            lo, hi = sorted((rng.min / self.scale, rng.max / self.scale))
            inside = real[(real >= lo) & (real <= hi)]
            if len(inside):
                real = inside
        return float(real[np.argmin(np.abs(real - linear))] * self.scale)

    def inverse(self, state: float) -> float:
        logger.info(
            "%s.inverse: brho %s state %s", self.__class__.__name__, self.brho, state
        )
        return float(np.polyval(self.coeffs, float(state) / self.scale)) / self.brho


class LinearCurve:
    """y(x), piecewise linear between the points, extended linearly from its
    end segments outside them (MATLAB's interp1(..., 'linear', 'extrap'))

    Used for excitation curves (current <-> field) and for energy scaling
    tables (energy <-> scaled current).
    """

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        pts = sorted(zip((float(v) for v in x), (float(v) for v in y)))
        if len(pts) < 2:
            raise ValueError(f"a curve needs at least 2 points, got {len(pts)}")
        self.x = np.array([p[0] for p in pts])
        self.y = np.array([p[1] for p in pts])
        if np.any(np.diff(self.x) == 0):
            raise ValueError("curve has repeated independent values")
        self._f = interp1d(self.x, self.y, kind="linear", fill_value="extrapolate", assume_sorted=True)

    @classmethod
    def from_points(cls, points: Sequence[CurvePoint]) -> "LinearCurve":
        return cls([p.indep for p in points], [p.dep for p in points])

    def __call__(self, x: float) -> float:
        return float(self._f(float(x)))

    def inverted(self) -> "LinearCurve":
        """x(y); needs y strictly monotonic in x"""
        steps = np.diff(self.y)
        if not (np.all(steps > 0) or np.all(steps < 0)):
            raise ValueError("curve is not strictly monotonic: it can not be inverted")
        return LinearCurve(self.y, self.x)


class MultiplierScaledByEnergyUnitConversion(StateConversion):
    def __init__(self, *, conv_data: MultiplyerScaledByEnergy, brho: float):
        """Todo: need actual brho or energy"""
        self.brho = float(brho)
        self.conv_data = conv_data

        # from energy to scaled current, and back
        self.fwd_interp = LinearCurve.from_points(conv_data.scale_by_energy)
        self.bwd_interp = self.fwd_interp.inverted()

    def forward(self, state: float) -> float:
        """corresponds to k2amp"""
        ref_field = state * self.brho
        t_brho = ref_field / self.conv_data.reference_multiplyer
        energy = calculate_brho_inverse(t_brho)
        scale = self.fwd_interp(energy)
        r = self.conv_data.range.interpolate(scale)
        return r

    def inverse(self, state: float) -> float:
        """corresponds to amp2k

        As ALS's amp2k: the energy on the ramp table for this current gives
        the field ``ref_k * brho(e)``; K is that field over the brho of the
        beam, the same brho forward() (k2amp) uses.
        """
        lambda_ = self.conv_data.range.scale(state)
        corresponding_energy = self.bwd_interp(lambda_)
        ref_k = self.conv_data.reference_multiplyer
        # Todo: fix data to be in eV
        t_brho = calculate_brho(corresponding_energy)
        correction = t_brho / self.brho
        return ref_k * correction


class UnitConversionAroundReferenceValue(StateConversion):
    """
    Todo:
        can we reuse state cache ...
    """
    def __init__(self, *, sub_obj: StateConversion, design_view_rcmd: ReadCommand, device_view_rcmd: ReadCommand, reference_cache = None):
        """
        Todo:
            type reference_cache as StateCache
        """
        self.sub_obj = sub_obj
        self.design_view_rcmd = design_view_rcmd
        self.device_view_rcmd = device_view_rcmd
        self.reference_cache = reference_cache

    def register_reference_cache(self, cache):
        assert callable(cache.get)
        self.reference_cache = cache

    def forward(self, state: float) -> float:
        dval = float(state)
        rev_val = self.get_reference_value(self.design_view_rcmd)
        total_val = dval + rev_val
        tmp = self.sub_obj.forward(total_val)
        rev_tgt = self.get_reference_value(self.device_view_rcmd)
        return tmp - rev_tgt

    def inverse(self, state: float) -> float:
        dval = float(state)
        rev_val = self.get_reference_value(self.device_view_rcmd)
        total_val = dval + rev_val
        tmp = self.sub_obj.inverse(total_val)
        rev_tgt = self.get_reference_value(self.design_view_rcmd)
        return tmp - rev_tgt

    def get_reference_value(self, rcmd: ReadCommand) -> float:
        if self.reference_cache is None:
            raise ValueError("reference_cache must be registered at this stage. Did you register it with translation service?")
        assert rcmd is not None
        rev_val = self.reference_cache.get(rcmd)
        if rev_val is None:
            raise ValueError(f"reference_cache {self.reference_cache} must provide reference data for {rcmd} but returned None. Did you populate the cache with an appropriate value")
        return rev_val

__all__ = [
    "EnergyDependentLinearUnitConversion",
    "LinearUnitConversion",
    "EnergyIndependentCurveUnitConversion",
    "LinearCurve",
    "PolynomialUnitConversion",
    "MultiplierScaledByEnergyUnitConversion",
    "calculate_brho_inverse",
    "calculate_brho",
]
