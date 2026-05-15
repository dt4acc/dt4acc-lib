from typing import Tuple, Sequence, Dict

import numpy as np
import at
from scipy.constants import speed_of_light


def estimate_shift(element, eps=1e-8):
    """
    Todo: get it upstreamed into pyat
    todo: currently this is very pyat specific element update
    Estimate the shift values for an element.

    Args:
        element: The element to estimate shift for.
        eps: Tolerance value for consistency check.

    Returns:
        np.ndarray: Computed shift values.

    Raises:
        AssertionError: If standard deviation exceeds the tolerance.
    """
    try:
        down_stream_shift = element.T1
    except AttributeError:
        down_stream_shift = np.zeros([6], float)
    try:
        up_stream_shift = element.T2
    except AttributeError:
        up_stream_shift = np.zeros([6], float)

    prep = np.array([down_stream_shift, -up_stream_shift])
    shift = prep.mean(axis=0)

    # shifts can be applied to more than one element, these are now
    # expected to have all the same shift.
    assert (np.absolute(prep.std(axis=0)) < eps).all()
    return shift


def update_shift(element, dx=None, dy=None):
    """
    Update the element shift.

    Args:
        dx: Shift in x-direction.
        dy: Shift in y-direction.

    Raises:
        AssertionError: If both dx and dy are None.
    """
    assert dx is not None or dy is not None, "Either dx or dy must be provided"

    shift = estimate_shift(element)

    dx = dx if dx is not None else shift[0]
    dy = dy if dy is not None else shift[1]

    # call AT shift element
    at.shift_elem(element, dx, dy)


def peek_kick(element, property_id: str) -> float:
    lut = dict(x_kick=0, y_kick=1)
    try:
        idx = lut[property_id]
    except KeyError as ke:
        raise AssertionError(f"Did not expect kick {property_id}")
    return float(element.KickAngle[idx])


def manipulate_kick(
    kick_angles: Tuple[float, float], kick_x=None, kick_y=None
) -> Tuple[float, float]:
    kick_angles = kick_angles.copy()
    if kick_x is not None:
        kick_angles[0] = kick_x
    if kick_y is not None:
        kick_angles[1] = kick_y
    return kick_angles


def check_multipole_index(idx):
    """Multipole indices according to European convention

    The multipole index is used for indexing into the polynomial arrays

    A 0 for dipole would result in a -1, this would then produce unexpected
    results
    """
    if idx == 0:
        raise AssertionError(
            "Using European multipole convention thus idx>0, but found {idx}"
        )


def update_magnetic_polynom(
    polynom: Sequence[float], coeffs: Dict[int, float], copy: bool = True
):
    """
    coeffs: coefficients to update. cofficient index in European Convention

    Todo:

        warning: does not check the multipole number
    """
    if copy:
        polynom = polynom.copy()
    for idx, coeff in coeffs.items():
        polynom[idx - 1] = coeff
    return polynom


def estimate_dipole_main_field(
        beam_energy: float,
        dipole_angle: float,
        path_length: float
):
    """dipole main field derived from energy

    Warning: only valid for electrons

    .. math::

        B = \\frac{\\theta}{ecL}\\sqrt{E^2 - m_e^2}

    Where did irho from Tracy go?
    """
    electron_rest_energy = 511e3
    ratio = dipole_angle /  (speed_of_light * path_length)
    E = np.sqrt(beam_energy - electron_rest_energy)
    r  = ratio * E
    return r


__all__ = [
    "peek_kick",
    "manipulate_kick",
    "estimate_shift",
    "update_shift",
    "update_magnetic_polynom",
    "check_multipole_index",
]
