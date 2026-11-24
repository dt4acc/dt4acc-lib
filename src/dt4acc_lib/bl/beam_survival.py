"""Beam survival: the computations, separate from the backend's work flow

Start vectors for checking if the beam is stored: the closed orbit and
particles at +- n beam sizes around it, in x and y.
"""
import math
from typing import List, Sequence, Tuple

import numpy as np

from dt4acc_lib.model.output.track import ParticleState


def beam_size(
    beta: Sequence[float], dispersion: Sequence[float], emittance: Tuple[float, float], sigma_e: float
) -> Tuple[float, float]:
    """(sigma_x, sigma_y) [m]: sqrt(emittance * beta + (dispersion * sigma_e)**2)

    ``beta``: (beta_x, beta_y); ``dispersion``: (eta_x, eta_x', eta_y, eta_y'),
    as pyAT's optics give them.
    """
    emittance_x, emittance_y = emittance
    sigma_x = math.sqrt(emittance_x * beta[0] + (dispersion[0] * sigma_e) ** 2)
    sigma_y = math.sqrt(emittance_y * beta[1] + (dispersion[2] * sigma_e) ** 2)
    return sigma_x, sigma_y


def survival_offsets(sigma: Tuple[float, float], n_sigma: float) -> List[Tuple[float, float]]:
    """(dx, dy) from the closed orbit: the closed orbit itself, +- n_sigma in x, +- n_sigma in y"""
    a_x, a_y = n_sigma * sigma[0], n_sigma * sigma[1]
    return [(0.0, 0.0), (a_x, 0.0), (-a_x, 0.0), (0.0, a_y), (0.0, -a_y)]


def survival_start_vectors(closed_orbit: Sequence[float], offsets: Sequence[Tuple[float, float]]) -> List[ParticleState]:
    """one particle per offset: the closed orbit, shifted by (dx, dy) in x and y"""
    r = []
    for dx, dy in offsets:
        p = np.asarray(closed_orbit, dtype=float).copy()
        p[0] += dx
        p[2] += dy
        r.append(ParticleState.from_sequence([float(v) for v in p]))
    return r


__all__ = ["beam_size", "survival_offsets", "survival_start_vectors"]
