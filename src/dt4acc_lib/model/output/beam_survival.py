from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .particle_losses import ParticleLosses


@dataclass
class BeamSurvivalSettings:
    """How the beam survival is checked: particles around the closed orbit,
    at the start of the lattice, tracked for ``n_turns``

    Without apertures in the lattice the dynamic aperture decides if a
    particle is lost; apertures, when added to the lattice, count without
    a change here.
    """

    #: turns to track
    n_turns: int = 1000
    #: start amplitudes, in beam sizes: the closed orbit, +- n_sigma in x and y
    n_sigma: float = 5.0
    #: vertical emittance as a fraction of the horizontal one (coupling)
    vertical_emittance_ratio: float = 0.01


@dataclass
class BeamSurvival:
    """Particles tracked around the closed orbit: is the beam stored

    The same tracking as the turn by turn data (``compute_track``), from
    start vectors derived from the optics (closed orbit, beam size).
    """

    #: a closed orbit exists and every particle survived ``n_turns``
    stored: bool
    #: closed orbit found
    closed_orbit: bool
    n_turns: int
    n_sigma: float
    #: beam size (x, y) at the start [m]
    sigma: Tuple[float, float]
    #: start offsets (x, y) from the closed orbit [m], one per particle
    offsets: Sequence[Tuple[float, float]] = field(default_factory=list)
    #: per particle: lost within ``n_turns``, where and when (None: no closed orbit, nothing tracked)
    losses: Optional[ParticleLosses] = None

    @property
    def lost(self) -> Sequence[bool]:
        """per particle: lost within ``n_turns``"""
        return [] if self.losses is None else list(self.losses.lost)
