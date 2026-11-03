from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class ParticleLosses:
    """Which tracked particles are lost, where and when: pyAT's loss map

    pyAT marks a particle lost when one of its coordinates gets non finite,
    one of x, px, y, py, delta exceeds 1 (``LIMIT_AMPLITUDE`` in pyAT's
    ``at.c``), or it hits a physical aperture of an element (``EApertures``,
    ``RApertures``). Its tracked coordinates are NaN from then on: the same
    a turn by turn reading counts as not survived. One entry per particle.
    The base for loss monitors.
    """

    #: lost within the tracked turns
    lost: Sequence[bool]
    #: turn it is lost in (None: not lost), counted from the start of the tracking
    turn: Sequence[Optional[int]] = field(default_factory=list)
    #: element (uid) it is lost at (None: not lost)
    element: Sequence[Optional[str]] = field(default_factory=list)
    #: coordinates (x, px, y, py, delta, ct) when it was lost (None: not lost)
    coord: Sequence[Optional[Tuple[float, ...]]] = field(default_factory=list)
