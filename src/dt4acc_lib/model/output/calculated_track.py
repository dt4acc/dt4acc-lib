from dataclasses import dataclass
from typing import Sequence


@dataclass
class CalculatedPosition:
    # this one is expected to be unique
    uid: str
    # family name: signal to user it is not necessarily unique
    fam_name: str
    x: float
    y: float


@dataclass
class CalculatedTrack:
    track : Sequence[CalculatedPosition]