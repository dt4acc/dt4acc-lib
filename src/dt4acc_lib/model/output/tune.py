from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Sequence, Optional






@dataclass
class CorrectionStat:
    mean: float
    std: float
    min: float
    max: float

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            "mean={self.mean:.3f},"
            " std={self.std:.3f},"
            " min={self.min:.3f},"
            " max={self.max:.3f}"
            ")"
        )

    def __mul__(self, other: float):
        return CorrectionStat(
            mean=self.mean * other,
            std=self.std * other,
            max=self.max * other,
            min=self.min * other,
        )


@dataclass
class Tune:
    """
    Todo:
        merge
         it with accml.app.tune.model.MeasuredTuneResponseItem?
    """

    #: horizontal
    x: float
    #: vertical
    y: float

    def __sub__(self, other):
        return Tune(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Tune(-self.x, -self.y)

    def __str__(self):
        return f"{self.__class__.__name__}(x={self.x:.4f}, y={self.y:.4f})"
