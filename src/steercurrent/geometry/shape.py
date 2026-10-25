"""Orthogonal closed centerlines and periodic path coordinates."""
from dataclasses import dataclass
import math
import numpy as np
from .validation import validate_shape


@dataclass(frozen=True)
class LoopShape:
    """Closed orthogonal centerline, without a repeated closing point.

    The first point is the battery center and lies inside a straight section.
    Distances are periodic and increase in point order. At a corner, direction()
    returns the outgoing tangent. Coordinates and all lengths use the same units.
    """
    points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        validate_shape(self)

    @property
    def lengths(self):
        p = np.asarray(self.points)
        return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)

    @property
    def length(self):
        # Use the endpoint of the same cumulative distances used for segment cuts.
        # A separate sum can differ by one floating-point bit and create a tiny
        # extra CAD segment where the loop closes.
        return float(self.distances[-1])

    @property
    def distances(self):
        return np.r_[0., np.cumsum(self.lengths)]

    @property
    def tangents(self):
        p = np.asarray(self.points)
        return (np.roll(p, -1, axis=0) - p) / self.lengths[:, None]

    def _locate(self, distance):
        if not math.isfinite(distance):
            raise ValueError("Path distance must be finite.")
        s = distance % self.length
        i = min(int(np.searchsorted(self.distances, s, side='right') - 1), len(self.points) - 1)
        return i, s - self.distances[i]

    def position(self, distance):
        i, offset = self._locate(distance)
        return np.asarray(self.points[i]) + offset * self.tangents[i]

    def direction(self, distance):
        i, _ = self._locate(distance)
        return self.tangents[i].copy()
