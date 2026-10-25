"""A fixed circuit configuration embedded in a compatible loop shape."""
from dataclasses import asdict, dataclass
import numpy as np
from steercurrent.geometry.shape import LoopShape
from steercurrent.geometry.configuration import CircuitConfig
from .validation import validate_geometry


@dataclass(frozen=True)
class CircuitGeometry:
    """A fixed configuration embedded in a compatible sharp orthogonal shape.

    Length agreement uses relative tolerance 1e-10.
    Replacing shape preserves the configuration and revalidates all clearances.
    """
    config: CircuitConfig
    shape: LoopShape

    def __post_init__(self):
        self.validate()

    def validate(self):
        validate_geometry(self)

    # turns the zero-width centerline into the finite-width conductor,
    # returning one quadrilateral per centerline segment.
    # The result is a list of arrays, each shaped (4, 2): four polygon vertices, each with (x, y) coordinates.
    # These polygons cover only the conductor strip, including wires, resistors, and the battery.
    def strip_polygons(self):
        p, t, a = np.asarray(
            self.shape.points), self.shape.tangents, self.config.width / 2
        left = np.column_stack((-t[:, 1], t[:, 0]))
        previous = np.roll(left, 1, axis=0)
        offsets = a * (left + previous) / \
            (1 + np.sum(left * previous, axis=1))[:, None]
        plus, minus = p + offsets, p - offsets
        polygons = []
        for i in range(len(p)):
            j = (i + 1) % len(p)
            if min((plus[j] - plus[i]) @ t[i], (minus[j] - minus[i]) @ t[i]) <= 0:
                raise ValueError(
                    "Conductor width collapses a segment or cavity near a corner.")
            polygons.append(np.array([minus[i], minus[j], plus[j], plus[i]]))
        return polygons

    def to_dict(self):
        """JSON geometry with separate configuration and shape."""
        return {'config': self.config.to_dict(), 'shape': asdict(self.shape)}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['config'] = CircuitConfig.from_dict(data['config'])
        data['shape'] = LoopShape(**data['shape'])
        return cls(**data)
