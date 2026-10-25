"""Mesh-resolution and exterior settings."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MeshOptions:
    size: float | None = None
    local_size: float | None = None
    refinement_distance: float | None = None

    def resolved(self, width):
        size = width / 4 if self.size is None else self.size
        local = size / 2 if self.local_size is None else self.local_size
        distance = width if self.refinement_distance is None else self.refinement_distance
        if not all(math.isfinite(x) and x > 0 for x in (size, local, distance)) or local > size:
            raise ValueError(
                "Mesh sizes must be positive and local_size must not exceed size.")
        return {"size": size, "local_size": local, "refinement_distance": distance}


@dataclass(frozen=True)
class ExteriorOptions:
    """Far rectangle scales the conductor bounding box about its center.

    Fixed conductor-boundary edges are retained regardless of near_size.
    """
    outer_scale: float = 4.0
    near_size: float | None = None
    far_size: float | None = None
    transition_distance: float | None = None

    def resolved(self, *, span, conductor_size):
        """Resolve exterior defaults from the conductor span and target mesh size."""
        if not all(math.isfinite(v) and v > 0 for v in (span, conductor_size)):
            raise ValueError(
                "Conductor span and mesh size must be finite and positive.")
        settings = dict(outer_scale=self.outer_scale,
                        near_size=conductor_size if self.near_size is None else self.near_size,
                        far_size=span/4 if self.far_size is None else self.far_size,
                        transition_distance=span if self.transition_distance is None else self.transition_distance)
        if (not all(math.isfinite(v) and v > 0 for v in settings.values()) or
                self.outer_scale <= 1 or settings['far_size'] < settings['near_size']):
            raise ValueError(
                "Exterior needs outer_scale > 1 and positive sizes with far_size >= near_size.")
        return settings
