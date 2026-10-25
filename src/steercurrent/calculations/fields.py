"""Electric fields Data."""
from dataclasses import dataclass
import numpy as np
from steercurrent.meshing.model import MeshData


@dataclass
class FieldData:
    mesh: MeshData
    potential: np.ndarray           # (nvertices,)
    electric_field: np.ndarray      # (ntriangles, 2)
    current_density: np.ndarray     # (ntriangles, 2), zero in air
    permittivity: float = 1.
    total_charge: float = 0.

    def __post_init__(self):
        for name, shape in (('potential', (len(self.mesh.vertices),)),
                            ('electric_field', (len(self.mesh.triangles), 2)),
                            ('current_density', (len(self.mesh.triangles), 2))):
            values = getattr(self, name)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f'{name} must be finite with shape {shape}.')
        if not np.isfinite(self.permittivity) or self.permittivity <= 0:
            raise ValueError('Permittivity must be finite and positive.')
        if not np.isfinite(self.total_charge):
            raise ValueError('Total charge must be finite.')
