"""Recover charge density from electric fields."""
from dataclasses import dataclass
import numpy as np
from steercurrent.meshing.model import MeshData
from steercurrent.calculations.fields import FieldData


@dataclass
class ChargeTrace:
    """One density per edge."""
    mesh: MeshData
    edge_ids: np.ndarray
    density: np.ndarray
    regions: tuple[int, int] | None = None

    @property
    def charge_per_depth(self):
        return self.density * self.mesh.edge_lengths[self.edge_ids]


@dataclass
class ChargeResult:
    fields: FieldData
    # surfaces: conductor-air boundary; keys are "outer" and "cavity"
    surfaces: dict[str, ChargeTrace]
    # interfaces: boundaries inside conductor; such as wire-resistor, wire-battery
    interfaces: dict[str, ChargeTrace]
    bulk_density: np.ndarray  # (ntriangles,), zero under the current model


def calculate_charges(fields: FieldData):
    """Recover surface/interface charge.

    Assumes uniform epsilon (fields.permittivity), piecewise-constant
    conductivity and uniform battery driving field. Bulk charge is then zero.
    Same-material FEM gradient jumps are numerical errors.
    No correction is applied to force the recovered charges to sum to zero.
    """
    mesh = fields.mesh
    epsilon = fields.permittivity
    electric = fields.electric_field
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Permittivity must be finite and positive.')
    if electric.shape != (len(mesh.triangles), 2) or not np.all(np.isfinite(electric)):
        raise ValueError(
            'Electric field must be a finite array of shape (ntriangles, 2).')
    normals = mesh.edge_normals

    def trace(name, regions=None):
        edges = mesh.edge_sets[name]
        first, second = mesh.edge_triangles[edges].T
        if np.any(second < 0):
            raise ValueError(
                'Charge surfaces require triangles on both sides.')
        jump = electric[second] - electric[first]
        density = epsilon * np.sum(jump * normals[edges], axis=1)
        return ChargeTrace(mesh, edges, density, regions)

    surfaces = {name: trace(name) for name in ('outer', 'cavity')}
    interfaces = {name: trace(name, regions)
                  for name, regions in mesh.interface_regions.items()}
    return ChargeResult(fields, surfaces, interfaces, np.zeros(len(mesh.triangles)))
