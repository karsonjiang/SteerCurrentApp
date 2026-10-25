"""Cavity and outside electrostatic potential."""
from dataclasses import dataclass
import numpy as np
from steercurrent.calculations.fem import assemble_p1
from steercurrent.calculations.fem import solve_dirichlet
from steercurrent.calculations.conduction import ConductionResult
from steercurrent.calculations.fields import FieldData


@dataclass
class ExteriorResult:
    conduction: ConductionResult
    fields: FieldData
    potential_offset: float

    @property
    def mesh(self):
        return self.fields.mesh

    @property
    def potential(self):
        return self.fields.potential

    @property
    def electric_field(self):
        return self.fields.electric_field

    @property
    def permittivity(self):
        return self.fields.permittivity


def solve_exterior(conduction, *, permittivity=1.):
    """Fill air V/E on the existing mesh, choosing the potential offset to achieve neutral charges.
    """
    if not np.isfinite(permittivity) or permittivity <= 0:
        raise ValueError('Permittivity must be finite and positive.')
    mesh = conduction.mesh
    conductor_cells = mesh.triangle_sets['conductor']
    conductor_nodes = np.unique(mesh.triangles[conductor_cells])
    if not np.all(np.isfinite(conduction.potential[conductor_nodes])):
        raise ValueError('Conductor potential must be finite.')
    air_cells = np.sort(
        np.r_[mesh.triangle_sets['outside'], mesh.triangle_sets['cavity']])
    system = assemble_p1(mesh, air_cells)
    matrix = system.matrix
    interface = system.dofs(np.unique(mesh.edges[mesh.edge_sets['conductor']]))
    far = system.dofs(np.unique(mesh.edges[mesh.edge_sets['far']]))
    fixed = np.union1d(interface, far)
    values = np.zeros(len(system.vertex_ids))
    values[interface] = conduction.potential[system.vertex_ids[interface]]
    unshifted = solve_dirichlet(matrix, system.source, values, fixed)
    values[interface] = 1.
    response = solve_dirichlet(matrix, system.source, values, fixed)
    flux = float((matrix@unshifted)[far].sum())
    response_flux = float((matrix@response)[far].sum())
    if not np.isfinite(response_flux) or response_flux >= 0:
        raise RuntimeError('Invalid constant-potential response.')
    offset = -flux/response_flux
    v = unshifted+offset*response
    potential = conduction.potential.copy()
    potential[conductor_nodes] += offset
    potential[system.vertex_ids] = v
    electric = conduction.electric_field.copy()
    electric[air_cells] = -system.gradient(v)
    residual = matrix@v
    total_charge = float(-permittivity*residual[far].sum())
    fields = FieldData(mesh, potential, electric,
                       conduction.current_density.copy(), permittivity, total_charge)
    return ExteriorResult(conduction, fields, float(offset))
