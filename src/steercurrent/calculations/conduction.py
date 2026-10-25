"""Conduction model in an insulated single-loop conductor."""
from dataclasses import dataclass
import numpy as np
from steercurrent.calculations.fem import assemble_p1
from steercurrent.calculations.fem import solve_dirichlet
from steercurrent.meshing.model import CircuitMesh
from steercurrent.meshing.validation import validate_combined_mesh


@dataclass
class ConductionResult:
    """Fields indexed by the ONE unified mesh; air V/E/rho are unsolved (NaN).

    Current and driving field are zero in air.
    """
    circuit: CircuitMesh
    potential: np.ndarray           # (nv,)
    electric_field: np.ndarray      # (nt,2)
    current_density: np.ndarray     # (nt,2)
    driving_field: np.ndarray       # (nt,2)
    resistivity: np.ndarray         # (nt,)

    @property
    def mesh(self):
        return self.circuit.mesh


def solve_conduction(circuit, *, gauge_dof=None):
    """Query conductor triangles from the unified mesh and solve J=gamma(E+F).

    gauge_dof, if given, is a unified vertex ID.
    """
    validate_combined_mesh(circuit, check_coverage=False)
    mesh = circuit.mesh
    cells = mesh.triangle_sets['conductor']
    vertices = np.unique(mesh.triangles[cells])
    if gauge_dof is None:
        pin = 0
    else:
        if not isinstance(gauge_dof, (int, np.integer)) or gauge_dof not in vertices:
            raise ValueError('gauge_dof must be a conductor vertex ID.')
        pin = int(np.flatnonzero(vertices == gauge_dof)[0])
    labels = mesh.region_ids[cells]
    rho = np.array([circuit.regions[int(r)].resistivity for r in labels])
    b = circuit.battery
    driving = np.zeros((len(cells), 2))
    driving[labels == b.region_id] = b.emf/b.length*np.asarray(b.direction)
    system = assemble_p1(mesh, cells, coefficient=1/rho, driving=driving)
    matrix, source = system.matrix, system.source
    v = solve_dirichlet(matrix, source, np.zeros(len(vertices)), [pin])
    # The mean of a linear function on a triangle is its three nodal values averaged.
    v -= np.sum(v[system.triangles].mean(axis=1)
                * system.areas)/system.areas.sum()
    electric = -system.gradient(v)
    current = (electric+driving)/rho[:, None]
    potential = np.full(len(mesh.vertices), np.nan)
    field = np.full((len(mesh.triangles), 2), np.nan)
    j, f = np.zeros_like(field), np.zeros_like(field)
    resistivity = np.full(len(mesh.triangles), np.nan)
    potential[vertices], field[cells], j[cells], f[cells], resistivity[cells] = v, electric, current, driving, rho
    return ConductionResult(circuit, potential, field, j, f, resistivity)
