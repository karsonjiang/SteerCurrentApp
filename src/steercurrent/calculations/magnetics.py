"""Calculate Bz field."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from steercurrent.calculations.conduction import ConductionResult
from steercurrent.calculations.fem import assemble_p1
from steercurrent.calculations.fem import solve_dirichlet
from steercurrent.meshing.model import MeshData
from steercurrent.meshing.validation import validate_combined_mesh


@dataclass
class MagneticResult:
    """Continuous Bz and constant triangle gradients.

    Uniform permeability; no applied external field.
    Positive Bz points out of the xy plane.
    """
    mesh: MeshData
    bz: np.ndarray                  # (nvertices,)
    gradient_bz: np.ndarray         # (ntriangles, 2)
    permeability: float

    @property
    def element_bz(self):
        """Triangle-average Bz, also its value at the triangle center."""
        return self.bz[self.mesh.triangles].mean(axis=1)

    @property
    def ampere_current(self):
        """Current reconstructed from curl(B)/mu, used by tests."""
        return np.column_stack((self.gradient_bz[:, 1], -self.gradient_bz[:, 0])) / self.permeability


def solve_magnetics(conduction: ConductionResult, *, permeability=1.):
    """Minimize integral_conductor |grad(Bz) - mu*(-Jy,Jx)|^2 dA.

    Exterior Bz=0 and cavity Bz=C are exact constraints. C is solved, together
    with vertex values inside the conductor. Shared boundary vertices enforce continuity.
    """
    if not np.isfinite(permeability) or permeability <= 0:
        raise ValueError('Permeability must be finite and positive.')
    validate_combined_mesh(conduction.circuit, check_coverage=False)
    mesh = conduction.mesh
    current = conduction.current_density
    if current.shape != (len(mesh.triangles), 2) or not np.all(np.isfinite(current)):
        raise ValueError('Current must be finite with shape (ntriangles, 2).')
    air = np.r_[mesh.triangle_sets['outside'], mesh.triangle_sets['cavity']]
    if np.any(current[air] != 0):
        raise ValueError('The magnetic model requires current-free air.')
    cells = mesh.triangle_sets['conductor']
    target = permeability * \
        np.column_stack((-current[cells, 1], current[cells, 0]))
    system = assemble_p1(mesh, cells, driving=target)
    outer = system.dofs(np.unique(mesh.edges[mesh.edge_sets['outer']]))
    cavity = system.dofs(np.unique(mesh.edges[mesh.edge_sets['cavity']]))
    if np.intersect1d(outer, cavity).size:
        raise ValueError('Outer and cavity surfaces must not share vertices.')

    # Tie every cavity-boundary value to one unknown. Other conductor vertices
    # keep individual unknowns. P expands reduced values back to vertex values.
    n = len(system.vertex_ids)
    groups = np.arange(n)
    groups[cavity] = n
    _, reduced_ids = np.unique(groups, return_inverse=True)
    size = int(reduced_ids.max()) + 1
    expansion = coo_matrix(
        (np.ones(n), (np.arange(n), reduced_ids)), shape=(n, size)).tocsr()
    matrix = (expansion.T @ system.matrix @ expansion).tocsr()
    source = np.asarray(expansion.T @ system.source)
    fixed = np.unique(reduced_ids[outer])
    reduced = solve_dirichlet(matrix, source, np.zeros(size), fixed)
    values = np.asarray(expansion @ reduced)
    cavity_value = float(reduced[reduced_ids[cavity[0]]])

    bz = np.zeros(len(mesh.vertices))
    bz[system.vertex_ids] = values
    cavity_nodes = np.unique(mesh.triangles[mesh.triangle_sets['cavity']])
    bz[cavity_nodes] = cavity_value
    gradient = np.zeros((len(mesh.triangles), 2))
    gradient[cells] = system.gradient(values)

    return MagneticResult(mesh, bz, gradient, float(permeability))
