"""Sparse Dirichlet solves for finite-element method."""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve


@dataclass
class P1System:
    """Temporary arrays for the selected triangles.

    vertex_ids maps local equation numbers to unified mesh vertices.
    triangles holds local equation numbers in the selected triangle order.
    gradients[t, i] is the constant 2D gradient of basis function i on triangle t.
    """
    vertex_ids: np.ndarray
    triangles: np.ndarray
    areas: np.ndarray
    gradients: np.ndarray
    matrix: csr_matrix
    source: np.ndarray

    def dofs(self, global_vertex_ids):
        """Translate global_vertex_ids to local equation numbers.
        global_vertex_ids should be a subset of vertex stored in self.vertex_ids.
        """
        ids = np.asarray(global_vertex_ids, dtype=np.int64)
        local = np.searchsorted(self.vertex_ids, ids)
        if np.any(local >= len(self.vertex_ids)) or not np.array_equal(self.vertex_ids[local], ids):
            raise ValueError(
                'Vertices of global_vertex_ids does not belong to self.vertex_ids.')
        return local

    def gradient(self, potential):
        """grad(V) = sum_i V_i grad(N_i), one vector per selected triangle."""
        vertex_values = potential[self.triangles]  # (ntriangles, 3)
        contributions = vertex_values[:, :, None] * self.gradients
        return contributions.sum(axis=1)  # Add the three vertex contributions.


def assemble_p1(mesh, triangle_ids, coefficient=1., driving=None):
    """Assemble integral c grad(V).grad(v) = integral c F.grad(v).

    c and F are constant per triangle. Each integral
    is its constant integrand times triangle area. Rows use compact vertex IDs
    so triangles outside the selected domain introduce no empty equations.
    """
    cells = np.asarray(triangle_ids, dtype=np.int64)
    if cells.ndim != 1 or not len(cells) or np.any(cells < 0) or np.any(cells >= len(mesh.triangles)) or len(np.unique(cells)) != len(cells):
        raise ValueError(
            'Select a nonempty set of distinct valid triangle IDs.')
    # inverse stores the local vertex number for each entry in the selected triangles.
    vertices, inverse = np.unique(mesh.triangles[cells], return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    points = mesh.vertices[mesh.triangles[cells]]
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    if not np.all(np.isfinite(determinant)) or np.any(determinant == 0):
        raise ValueError('Triangles must have finite, nonzero area.')
    areas = np.abs(determinant) / 2
    # Signed determinant makes the gradients correct for either vertex winding.
    gradients = np.empty((len(cells), 3, 2))
    gradients[:, 1] = np.column_stack(
        (b[:, 1], -b[:, 0])) / determinant[:, None]
    gradients[:, 2] = np.column_stack(
        (-a[:, 1], a[:, 0])) / determinant[:, None]
    gradients[:, 0] = -gradients[:, 1] - gradients[:, 2]

    coefficient = np.broadcast_to(np.asarray(
        coefficient, dtype=float), (len(cells),))
    driving = np.zeros((len(cells), 2)) if driving is None else np.asarray(
        driving, dtype=float)
    if not np.all(np.isfinite(coefficient)) or np.any(coefficient <= 0):
        raise ValueError('Coefficient must be finite and positive.')
    if driving.shape != (len(cells), 2) or not np.all(np.isfinite(driving)):
        raise ValueError(
            'Driving field must be finite with shape (nselected_triangles, 2).')

    # Each triangle contributes a 3x3 matrix and a three-entry source vector.
    weight = coefficient * areas
    # Per triangle: (3, 2) @ (2, 3) gives all nine gradient dot products.
    gradient_products = gradients @ gradients.transpose(0, 2, 1)
    element_matrix = weight[:, None, None] * gradient_products
    # Dot the driving field with each of the triangle's three basis gradients.
    driving_products = np.sum(gradients * driving[:, None, :], axis=2)
    element_source = weight[:, None] * driving_products
    # turn from triangle indexed quantities to vertices indexed
    # sum each triangle’s contributions into a system indexed by vertices
    rows = np.broadcast_to(triangles[:, :, None], element_matrix.shape)
    columns = np.broadcast_to(triangles[:, None, :], element_matrix.shape)
    n = len(vertices)
    # COO -> CSR adds contributions at shared vertices automatically.
    matrix = coo_matrix(
        (element_matrix.ravel(), (rows.ravel(), columns.ravel())), shape=(n, n)).tocsr()
    source = np.bincount(
        triangles.ravel(), weights=element_source.ravel(), minlength=n)
    return P1System(vertices, triangles, areas, gradients, matrix, source)


def solve_dirichlet(matrix, source, values, fixed):
    """Move prescribed values to the RHS and then solve: K_ff V_f = b_f - K_fd V_d.

    values and fixed use this linear system's local equation numbers.
    """
    n = matrix.shape[0]
    fixed = np.unique(np.asarray(fixed, dtype=np.int64))
    values, source = np.asarray(
        values, dtype=float), np.asarray(source, dtype=float)
    if matrix.shape != (n, n) or values.shape != (n,) or source.shape != (n,):
        raise ValueError(
            'Matrix, source, and prescribed values have incompatible shapes.')
    if not len(fixed) or np.any(fixed < 0) or np.any(fixed >= n):
        raise ValueError(
            'At least one valid prescribed potential is required.')
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(values[fixed])):
        raise ValueError('Source and prescribed potentials must be finite.')
    count, components = connected_components(matrix != 0, directed=False)
    if len(np.unique(components[fixed])) != count:
        raise ValueError(
            'Each disconnected component needs a prescribed potential.')
    free = np.setdiff1d(np.arange(n), fixed)
    potential = np.zeros(n)
    potential[fixed] = values[fixed]
    if len(free):
        rhs = source[free] - matrix[free][:, fixed] @ potential[fixed]
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            try:
                potential[free] = spsolve(matrix[free][:, free], rhs)
            except MatrixRankWarning as error:
                raise RuntimeError(
                    'The reduced potential system is singular.') from error
    if not np.all(np.isfinite(potential)):
        raise RuntimeError('Potential solve produced nonfinite values.')
    return potential
