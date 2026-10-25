"""Independent scikit-fem views and current sampling for physics tests only."""

from dataclasses import dataclass

import numpy as np
from skfem import MeshTri


@dataclass
class _FEMView:
    mesh: MeshTri
    vertex_ids: np.ndarray  # local vertex -> unified vertex
    triangle_ids: np.ndarray  # local triangle -> unified triangle


def solver_view(mesh, triangle_ids):
    """Build a library view for independent field checks and point sampling."""
    cells = np.asarray(triangle_ids, dtype=np.int64)
    vertices, inverse = np.unique(mesh.triangles[cells], return_inverse=True)
    local = MeshTri(mesh.vertices[vertices].T, inverse.reshape(-1, 3).T, sort_t=False)
    lookup = {
        tuple(edge): i
        for i, edge in enumerate(np.sort(vertices[local.facets.T], axis=1))
    }
    boundaries = {
        name: np.array(
            [
                lookup[tuple(mesh.edges[i])]
                for i in ids
                if tuple(mesh.edges[i]) in lookup
            ],
            dtype=np.int32,
        )
        for name, ids in mesh.edge_sets.items()
    }
    local = local.with_boundaries(
        {name: ids for name, ids in boundaries.items() if len(ids)}
    )
    local = local.with_subdomains(
        {
            name: np.flatnonzero(np.isin(cells, ids))
            for name, ids in mesh.triangle_sets.items()
        }
    )
    return _FEMView(local, vertices, cells)


def section_current(result, start, end, normal, *, samples=512):
    """Midpoint current integral for a valid cut with a perpendicular unit normal."""
    start, end, normal = (np.asarray(v, dtype=float) for v in (start, end, normal))
    view = solver_view(result.mesh, result.mesh.triangle_sets["conductor"])
    points = start[:, None] + (end - start)[:, None] * (
        (np.arange(samples) + 0.5) / samples
    )
    cells = view.triangle_ids[view.mesh.element_finder()(*points)]
    return float(
        np.mean(result.current_density[cells] @ normal) * np.linalg.norm(end - start)
    )
