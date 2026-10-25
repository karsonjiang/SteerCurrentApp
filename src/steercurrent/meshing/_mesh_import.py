"""Join conductor and air meshes and build the arrays used by calculations."""
import numpy as np
from scipy.spatial import cKDTree
from steercurrent.meshing.model import MeshData


def _make_mesh_data(vertices, triangles, region_ids, region_names, *, edge_vertices=None,
                    triangle_sets=None, interface_regions=None):
    """Build MeshData from vertices, triangles, and region IDs.

    Find each edge and its neighboring triangles. Convert named edge sets
    from pairs of vertex IDs to edge IDs, and reject invalid input.
    """
    vertices = np.array(vertices, dtype=float, copy=True)
    triangles = np.array(triangles, dtype=np.int64, copy=True)
    labels = np.array(region_ids, dtype=np.int64, copy=True)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or not np.all(np.isfinite(vertices)) or
            triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles) or
            labels.shape != (len(triangles),) or np.any(triangles < 0) or
            np.any(triangles >= len(vertices))):
        raise ValueError(
            'Invalid mesh coordinates, triangle indices, or region labels.')
    if not set(labels) <= set(region_names):
        raise ValueError('Every triangle region needs a name.')
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError('Duplicate triangles.')
    pairs = np.sort(
        triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold edge: more than two adjacent triangles.')
    cells = np.repeat(np.arange(len(triangles)), 3)
    order = np.lexsort((cells, labels[cells], inverse))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    adjacent = np.full((len(edges), 2), -1, dtype=np.int64)
    adjacent[:, 0] = cells[order[starts]]
    shared = counts == 2
    adjacent[shared, 1] = cells[order[starts[shared]+1]]
    lookup = {tuple(edge): i for i, edge in enumerate(edges)}
    try:
        sets = {name: np.unique([lookup[tuple(sorted(pair))] for pair in pairs]).astype(np.int64)
                for name, pairs in (edge_vertices or {}).items()}
    except KeyError as error:
        raise ValueError('Named edge does not exist in the mesh.') from error
    cell_sets = {name: np.unique(ids).astype(np.int64)
                 for name, ids in (triangle_sets or {}).items()}
    if any(np.any(ids < 0) or np.any(ids >= len(triangles)) for ids in cell_sets.values()):
        raise ValueError('Triangle set contains invalid IDs.')
    interfaces = dict(interface_regions or {})
    result = MeshData(vertices, triangles, labels, edges, adjacent, dict(
        region_names), sets, cell_sets, interfaces)
    if np.any(result.triangle_areas <= 0):
        raise ValueError('Degenerate triangle.')
    # Two adjacent triangles must lie on opposite sides of their common edge.
    displacement = result.triangle_centers[adjacent[shared, 1]
                                           ]-result.edge_midpoints[shared]
    if np.any(np.sum(result.edge_normals[shared]*displacement, axis=1) <= 0):
        raise ValueError('Adjacent triangles overlap across an edge.')
    if any(name not in sets for name in interfaces):
        raise ValueError('Interface needs a named edge set.')
    # Prevent array edits that could leave coordinates and connectivity inconsistent.
    for array in (vertices, triangles, labels, edges, adjacent, *sets.values(), *cell_sets.values()):
        array.flags.writeable = False
    return result


def _map_conductor_boundary(mesh, conductor):
    """Match air boundary vertices to conductor vertices so the meshes can be joined.

    The returned array has one entry per air-mesh vertex:
    mapping[i] >= 0: the matching conductor vertex ID.
    mapping[i] == -1: this air vertex isn’t on the conductor boundary.
    """
    source = conductor.mesh
    if not mesh.edge_vertices or not {'conductor', 'conductor_outer', 'conductor_cavity'} <= mesh.edge_vertices.keys():
        raise ValueError('Exterior mesh needs conductor boundary labels.')
    # Collects boundary vertices from both meshes
    source_nodes = np.unique(source.edge_vertices['conductor'])
    target_nodes = np.unique(mesh.edge_vertices['conductor'])
    if not len(source_nodes) or not len(target_nodes):
        raise ValueError('Conductor boundary must be nonempty in both meshes.')
    # Matches them by coordinates using nearest-neighbor search
    distances, indices = cKDTree(source.vertices[source_nodes]).query(
        mesh.vertices[target_nodes])
    tolerance = np.ptp(source.vertices, axis=0).max() * 1e-10
    if (len(target_nodes) != len(source_nodes) or len(np.unique(indices)) != len(source_nodes) or
            np.max(distances) > tolerance):
        raise ValueError(
            "Exterior mesh must preserve all conductor boundary vertices exactly.")
    mapping = np.full(len(mesh.vertices), -1, dtype=np.int64)
    mapping[target_nodes] = source_nodes[indices]
    # Matching points must also form the same edges on each boundary.
    for target_name, source_name in [('conductor_outer', 'outer'), ('conductor_cavity', 'cavity')]:
        a = np.sort(
            mapping[mesh.edge_vertices[target_name]], axis=1)
        b = np.sort(source.edge_vertices[source_name], axis=1)
        if set(map(tuple, a)) != set(map(tuple, b)):
            raise ValueError(
                "Exterior mesh must preserve the conductor boundary edges and labels.")
    return mapping


def combine_meshes(conductor, domain):
    source, air = conductor.mesh, domain.mesh
    # Ensure every air triangle receives exactly one label before allocating it.
    if not air.triangle_sets or not {'outside', 'cavity'} <= air.triangle_sets.keys():
        raise ValueError('Exterior mesh needs outside and cavity triangle sets.')
    partition = np.concatenate([air.triangle_sets[name]
                               for name in ('outside', 'cavity')])
    if not np.array_equal(np.sort(partition), np.arange(len(air.triangles))):
        raise ValueError('Outside and cavity triangle sets must partition the air mesh.')
    if not air.edge_vertices or 'far' not in air.edge_vertices:
        raise ValueError('Exterior mesh needs a far boundary label.')
    # This check requires both original meshes and cannot be deferred until after merging.
    mapping = _map_conductor_boundary(air, conductor)
    # Only air-only vertices are appended; shared boundary vertices keep one ID.
    new = mapping < 0
    mapping[new] = len(source.vertices) + np.arange(np.count_nonzero(new))
    vertices = np.vstack((source.vertices, air.vertices[new]))
    triangles = np.vstack((source.triangles, mapping[air.triangles]))
    region_names = {key: r.name for key, r in conductor.regions.items()}
    air_labels = np.empty(len(air.triangles), dtype=int)
    cell_sets = {'conductor': np.arange(len(source.triangles))}
    for i, name in enumerate(('outside', 'cavity'), start=max(region_names)+1):
        region_names[i] = name
        air_labels[air.triangle_sets[name]] = i
        cell_sets[name] = len(source.triangles) + air.triangle_sets[name]
    edge_vertices = dict(source.edge_vertices)
    edge_vertices['far'] = mapping[air.edge_vertices['far']]
    return _make_mesh_data(vertices, triangles, np.r_[conductor.region_ids, air_labels], region_names,
                           edge_vertices=edge_vertices, triangle_sets=cell_sets,
                           interface_regions=conductor.interface_regions)
