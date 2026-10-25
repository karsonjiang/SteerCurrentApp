"""Circuit validation on the final, unified MeshData arrays."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _boundary_signed_area(mesh, edges):
    """Calculate the enclosed area: positive for outer/far, negative for cavity.
    Sum midpoint dot outward normal times edge length, then divide by two.

    "outer" → positive area enclosed by the conductor’s outer boundary, including both conductor and cavity.
    "cavity" → negative area of the cavity.
    "far" → positive area of the entire simulation domain.
    """
    # Normals point from conductor to air, or outward at the far boundary.
    midpoints = mesh.edge_midpoints[edges] - mesh.vertices[0]
    return float(.5 * np.sum(np.sum(midpoints * mesh.edge_normals[edges], axis=1)
                             * mesh.edge_lengths[edges]))


def _validate_triangle_coverage(mesh):
    """Reject overlapping triangles and edges that meet only partway along an edge."""
    import shapely

    # Each polygon is the three (x, y) coordinates of one triangle.
    polygons = shapely.polygons(mesh.vertices[mesh.triangles])
    # Shared edges must match exactly. Gaps are checked by the other mesh checks.
    if not shapely.coverage_is_valid(polygons):
        raise ValueError('Triangles overlap or their shared edges do not match.')


def _validate_materials(circuit):
    """Check material properties and battery data against the mesh's region IDs."""
    m = circuit.mesh
    labels = m.region_ids[m.triangle_sets['conductor']]
    if set(labels) != set(circuit.regions):
        raise ValueError('Material metadata must match conductor regions.')
    if any(not np.isfinite(r.resistivity) or r.resistivity <= 0 for r in circuit.regions.values()):
        raise ValueError('Resistivity must be finite and positive.')
    if any(r.material_kind not in {'wire', 'resistor', 'battery'} for r in circuit.regions.values()):
        raise ValueError('Unknown conductor material kind.')
    if any(m.region_names[key] != r.name for key, r in circuit.regions.items()):
        raise ValueError('Material names must match mesh region names.')
    b = circuit.battery
    if ([key for key, r in circuit.regions.items() if r.material_kind == 'battery'] != [b.region_id] or
            not np.isfinite(b.emf) or not np.isfinite(b.length) or b.length <= 0 or
            np.asarray(b.direction).shape != (2,) or not np.isclose(np.linalg.norm(b.direction), 1)):
        raise ValueError(
            'Exactly one battery with finite EMF, positive length, and unit direction is required.')


def _validate_interfaces(circuit):
    """Check which triangles meet at each conductor surface and material interface."""
    m = circuit.mesh
    n = len(m.triangles)
    groups = [m.triangle_sets[name] for name in ('conductor', 'outside', 'cavity')]
    conductor = np.zeros(n, dtype=bool)
    conductor[groups[0]] = True
    for name in ('outer', 'cavity'):
        pair = m.edge_triangles[m.edge_sets[name]]
        if np.any(pair < 0) or not np.all(conductor[pair[:, 0]]) or np.any(conductor[pair[:, 1]]):
            raise ValueError('Surface edges must point conductor-to-air.')
        expected = groups[1] if name == 'outer' else groups[2]
        if not np.all(np.isin(pair[:, 1], expected)):
            raise ValueError('Surface edge faces the wrong air region.')
    if not np.array_equal(np.sort(np.r_[m.edge_sets['outer'], m.edge_sets['cavity']]), m.edge_sets['conductor']):
        raise ValueError(
            'Conductor boundary must be the disjoint outer/cavity union.')
    adjacent = m.edge_triangles[m.edge_triangles[:, 1] >= 0]
    shared = np.flatnonzero(m.edge_triangles[:, 1] >= 0)
    surface = shared[conductor[adjacent[:, 0]] != conductor[adjacent[:, 1]]]
    if not np.array_equal(surface, m.edge_sets['conductor']):
        raise ValueError(
            'Conductor surface tags must cover every conductor/air edge.')
    material = shared[conductor[adjacent[:, 0]] & conductor[adjacent[:, 1]] &
                      (m.region_ids[adjacent[:, 0]] != m.region_ids[adjacent[:, 1]])]
    tagged = np.concatenate([m.edge_sets[name] for name in m.interface_regions]
                            ) if m.interface_regions else np.array([], dtype=int)
    if not np.array_equal(material, np.sort(tagged)):
        raise ValueError(
            'Material interfaces must cover each material transition once.')
    for name, regions in m.interface_regions.items():
        if len(regions) != 2 or regions[0] >= regions[1] or any(key not in circuit.regions for key in regions):
            raise ValueError(
                'Interface needs two ordered conductor region IDs.')
        pair = m.edge_triangles[m.edge_sets[name]]
        if np.any(pair < 0) or not np.all(m.region_ids[pair] == regions):
            raise ValueError(
                'Interface edges do not separate the declared regions.')


def _validate_connectivity(m):
    """Check that the three domains are connected and have the expected holes."""
    n = len(m.triangles)
    groups = [m.triangle_sets[name] for name in ('conductor', 'outside', 'cavity')]
    adjacent = m.edge_triangles[m.edge_triangles[:, 1] >= 0]
    # Air has two separate domains; all their contact must be through conductor.
    domains = np.empty(n, dtype=int)
    for index, ids in enumerate(groups):
        domains[ids] = index
    if np.any(np.all(domains[adjacent] > 0, axis=1) & (domains[adjacent[:, 0]] != domains[adjacent[:, 1]])):
        raise ValueError('Outside air and cavity must not share an edge.')
    if not np.all(domains[m.edge_triangles[m.edge_sets['far'], 0]] == 1):
        raise ValueError('Far boundary must belong to outside air.')

    for index, ids in enumerate(groups):
        pairs = adjacent[np.all(np.isin(adjacent, ids), axis=1)]
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(
            n, n)).tocsr()[ids][:, ids]
        if connected_components(graph, directed=False, return_labels=False) != 1:
            raise ValueError('Each domain must be connected.')
        # V - E + T = 1 - number of holes for a connected planar domain.
        touches = np.isin(m.edge_triangles, ids).any(axis=1)
        euler = len(np.unique(m.triangles[ids])) - \
            np.count_nonzero(touches) + len(ids)
        if euler != (1 if index == 2 else 0):
            raise ValueError(
                'Conductor and outside must each have one hole; cavity must have none.')


def _validate_boundaries(m):
    """Check closed boundary loops, their orientation, and the areas they enclose."""
    groups = [m.triangle_sets[name] for name in ('conductor', 'outside', 'cavity')]
    areas = m.triangle_areas
    boundary_areas = {}
    for name in ('outer', 'cavity', 'far'):
        edges = m.edge_sets[name]
        if not len(edges):
            raise ValueError('Each boundary component must be nonempty.')
        nodes, inverse, degrees = np.unique(
            m.edges[edges], return_inverse=True, return_counts=True)
        endpoints = inverse.reshape(-1, 2)
        graph = coo_matrix((np.ones(len(edges)), endpoints.T),
                           shape=(len(nodes), len(nodes)))
        if np.any(degrees != 2) or connected_components(graph, directed=False, return_labels=False) != 1:
            raise ValueError(
                'Each boundary label must describe one closed boundary component.')
        boundary_areas[name] = _boundary_signed_area(m, edges)
    if boundary_areas['outer'] <= 0 or boundary_areas['cavity'] >= 0 or boundary_areas['far'] <= 0:
        raise ValueError(
            'Outer, cavity, or far boundary orientation is incorrect.')

    # Enclosed areas must match the triangles assigned to each domain.
    expected_areas = (boundary_areas['outer'] + boundary_areas['cavity'],
                      boundary_areas['far'] - boundary_areas['outer'],
                      -boundary_areas['cavity'])
    if any(not np.isclose(areas[ids].sum(), expected, rtol=1e-9, atol=0)
           for ids, expected in zip(groups, expected_areas)):
        raise ValueError(
            'Domain triangles do not fill their enclosing boundaries.')
    return boundary_areas


def validate_combined_mesh(circuit, *, check_coverage=True):
    """Validate circuit topology and geometry using only the final MeshData.

    Basic array/index invariants are enforced when MeshData is constructed.
    Solvers may skip the triangle coverage check after mesh generation.
    """
    m = circuit.mesh
    n = len(m.triangles)
    areas = m.triangle_areas
    if not np.all(np.isfinite(m.vertices)) or not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError(
            'Mesh has nonfinite coordinates or degenerate triangles.')
    if not {'conductor', 'outside', 'cavity'} <= m.triangle_sets.keys():
        raise ValueError(
            'Mesh needs conductor, outside, and cavity triangle sets.')
    if not {'conductor', 'outer', 'cavity', 'far'} <= m.edge_sets.keys():
        raise ValueError(
            'Mesh needs conductor, outer, cavity, and far edge sets.')
    groups = [m.triangle_sets[name]
              for name in ('conductor', 'outside', 'cavity')]
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n)):
        raise ValueError(
            'Conductor, outside, and cavity must partition all triangles.')
    if any(not len(ids) for ids in groups):
        raise ValueError('Each domain needs triangles.')
    if not np.array_equal(np.flatnonzero(m.edge_triangles[:, 1] < 0), m.edge_sets['far']):
        raise ValueError(
            'Only far-boundary edges may have a missing neighbor.')
    _validate_materials(circuit)
    _validate_interfaces(circuit)
    _validate_connectivity(m)
    boundary_areas = _validate_boundaries(m)

    if check_coverage:
        _validate_triangle_coverage(m)
    return {'vertices': len(m.vertices), 'triangles': n, 'area': float(areas.sum()),
            'conductor_triangles': len(groups[0]), 'air_triangles': len(groups[1])+len(groups[2]),
            'region_areas': {name: float(areas[ids].sum()) for name, ids in m.triangle_sets.items()},
            'boundary_areas': boundary_areas,
            'interface_facets': sum(len(m.edge_sets[name]) for name in m.interface_regions)}
