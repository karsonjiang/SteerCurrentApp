"""Current continuity and power conservation checks for conductor solutions."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _cross_sections(mesh, edges):
    """Split an interface into disconnected cuts, such as the two ends of a resistor."""
    nodes, inverse = np.unique(mesh.edges[edges], return_inverse=True)
    endpoints = inverse.reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(len(edges)), (endpoints[:, 0], endpoints[:, 1])),
        shape=(len(nodes), len(nodes)),
    )
    count, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels[endpoints[:, 0]] == i) for i in range(count)]


def conduction_diagnostics(result):
    """Integrate power and compare current through material interfaces."""
    m = result.mesh
    cells = m.triangle_sets["conductor"]
    j = result.current_density
    areas = m.triangle_areas[cells]
    joule = float(
        np.sum(result.resistivity[cells] * np.sum(j[cells] ** 2, axis=1) * areas)
    )
    emf = float(np.sum(np.sum(j[cells] * result.driving_field[cells], axis=1) * areas))
    edges = m.edge_sets["conductor"]
    first = m.edge_triangles[edges, 0]
    normals, lengths = m.edge_normals, m.edge_lengths
    jn = np.sum(j[first] * normals[edges], axis=1)
    flux_l2 = float(np.sqrt(np.sum(jn**2 * lengths[edges])))
    j_l2 = float(np.sqrt(np.sum(np.sum(j[first] ** 2, axis=1) * lengths[edges])))
    sections = []
    for name, pair in m.interface_regions.items():
        edges = m.edge_sets[name]
        for indices in _cross_sections(m, edges):
            selected = edges[indices]
            a, b = m.edge_triangles[selected].T
            values = [
                float(
                    np.sum(
                        np.sum(j[side] * normals[selected], axis=1) * lengths[selected]
                    )
                )
                for side in (a, b)
            ]
            sections.append(
                dict(
                    interface=name,
                    regions=list(pair),
                    current_from_first=values[0],
                    current_from_second=values[1],
                )
            )
    battery = result.circuit.battery
    reference = abs(emf / battery.emf) if battery.emf else 0.0
    magnitudes = [
        abs(s[key])
        for s in sections
        for key in ("current_from_first", "current_from_second")
    ]
    return dict(
        joule_power=joule,
        emf_power=emf,
        power_balance_relative_error=abs(joule - emf)
        / max(abs(joule), abs(emf), 1e-30),
        source_average_current=emf / battery.emf if battery.emf else 0.0,
        boundary_normal_current_relative=flux_l2 / max(j_l2, 1e-30),
        boundary_net_current=float(np.sum(jn * lengths[m.edge_sets["conductor"]])),
        interface_cross_sections=sections,
        cross_section_max_relative_deviation=max(
            (abs(i - reference) / max(reference, 1e-30) for i in magnitudes),
            default=0.0,
        ),
    )
