"""Exterior field diagnostics."""

import numpy as np


def exterior_diagnostics(result):
    """Measure boundary-value errors and integrate electric energy in air."""
    conduction, fields = result.conduction, result.fields
    offset = result.potential_offset
    mesh, electric = fields.mesh, fields.electric_field
    permittivity = fields.permittivity
    air_cells = np.sort(
        np.r_[mesh.triangle_sets["outside"], mesh.triangle_sets["cavity"]]
    )
    edges = mesh.edge_sets["far"]
    interface = np.unique(mesh.edges[mesh.edge_sets["conductor"]])
    far = np.unique(mesh.edges[edges])
    v = fields.potential
    return dict(
        area=float(mesh.triangle_areas[air_cells].sum()),
        region_areas={
            name: float(mesh.triangle_areas[mesh.triangle_sets[name]].sum())
            for name in ("outside", "cavity")
        },
        boundary_potential_error=float(
            np.max(np.abs(v[interface] - (conduction.potential[interface] + offset)))
        ),
        far_potential_error=float(np.max(np.abs(v[far]))),
        exterior_field_energy=float(
            0.5
            * permittivity
            * np.sum(
                np.sum(electric[air_cells] ** 2, axis=1)
                * mesh.triangle_areas[air_cells]
            )
        ),
    )
