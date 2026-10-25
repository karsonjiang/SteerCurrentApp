"""Magnetic recovery diagnostics and integrated magnetic energy."""

import numpy as np


def magnetic_diagnostics(result, conduction):
    """Compare grad(Bz) with the prescribed current and integrate magnetic energy."""
    mesh, bz, gradient = result.mesh, result.bz, result.gradient_bz
    permeability = result.permeability
    cells = mesh.triangle_sets["conductor"]
    current = conduction.current_density[cells]
    target = permeability * np.column_stack((-current[:, 1], current[:, 0]))
    cavity_nodes = np.unique(mesh.triangles[mesh.triangle_sets["cavity"]])
    # Bz is constant in the cavity; read its value from the completed field.
    cavity_value = float(bz[cavity_nodes[0]])
    error = gradient[cells] - target
    error_l2 = float(
        np.sqrt(np.sum(np.sum(error**2, axis=1) * mesh.triangle_areas[cells]))
    )
    target_l2 = float(
        np.sqrt(np.sum(np.sum(target**2, axis=1) * mesh.triangle_areas[cells]))
    )
    # Exact integral of the square of a linear field on each triangle:
    # integral B^2 = A/12 * ((B0+B1+B2)^2 + B0^2+B1^2+B2^2).
    nodal = bz[mesh.triangles]
    energy = np.sum(
        mesh.triangle_areas * (nodal.sum(axis=1) ** 2 + (nodal**2).sum(axis=1))
    ) / (24 * permeability)
    diagnostics = {
        "cavity_bz": cavity_value,
        "ampere_gradient_relative_error": error_l2 / max(target_l2, 1e-30),
        "magnetic_energy_per_depth": float(energy),
    }
    return diagnostics
