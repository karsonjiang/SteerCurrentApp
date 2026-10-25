"""Local, regional, and global energy-flow diagnostics."""

import numpy as np


def poynting_diagnostics(result):
    """Compare edge energy flow with volume absorption, locally and by material."""
    mesh, electric = result.mesh, result.exterior.electric_field
    flux, divergence = result.energy_flux, result.divergence
    electrical = result.electrical_power_density
    joule, source, residual = (
        result.joule_density,
        result.source_power_density,
        result.balance_residual,
    )
    areas = mesh.triangle_areas
    traces = result.edge_fluxes()
    first, second = mesh.edge_triangles.T
    shared = second >= 0
    cell_flux = np.zeros(len(mesh.triangles))
    # Both traces use first-to-second normals; reverse the second for outward flux.
    np.add.at(cell_flux, first, traces[:, 0])
    np.add.at(cell_flux, second[shared], -traces[shared, 1])
    regional = {}
    for key, name in mesh.region_names.items():
        selected = mesh.region_ids == key
        side1 = selected[first]
        side2 = np.zeros(len(first), dtype=bool)
        side2[shared] = selected[second[shared]]
        # Include only edges on this material region's boundary.
        outward = float(
            traces[side1 & ~side2, 0].sum() - traces[side2 & ~side1, 1].sum()
        )
        absorbed = float(np.sum(electrical[selected] * areas[selected]))
        regional[name] = {
            "outward_flux_per_depth": outward,
            "electrical_absorption_per_depth": absorbed,
            "joule_power_per_depth": float(np.sum(joule[selected] * areas[selected])),
            "source_power_per_depth": float(np.sum(source[selected] * areas[selected])),
        }

    def norm(values):
        return float(np.sqrt(np.sum(values**2 * areas)))

    orthogonal = np.sum(flux * electric, axis=1)
    orthogonal_scale = np.linalg.norm(flux, axis=1) * np.linalg.norm(electric, axis=1)
    joule_total, source_total = float(joule @ areas), float(source @ areas)
    return {
        "regions": regional,
        "joule_power_per_depth": joule_total,
        "source_power_per_depth": source_total,
        "global_power_balance_relative_error": abs(joule_total - source_total)
        / max(abs(joule_total), abs(source_total), 1e-30),
        "local_balance_relative_l2": norm(residual) / max(norm(electrical), 1e-30),
        "ohm_power_identity_error_l2": norm(electrical - (joule - source)),
        "orthogonality_relative_max": float(
            np.max(np.abs(orthogonal) / np.maximum(orthogonal_scale, 1e-30))
        ),
        "edge_divergence_consistency_l2": norm(cell_flux / areas - divergence),
        "normal_flux_jump_l2": float(
            np.sqrt(
                np.sum(
                    (traces[shared, 0] - traces[shared, 1]) ** 2
                    / mesh.edge_lengths[shared]
                )
            )
        ),
        "far_outward_power_per_depth": float(traces[mesh.edge_sets["far"], 0].sum()),
    }
