"""A deliberately incomplete field model for the battery-only demo."""
from dataclasses import dataclass

import numpy as np

from steercurrent.meshing.model import CircuitMesh, MeshData


@dataclass(frozen=True)
class BatteryOnlyResult:
    """Equal/opposite terminal charges and their field at triangle centers."""

    mesh: MeshData
    electric_field: np.ndarray  # (ntriangles, 2)
    charge_edges: np.ndarray    # mesh edge IDs at the two battery terminals
    charge_density: np.ndarray  # (ncharge_edges,), charge per unit edge length/depth


def _field_from_edges(mesh, points, edges, density, permittivity):
    """Sum the 2D Coulomb field of uniformly charged finite mesh edges."""
    field = np.zeros((len(points), 2))
    nodes, weights = np.polynomial.legendre.leggauss(12)
    for edge, charge in zip(edges, density):
        start, end = mesh.vertices[mesh.edges[edge]]
        samples = (start+end)/2 + nodes[:, None]*(end-start)/2
        difference = points[:, None, :] - samples[None, :, :]
        radius2 = np.sum(difference**2, axis=2)
        if np.any(radius2 == 0):
            raise ValueError('Evaluate the toy field away from the charge sheets.')
        field += (charge*mesh.edge_lengths[edge]/(4*np.pi*permittivity)
                  * np.sum(weights[None, :, None]*difference/radius2[:, :, None], axis=1))
    return field


def battery_only_field(circuit: CircuitMesh, *, permittivity=1.) -> BatteryOnlyResult:
    """Treat only the battery terminals as charged capacitor plates.

    The sheet strength is calibrated so the area-averaged battery field along
    its axis equals -EMF / battery length, the zero-internal-resistance limit.
    No charge is placed on wire surfaces or resistor interfaces. Consequently
    this is a comparison model, not a solution to the conducting circuit.
    """
    if not np.isfinite(permittivity) or permittivity <= 0:
        raise ValueError('Permittivity must be finite and positive.')
    mesh, battery = circuit.mesh, circuit.battery
    terminal_groups = [
        mesh.edge_sets[name] for name, regions in mesh.interface_regions.items()
        if battery.region_id in regions
    ]
    if not terminal_groups:
        raise ValueError('Battery terminal interfaces are missing from the mesh.')
    terminal_edges = np.concatenate(terminal_groups)

    direction = np.asarray(battery.direction)
    projection = mesh.edge_midpoints[terminal_edges] @ direction
    middle = (projection.min()+projection.max())/2
    low_terminal = projection < middle
    if not np.any(low_terminal) or np.all(low_terminal):
        raise ValueError('The battery must have two distinct terminal faces.')
    # Trial signs set the response; the calibration below gives the physical signs.
    unit_density = np.where(low_terminal, 1., -1.)
    field = _field_from_edges(mesh, mesh.triangle_centers, terminal_edges,
                              unit_density, permittivity)
    battery_cells = mesh.triangle_sets['conductor']
    battery_cells = battery_cells[mesh.region_ids[battery_cells] == battery.region_id]
    area = mesh.triangle_areas[battery_cells]
    response = np.average(field[battery_cells] @ direction, weights=area)
    if not np.isfinite(response) or response == 0:
        raise ValueError('Could not calibrate the battery terminal charges.')
    strength = (-battery.emf/battery.length)/response
    return BatteryOnlyResult(mesh, strength*field, terminal_edges,
                             strength*unit_density)
