"""Translate browser requests into core calls and return only display data."""
import time

import numpy as np

from steercurrent.geometry import CircuitGeometry


def validate_request(data):
    """Validate geometry and bound the work accepted by this interactive prototype."""
    if not isinstance(data, dict):
        raise ValueError('Send a circuit configuration object.')
    raw = data['geometry']
    # Bound list sizes before the core's geometric intersection checks run.
    if len(raw['shape']['points']) > 32 or len(raw['config'].get('resistors', [])) > 6:
        raise ValueError('Use at most 32 turning points and 6 resistors.')
    geometry = CircuitGeometry.from_dict(raw)
    config = geometry.config
    if not 24 <= config.length <= 80 or not .5 <= config.width <= 2:
        raise ValueError(
            'This prototype supports loop lengths 24–80 and widths 0.5–2.')
    if np.max(np.abs(geometry.shape.points)) > 100:
        raise ValueError('Keep the circuit coordinates within ±100.')
    if not 0 <= config.battery.emf <= 10:
        raise ValueError('Battery strength must be between 0 and 10.')
    materials = [config.wire_resistivity, config.battery.resistivity,
                 *(r.resistivity for r in config.resistors)]
    if any(not .001 <= value <= 1000 for value in materials):
        raise ValueError('Resistivity must be between 0.001 and 1000.')
    if data.get('resolution', 'quick') not in ('quick', 'fine'):
        raise ValueError('Choose quick or fine resolution.')
    return geometry


def simulate(data):
    """Run in a worker process, where Gmsh owns the main thread and its own state."""
    from steercurrent.meshing import generate_mesh, MeshOptions, ExteriorOptions
    from steercurrent.calculations import (
        solve_conduction, solve_exterior, calculate_charges,
        solve_magnetics, calculate_poynting, battery_only_field,
    )

    started = time.monotonic()
    geometry = validate_request(data)
    size = geometry.config.width * \
        (.55 if data.get('resolution', 'quick') == 'quick' else .3)
    circuit = generate_mesh(geometry, MeshOptions(size=size, local_size=size*.7),
                            exterior_options=ExteriorOptions(outer_scale=3, near_size=size,
                                                             far_size=geometry.config.length/12))
    conduction = solve_conduction(circuit)
    exterior = solve_exterior(conduction)
    charges = calculate_charges(exterior.fields)
    toy = battery_only_field(circuit, permittivity=exterior.permittivity)
    magnetic = solve_magnetics(conduction)
    energy = calculate_poynting(exterior, magnetic)
    mesh = circuit.mesh

    def array(values):
        # JSON carries finite arrays only; air NaNs are never exported.
        return np.asarray(values).tolist()

    traces = {}
    for name, trace in {**charges.surfaces, **charges.interfaces}.items():
        traces[name] = {'edges': array(
            mesh.edges[trace.edge_ids]), 'density': array(trace.density)}
    cells = mesh.triangle_sets['conductor']
    source_power = float(
        np.sum(energy.source_power_density[cells]*mesh.triangle_areas[cells]))
    resistor_cells = cells[np.isin(mesh.region_ids[cells], [
        region_id for region_id, region in circuit.regions.items()
        if region.material_kind == 'resistor'
    ])]
    resistor_power = float(np.sum(energy.joule_density[resistor_cells]
                                  * mesh.triangle_areas[resistor_cells]))
    current = source_power / \
        geometry.config.battery.emf if geometry.config.battery.emf else 0.
    return {
        'geometry': geometry.to_dict(),
        'vertices': array(mesh.vertices), 'triangles': array(mesh.triangles),
        'centers': array(mesh.triangle_centers), 'regions': array(mesh.region_ids),
        'region_names': mesh.region_names,
        'potential': array(exterior.potential), 'electric': array(exterior.electric_field),
        'toy_electric': array(toy.electric_field),
        'toy_charges': {'edges': array(mesh.edges[toy.charge_edges]),
                        'density': array(toy.charge_density)},
        'current': array(conduction.current_density), 'magnetic': array(magnetic.bz),
        'energy': array(energy.energy_flux),
        'power_exchange': array(energy.source_power_density - energy.joule_density),
        'charges': traces,
        'summary': {'seconds': time.monotonic()-started, 'triangles': len(mesh.triangles),
                    'current': current, 'source_power': source_power,
                    'resistor_power': resistor_power,
                    'joule_power': float(np.sum(energy.joule_density*mesh.triangle_areas))},
    }
