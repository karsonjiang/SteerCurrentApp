"""The toy battery plates reproduce the intended source field."""
from dataclasses import replace

import numpy as np
import pytest

from steercurrent import MeshOptions, solve_conduction, solve_exterior, calculate_charges
from steercurrent.calculations import battery_only_field

from circuit_fixtures import rectangle_mesh


@pytest.fixture(scope='module')
def circuit():
    return rectangle_mesh(options=MeshOptions(size=.5))


def test_terminal_charge_field(circuit):
    toy = battery_only_field(circuit)
    mesh, battery = circuit.mesh, circuit.battery
    battery_cells = np.flatnonzero(mesh.region_ids == battery.region_id)
    mean = np.average(toy.electric_field[battery_cells] @ battery.direction,
                      weights=mesh.triangle_areas[battery_cells])
    assert mean == pytest.approx(-battery.emf/battery.length)
    assert toy.electric_field.shape == (len(mesh.triangles), 2)
    assert len(toy.charge_edges) == len(toy.charge_density)
    assert np.all(np.isfinite(toy.electric_field))
    assert np.sum(toy.charge_density *
                  mesh.edge_lengths[toy.charge_edges]) == pytest.approx(0, abs=1e-12)
    projection = mesh.edge_midpoints[toy.charge_edges] @ battery.direction
    assert np.all(toy.charge_density[projection > projection.mean()] > 0)
    assert np.all(toy.charge_density[projection < projection.mean()] < 0)
    # The actual circuit has the same terminal polarities, though its densities differ.
    real = calculate_charges(solve_exterior(solve_conduction(circuit)).fields)
    terminal = next(trace for trace in real.interfaces.values()
                    if battery.region_id in trace.regions)
    actual_projection = mesh.edge_midpoints[terminal.edge_ids] @ battery.direction
    assert terminal.charge_per_depth[actual_projection >
                                     actual_projection.mean()].sum() > 0
    assert terminal.charge_per_depth[actual_projection <
                                     actual_projection.mean()].sum() < 0
    # Changing epsilon changes the required charge, but not the calibrated E.
    doubled = battery_only_field(circuit, permittivity=2)
    np.testing.assert_allclose(doubled.charge_density, 2*toy.charge_density)
    np.testing.assert_allclose(doubled.electric_field, toy.electric_field)


def test_zero_and_reversed_emf(circuit):
    normal = battery_only_field(circuit)
    zero = battery_only_field(
        replace(circuit, battery=replace(circuit.battery, emf=0)))
    reversed_source = battery_only_field(
        replace(circuit, battery=replace(circuit.battery, emf=-circuit.battery.emf)))
    np.testing.assert_array_equal(zero.electric_field, 0)
    np.testing.assert_allclose(
        reversed_source.electric_field, -normal.electric_field)
    np.testing.assert_allclose(
        reversed_source.charge_density, -normal.charge_density)


def test_invalid_permittivity(circuit):
    with pytest.raises(ValueError, match='Permittivity'):
        battery_only_field(circuit, permittivity=0)
