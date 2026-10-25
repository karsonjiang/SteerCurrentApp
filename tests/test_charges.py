"""Surface charges: field-jump signs, material interfaces, and convergence."""

from dataclasses import replace

import numpy as np
import pytest

from steercurrent import (
    ExteriorOptions,
    MeshOptions,
    solve_conduction,
    solve_exterior,
    calculate_charges,
)

from circuit_fixtures import rectangle_mesh
from physics_diagnostics.charges import charge_diagnostics


@pytest.fixture(scope="module")
def exterior():
    c = rectangle_mesh(
        options=MeshOptions(size=0.4),
        exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
    )
    return solve_exterior(solve_conduction(c))


def test_constant_field_jump_and_normals(exterior):
    """Analytical traces test signs, units, facet matching, and both air regions."""
    mesh = exterior.mesh
    electric = np.tile([-1.0, 4.0], (len(mesh.triangles), 1))
    electric[mesh.triangle_sets["conductor"]] = [2.0, -3.0]
    r = calculate_charges(
        replace(exterior.fields, electric_field=electric, permittivity=2.0)
    )
    center = np.array([[5.0], [3.5]])
    for name, trace in r.surfaces.items():
        midpoint = trace.mesh.edge_midpoints[trace.edge_ids].T
        # Axis-aligned rectangular boundaries; construct normals independently.
        displacement = midpoint - center
        half = np.array([[5.5], [4.0]]) if name == "outer" else np.array([[4.5], [3.0]])
        normal = np.zeros_like(displacement)
        for axis in (0, 1):
            on_face = np.isclose(np.abs(displacement[axis]), half[axis, 0])
            normal[axis, on_face] = np.sign(displacement[axis, on_face])
        if name == "cavity":
            normal *= -1
        np.testing.assert_allclose(
            trace.mesh.edge_normals[trace.edge_ids].T, normal, atol=1e-12
        )
        expected = 2 * np.sum(np.array([[-3.0], [7.0]]) * normal, axis=0)
        np.testing.assert_allclose(trace.density, expected, atol=1e-12)
        lengths = trace.mesh.edge_lengths[trace.edge_ids]
        np.testing.assert_allclose(
            trace.charge_per_depth, expected * lengths, atol=1e-12
        )
        assert abs(trace.charge_per_depth.sum()) < 1e-12
    for trace in r.interfaces.values():
        np.testing.assert_allclose(trace.density, 0, atol=1e-12)
    np.testing.assert_array_equal(r.bulk_density, 0)


def test_material_fields_and_battery(exterior):
    m = exterior.mesh
    electric = np.column_stack((m.region_ids * 2.0, -m.region_ids * 3.0))
    r = calculate_charges(replace(exterior.fields, electric_field=electric))
    for trace in [*r.surfaces.values(), *r.interfaces.values()]:
        pair = m.edge_triangles[trace.edge_ids]
        jump = (m.region_ids[pair[:, 1]] - m.region_ids[pair[:, 0]])[:, None] * [
            2.0,
            -3.0,
        ]
        np.testing.assert_allclose(
            trace.density, np.sum(jump * m.edge_normals[trace.edge_ids], axis=1)
        )
        displacement = m.triangle_centers[pair[:, 1]] - m.triangle_centers[pair[:, 0]]
        assert np.all(np.sum(displacement * m.edge_normals[trace.edge_ids], axis=1) > 0)
    battery = exterior.conduction.circuit.battery.region_id
    battery_trace = next(t for t in r.interfaces.values() if battery in t.regions)
    assert np.max(np.abs(battery_trace.density)) > 0


def test_scaling_zero_and_balance(exterior):
    reference = calculate_charges(exterior.fields)
    for factor in (0.0, -2.0):
        r = calculate_charges(
            replace(exterior.fields, electric_field=factor * exterior.electric_field)
        )
        for name, trace in {**r.surfaces, **r.interfaces}.items():
            expected = {**reference.surfaces, **reference.interfaces}[name]
            np.testing.assert_allclose(
                trace.density, factor * expected.density, atol=1e-12
            )
    doubled = calculate_charges(replace(exterior.fields, permittivity=2))
    assert charge_diagnostics(doubled)["total_charge_per_depth"] == pytest.approx(
        2 * charge_diagnostics(reference)["total_charge_per_depth"]
    )
    all_facets = np.concatenate(
        [
            t.edge_ids
            for t in [*reference.surfaces.values(), *reference.interfaces.values()]
        ]
    )
    assert len(np.unique(all_facets)) == len(all_facets)
    assert charge_diagnostics(reference)["relative_charge_imbalance"] < 0.03
    with pytest.raises(ValueError, match="shape"):
        calculate_charges(replace(exterior.fields, electric_field=np.zeros((2, 1))))


def test_integrated_segment_convergence():
    charges = []
    for h in (0.5, 0.25, 0.125):
        c = rectangle_mesh(
            MeshOptions(size=h),
            exterior_options=ExteriorOptions(
                outer_scale=3, near_size=h / 2, far_size=1
            ),
        )
        r = calculate_charges(solve_exterior(solve_conduction(c)).fields)
        trace = r.surfaces["outer"]
        endpoints = trace.mesh.vertices[trace.mesh.edges[trace.edge_ids]].transpose(
            2, 1, 0
        )
        # Integrate the fixed segment x=10.5, 2<=y<=5, including fractional edges.
        on_right = np.all(np.isclose(endpoints[0], 10.5), axis=0)
        overlap = np.maximum(
            0,
            np.minimum(endpoints[1].max(axis=0), 5)
            - np.maximum(endpoints[1].min(axis=0), 2),
        )
        charges.append(float(np.sum(trace.density[on_right] * overlap[on_right])))
    assert abs(charges[2] - charges[1]) < abs(charges[1] - charges[0])
