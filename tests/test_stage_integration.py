"""Different calculation stages use one authoritative mesh, independent of storage ordering."""

from dataclasses import replace

import numpy as np
import pytest

from steercurrent import (
    MeshOptions,
    ExteriorOptions,
    solve_conduction,
    solve_exterior,
    calculate_charges,
    validate_combined_mesh,
)
from steercurrent.meshing._mesh_import import _make_mesh_data

from circuit_fixtures import rectangle_mesh
from physics_diagnostics.charges import charge_diagnostics


@pytest.fixture(scope="module")
def circuit():
    return rectangle_mesh(
        options=MeshOptions(size=0.4),
        exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
    )


def test_calculations_do_not_remesh(circuit, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A later stage attempted to generate geometry.")

    import steercurrent.meshing._exterior_meshing as air
    import steercurrent.meshing._conductor_meshing as conductor

    monkeypatch.setattr(air, "generate_exterior_mesh", forbidden)
    monkeypatch.setattr(conductor, "generate_conductor_mesh", forbidden)
    import gmsh

    monkeypatch.setattr(gmsh, "initialize", forbidden)
    step2 = solve_conduction(circuit)
    m = circuit.mesh
    cond_nodes = np.unique(m.triangles[m.triangle_sets["conductor"]])
    air_only_nodes = np.setdiff1d(np.arange(len(m.vertices)), cond_nodes)
    assert np.all(np.isnan(step2.potential[air_only_nodes]))
    assert np.all(np.isnan(step2.electric_field[m.triangle_sets["outside"]]))
    assert np.all(step2.current_density[m.triangle_sets["outside"]] == 0)
    step3 = solve_exterior(step2)
    step4 = calculate_charges(step3.fields)
    assert circuit.mesh is step2.mesh is step3.mesh is step4.fields.mesh
    assert np.all(np.isfinite(step3.potential))
    assert np.all(np.isfinite(step3.electric_field))


def test_permuted_global_ids(circuit):
    m = circuit.mesh
    rng = np.random.default_rng(781)
    vertices = rng.permutation(len(m.vertices))
    triangles = rng.permutation(len(m.triangles))
    vertex_inverse, triangle_inverse = np.argsort(
        vertices), np.argsort(triangles)
    reordered = _make_mesh_data(
        m.vertices[vertices],
        vertex_inverse[m.triangles[triangles]],
        m.region_ids[triangles],
        m.region_names,
        edge_vertices={
            name: vertex_inverse[m.edges[ids]] for name, ids in m.edge_sets.items()
        },
        triangle_sets={
            name: triangle_inverse[ids] for name, ids in m.triangle_sets.items()
        },
        interface_regions=m.interface_regions,
    )
    changed = replace(circuit, mesh=reordered)
    validate_combined_mesh(changed)
    expected = solve_exterior(solve_conduction(circuit))
    actual = solve_exterior(solve_conduction(changed))
    np.testing.assert_allclose(
        actual.potential, expected.potential[vertices], atol=2e-10
    )
    np.testing.assert_allclose(
        actual.electric_field, expected.electric_field[triangles], atol=2e-10
    )
    a, b = calculate_charges(expected.fields), calculate_charges(actual.fields)
    for name, value in charge_diagnostics(a)["charge_per_depth_by_group"].items():
        assert charge_diagnostics(b)["charge_per_depth_by_group"][
            name
        ] == pytest.approx(value, abs=2e-10)
