"""Conduction: analytical solutions, current conservation, and mesh convergence."""

from dataclasses import replace

import numpy as np
import pytest
from skfem import MeshTri

from steercurrent import MeshOptions, Region, solve_conduction
from steercurrent.calculations.fem import assemble_p1, solve_dirichlet
from steercurrent.meshing._mesh_import import _make_mesh_data

from circuit_fixtures import rectangle_mesh
from fem_helpers import solver_view, section_current
from physics_diagnostics.conduction import conduction_diagnostics


@pytest.fixture(scope="module")
def reference():
    return solve_conduction(rectangle_mesh())


@pytest.mark.parametrize("rho_right", [1.0, 20.0])
def test_analytical_straight_conductor(rho_right):
    mesh = MeshTri.init_tensor(np.linspace(0, 4, 17), np.linspace(0, 1, 5))
    rho = np.where(mesh.p[:, mesh.t].mean(axis=1)[0] < 2, 1.0, rho_right)
    prescribed = {
        int(i): float(mesh.p[0, i] == 0)
        for i in np.flatnonzero((mesh.p[0] == 0) | (mesh.p[0] == 4))
    }
    data = _make_mesh_data(
        mesh.p.T, mesh.t.T, np.ones(mesh.nelements, dtype=int), {1: "conductor"}
    )
    system = assemble_p1(data, np.arange(mesh.nelements), coefficient=1 / rho)
    matrix, source = system.matrix, system.source
    values = np.zeros(len(data.vertices))
    fixed = np.array(list(prescribed))
    values[fixed] = list(prescribed.values())
    potential = solve_dirichlet(matrix, source, values, fixed)
    current = 1 / (2 + 2 * rho_right)
    expected = 1 - current * (
        np.minimum(mesh.p[0], 2) + rho_right * np.maximum(mesh.p[0] - 2, 0)
    )
    np.testing.assert_allclose(potential, expected, atol=2e-13)
    electric = -system.gradient(potential).T
    np.testing.assert_allclose(electric[0] / rho, current, atol=2e-13)
    np.testing.assert_allclose(electric[1], 0, atol=2e-13)
    free = np.setdiff1d(np.arange(len(data.vertices)), list(prescribed))
    np.testing.assert_allclose((matrix @ potential - source)[free], 0, atol=2e-13)


def test_reference_conservation_and_stage_interface(reference):
    r = reference
    d = conduction_diagnostics(r)
    assert r.potential.shape == (len(r.mesh.vertices),)
    assert r.current_density.shape == (len(r.mesh.triangles), 2)
    cells = r.mesh.triangle_sets["conductor"]
    assert np.sum(
        r.potential[r.mesh.triangles[cells]].mean(axis=1) * r.mesh.triangle_areas[cells]
    ) == pytest.approx(0, abs=1e-12)
    assert d["power_balance_relative_error"] < 1e-9
    assert d["emf_power"] > 0
    assert len(d["interface_cross_sections"]) == 4
    assert d["cross_section_max_relative_deviation"] < 0.002
    assert d["boundary_normal_current_relative"] < 0.1
    # Direct P1 boundary traces have discretization error; weak conservation
    # is checked in the FEM tests, and trace convergence is checked under refinement.
    assert abs(d["boundary_net_current"]) < 0.01 * d["source_average_current"]
    # Independently sample four cuts, with normals following clockwise circulation.
    cuts = [
        ((2, -0.5), (2, 0.5), (-1, 0)),
        ((9.5, 3.5), (10.5, 3.5), (0, -1)),
        ((5, 6.5), (5, 7.5), (1, 0)),
        ((-0.5, 3.5), (0.5, 3.5), (0, 1)),
    ]
    for cut in cuts:
        current = section_current(r, *cut)
        assert current == pytest.approx(d["source_average_current"], rel=0.002)
        assert section_current(r, *cut, samples=1024) == pytest.approx(
            current, rel=0.001
        )
    # Battery is the sole source; exporting E as J/gamma would miss this distinction.
    battery = r.mesh.region_ids == r.circuit.battery.region_id
    np.testing.assert_allclose(r.driving_field[~battery], 0)
    np.testing.assert_allclose(r.driving_field[battery, 1], 0.5)
    np.testing.assert_allclose(
        r.current_density[cells] * r.resistivity[cells, None],
        r.electric_field[cells] + r.driving_field[cells],
    )
    # The next stage reads boundary vertex IDs directly from the unified mesh.
    dofs = np.unique(r.mesh.edges[r.mesh.edge_sets["conductor"]])
    assert np.all(np.isfinite(r.potential[dofs]))


def test_gauge_scaling_and_reversal(reference):
    c = reference.circuit
    alternate = solve_conduction(
        c,
        gauge_dof=int(
            np.unique(c.mesh.triangles[c.mesh.triangle_sets["conductor"]])[-1]
        ),
    )
    np.testing.assert_allclose(alternate.potential, reference.potential, atol=2e-11)
    np.testing.assert_allclose(
        alternate.current_density, reference.current_density, atol=2e-11
    )
    for factor in (0, 2, -1):
        scaled = solve_conduction(
            replace(c, battery=replace(c.battery, emf=factor * c.battery.emf))
        )
        for name in ("potential", "electric_field", "current_density"):
            np.testing.assert_allclose(
                getattr(scaled, name), factor * getattr(reference, name), atol=2e-11
            )
        assert conduction_diagnostics(scaled)["joule_power"] == pytest.approx(
            factor**2 * conduction_diagnostics(reference)["joule_power"], abs=1e-12
        )
        assert conduction_diagnostics(scaled)["emf_power"] == pytest.approx(
            conduction_diagnostics(scaled)["joule_power"], abs=1e-12
        )
        assert all(
            np.isfinite(value)
            for value in conduction_diagnostics(scaled).values()
            if isinstance(value, (int, float))
        )


def test_geometry_independent_rotated_conduction(reference):
    c = reference.circuit
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = replace(
        c,
        mesh=replace(c.mesh, vertices=c.mesh.vertices @ rotation.T),
        battery=replace(c.battery, direction=tuple(rotation @ c.battery.direction)),
    )
    result = solve_conduction(rotated)
    np.testing.assert_allclose(result.potential, reference.potential, atol=2e-11)
    np.testing.assert_allclose(
        result.current_density, reference.current_density @ rotation.T, atol=2e-11
    )


def test_multiple_resistors(reference):
    c = reference.circuit
    m = c.mesh
    labels = m.region_ids.copy()
    centers = m.triangle_centers
    key = max(m.region_names) + 1
    labels[
        np.isin(np.arange(len(labels)), m.triangle_sets["conductor"])
        & (centers[:, 1] < 0.5)
        & (centers[:, 0] > 4)
        & (centers[:, 0] < 6)
    ] = key
    # Keep all conductor IDs below air IDs so surface orientation stays canonical.
    air = np.r_[m.triangle_sets["outside"], m.triangle_sets["cavity"]]
    old_air_ids = np.unique(labels[air])
    names = {k: v for k, v in m.region_names.items() if k not in old_air_ids}
    names[key] = "resistor_2"
    for i, old in enumerate(old_air_ids, start=key + 1):
        labels[np.isin(np.arange(len(labels)), air) & (m.region_ids == old)] = i
        names[i] = m.region_names[old]
    regions = {**c.regions, key: Region("resistor_2", "resistor", 30)}
    interior = np.flatnonzero(m.edge_triangles[:, 1] >= 0)
    pairs = np.sort(labels[m.edge_triangles[interior]], axis=1)
    edge_sets = {name: m.edges[ids] for name, ids in m.edge_sets.items()}
    name = f"interface_1_{key}"
    edge_sets[name] = m.edges[interior[np.all(pairs == [1, key], axis=1)]]
    mesh = _make_mesh_data(
        m.vertices,
        m.triangles,
        labels,
        names,
        edge_vertices=edge_sets,
        triangle_sets=m.triangle_sets,
        interface_regions={**m.interface_regions, name: (1, key)},
    )
    result = solve_conduction(replace(c, mesh=mesh, regions=regions))
    assert (
        conduction_diagnostics(result)["source_average_current"]
        < conduction_diagnostics(reference)["source_average_current"]
    )
    assert conduction_diagnostics(result)["power_balance_relative_error"] < 1e-9


def test_mesh_convergence():
    results = [
        solve_conduction(rectangle_mesh(MeshOptions(size=h))) for h in (0.4, 0.2, 0.1)
    ]
    currents = np.array(
        [conduction_diagnostics(r)["source_average_current"] for r in results]
    )
    assert abs(currents[2] - currents[1]) < abs(currents[1] - currents[0])
    assert abs(currents[2] - currents[1]) / currents[2] < 0.002
    # Natural boundary conditions converge weakly; recovered P1 flux is not exactly zero.
    boundary_errors = [
        conduction_diagnostics(r)["boundary_normal_current_relative"] for r in results
    ]
    assert boundary_errors[-1] < boundary_errors[0]
    for r in results:
        assert conduction_diagnostics(r)["power_balance_relative_error"] < 1e-8
    # Compare field values away from corners/interfaces.
    points = np.array([[2.0, 10.0, 5.0, 0.0], [0.0, 3.5, 7.0, 3.5]])
    fields = []
    for r in results:
        view = solver_view(r.mesh, r.mesh.triangle_sets["conductor"])
        fields.append(
            r.current_density[view.triangle_ids[view.mesh.element_finder()(*points)]]
        )
    np.testing.assert_allclose(fields[-1], fields[-2], rtol=0.02, atol=2e-5)


def test_invalid_inputs(reference):
    with pytest.raises(ValueError, match="gauge_dof"):
        solve_conduction(reference.circuit, gauge_dof=-1)
    c = reference.circuit
    with pytest.raises(ValueError, match="battery"):
        solve_conduction(replace(c, battery=replace(c.battery, direction=(1.0,))))
    with pytest.raises(ValueError, match="Resistivity"):
        solve_conduction(
            replace(c, regions={**c.regions, 1: replace(c.regions[1], resistivity=-1)})
        )
