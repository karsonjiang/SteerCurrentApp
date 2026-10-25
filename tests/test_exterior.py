"""Air potential: boundary values, potential global shift, and truncation error."""

from dataclasses import replace

import numpy as np
import pytest
from skfem import Basis, ElementTriP1, FacetBasis

from steercurrent import (
    ExteriorOptions,
    validate_combined_mesh,
    MeshOptions,
    solve_conduction,
    solve_exterior,
)
from steercurrent.calculations.fem import assemble_p1, solve_dirichlet

from circuit_fixtures import rectangle_mesh
from fem_helpers import solver_view
from physics_diagnostics.exterior import exterior_diagnostics


def air_view(mesh):
    return solver_view(
        mesh,
        np.sort(np.r_[mesh.triangle_sets["outside"],
                mesh.triangle_sets["cavity"]]),
    )


@pytest.fixture(scope="module")
def conduction():
    return solve_conduction(
        rectangle_mesh(
            options=MeshOptions(size=0.4),
            exterior_options=ExteriorOptions(
                outer_scale=3, near_size=0.2, far_size=1),
        )
    )


@pytest.fixture(scope="module")
def exterior(conduction):
    return solve_exterior(conduction)


def test_boundary_mapping_and_neutrality(exterior):
    r = exterior
    d = exterior_diagnostics(r)
    assert d["region_areas"]["cavity"] == pytest.approx(54)
    assert d["area"] == pytest.approx(11 * 8 * 9 - 34)
    assert d["boundary_potential_error"] < 1e-13
    assert d["far_potential_error"] < 1e-13
    assert abs(r.fields.total_charge) < 1e-11
    assert r.electric_field.shape == (len(r.mesh.triangles), 2)
    assert r.fields.mesh is r.conduction.mesh
    ids = np.unique(r.mesh.edges[r.mesh.edge_sets["conductor"]])
    np.testing.assert_allclose(
        r.potential[ids], r.conduction.potential[ids] + r.potential_offset, atol=1e-13
    )
    assert (
        r.potential.min()
        >= np.nanmin(r.conduction.potential) + r.potential_offset - 1e-10
    )
    assert (
        r.potential.max()
        <= np.nanmax(r.conduction.potential) + r.potential_offset + 1e-10
    )


def test_gauge_invariance_scaling_and_zero(exterior):
    r = exterior
    original = r.conduction.potential.copy()
    shifted = solve_exterior(replace(r.conduction, potential=original + 7))
    np.testing.assert_allclose(shifted.potential, r.potential, atol=5e-12)
    np.testing.assert_allclose(
        shifted.electric_field, r.electric_field, atol=5e-11)
    assert shifted.potential_offset == pytest.approx(
        r.potential_offset - 7, abs=5e-12)
    np.testing.assert_array_equal(r.conduction.potential, original)
    for multiplier in (0, 2, -1):
        changed = solve_exterior(
            replace(
                r.conduction,
                potential=multiplier * original,
                electric_field=multiplier * r.conduction.electric_field,
                current_density=multiplier * r.conduction.current_density,
            )
        )
        np.testing.assert_allclose(
            changed.potential, multiplier * r.potential, atol=1e-12
        )
        np.testing.assert_allclose(
            changed.electric_field, multiplier * r.electric_field, atol=1e-12
        )
    eps = solve_exterior(r.conduction, permittivity=3)
    np.testing.assert_allclose(eps.potential, r.potential, atol=1e-12)
    assert exterior_diagnostics(eps)["exterior_field_energy"] == pytest.approx(
        3 * exterior_diagnostics(r)["exterior_field_energy"]
    )


def test_analytical_linear_laplace_both_components(exterior):
    view = air_view(exterior.mesh)
    system = assemble_p1(exterior.mesh, view.triangle_ids)
    basis = Basis(view.mesh, ElementTriP1())
    mesh = basis.mesh
    exact = 2 * mesh.p[0] - 3 * mesh.p[1] + 0.7
    prescribed = basis.get_dofs().all()
    v = solve_dirichlet(system.matrix, system.source, exact, prescribed)
    np.testing.assert_allclose(v, exact, atol=1e-10)
    gradient = basis.interpolate(v).grad
    np.testing.assert_allclose(gradient[0], 2, atol=1e-10)
    np.testing.assert_allclose(gradient[1], -3, atol=1e-10)
    # Constant cavity boundary values give a constant cavity, independent of outside.
    data = np.zeros(basis.N)
    data[basis.get_dofs("cavity").all()] = 1
    v = solve_dirichlet(system.matrix, system.source, data, prescribed)
    cavity_nodes = np.unique(mesh.t[:, mesh.subdomains["cavity"]])
    outside_nodes = np.unique(mesh.t[:, mesh.subdomains["outside"]])
    np.testing.assert_allclose(v[cavity_nodes], 1, atol=1e-12)
    np.testing.assert_allclose(v[outside_nodes], 0, atol=1e-12)


def test_harmonic_quadratic_and_boundary_gradient_convergence():
    errors, flux_errors = [], []
    for h in (0.4, 0.2):
        c = rectangle_mesh(
            MeshOptions(size=h),
            exterior_options=ExteriorOptions(
                outer_scale=2, near_size=h / 2, far_size=2 * h
            ),
        )
        view = air_view(c.mesh)
        system = assemble_p1(c.mesh, view.triangle_ids)
        mesh = view.mesh
        basis = Basis(mesh, ElementTriP1())
        data = mesh.p[0] ** 2 - mesh.p[1] ** 2
        v = solve_dirichlet(system.matrix, system.source,
                            data, basis.get_dofs().all())
        xy = basis.global_coordinates()
        error = basis.interpolate(v) - (xy[0] ** 2 - xy[1] ** 2)
        errors.append(float(np.sqrt(np.sum(error**2 * basis.dx))))
        boundary = FacetBasis(mesh, ElementTriP1(), facets="conductor")
        xy = boundary.global_coordinates()
        exact_gradient = np.array([2 * xy[0], -2 * xy[1]])
        normal_error = np.sum(
            (boundary.interpolate(v).grad - exact_gradient) * boundary.normals, axis=0
        )
        flux_errors.append(
            float(np.sqrt(np.sum(normal_error**2 * boundary.dx))))
    assert errors[1] < 0.6 * errors[0]
    assert flux_errors[1] < 0.8 * flux_errors[0]


def test_outer_boundary_sensitivity(conduction):
    probes = np.array([[12.0, 5.0, 3.0], [3.5, 9.0, 3.0]])
    results = [
        solve_exterior(
            solve_conduction(
                rectangle_mesh(
                    options=MeshOptions(size=0.4),
                    exterior_options=ExteriorOptions(
                        outer_scale=scale, near_size=0.15, far_size=1.0
                    ),
                )
            )
        )
        for scale in (2, 4, 8)
    ]
    potentials = []
    for r in results:
        view = air_view(r.mesh)
        potentials.append(
            Basis(view.mesh, ElementTriP1()).interpolator(r.potential[view.vertex_ids])(
                probes
            )
        )
    # The change decreases when the box is enlarged again. This tests truncation,
    # not just the zero-charge constraint which is exact even on the smallest box.
    assert np.linalg.norm(potentials[2] - potentials[1]) < 0.5 * np.linalg.norm(
        potentials[1] - potentials[0]
    )
    # The cavity E cannot depend on a far-boundary potential offset.
    point = np.array([[3.0], [3.0]])
    fields = []
    for r in results:
        view = air_view(r.mesh)
        fields.append(
            r.electric_field[view.triangle_ids[view.mesh.element_finder()(*point)]]
        )
    np.testing.assert_allclose(fields[2], fields[0], atol=1e-4, rtol=0.01)


def test_rotated_translated_shape(conduction):
    c = conduction.circuit
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    rotated = replace(
        c,
        mesh=replace(c.mesh, vertices=c.mesh.vertices @
                     rotation.T + [20.0, -5.0]),
        battery=replace(c.battery, direction=tuple(
            rotation @ c.battery.direction)),
    )
    result = solve_exterior(solve_conduction(rotated))
    assert exterior_diagnostics(result)["boundary_potential_error"] < 1e-12
    assert abs(result.fields.total_charge) < 1e-11


def test_invalid_options_and_mapping(conduction, exterior):
    for options in (
        ExteriorOptions(outer_scale=1),
        ExteriorOptions(near_size=0),
        ExteriorOptions(far_size=float("nan")),
        ExteriorOptions(near_size=2, far_size=1),
    ):
        with pytest.raises(ValueError):
            rectangle_mesh(options=MeshOptions(
                size=0.4), exterior_options=options)
    with pytest.raises(ValueError, match="Permittivity"):
        solve_exterior(conduction, permittivity=0)
    wrong = conduction.mesh.edge_triangles.copy()
    wrong[conduction.mesh.edge_sets["outer"], 1] = -1
    with pytest.raises(ValueError, match="missing neighbor"):
        validate_combined_mesh(
            replace(
                conduction.circuit, mesh=replace(
                    conduction.mesh, edge_triangles=wrong)
            )
        )
