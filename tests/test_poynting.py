"""Energy flow: exact edge integrals, power balance, and convergence."""

from dataclasses import replace

import numpy as np
import pytest

from steercurrent import (
    MeshOptions,
    ExteriorOptions,
    solve_conduction,
    solve_exterior,
    solve_magnetics,
    calculate_poynting,
)

from circuit_fixtures import rectangle_mesh
from physics_diagnostics.poynting import poynting_diagnostics


@pytest.fixture(scope="module")
def results():
    c = solve_conduction(
        rectangle_mesh(
            options=MeshOptions(size=0.4),
            exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
        )
    )
    return solve_exterior(c), solve_magnetics(c)


def test_manufactured_fields_and_exact_edge_integration(results):
    exterior, magnetic = results
    mesh = exterior.mesh
    electric = np.tile([2.0, -3.0], (len(mesh.triangles), 1))
    gradient = np.tile([0.4, 0.2], (len(mesh.triangles), 1))
    mu = 2.0
    current = np.tile([0.1, -0.2], (len(mesh.triangles), 1))
    potential = -2 * mesh.vertices[:, 0] + 3 * mesh.vertices[:, 1]
    bz = 1 + 0.4 * mesh.vertices[:, 0] + 0.2 * mesh.vertices[:, 1]
    # Manufactured data tests the energy calculation, not the circuit model.
    e = replace(
        exterior,
        conduction=replace(exterior.conduction, current_density=current),
        fields=replace(
            exterior.fields,
            potential=potential,
            electric_field=electric,
            current_density=current,
        ),
    )
    b = replace(magnetic, bz=bz, gradient_bz=gradient, permeability=mu)
    result = calculate_poynting(e, b)
    expected_b = (
        1 + 0.4 * mesh.triangle_centers[:, 0] + 0.2 * mesh.triangle_centers[:, 1]
    )
    np.testing.assert_allclose(
        result.energy_flux, expected_b[:, None] * np.array([-3.0, -2.0]) / mu
    )
    np.testing.assert_allclose(result.divergence, -0.8)
    np.testing.assert_allclose(result.balance_residual, 0, atol=1e-14)
    mid = mesh.edge_midpoints
    expected = (
        (1 + 0.4 * mid[:, 0] + 0.2 * mid[:, 1])
        / mu
        * (mesh.edge_normals @ [-3.0, -2.0])
        * mesh.edge_lengths
    )
    flux = result.edge_fluxes()
    np.testing.assert_allclose(flux[:, 0], expected, atol=1e-13)
    shared = mesh.edge_triangles[:, 1] >= 0
    np.testing.assert_allclose(flux[shared, 1], expected[shared], atol=1e-13)
    assert np.all(np.isnan(flux[~shared, 1]))
    assert poynting_diagnostics(result)["edge_divergence_consistency_l2"] < 1e-10


def test_energy_flow_and_power_balance(results):
    result = calculate_poynting(*results)
    d = poynting_diagnostics(result)
    assert result.mesh is results[0].mesh
    assert d["global_power_balance_relative_error"] < 1e-9
    assert d["orthogonality_relative_max"] < 1e-13
    assert d["normal_flux_jump_l2"] < 1e-10
    assert d["edge_divergence_consistency_l2"] < 1e-10
    assert d["ohm_power_identity_error_l2"] < 1e-11
    assert d["local_balance_relative_l2"] < 0.03
    assert d["far_outward_power_per_depth"] == 0
    for name in ("wire", "resistor_1"):
        r = d["regions"][name]
        assert r["outward_flux_per_depth"] < 0
        assert -r["outward_flux_per_depth"] == pytest.approx(
            r["joule_power_per_depth"], rel=0.02
        )
    battery = d["regions"]["battery"]
    assert battery["outward_flux_per_depth"] > 0
    assert battery["electrical_absorption_per_depth"] < 0
    assert battery["source_power_per_depth"] > battery["joule_power_per_depth"]
    assert battery["outward_flux_per_depth"] == pytest.approx(
        battery["source_power_per_depth"] - battery["joule_power_per_depth"], rel=0.02
    )
    np.testing.assert_array_equal(
        result.energy_flux[result.mesh.triangle_sets["outside"]], 0
    )


def test_scaling_permeability_zero_and_gauge(results):
    exterior, magnetic = results
    reference = calculate_poynting(exterior, magnetic)
    for factor in (0.0, -1.0, 2.0):
        c = exterior.conduction
        changed = solve_conduction(
            replace(
                c.circuit,
                battery=replace(c.circuit.battery, emf=factor * c.circuit.battery.emf),
            )
        )
        result = calculate_poynting(solve_exterior(changed), solve_magnetics(changed))
        np.testing.assert_allclose(
            result.energy_flux, factor**2 * reference.energy_flux, atol=1e-12
        )
        np.testing.assert_allclose(
            result.electrical_power_density,
            factor**2 * reference.electrical_power_density,
            atol=1e-12,
        )
    increased = calculate_poynting(
        exterior, solve_magnetics(exterior.conduction, permeability=3)
    )
    np.testing.assert_allclose(increased.energy_flux, reference.energy_flux, atol=1e-12)
    shifted = replace(
        exterior, fields=replace(exterior.fields, potential=exterior.potential + 7)
    )
    np.testing.assert_array_equal(
        calculate_poynting(shifted, magnetic).energy_flux, reference.energy_flux
    )


def test_local_balance_refines():
    errors = []
    for h in (0.4, 0.2):
        c = solve_conduction(
            rectangle_mesh(
                options=MeshOptions(size=h),
                exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
            )
        )
        r = calculate_poynting(solve_exterior(c), solve_magnetics(c))
        errors.append(poynting_diagnostics(r)["local_balance_relative_l2"])
    assert errors[1] < errors[0]


def test_invalid_fields(results):
    e, b = results
    with pytest.raises(ValueError, match="same mesh"):
        calculate_poynting(e, replace(b, mesh=replace(b.mesh)))
    with pytest.raises(ValueError, match="Permeability"):
        calculate_poynting(e, replace(b, permeability=0))
    with pytest.raises(ValueError, match="shapes"):
        calculate_poynting(e, replace(b, bz=np.zeros(1)))
