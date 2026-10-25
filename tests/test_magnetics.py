"""Magnetic recovery: exact fields, Ampere law, scaling, and convergence."""

from dataclasses import replace

import numpy as np
import pytest

from steercurrent import MeshOptions, ExteriorOptions, solve_conduction, solve_magnetics
from steercurrent.meshing._mesh_import import _make_mesh_data

from circuit_fixtures import rectangle_mesh
from fem_helpers import section_current
from physics_diagnostics.conduction import conduction_diagnostics
from physics_diagnostics.magnetics import magnetic_diagnostics


@pytest.fixture(scope="module")
def conduction():
    return solve_conduction(
        rectangle_mesh(
            options=MeshOptions(size=0.4),
            exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
        )
    )


def test_exact_manufactured_stream_function(conduction):
    """Recover a known continuous P1 Bz with independent affine gradients."""
    mesh = conduction.mesh
    mu = 2.3
    exact = 0.03 * np.sin(mesh.vertices[:, 0]) + 0.02 * mesh.vertices[:, 1]
    outside = np.unique(mesh.triangles[mesh.triangle_sets["outside"]])
    cavity = np.unique(mesh.triangles[mesh.triangle_sets["cavity"]])
    exact[outside] = 0
    exact[cavity] = -0.8
    cells = mesh.triangle_sets["conductor"]
    points = mesh.vertices[mesh.triangles[cells]]
    affine = np.concatenate((np.ones((*points.shape[:2], 1)), points), axis=2)
    coefficients = np.linalg.solve(affine, exact[mesh.triangles[cells]][:, :, None])[
        :, :, 0
    ]
    current = np.zeros_like(conduction.current_density)
    current[cells] = np.column_stack((coefficients[:, 2], -coefficients[:, 1])) / mu
    manufactured = replace(conduction, current_density=current)
    result = solve_magnetics(manufactured, permeability=mu)
    assert result.mesh is mesh
    np.testing.assert_allclose(result.bz, exact, atol=2e-12)
    np.testing.assert_allclose(result.ampere_current, current, atol=3e-11)
    assert (
        magnetic_diagnostics(result, manufactured)["ampere_gradient_relative_error"]
        < 1e-11
    )
    # Independent degree-2 triangle quadrature checks the B^2 energy formula.
    barycentric = np.array(
        [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
    )
    values = exact[mesh.triangles] @ barycentric.T
    energy = np.sum(mesh.triangle_areas * np.mean(values**2, axis=1)) / (2 * mu)
    assert magnetic_diagnostics(result, manufactured)[
        "magnetic_energy_per_depth"
    ] == pytest.approx(energy, rel=1e-11)


def test_air_constants_sign_and_ampere_law(conduction):
    result = solve_magnetics(conduction)
    mesh = result.mesh
    outside = np.unique(mesh.triangles[mesh.triangle_sets["outside"]])
    cavity = np.unique(mesh.triangles[mesh.triangle_sets["cavity"]])
    np.testing.assert_array_equal(result.bz[outside], 0)
    np.testing.assert_array_equal(
        result.bz[cavity], magnetic_diagnostics(result, conduction)["cavity_bz"]
    )
    assert (
        magnetic_diagnostics(result, conduction)["cavity_bz"] < 0
    )  # clockwise current: into the plane
    measured = section_current(conduction, (-0.5, 3.5), (0.5, 3.5), (0, 1))
    assert -magnetic_diagnostics(result, conduction)["cavity_bz"] == pytest.approx(
        measured, rel=0.015
    )
    assert (
        magnetic_diagnostics(result, conduction)["ampere_gradient_relative_error"]
        < 0.12
    )
    np.testing.assert_array_equal(
        result.gradient_bz[
            np.r_[mesh.triangle_sets["outside"], mesh.triangle_sets["cavity"]]
        ],
        0,
    )


def test_scaling_reversal_gauge_and_invalid_inputs(conduction):
    reference = solve_magnetics(conduction)
    original = conduction.current_density.copy()
    for factor in (0.0, 2.0, -1.0):
        scaled_conduction = replace(conduction, current_density=factor * original)
        changed = solve_magnetics(scaled_conduction)
        np.testing.assert_allclose(changed.bz, factor * reference.bz, atol=1e-13)
        assert magnetic_diagnostics(changed, scaled_conduction)[
            "magnetic_energy_per_depth"
        ] == pytest.approx(
            factor**2
            * magnetic_diagnostics(reference, conduction)["magnetic_energy_per_depth"],
            abs=1e-14,
        )
    changed = solve_magnetics(conduction, permeability=3.0)
    np.testing.assert_allclose(changed.bz, 3 * reference.bz, atol=1e-13)
    np.testing.assert_allclose(
        changed.ampere_current, reference.ampere_current, atol=1e-12
    )
    assert magnetic_diagnostics(changed, conduction)[
        "magnetic_energy_per_depth"
    ] == pytest.approx(
        3 * magnetic_diagnostics(reference, conduction)["magnetic_energy_per_depth"]
    )
    shifted = solve_magnetics(replace(conduction, potential=conduction.potential + 7))
    np.testing.assert_array_equal(shifted.bz, reference.bz)
    np.testing.assert_array_equal(conduction.current_density, original)
    for mu in (0, -1, np.nan):
        with pytest.raises(ValueError, match="Permeability"):
            solve_magnetics(conduction, permeability=mu)
    with pytest.raises(ValueError, match="shape"):
        solve_magnetics(replace(conduction, current_density=np.zeros((2, 1))))
    current = original.copy()
    current[conduction.mesh.triangle_sets["cavity"][0]] = 1
    with pytest.raises(ValueError, match="current-free"):
        solve_magnetics(replace(conduction, current_density=current))


def test_refinement():
    errors, currents = [], []
    for h in (0.4, 0.2, 0.1):
        c = solve_conduction(
            rectangle_mesh(
                MeshOptions(size=h),
                exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
            )
        )
        r = solve_magnetics(c)
        errors.append(magnetic_diagnostics(r, c)["ampere_gradient_relative_error"])
        currents.append(
            abs(
                -magnetic_diagnostics(r, c)["cavity_bz"]
                - conduction_diagnostics(c)["source_average_current"]
            )
        )
    assert errors[-1] < 0.8 * errors[0]
    assert currents[-1] < currents[0]


def test_permuted_rotated_mesh(conduction):
    mesh = conduction.mesh
    rng = np.random.default_rng(712)
    vertices = rng.permutation(len(mesh.vertices))
    triangles = rng.permutation(len(mesh.triangles))
    vi, ti = np.argsort(vertices), np.argsort(triangles)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    changed = _make_mesh_data(
        mesh.vertices[vertices] @ rotation.T + [20, -5],
        vi[mesh.triangles[triangles]],
        mesh.region_ids[triangles],
        mesh.region_names,
        edge_vertices={
            name: vi[mesh.edges[ids]] for name, ids in mesh.edge_sets.items()
        },
        triangle_sets={name: ti[ids] for name, ids in mesh.triangle_sets.items()},
        interface_regions=mesh.interface_regions,
    )
    circuit = replace(
        conduction.circuit,
        mesh=changed,
        battery=replace(
            conduction.circuit.battery,
            direction=tuple(rotation @ conduction.circuit.battery.direction),
        ),
    )
    reference = solve_magnetics(conduction)
    result = solve_magnetics(solve_conduction(circuit))
    np.testing.assert_allclose(result.bz, reference.bz[vertices], atol=1e-12)
    np.testing.assert_allclose(
        result.gradient_bz, reference.gradient_bz[triangles] @ rotation.T, atol=1e-12
    )
