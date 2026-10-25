"""Independent analytical and library-reference checks for explicit assembly and fem."""

import numpy as np
import pytest
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, asm
from skfem.helpers import dot, grad

from steercurrent import MeshOptions, ExteriorOptions, solve_conduction, solve_exterior
from steercurrent.calculations.fem import assemble_p1, solve_dirichlet
from steercurrent.meshing._mesh_import import _make_mesh_data

from circuit_fixtures import rectangle_mesh


@BilinearForm
def stiffness(u, v, w):
    return w.c * dot(grad(u), grad(v))


@LinearForm
def source(v, w):
    return w.c * (w.fx * grad(v)[0] + w.fy * grad(v)[1])


@pytest.mark.parametrize("reverse", [False, True])
def test_matrix_and_source_against_independent_library(reverse):
    native = MeshTri.init_tensor(np.linspace(0, 3, 8), np.linspace(0, 1, 5))
    triangles = native.t.T.copy()
    if reverse:
        triangles[::2] = triangles[::2, ::-1]
    mesh = _make_mesh_data(
        native.p.T, triangles, np.ones(len(triangles), dtype=int), {
            1: "material"}
    )
    rng = np.random.default_rng(72)
    coefficient = rng.uniform(0.05, 10, len(triangles))
    driving = rng.normal(size=(len(triangles), 2))
    system = assemble_p1(mesh, np.arange(len(triangles)), coefficient, driving)
    basis = Basis(MeshTri(mesh.vertices.T, triangles.T,
                  sort_t=False), ElementTriP1())
    expected_matrix = asm(stiffness, basis, c=coefficient[:, None])
    expected_source = asm(
        source,
        basis,
        c=coefficient[:, None],
        fx=driving[:, 0, None],
        fy=driving[:, 1, None],
    )
    np.testing.assert_allclose(
        system.matrix.toarray(), expected_matrix.toarray(), atol=1e-13
    )
    np.testing.assert_allclose(system.source, expected_source, atol=1e-14)
    np.testing.assert_allclose(
        system.gradient(2 * mesh.vertices[:, 0] - 3 * mesh.vertices[:, 1] + 5),
        np.tile([2.0, -3.0], (len(triangles), 1)),
        atol=1e-13,
    )


def test_uniform_driving_field_has_exact_zero_current():
    native = MeshTri.init_tensor(np.linspace(0, 4, 9), np.linspace(0, 1, 4))
    mesh = _make_mesh_data(
        native.p.T, native.t.T, np.ones(
            native.nelements, dtype=int), {1: "battery"}
    )
    driving = np.tile([0.3, -0.7], (native.nelements, 1))
    system = assemble_p1(
        mesh, np.arange(native.nelements), coefficient=2.0, driving=driving
    )
    values = np.zeros(len(mesh.vertices))
    values[0] = 1.2
    potential = solve_dirichlet(system.matrix, system.source, values, [0])
    expected = mesh.vertices @ driving[0] + 1.2
    np.testing.assert_allclose(potential, expected, atol=1e-12)
    np.testing.assert_allclose(system.gradient(potential), driving, atol=1e-12)


def test_dirichlet_elimination_and_disconnected_components():
    mesh = _make_mesh_data(
        [[0, 0], [1, 0], [0, 1], [3, 0], [4, 0], [3, 1]],
        [[0, 1, 2], [3, 4, 5]],
        [1, 1],
        {1: "air"},
    )
    system = assemble_p1(mesh, [0, 1])
    values = np.array([2.0, 0, 0, -3.0, 0, 0])
    potential = solve_dirichlet(system.matrix, system.source, values, [0, 3])
    np.testing.assert_allclose(potential, [2, 2, 2, -3, -3, -3])
    np.testing.assert_array_equal(
        solve_dirichlet(system.matrix, system.source,
                        values, np.arange(6)), values
    )
    for fixed in ([], [0], [-1], [6]):
        with pytest.raises(ValueError):
            solve_dirichlet(system.matrix, system.source, values, fixed)
    with pytest.raises(ValueError):
        assemble_p1(mesh, [0], coefficient=0)
    with pytest.raises(ValueError):
        assemble_p1(mesh, [0], driving=np.array([[np.nan, 0]]))
    with pytest.raises(ValueError):
        system.dofs([100])
    with pytest.raises(ValueError):
        assemble_p1(mesh, [0, 0])


def test_final_fields_satisfy_assembled_equations():
    """Rebuild equations in the test without retaining solver intermediates."""
    from steercurrent import solve_magnetics

    circuit = rectangle_mesh(
        options=MeshOptions(size=0.4),
        exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
    )
    conduction = solve_conduction(circuit)
    exterior = solve_exterior(conduction)
    magnetic = solve_magnetics(conduction)
    mesh = circuit.mesh
    cells = mesh.triangle_sets["conductor"]

    system = assemble_p1(
        mesh, cells, 1 /
        conduction.resistivity[cells], conduction.driving_field[cells]
    )
    residual = system.matrix @ conduction.potential[system.vertex_ids] - system.source
    assert np.linalg.norm(residual) / np.linalg.norm(system.source) < 1e-10

    air = np.r_[mesh.triangle_sets["outside"], mesh.triangle_sets["cavity"]]
    system = assemble_p1(mesh, air)
    boundary = np.unique(
        mesh.edges[np.r_[mesh.edge_sets["conductor"], mesh.edge_sets["far"]]]
    )
    fixed = system.dofs(boundary)
    free = np.setdiff1d(np.arange(len(system.vertex_ids)), fixed)
    residual = system.matrix @ exterior.potential[system.vertex_ids]
    assert np.linalg.norm(residual[free]) / \
        np.linalg.norm(residual[fixed]) < 1e-10
    far = system.dofs(np.unique(mesh.edges[mesh.edge_sets["far"]]))
    assert abs(residual[far].sum()) / np.abs(residual[fixed]).sum() < 1e-11

    current = conduction.current_density[cells]
    target = magnetic.permeability * \
        np.column_stack((-current[:, 1], current[:, 0]))
    system = assemble_p1(mesh, cells, driving=target)
    residual = system.matrix @ magnetic.bz[system.vertex_ids] - system.source
    outer = system.dofs(np.unique(mesh.edges[mesh.edge_sets["outer"]]))
    cavity = system.dofs(np.unique(mesh.edges[mesh.edge_sets["cavity"]]))
    free = np.setdiff1d(np.arange(len(system.vertex_ids)),
                        np.r_[outer, cavity])
    # Cavity vertices share one unknown, so their equations are summed.
    reduced_residual = np.r_[residual[free], residual[cavity].sum()]
    assert np.linalg.norm(reduced_residual) / \
        np.linalg.norm(system.source) < 1e-10
