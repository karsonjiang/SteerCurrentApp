"""Unified mesh arrays: adjacency, orientation, and invariance under rotation."""

from dataclasses import replace

import numpy as np
import pytest

from steercurrent import (
    MeshOptions,
    solve_conduction,
    solve_exterior,
    ExteriorOptions,
    calculate_charges,
)
from steercurrent.meshing._mesh_import import _make_mesh_data

from circuit_fixtures import rectangle_mesh


def test_explicit_triangle_geometry_and_orientation():
    # Clockwise and counterclockwise input triangles, regions deliberately reversed.
    m = _make_mesh_data(
        [[0, 0], [2, 0], [2, 1], [0, 1]],
        [[0, 1, 2], [0, 3, 2]],
        [9, 2],
        {9: "b", 2: "a"},
        edge_vertices={"interface": [[0, 2]]},
        interface_regions={"interface": (2, 9)},
    )
    np.testing.assert_allclose(m.triangle_areas, [1, 1])
    edge = m.edge_sets["interface"][0]
    np.testing.assert_array_equal(m.edge_triangles[edge], [1, 0])
    np.testing.assert_allclose(m.edge_lengths[edge], np.sqrt(5))
    np.testing.assert_allclose(m.edge_normals[edge], [1 / np.sqrt(5), -2 / np.sqrt(5)])
    outer = np.flatnonzero(m.edge_triangles[:, 1] == -1)
    assert len(outer) == 4
    np.testing.assert_allclose(
        (m.edge_normals[outer] * m.edge_lengths[outer, None]).sum(axis=0), 0, atol=1e-14
    )
    with pytest.raises(ValueError):
        m.vertices[0] = 0


def test_invalid_topology():
    v = [[0, 0], [1, 0], [0, 1], [0, -1], [1, 1]]
    for triangles in (
        [[0, 1, 2], [0, 1, 3], [0, 1, 4]],
        [[0, 1, 2], [0, 1, 2]],
        [[0, 0, 1]],
        [[0, 1, 7]],
    ):
        with pytest.raises(ValueError):
            _make_mesh_data(
                v, triangles, np.ones(len(triangles), dtype=int), {1: "material"}
            )
    with pytest.raises(ValueError, match="overlap"):
        _make_mesh_data(v, [[0, 1, 2], [0, 1, 4]], [1, 1], {1: "material"})


def test_combined_mesh_and_charge_rotation():
    c = rectangle_mesh(
        options=MeshOptions(size=0.4),
        exterior_options=ExteriorOptions(outer_scale=2, far_size=1),
    )
    r = solve_exterior(solve_conduction(c))
    fields = r.fields
    m = fields.mesh
    assert m is c.mesh
    assert m is r.conduction.mesh
    np.testing.assert_allclose(m.triangle_areas.sum(), 11 * 8 * 4)
    np.testing.assert_array_equal(
        np.flatnonzero(m.edge_triangles[:, 1] < 0), m.edge_sets["far"]
    )
    cells = m.triangle_sets["conductor"]
    nodes = np.unique(m.triangles[cells])
    np.testing.assert_allclose(
        fields.potential[nodes], r.conduction.potential[nodes] + r.potential_offset
    )
    np.testing.assert_allclose(
        fields.electric_field[cells], r.conduction.electric_field[cells]
    )
    a = calculate_charges(fields)
    # Rotation preserves charges; geometry and vector fields transform together.
    rotation = np.array([[0, -1], [1, 0]])
    rotated = _make_mesh_data(
        m.vertices @ rotation.T + [20, -5],
        m.triangles,
        m.region_ids,
        m.region_names,
        edge_vertices={name: m.edges[ids] for name, ids in m.edge_sets.items()},
        triangle_sets=m.triangle_sets,
        interface_regions=m.interface_regions,
    )
    changed = calculate_charges(
        replace(fields, mesh=rotated, electric_field=fields.electric_field @ rotation.T)
    )
    for name, trace in {**a.surfaces, **a.interfaces}.items():
        np.testing.assert_allclose(
            trace.density,
            {**changed.surfaces, **changed.interfaces}[name].density,
            atol=1e-12,
        )
