"""Mesh generation and validation: region labels, edge normals, and invalid geometry."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from steercurrent import CircuitMesh, CircuitGeometry, CircuitConfig, LoopShape, ResistorSpec
from steercurrent import validate_combined_mesh
from steercurrent.meshing._conductor_meshing import Battery
from steercurrent.meshing._conductor_meshing import generate_conductor_mesh
from steercurrent.meshing._exterior_meshing import generate_exterior_mesh
from steercurrent.meshing._mesh_import import combine_meshes, _make_mesh_data
from steercurrent.meshing.options import MeshOptions, ExteriorOptions
from steercurrent.meshing.validation import (
    _validate_triangle_coverage,
    _boundary_signed_area,
)

from circuit_fixtures import rectangle_conductor


def _complete(conductor):
    air = generate_exterior_mesh(
        conductor, ExteriorOptions(outer_scale=2, far_size=1), conductor_size=0.25
    )
    return CircuitMesh(
        combine_meshes(conductor, air),
        conductor.regions,
        conductor.battery,
    )


@pytest.fixture(scope="module")
def sharp():
    return rectangle_conductor()


@pytest.fixture(scope="module")
def circuit(sharp):
    return _complete(sharp)


def test_reference_geometry_and_physics_contract(circuit):
    data = circuit.mesh
    report = validate_combined_mesh(circuit)
    assert report["region_areas"]["conductor"] == pytest.approx(34)
    for key, expected in [(1, 29), (2, 2), (3, 3)]:
        assert data.triangle_areas[data.region_ids == key].sum() == pytest.approx(
            expected
        )
    nodes = np.unique(data.triangles[data.triangle_sets["conductor"]])
    np.testing.assert_allclose(data.vertices[nodes].min(axis=0), [-0.5, -0.5])
    np.testing.assert_allclose(data.vertices[nodes].max(axis=0), [10.5, 7.5])
    assert circuit.battery == Battery(2, 1, 2, (0, 1))
    assert [circuit.regions[i].resistivity for i in (1, 2, 3)] == [1, 0.1, 20]
    assert len(data.interface_regions) == 2
    for name, regions in data.interface_regions.items():
        edges = data.edge_sets[name]
        ends = data.edges[edges]
        lengths = data.edge_lengths[edges]
        assert lengths.sum() == pytest.approx(2)
        # Two disjoint, width-one cross-sections per component.
        nodes, inverse = np.unique(ends, return_inverse=True)
        endpoints = inverse.reshape(-1, 2)
        adjacency = coo_matrix(
            (np.ones(len(edges)), endpoints.T), shape=(len(nodes), len(nodes))
        )
        n, labels = connected_components(adjacency, directed=False)
        assert n == 2
        for component in range(n):
            assert lengths[labels[endpoints[:, 0]] == component].sum() == pytest.approx(
                1
            )
        first, second = data.edge_triangles[edges].T
        np.testing.assert_array_equal(data.region_ids[first], regions[0])
        np.testing.assert_array_equal(data.region_ids[second], regions[1])
        normals = data.edge_normals[edges]
        centers = data.triangle_centers[second] - data.triangle_centers[first]
        assert np.all(np.sum(centers * normals, axis=1) > 0)
        xy = data.edge_midpoints[edges]
        expected = np.zeros(normals.shape)
        if regions == (1, 3):
            expected[:, 0] = np.where(xy[:, 0] < 5, 1, -1)
        else:
            expected[:, 1] = np.where(xy[:, 1] < 3.5, 1, -1)
        np.testing.assert_allclose(normals, expected, atol=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
def test_boundary_areas_with_translation_and_vertex_winding(circuit, reverse):
    data = circuit.mesh
    triangles = data.triangles[:, ::-1] if reverse else data.triangles
    mesh = replace(data, vertices=data.vertices + [123.0, -456.0], triangles=triangles)
    for name, expected in [("outer", 88), ("cavity", -54)]:
        assert _boundary_signed_area(mesh, mesh.edge_sets[name]) == pytest.approx(
            expected
        )
    validate_combined_mesh(replace(circuit, mesh=mesh))


def test_reversed_boundary_labels_rejected(circuit):
    m = circuit.mesh
    mesh = replace(
        m,
        edge_sets={
            **m.edge_sets,
            "outer": m.edge_sets["cavity"],
            "cavity": m.edge_sets["outer"],
        },
    )
    with pytest.raises(ValueError, match="wrong air region"):
        validate_combined_mesh(replace(circuit, mesh=mesh))


def test_wrong_interface_regions_rejected(circuit):
    m = circuit.mesh
    name = next(iter(m.interface_regions))
    wrong = (1, 3) if m.interface_regions[name] == (1, 2) else (1, 2)
    mesh = replace(m, interface_regions={**m.interface_regions, name: wrong})
    with pytest.raises(ValueError, match="declared regions"):
        validate_combined_mesh(replace(circuit, mesh=mesh))


def test_folded_cavity_geometry_rejected(circuit):
    m = circuit.mesh
    interior = np.setdiff1d(
        np.unique(m.triangles[m.triangle_sets["cavity"]]),
        np.unique(m.edges[m.edge_sets["cavity"]]),
    )
    node = interior[
        np.argmin(np.linalg.norm(m.vertices[interior] - [5.0, 3.5], axis=1))
    ]
    vertices = m.vertices.copy()
    vertices[node] += [20.0, 0.0]
    with pytest.raises(ValueError, match="enclosing boundaries"):
        validate_combined_mesh(replace(circuit, mesh=replace(m, vertices=vertices)))


def test_merge_rejects_bad_mapping_and_incomplete_air_partition(sharp):
    air = generate_exterior_mesh(
        sharp, ExteriorOptions(outer_scale=2, far_size=1), conductor_size=0.25
    )
    vertices = air.mesh.vertices.copy()
    boundary_node = air.mesh.edge_vertices["conductor"][0, 0]
    vertices[boundary_node] += [0.01, 0.0]
    with pytest.raises(ValueError, match="preserve all conductor boundary vertices"):
        combine_meshes(sharp, replace(air, mesh=replace(air.mesh, vertices=vertices)))
    mesh = replace(
        air.mesh,
        triangle_sets={
            **air.mesh.triangle_sets,
            "outside": air.mesh.triangle_sets["outside"][1:],
        },
    )
    with pytest.raises(ValueError, match="partition"):
        combine_meshes(sharp, replace(air, mesh=mesh))


def test_refinement():
    coarse = _complete(rectangle_conductor(MeshOptions(size=0.4)))
    fine = _complete(rectangle_conductor(MeshOptions(size=0.2)))
    assert len(fine.mesh.triangle_sets["conductor"]) > len(
        coarse.mesh.triangle_sets["conductor"]
    )
    for circuit in (coarse, fine):
        report = validate_combined_mesh(circuit)
        assert report["region_areas"]["conductor"] == pytest.approx(34)
        for key, expected in [(1, 29), (2, 2), (3, 3)]:
            m = circuit.mesh
            assert m.triangle_areas[m.region_ids == key].sum() == pytest.approx(
                expected
            )


def test_resized_loop_closes_without_extra_interface():
    # These edited U-loop coordinates made np.sum(lengths) differ from the last
    # cumulative distance. The mismatch used to duplicate the battery-center edge.
    shape = LoopShape(((0, 0), (0, 6.145951819140464),
                       (4, 6.145951819140464), (4, 1), (6, 1),
                       (6, 6.145951819140464),
                       (7.10000000521077, 6.145951819140464),
                       (7.10000000521077, -6.705024186521769),
                       (0, -6.705024186521769)))
    geometry = CircuitGeometry(
        CircuitConfig(length=shape.length, resistors=(
            ResistorSpec('R_corner', 14.291903638280928, 18.291903638280928),
            ResistorSpec('R_bottom', 35.17772035776226, 39.17772035776226),
        )), shape)
    assert shape.length == shape.distances[-1]
    conductor = generate_conductor_mesh(geometry, MeshOptions(size=0.5))
    assert set(conductor.interface_regions) == {'interface_1_2', 'interface_1_3', 'interface_1_4'}


@pytest.mark.parametrize(
    "options",
    [
        MeshOptions(size=0),
        MeshOptions(local_size=1),
        MeshOptions(refinement_distance=-1),
    ],
)
def test_invalid_mesh_options(options):
    with pytest.raises(ValueError):
        rectangle_conductor(options=options)


@pytest.mark.parametrize(
    "second, valid",
    [
        ([(0, 0), (2, 0), (2, -2)], True),  # Full shared edge.
        ([(2, 0), (3, 0), (2, 1)], True),  # Shared vertex.
        ([(3, 0), (4, 0), (3, 1)], True),  # Separate triangles.
        ([(0.2, 0.2), (1, 0.2), (0.2, 1)], False),  # Containment.
        ([(1, 0), (3, 0), (1, 2)], False),  # Partial overlap.
        ([(0, 0), (2, 0), (0, 2)], False),  # Duplicate triangle.
        ([(0, 0), (1, 0), (0, -1)], False),  # Part of a shared edge.
        ([(1, 0), (2, -1), (0, -1)], False),  # Vertex in another edge.
        ([(2 - 1e-12, 0), (3, 0), (2 - 1e-12, 1)], False),  # Tiny overlap.
    ],
)
def test_triangle_coverage(second, valid):
    vertices = np.array([(0, 0), (2, 0), (0, 2), *second], dtype=float)
    mesh = _make_mesh_data(vertices, [[0, 1, 2], [3, 4, 5]], [1, 1], {1: "test"})
    if valid:
        _validate_triangle_coverage(mesh)
    else:
        with pytest.raises(ValueError, match="overlap or their shared edges"):
            _validate_triangle_coverage(mesh)
