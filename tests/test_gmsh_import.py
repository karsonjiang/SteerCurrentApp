"""The mesher's IDs and overlapping labels must survive conversion to arrays."""

import gmsh
import numpy as np

from steercurrent.meshing._gmsh_import import read_gmsh_mesh


def test_sparse_unordered_ids_and_overlapping_groups():
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("import-test")
        for surface in (101, 102):
            gmsh.model.addDiscreteEntity(2, surface)
        # A unit square, deliberately using shuffled vertex IDs with gaps.
        gmsh.model.mesh.addNodes(
            2, 101, [40, 10, 70, 20], [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
        )
        gmsh.model.mesh.addElementsByType(101, 2, [900], [10, 40, 70])
        gmsh.model.mesh.addElementsByType(102, 2, [500], [10, 70, 20])
        gmsh.model.addPhysicalGroup(2, [101], 11, "wire")
        gmsh.model.addPhysicalGroup(2, [102], 12, "battery")
        gmsh.model.addPhysicalGroup(2, [101, 102], 13, "all")
        gmsh.model.addDiscreteEntity(1, 201)
        gmsh.model.mesh.addElementsByType(201, 1, [800], [40, 10])
        # Same edge in two groups; curve tag 11 also coexists with surface tag 11.
        gmsh.model.addPhysicalGroup(1, [201], 11, "outer")
        gmsh.model.addPhysicalGroup(1, [201], 13, "conductor")
        mesh = read_gmsh_mesh(gmsh)
    finally:
        gmsh.finalize()
    np.testing.assert_array_equal(mesh.vertices, [[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_array_equal(mesh.triangles, [[0, 3, 1], [0, 2, 3]])
    for name, expected in [("wire", [1]), ("battery", [0]), ("all", [0, 1])]:
        np.testing.assert_array_equal(mesh.triangle_sets[name], expected)
    for name in ("outer", "conductor"):
        np.testing.assert_array_equal(mesh.edge_vertices[name], [[0, 2]])
