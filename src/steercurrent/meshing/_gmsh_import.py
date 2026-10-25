"""Read Gmsh's result into plain arrays before closing the mesher."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MeshArrays:
    """Temporary mesher output, before building the final mesh's edge adjacency.

    All vertex and triangle IDs are row indices, starting at zero.
    Named edge groups contain pairs of vertex IDs, not Gmsh curve IDs.
    """
    vertices: np.ndarray                          # (nvertices, 2): x, y
    triangles: np.ndarray                         # (ntriangles, 3): vertex IDs
    triangle_sets: dict[str, np.ndarray]          # name -> triangle IDs
    edge_vertices: dict[str, np.ndarray]          # name -> (nedges, 2) vertex IDs


def read_gmsh_mesh(gmsh):
    """Read a generated linear triangle mesh and its named regions and edges.

    Gmsh numbers vertices and elements with tags that may have gaps and may
    arrive out of order. Sort them and translate those tags to array rows here.
    No Gmsh objects or tags are kept in the returned data.
    """
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    node_order = np.argsort(node_tags)
    node_tags = node_tags[node_order]
    vertices = coordinates.reshape(-1, 3)[node_order, :2].copy()
    # Gmsh's numeric codes for two-vertex edges and three-vertex triangles.
    line_type, triangle_type = 1, 2
    triangle_tags, triangle_nodes = gmsh.model.mesh.getElementsByType(triangle_type)
    triangle_order = np.argsort(triangle_tags)
    triangle_tags = triangle_tags[triangle_order]
    triangles = np.searchsorted(node_tags, triangle_nodes.reshape(-1, 3)[triangle_order])
    triangle_sets, edge_vertices = {}, {}
    for dimension, group in gmsh.model.getPhysicalGroups():
        if dimension not in (1, 2):
            continue
        name = gmsh.model.getPhysicalName(dimension, group)
        parts = []
        # A named group may cover several CAD curves or surfaces.
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dimension, group):
            element_type = triangle_type if dimension == 2 else line_type
            tags, nodes = gmsh.model.mesh.getElementsByType(element_type, entity)
            parts.append(np.searchsorted(triangle_tags, tags) if dimension == 2
                         else np.searchsorted(node_tags, nodes).reshape(-1, 2))
        if dimension == 2:
            triangle_sets[name] = np.unique(np.concatenate(parts))
        else:
            # Keep an edge in every group it belongs to, e.g. outer and conductor.
            edge_vertices[name] = np.unique(np.sort(np.concatenate(parts), axis=1), axis=0)
    return MeshArrays(vertices, triangles, triangle_sets, edge_vertices)
