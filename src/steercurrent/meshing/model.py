"""Unified mesh with circuit material and battery metadata."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MeshData:
    """The mesh stored as arrays, with one vertex, triangle, or edge per row.

    Vertex, triangle, and edge IDs are row indices in their corresponding arrays.

    An edge normal points from edge_triangles[:, 0] to [:, 1]. The first
    triangle has the lower region ID (then lower triangle ID). A boundary
    edge has second=-1 and an outward normal. Never index fields with -1.
    Merging assigns air regions higher IDs than conductor regions,
    so shared surface normals point conductor-to-air.
    """
    vertices: np.ndarray             # (nvertices, 2)
    triangles: np.ndarray            # (ntriangles, 3)
    region_ids: np.ndarray           # (ntriangles,)
    edges: np.ndarray                # (nedges, 2), ascending vertex IDs
    # (nedges, 2), second=-1 at domain boundary
    edge_triangles: np.ndarray
    region_names: dict[int, str]
    # keys: Boundary/interface names, such as "outer", "far", "interface_1_2"
    edge_sets: dict[str, np.ndarray]
    # keys: Domain names: "conductor", "outside", "cavity"
    triangle_sets: dict[str, np.ndarray]
    # keys: Material-interface names, such as "interface_1_2"
    interface_regions: dict[str, tuple[int, int]]

    @property
    def edge_lengths(self):
        return np.linalg.norm(self.vertices[self.edges[:, 1]]-self.vertices[self.edges[:, 0]], axis=1)

    @property
    def edge_midpoints(self):
        return self.vertices[self.edges].mean(axis=1)

    @property
    def triangle_centers(self):
        return self.vertices[self.triangles].mean(axis=1)

    @property
    def triangle_areas(self):
        points = self.vertices[self.triangles]
        a, b = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
        return .5*np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])

    @property
    def edge_normals(self):
        tangent = self.vertices[self.edges[:, 1]] - \
            self.vertices[self.edges[:, 0]]
        normals = np.column_stack(
            (tangent[:, 1], -tangent[:, 0]))/self.edge_lengths[:, None]
        # Point away from the first adjacent triangle, toward the second or air.
        direction = self.edge_midpoints - \
            self.triangle_centers[self.edge_triangles[:, 0]]
        normals[np.sum(normals*direction, axis=1) < 0] *= -1
        return normals


@dataclass(frozen=True)
class Region:
    name: str
    material_kind: str
    resistivity: float


@dataclass(frozen=True)
class Battery:
    region_id: int
    emf: float
    length: float
    direction: tuple[float, float]


@dataclass
class CircuitMesh:
    mesh: MeshData
    regions: dict[int, Region]  # conducting materials; air is identified by triangle_sets
    battery: Battery
