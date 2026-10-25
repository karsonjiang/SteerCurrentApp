"""Shared straight-edge CAD construction and Gmsh mesh import."""
from ._gmsh_import import read_gmsh_mesh


class CAD:
    def __init__(self, gmsh):
        self.g = gmsh.model.geo
        self.points = {}
        self.curves = {}
        self.surfaces = {}
        self.uses = {}

    def _point(self, xy):
        key = tuple(round(float(x), 12) for x in xy)
        if key not in self.points:
            self.points[key] = self.g.addPoint(*key, 0)
        return self.points[key]

    def curve(self, start, end):
        p, q = self._point(start), self._point(end)
        key = (min(p, q), max(p, q))
        if key not in self.curves:
            self.curves[key] = (self.g.addLine(p, q), p)
        tag, first = self.curves[key]
        return tag if p == first else -tag

    def surface(self, edges, region):
        tag = self.g.addPlaneSurface([self.g.addCurveLoop(edges)])
        self.surfaces.setdefault(region, []).append(tag)
        for edge in edges:
            self.uses.setdefault(abs(edge), []).append(region)


def set_mesh_sizes(gmsh, *, points=(), curves=(), near_size, far_size,
                   transition_distance, curve_samples=100):
    """Use small triangles near selected points/curves, growing with distance.

    curve_samples controls how many points Gmsh samples along each curve when
    estimating distance. Short mesh edges need fewer samples than long CAD curves.
    """
    gmsh.option.setNumber("Mesh.MeshSizeMin", near_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", far_size)
    # Use this size rule rather than Gmsh's other automatic size estimates.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "PointsList", list(points))
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", list(curves))
    gmsh.model.mesh.field.setNumber(distance, "Sampling", curve_samples)
    threshold = gmsh.model.mesh.field.add("Threshold")
    for key, value in {"InField": distance, "SizeMin": near_size,
                       "SizeMax": far_size, "DistMin": 0,
                       "DistMax": transition_distance}.items():
        gmsh.model.mesh.field.setNumber(threshold, key, value)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)


def mesh_cad(gmsh, cad, settings, interfaces):
    """Set triangle sizes, generate the mesh, and load its named regions and edges."""
    set_mesh_sizes(gmsh, points=cad.points.values(),
                   curves=[curve for curves in interfaces.values() for curve in curves],
                   near_size=settings['local_size'], far_size=settings['size'],
                   transition_distance=settings['refinement_distance'])
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)
    return read_gmsh_mesh(gmsh)
