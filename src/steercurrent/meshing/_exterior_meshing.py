"""Generate the air mesh inside the cavity and outside the conductor."""
from dataclasses import dataclass
import numpy as np
from .options import ExteriorOptions
from ._gmsh_import import MeshArrays, read_gmsh_mesh
from ._cad import set_mesh_sizes


@dataclass
class _AirImport:
    mesh: MeshArrays


def generate_exterior_mesh(conductor, options=None, *, conductor_size):
    """Fill the cavity and surrounding air with triangles.

    Reuse the conductor's boundary vertices and edges so the meshes fit together.
    """
    source = conductor.mesh
    settings = (options or ExteriorOptions()).resolved(
        span=float(np.ptp(source.vertices, axis=0).max()), conductor_size=conductor_size)
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError(
            "Gmsh is already initialized; generate the exterior outside an active session.")
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('exterior')
        geo = gmsh.model.geo
        nodes = np.unique(source.edge_vertices['conductor'])
        point_ids = {int(i): geo.addPoint(*source.vertices[i], 0) for i in nodes}
        curves, loops = {}, {}
        for name in ('outer', 'cavity'):
            curves[name] = [geo.addLine(point_ids[int(a)], point_ids[int(b)])
                            for a, b in source.edge_vertices[name]]
            found = geo.addCurveLoops(curves[name])
            if len(found) != 1:
                raise ValueError(
                    f"Expected one closed conductor {name} boundary.")
            loops[name] = int(found[0])
        lower, upper = source.vertices.min(axis=0), source.vertices.max(axis=0)
        center, half = (lower+upper)/2, (upper-lower)*settings['outer_scale']/2
        x0, y0 = center-half
        x1, y1 = center+half
        corners = [geo.addPoint(x, y, 0) for x, y in [
            (x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
        far = [geo.addLine(corners[i], corners[(i+1) % 4]) for i in range(4)]
        far_loop = geo.addCurveLoop(far)
        outside = geo.addPlaneSurface([far_loop, loops['outer']])
        cavity = geo.addPlaneSurface([loops['cavity']])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [outside], name='outside')
        gmsh.model.addPhysicalGroup(2, [cavity], name='cavity')
        gmsh.model.addPhysicalGroup(1, curves['outer'], name='conductor_outer')
        gmsh.model.addPhysicalGroup(
            1, curves['cavity'], name='conductor_cavity')
        gmsh.model.addPhysicalGroup(
            1, curves['outer']+curves['cavity'], name='conductor')
        gmsh.model.addPhysicalGroup(1, far, name='far')
        for curve in curves['outer']+curves['cavity']:
            gmsh.model.mesh.setTransfiniteCurve(curve, 2)
        set_mesh_sizes(gmsh, curves=curves['outer']+curves['cavity'],
                       near_size=settings['near_size'], far_size=settings['far_size'],
                       transition_distance=settings['transition_distance'], curve_samples=5)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.model.mesh.generate(2)
        mesh = read_gmsh_mesh(gmsh)
        return _AirImport(mesh)
    finally:
        gmsh.finalize()
