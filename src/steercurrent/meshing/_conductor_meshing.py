"""Build conductor surfaces from the loop geometry and mesh them with Gmsh."""
import numpy as np
from dataclasses import dataclass
from ._gmsh_import import MeshArrays
from steercurrent.meshing._cad import CAD
from steercurrent.meshing._cad import mesh_cad
from steercurrent.meshing.options import MeshOptions
from steercurrent.meshing.model import Battery
from steercurrent.meshing.model import Region


@dataclass
class _ConductorImport:
    mesh: MeshArrays
    regions: dict[int, Region]
    battery: Battery
    interface_regions: dict[str, tuple[int, int]]

    @property
    def region_ids(self):
        """Return one region ID per triangle, using the named triangle groups."""
        labels = np.empty(len(self.mesh.triangles), dtype=np.int64)
        coverage = np.zeros(len(self.mesh.triangles), dtype=np.int64)
        for key, region in self.regions.items():
            elements = self.mesh.triangle_sets[region.name]
            labels[elements] = key
            np.add.at(coverage, elements, 1)
        if np.any(coverage != 1):
            raise ValueError(
                "Each triangle must belong to exactly one circuit region.")
        return labels


def _make_conductor_mesh(mesh, regions, battery, interface_regions):
    """Attach material and battery information to the mesher's arrays."""
    if len({r.name for r in regions.values()}) != len(regions) or any(':' in r.name for r in regions.values()):
        raise ValueError("Region names must be unique and cannot contain ':'.")
    if not mesh.triangle_sets or any(r.name not in mesh.triangle_sets or not len(mesh.triangle_sets[r.name])
                                  for r in regions.values()):
        raise ValueError(
            "Each circuit region needs a nonempty named triangle set.")
    if not mesh.edge_vertices or not {"outer", "cavity", "conductor"} <= mesh.edge_vertices.keys():
        raise ValueError(
            "Mesh needs outer, cavity, and conductor boundary labels.")
    # The air mesher needs the endpoints of every outer and cavity edge.
    boundary = np.concatenate([mesh.edge_vertices[name] for name in ('outer', 'cavity')])
    if not len(boundary) or set(map(tuple, boundary)) != set(map(tuple, mesh.edge_vertices['conductor'])):
        raise ValueError('Outer/cavity labels must partition the conductor boundary.')
    conductor = _ConductorImport(mesh, regions, battery, interface_regions)
    conductor.region_ids  # Check that every triangle belongs to exactly one region.
    return conductor


def generate_conductor_mesh(geometry, options=None):
    geometry.validate()
    config = geometry.config
    loop, b = geometry.shape, config.battery
    settings = (options or MeshOptions()).resolved(config.width)
    regions = {1: Region('wire', 'wire', config.wire_resistivity),
               2: Region('battery', 'battery', b.resistivity)}
    spans = [(0., b.length / 2, 2),
             (loop.length - b.length / 2, loop.length, 2)]
    for key, r in enumerate(config.resistors, 3):
        regions[key] = Region(r.id, 'resistor', r.resistivity)
        spans.append((r.start, r.end, key))
    cuts = sorted({s for start, end, _ in spans for s in (start, end)})
    p = np.asarray(loop.points)
    # Left of a CCW traversal is the cavity; right is the outer boundary.
    # CCW: counter clock wise or not
    ccw = np.sum(p[:, 0] * np.roll(p[:, 1], -1) -
                 p[:, 1] * np.roll(p[:, 0], -1)) > 0
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError(
            'Gmsh is already initialized; call outside an active session.')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('loop_circuit')
        cad = CAD(gmsh)
        cad.surfaces = {key: [] for key in regions}
        outer, cavity = [], []
        for i, polygon in enumerate(geometry.strip_polygons()):
            # split each conductor-strip polygon at battery/resistor endpoints, create a CAD surface for each piece, and label its boundary edges
            # find where to split the segment
            start, end = loop.distances[i:i + 2]
            distances = [start] + [s for s in cuts if start < s < end] + [end]
            # construct the cross-section
            normal = np.array(
                [-loop.tangents[i, 1], loop.tangents[i, 0]]) * config.width / 2
            sections = [(polygon[0], polygon[3])]
            sections += [(loop.position(s) - normal, loop.position(s) + normal)
                         for s in distances[1:-1]]
            sections.append((polygon[1], polygon[2]))
            for k, (lo, hi) in enumerate(zip(distances[:-1], distances[1:])):
                # assign each piece a material
                mid = (lo + hi) / 2
                region = next((key for a, z, key in spans if a <= mid < z), 1)
                # build the quadrilateral
                right0, left0 = sections[k]
                right1, left1 = sections[k + 1]
                right = cad.curve(right0, right1)
                left = cad.curve(left1, left0)
                cad.surface([right, cad.curve(right1, left1),
                            left, cad.curve(left0, right0)], region)
                # For a counterclockwise loop, the right side faces the outside, and the left side faces the cavity.
                # For a clockwise loop, those roles reverse.
                (outer if ccw else cavity).append(abs(right))
                (cavity if ccw else outer).append(abs(left))
        cad.g.synchronize()
        for key, region in regions.items():
            gmsh.model.addPhysicalGroup(2, cad.surfaces[key], key, region.name)
        interfaces = {}
        # if a curve is used by more than one region, add to interfaces
        for curve, uses in cad.uses.items():
            if len(set(uses)) > 1:
                interfaces.setdefault(tuple(sorted(uses)), []).append(curve)
        for tag, name, curves in [(11, 'outer', outer), (12, 'cavity', cavity),
                                  (13, 'conductor', outer + cavity)]:
            gmsh.model.addPhysicalGroup(1, curves, tag, name)
        for tag, (pair, curves) in enumerate(sorted(interfaces.items()), 20):
            gmsh.model.addPhysicalGroup(
                1, curves, tag, f'interface_{pair[0]}_{pair[1]}')
        mesh = mesh_cad(gmsh, cad, settings, interfaces)
        direction = tuple(b.polarity * loop.direction(0))
        return _make_conductor_mesh(mesh, regions, Battery(2, b.emf, b.length, direction),
                                    {f'interface_{a}_{b}': (a, b) for a, b in interfaces})
    finally:
        gmsh.finalize()
