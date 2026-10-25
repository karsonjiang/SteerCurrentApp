"""Build the complete unified conductor, cavity, and outside mesh."""
from steercurrent.meshing.model import CircuitMesh
from .options import MeshOptions


def generate_mesh(geometry, options=None, *, exterior_options=None):
    """Mesh a sharp CircuitGeometry, including its cavity and exterior."""
    from steercurrent.meshing._conductor_meshing import generate_conductor_mesh
    from steercurrent.meshing._exterior_meshing import generate_exterior_mesh
    from steercurrent.meshing._mesh_import import combine_meshes
    from steercurrent.meshing.validation import validate_combined_mesh
    options = options or MeshOptions()
    conductor_size = options.resolved(geometry.config.width)["size"]
    conductor = generate_conductor_mesh(geometry, options)
    exterior = generate_exterior_mesh(
        conductor, exterior_options, conductor_size=conductor_size)
    result = CircuitMesh(combine_meshes(conductor, exterior), conductor.regions, conductor.battery)
    validate_combined_mesh(result)
    return result
