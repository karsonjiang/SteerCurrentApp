"""Test-only circuit data; all geometry goes through the generic mesher."""

from steercurrent import (
    CircuitConfig,
    CircuitGeometry,
    LoopShape,
    ResistorSpec,
    generate_mesh,
)

def rectangle_geometry():
    return CircuitGeometry(
        CircuitConfig(length=34, resistors=(ResistorSpec("resistor_1", 7, 10),)),
        LoopShape(((0, 3.5), (0, 7), (10, 7), (10, 0), (0, 0))),
    )


def rectangle_mesh(options=None, **kwargs):
    return generate_mesh(rectangle_geometry(), options, **kwargs)


def rectangle_conductor(options=None, **kwargs):
    from steercurrent.meshing._conductor_meshing import generate_conductor_mesh

    return generate_conductor_mesh(rectangle_geometry(), options, **kwargs)
