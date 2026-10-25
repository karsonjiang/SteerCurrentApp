"""Ensure the independently usable layers stay independent as the app grows."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "imports, forbidden",
    [
        (
            "from steercurrent import CircuitConfig, LoopShape, CircuitGeometry",
            (
                "gmsh",
                "skfem",
                "scipy",
                "steercurrent.meshing",
                "steercurrent.calculations",
            ),
        ),
        (
            "from steercurrent.calculations import solve_conduction, solve_exterior, solve_magnetics",
            (
                "gmsh",
                "skfem",
                "steercurrent.meshing._conductor_meshing",
            ),
        ),
        (
            "import json; from pathlib import Path; "
            "from steercurrent import CircuitGeometry, generate_mesh, MeshOptions, ExteriorOptions, solve_conduction, solve_exterior, solve_magnetics, calculate_charges, calculate_poynting; "
            "geometry = CircuitGeometry.from_dict(json.loads(Path('tests/orthogonal_loop.json').read_text())); "
            "mesh = generate_mesh(geometry, MeshOptions(size=.5), "
            "exterior_options=ExteriorOptions(outer_scale=2, far_size=2)); "
            "c = solve_conduction(mesh); e = solve_exterior(c); b = solve_magnetics(c); "
            "calculate_charges(e.fields); calculate_poynting(e, b)",
            ("skfem", "meshio"),
        ),
    ],
)
def test_layer_imports(imports, forbidden):
    code = (
        f"{imports}\nimport sys\n"
        f"assert not any(n in sys.modules for n in {forbidden!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_all_public_exports_resolve():
    import steercurrent

    for name in steercurrent.__all__:
        assert getattr(steercurrent, name) is not None
