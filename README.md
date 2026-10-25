# Steering the Current

This is a local browser app that allows students to explore the physics behind simple circuits.
The app assumes a two-dimensional single-loop circuit. All physical quantities in the model
are invariant along the z direction (the depth direction), and as a result, reported total current,
charge, and power are per unit depth.

The app generates one unified mesh and calculates conduction, exterior electric fields,
interface charge, a magnetic field, and energy flow.
A battery-only charge model is included to motivate the distribution of charge at interfaces.

These results are rendered interactively in the browser. Users can reconfigure the circuit
and see the results update accordingly.

## Install the Python environment

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

## Run the local interactive app

```bash
.venv/bin/python -m steercurrent
```

Open **http://127.0.0.1:8000**. Drag resistors, reshape the loop,
and switch between physics views. Press Ctrl+C to stop.

## Run tests

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## Source structure

| Location        | Responsibility                                                                      |
| ---             | ---                                                                                 |
| `geometry/`     | Circuit configuration, loop shape, geometric validation, and dictionary conversion. |
| `meshing/`      | Unified mesh data, CAD construction, mesh generation, and topology validation.      |
| `calculations/` | Explicit finite element method, field calculations, and result data.                |
| `web/`          | Browser app, request handling, and interactive field display.                       |

Naming convention: `_module.py` is internal to the package.
A leading underscore on a function, method, or class marks a helper used only within its own module
in production; tests may import these helpers without changing their visibility.


## Python usage

### Imports

```python
import json
from pathlib import Path

from steercurrent import (
    CircuitConfig, CircuitGeometry, LoopShape, ResistorSpec
    generate_mesh, MeshOptions, ExteriorOptions,
    solve_conduction, solve_exterior, calculate_charges,
    solve_magnetics, calculate_poynting,
)
```

### Circuit configuration

```python
config = CircuitConfig(length=40, resistors=(
    ResistorSpec('R_corner', start=10, end=14, resistivity=20),
    ResistorSpec('R_bottom', start=26, end=30, resistivity=8),
))
shape = LoopShape(((0, 0), (0, 4), (8, 4), (8, 0),
                   (12, 0), (12, -4), (0, -4)))
geometry = CircuitGeometry(config, shape)
```

`CircuitConfig` stores fixed loop length, conductor width, one battery, resistor intervals, and material resistivities.
`LoopShape` stores an ordered list of points that define an orthogonal centerline.
The first point is the battery center (the battery must be inside a straight section).
Point order defines positive distance along the loop.

`CircuitGeometry(config, shape)` requires the shape's length to match the fixed
loop length within a relative tolerance of `1e-10`.
Changing shape preserves component distances. For example, a resistor can
become L-shaped when a corner moves into its fixed interval.

```python
geometry = CircuitGeometry.from_dict(
    json.loads(Path('tests/orthogonal_loop.json').read_text())
)
```

`to_dict()` and `from_dict()` convert the circuit configuration used by
the browser and tests.

### Mesh and physical options

```python
circuit = generate_mesh(
    geometry, MeshOptions(size=.25, local_size=.125),
    exterior_options=ExteriorOptions(outer_scale=4),
)
```

Conductor `MeshOptions`:

- `size`: global target size, default width/4.
- `local_size`: target near corners/interfaces, default size/2.
- `refinement_distance`: transition distance to global size, default width.

Air `ExteriorOptions`:

- `outer_scale`: far rectangle size relative to conductor bounds, default 4.
- `near_size`: target near conductor, default resolved conductor `size`.
- `far_size`: distant target, default conductor span/4.
- `transition_distance`: near-to-far transition distance, default conductor span.

Gmsh generates triangles, first inside the conductor and then in the exterior air.
Conductor and air share boundary nodes exactly.
The unified `MeshData` is designed so that downstream code is independent of
Gmsh's internal structure.
`MeshData` stores vertices `(nv, 2)`, triangles `(nt, 3)`, region IDs
`(nt,)`, edges `(ne, 2)`, adjacent triangle IDs `(ne, 2)`, and named sets.
The edge normal points from the first adjacent triangle to the second, or
outward at the far domain boundary where the second ID is `-1`.

### Calculations of physical quantities

```python
conduction = solve_conduction(circuit)
exterior = solve_exterior(conduction)
charges = calculate_charges(exterior.fields)
magnetic = solve_magnetics(conduction)
energy = calculate_poynting(exterior, magnetic)
assert circuit.mesh is exterior.mesh is magnetic.mesh
```

Uniform permittivity and permeability default to 1 and can be supplied through
`solve_exterior()` and `solve_magnetics()`.

Conduction and exterior electrostatics assemble P1 triangle equations explicitly
using NumPy; SciPy solves sparse linear systems.
The conductor solve gives potential, E, and J. The exterior solve fills the
cavity and outside, choosing a constant potential offset that imposes total charge
neutrality. Charge density comes from one-sided electric-field jumps. Magnetic recovery
finds continuous Bz consistent with the current in a constrained least-squares
sense, and the final step calculates S = E × B / μ and energy flow.
In this strictly 2D model, Bz is constant in the cavity and zero outside; thus,
the Poynting flux outside is zero.

Detailed standalone derivations are in `doc/`:
[fem](doc/fem.pdf), [conduction](doc/conduction.pdf),
[exterior electrostatics](doc/exterior.pdf), [charges](doc/charges.pdf),
[the battery-only toy model](doc/battery_only.pdf),
[magnetics](doc/magnetics.pdf), and [Poynting flux](doc/poynting.pdf).

## Web app interface

See [the web app guide](src/steercurrent/web/README.md) for editing rules,
limitations, architecture, and browser tests.
