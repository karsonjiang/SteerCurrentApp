"""User geometry validation and integration with field calculations."""

from dataclasses import replace
import json

import numpy as np
import pytest

from steercurrent import (
    LoopShape,
    CircuitConfig,
    CircuitGeometry,
    BatterySpec,
    ResistorSpec,
    generate_mesh,
    MeshOptions,
    ExteriorOptions,
    solve_conduction,
    solve_exterior,
    calculate_charges,
    solve_magnetics,
    calculate_poynting,
)

from circuit_fixtures import rectangle_mesh
from physics_diagnostics.conduction import conduction_diagnostics
from physics_diagnostics.poynting import poynting_diagnostics


POINTS = ((0, 0), (0, 4), (8, 4), (8, 0), (12, 0), (12, -4), (0, -4))


def example():
    # R_corner spans the turn at s=12; R_bottom is straight.
    return CircuitGeometry(
        CircuitConfig(
            length=40,
            resistors=(
                ResistorSpec("R_corner", 10, 14),
                ResistorSpec("R_bottom", 26, 30, 8),
            ),
        ),
        LoopShape(POINTS),
    )


def test_coordinates_and_json():
    g = example()
    assert g.shape.length == 40
    np.testing.assert_array_equal(g.shape.position(12), [8, 4])
    np.testing.assert_array_equal(g.shape.direction(12), [0, -1])
    np.testing.assert_array_equal(g.shape.position(-1), [0, -1])
    np.testing.assert_array_equal(g.shape.position(40), [0, 0])
    assert CircuitGeometry.from_dict(json.loads(json.dumps(g.to_dict()))) == g
    with pytest.raises(ValueError, match="finite"):
        g.shape.position(float("nan"))


@pytest.mark.parametrize(
    "points",
    [
        ((0, 0), (1, 4), (8, 4), (8, -4), (0, -4)),  # diagonal
        ((0, 0), (0, 4), (8, 4), (8, -4), (0, -4), (0, 0)),  # repeated closure
        ((0, 0), (0, 4), (8, 4), (8, 0), (-2, 0), (-2, -4), (0, -4)),  # crossing
        ((0, 0), (0, 4), (8, 4), (8, -4), (4, -4)),  # diagonal closing edge
        ((0, 0), (0, 4), (0, 2), (8, 2), (8, -4), (0, -4)),  # retrace
    ],
)
def test_invalid_loops(points):
    with pytest.raises(ValueError):
        LoopShape(points)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": -1},
        {"width": 5},
        {"battery": BatterySpec(length=8)},
        {"battery": BatterySpec(polarity=0)},
        {"resistors": (ResistorSpec("R", 11.8, 14),)},  # near corner
        {"resistors": (ResistorSpec("R", 0.5, 2),)},  # battery overlap
        {"resistors": (ResistorSpec("R", 6, 9), ResistorSpec("S", 8, 10))},
        {"resistors": (ResistorSpec("R", 6, 9), ResistorSpec("R", 26, 30))},
        {"resistors": (ResistorSpec("outside", 6, 9),)},
        {"resistors": (ResistorSpec("R", 6, 9, float("nan")),)},
    ],
)
def test_invalid_circuits(changes):
    with pytest.raises(ValueError):
        replace(example(), config=replace(example().config, **changes))


def test_thick_strip_contact():
    # Long legs, but two nonadjacent strips close the narrow indentation.
    loop = LoopShape(
        ((0, 0), (0, 8), (10, 8), (10, 5), (1, 5), (1, 2), (10, 2), (10, -8), (0, -8))
    )
    with pytest.raises(ValueError, match="overlap or touch"):
        CircuitGeometry(CircuitConfig(length=loop.length, width=1), loop)


@pytest.mark.parametrize("mirror", [1, -1])
def test_nonrectangular_all_steps(mirror):
    g = example()
    g = replace(g, shape=LoopShape(tuple((mirror * x, y) for x, y in g.shape.points)))
    circuit = generate_mesh(
        g,
        MeshOptions(size=0.5),
        exterior_options=ExteriorOptions(outer_scale=2, far_size=2),
    )
    m = circuit.mesh
    assert m.triangle_areas[m.triangle_sets["conductor"]].sum() == pytest.approx(
        g.shape.length * g.config.width
    )
    for key, r in circuit.regions.items():
        if r.material_kind == "resistor":
            spec = next(s for s in g.config.resistors if s.id == r.name)
            assert m.triangle_areas[m.region_ids == key].sum() == pytest.approx(
                (spec.end - spec.start) * g.config.width
            )
    c = solve_conduction(circuit)
    e = solve_exterior(c)
    q = calculate_charges(e.fields)
    b = solve_magnetics(c)
    energy = calculate_poynting(e, b)
    assert energy.mesh is q.fields.mesh is m
    assert np.isfinite(e.electric_field).all()
    assert poynting_diagnostics(energy)["global_power_balance_relative_error"] < 1e-9


def test_reference_and_polarity():
    g = CircuitGeometry(
        CircuitConfig(length=34, resistors=(ResistorSpec("resistor_1", 7, 10),)),
        LoopShape(((0, 0), (0, 3.5), (10, 3.5), (10, -3.5), (0, -3.5))),
    )
    opts = MeshOptions(size=0.35)
    air = ExteriorOptions(outer_scale=2, far_size=2)
    mesh = generate_mesh(g, opts, exterior_options=air)
    c = solve_conduction(mesh)
    reference = solve_conduction(rectangle_mesh(options=opts, exterior_options=air))
    # Different CAD partitions/triangulations, same continuum circuit.
    assert conduction_diagnostics(c)["source_average_current"] == pytest.approx(
        conduction_diagnostics(reference)["source_average_current"], rel=0.003
    )
    reverse = replace(mesh, battery=replace(mesh.battery, direction=(0.0, -1.0)))
    np.testing.assert_allclose(
        solve_conduction(reverse).potential, -c.potential, atol=1e-12
    )


def test_wire_only_and_horizontal_battery():
    loop = LoopShape(((0, 0), (4, 0), (4, 6), (-4, 6), (-4, 0)))
    g = CircuitGeometry(
        CircuitConfig(length=28, battery=BatterySpec(polarity=-1)), loop
    )
    mesh = generate_mesh(
        g,
        MeshOptions(size=0.6),
        exterior_options=ExteriorOptions(outer_scale=2, far_size=2),
    )
    assert mesh.battery.direction == (-1.0, 0.0)
    assert {r.material_kind for r in mesh.regions.values()} == {"wire", "battery"}
    assert conduction_diagnostics(solve_conduction(mesh))["source_average_current"] > 0


def test_fixed_config_can_be_embedded_in_different_shapes():
    bent = example()
    # Same 40 units and material spans, but the corner resistor is now straight.
    straight = replace(
        bent, shape=LoopShape(((0, 0), (0, 4), (12, 4), (12, -4), (0, -4)))
    )
    assert straight.config is bent.config
    assert straight.config.resistors == bent.config.resistors
    np.testing.assert_array_equal(bent.shape.position(13), [8, 3])
    np.testing.assert_array_equal(straight.shape.position(13), [9, 4])
    for geometry in (bent, straight):
        circuit = generate_mesh(
            geometry,
            MeshOptions(size=0.6),
            exterior_options=ExteriorOptions(outer_scale=2, far_size=2),
        )
        assert (
            conduction_diagnostics(solve_conduction(circuit))["source_average_current"]
            > 0
        )


def test_length_mismatch_rejected_without_scaling():
    g = example()
    shape = LoopShape(((0, 0), (0, 4), (13, 4), (13, -4), (0, -4)))
    original = shape.points
    with pytest.raises(ValueError, match="does not match fixed circuit length"):
        replace(g, shape=shape)
    assert shape.points == original
    assert shape.length == 42


def test_length_tolerance_and_config_serialization():
    g = example()
    config = replace(g.config, length=40 + 1e-10)
    accepted = replace(g, config=config)
    assert accepted.shape is g.shape  # no normalization or scaling
    assert CircuitConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="fixed circuit length"):
        replace(g, config=replace(g.config, length=40 + 1e-6))
    for length in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            CircuitConfig(length=length)


def test_shape_deformation_keeps_battery_straight_and_component_cuts_clear():
    g = example()
    # Both shapes preserve length; the first brings a turn into the battery,
    # the second puts a turn on the resistor's start distance.
    with pytest.raises(ValueError, match="Battery must fit"):
        replace(g, shape=LoopShape(((0, 0), (0, 1), (16, 1), (16, -3), (0, -3))))
    with pytest.raises(ValueError, match="Component endpoints"):
        replace(g, shape=LoopShape(((0, 0), (0, 4), (6, 4), (6, -10), (0, -10))))
