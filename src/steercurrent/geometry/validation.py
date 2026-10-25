"""Geometry constraints, independent of meshing and field calculations."""
import math
import numpy as np


def validate_shape(self):
    p = np.asarray(self.points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 5 or not np.isfinite(p).all():
        raise ValueError(
            "Loop needs at least five finite (x, y) points, starting inside a straight section.")
    object.__setattr__(self, 'points', tuple(map(tuple, p.tolist())))
    d = np.roll(p, -1, axis=0) - p
    if not np.all(np.sum(d != 0, axis=1) == 1):
        raise ValueError(
            "Loop segments must be nonzero and axis-aligned; do not repeat the closing point.")
    t = d / np.linalg.norm(d, axis=1)[:, None]
    if not np.array_equal(t[0], t[-1]):
        raise ValueError(
            "The battery center must lie inside a straight section.")
    if np.any(np.sum(t * np.roll(t, 1, axis=0), axis=1) == -1):
        raise ValueError("Loop cannot retrace a segment.")
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if j == i + 1 or (i == 0 and j == len(p) - 1):
                continue
            a, b = p[i], p[(i + 1) % len(p)]
            c, e = p[j], p[(j + 1) % len(p)]
            if np.all(np.maximum(np.minimum(a, b), np.minimum(c, e)) <=
                      np.minimum(np.maximum(a, b), np.maximum(c, e))):
                raise ValueError(f"Loop segments {i} and {
                                 j} intersect or touch.")


def validate_config(self):
    b, L = self.battery, self.length
    if not all(math.isfinite(v) and v > 0 for v in
               (L, self.width, self.wire_resistivity, b.length, b.resistivity)):
        raise ValueError(
            "Loop length, width, battery length, and resistivities must be positive and finite.")
    if not math.isfinite(b.emf) or b.polarity not in (-1, 1) or b.length >= L:
        raise ValueError(
            "Battery requires finite EMF, polarity +1/-1, and length less than the loop.")
    names = {'wire', 'battery', 'outside', 'cavity', 'conductor', 'far'}
    previous = b.length / 2
    for r in sorted(self.resistors, key=lambda r: r.start):
        if (not isinstance(r.id, str) or not r.id.isidentifier() or r.id in names or
                r.id.startswith('interface_')):
            raise ValueError(
                "Resistor IDs must be unique identifiers and not reserved mesh names.")
        names.add(r.id)
        if not all(math.isfinite(v) for v in (r.start, r.end, r.resistivity)) or r.resistivity <= 0:
            raise ValueError(
                "Resistor distances must be finite and resistivity positive.")
        if not previous < r.start < r.end < L - b.length / 2:
            raise ValueError(
                "Components must not overlap or touch and must leave connecting wire.")
        previous = r.end


def validate_geometry(self):
    self.config.validate()
    c, shape = self.config, self.shape
    if not math.isclose(shape.length, c.length, rel_tol=1e-10, abs_tol=0.):
        raise ValueError(f"Shape length {
                         shape.length:g} does not match fixed circuit length {c.length:g}.")
    b, L, a = c.battery, shape.length, c.width / 2
    t = shape.tangents
    corners = shape.distances[:-1][np.any(t != np.roll(t, 1, axis=0), axis=1)]
    if np.any(np.minimum(corners, L - corners) <= b.length / 2 + a):
        raise ValueError(
            "Battery must fit on a straight section with clearance from corners.")
    for r in c.resistors:
        if any(np.any(np.abs(corners - s) <= a) for s in (r.start, r.end)):
            raise ValueError(
                "Component endpoints must be more than width/2 away from corners.")
    polygons = self.strip_polygons()
    for i, p in enumerate(polygons):
        for j in range(i + 1, len(polygons)):
            q = polygons[j]
            adjacent = j == i + 1 or (i == 0 and j == len(polygons) - 1)
            edges = np.concatenate(
                (np.roll(p, -1, axis=0) - p, np.roll(q, -1, axis=0) - q))
            axes = np.column_stack((-edges[:, 1], edges[:, 0]))
            axes /= np.linalg.norm(axes, axis=1)[:, None]
            pa, qa = p @ axes.T, q @ axes.T
            overlap = np.minimum(pa.max(0), qa.max(0)) - \
                np.maximum(pa.min(0), qa.min(0))
            tol = 1e-10 * max(self.config.width, self.shape.length)
            if np.all(overlap > tol if adjacent else overlap >= -tol):
                raise ValueError(f"Conductor strips {i} and {
                                 j} overlap or touch; reduce width or reshape loop.")
