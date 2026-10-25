"""Electromagnetic energy flux and local power fields."""
from dataclasses import dataclass
import numpy as np
from steercurrent.calculations.exterior import ExteriorResult
from steercurrent.calculations.magnetics import MagneticResult


@dataclass
class PoyntingResult:
    exterior: ExteriorResult
    magnetic: MagneticResult
    # (nt,2), triangle mean S (also center value)
    energy_flux: np.ndarray
    divergence: np.ndarray           # (nt,), elementwise div(S)
    # (nt,), J dot E; may be negative in battery
    electrical_power_density: np.ndarray
    joule_density: np.ndarray        # (nt,), rho |J|^2; zero in air
    source_power_density: np.ndarray  # (nt,), J dot F; zero outside battery
    balance_residual: np.ndarray     # (nt,), div(S) + J dot E

    @property
    def mesh(self):
        return self.exterior.mesh

    def edge_fluxes(self):
        """Integrated one-sided S.n on each edge, per unit z-depth.
        Calculates flux from both sides, but keeps them separate.

        Both columns use the mesh normal first->second.
        The second column is NaN at the far boundary (no second triangle).
        Since E is constant and Bz linear per triangle, midpoint Bz is exact.
        """
        mesh = self.mesh
        electric = self.exterior.electric_field
        rotated = np.column_stack((electric[:, 1], -electric[:, 0]))
        bz = self.magnetic.bz[mesh.edges].mean(axis=1)
        result = np.full((len(mesh.edges), 2), np.nan)
        for side in (0, 1):
            cells = mesh.edge_triangles[:, side]
            valid = cells >= 0
            result[valid, side] = (bz[valid]/self.magnetic.permeability *
                                   np.sum(rotated[cells[valid]]*mesh.edge_normals[valid], axis=1)*mesh.edge_lengths[valid])
        return result


def calculate_poynting(exterior, magnetic):
    """Compute S=E cross B/mu and diagnose steady div(S)=-J.E.

    S is linear separately on each triangle; energy_flux stores its mean.
    """
    if exterior.mesh is not magnetic.mesh:
        raise ValueError(
            'Electric and magnetic results must share the same mesh object.')
    mesh = exterior.mesh
    electric = exterior.electric_field
    current = exterior.conduction.current_density
    mu = magnetic.permeability
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError('Permeability must be finite and positive.')
    for array, shape in ((electric, (len(mesh.triangles), 2)), (current, (len(mesh.triangles), 2)),
                         (magnetic.gradient_bz, (len(mesh.triangles), 2)), (magnetic.bz, (len(mesh.vertices),))):
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(
                'Fields must be finite arrays with unified mesh shapes.')
    rotated = np.column_stack((electric[:, 1], -electric[:, 0]))
    flux = magnetic.element_bz[:, None]/mu * rotated
    divergence = np.sum(rotated*magnetic.gradient_bz, axis=1)/mu
    electrical = np.sum(current*electric, axis=1)
    cells = mesh.triangle_sets['conductor']
    joule, source = np.zeros(
        len(mesh.triangles)), np.zeros(len(mesh.triangles))
    joule[cells] = exterior.conduction.resistivity[cells] * \
        np.sum(current[cells]**2, axis=1)
    source[cells] = np.sum(
        current[cells]*exterior.conduction.driving_field[cells], axis=1)
    if not np.all(np.isfinite(joule)) or not np.all(np.isfinite(source)):
        raise ValueError('Conductor power data must be finite.')
    residual = divergence + electrical
    return PoyntingResult(exterior, magnetic, flux,
                          divergence, electrical, joule, source, residual)
