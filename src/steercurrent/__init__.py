"""Circuit geometry, meshing, and calculations public API.
"""
from importlib import import_module

_EXPORTS = {'MeshOptions': 'meshing.options',
            'ExteriorOptions': 'meshing.options',
            'Region': 'meshing.model',
            'Battery': 'meshing.model',
            'CircuitMesh': 'meshing.model',
            'MeshData': 'meshing.model',
            'FieldData': 'calculations.fields',
            'validate_combined_mesh': 'meshing.validation',
            'ConductionResult': 'calculations.conduction',
            'solve_conduction': 'calculations.conduction',
            'ExteriorResult': 'calculations.exterior',
            'solve_exterior': 'calculations.exterior',
            'ChargeTrace': 'calculations.charges',
            'ChargeResult': 'calculations.charges',
            'calculate_charges': 'calculations.charges',
            'MagneticResult': 'calculations.magnetics',
            'solve_magnetics': 'calculations.magnetics',
            'PoyntingResult': 'calculations.poynting',
            'calculate_poynting': 'calculations.poynting',
            'LoopShape': 'geometry.shape',
            'CircuitConfig': 'geometry.configuration',
            'CircuitGeometry': 'geometry.circuit',
            'BatterySpec': 'geometry.configuration',
            'ResistorSpec': 'geometry.configuration',
            'generate_mesh': 'meshing.generation'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
