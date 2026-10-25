"""Unified mesh data and generation; implementations load only on access."""
from importlib import import_module

_EXPORTS = {
    'CircuitMesh': 'model', 'Region': 'model', 'Battery': 'model',
    'MeshData': 'model',
    'MeshOptions': 'options', 'ExteriorOptions': 'options',
    'generate_mesh': 'generation',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module('.' + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
