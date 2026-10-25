"""Charge totals used by scaling, neutrality, and vertex-renumbering tests."""

import numpy as np


def charge_diagnostics(result):
    """Integrate each charge trace and measure net charge relative to its magnitude."""
    traces = {**result.surfaces, **result.interfaces}
    totals = {
        name: float(trace.charge_per_depth.sum()) for name, trace in traces.items()
    }
    total = sum(totals.values())
    absolute = sum(
        float(np.abs(trace.charge_per_depth).sum()) for trace in traces.values()
    )
    return {
        "charge_per_depth_by_group": totals,
        "total_charge_per_depth": total,
        "relative_charge_imbalance": abs(total) / absolute if absolute else 0.0,
    }
