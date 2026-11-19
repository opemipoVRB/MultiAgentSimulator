# src/calibration.py
"""
Pre-mission calibration: each drone's declared spec is measured, not typed in.

Each drone is run over a short distance twice, unloaded and carrying a known reference payload, outside the mission
clock and outside mission energy. The two battery readings give the declared unloaded consumption per cell and the
declared weight sensitivity. The readings come from the drone's own energy physics (`energy_needed_for_cells`), so a
heterogeneous fleet is measured drone by drone. Optional battery reading resolution and noise (default none, so the
measurement equals the true value).

Limit to state in the paper: the simulator's cost is linear in weight, so two points fit it exactly; real rotor power is
not linear in payload and a real calibration would use more points.
"""
import random
from typing import Dict


def calibrate(drone, cells: float = 10.0, reference_weight: float = 1.0, resolution: float = 0.0,
              noise_std: float = 0.0, rng: random.Random = None) -> Dict[str, float]:
    rng = rng or random.Random(f"calibration-{drone.agent_id}")

    def read(weight: float) -> float:
        e = float(drone.energy_needed_for_cells(cells, weight))
        if noise_std > 0.0:
            e += rng.gauss(0.0, noise_std)
        if resolution > 0.0:
            e = round(e / resolution) * resolution
        return e

    unloaded = read(0.0)
    loaded = read(reference_weight)
    if unloaded <= 0.0:
        raise ValueError("calibration measured no consumption")
    return {
        "unloaded_per_cell": round(unloaded / cells, 9),
        "weight_sensitivity": round((loaded / unloaded - 1.0) / reference_weight, 9),
    }
