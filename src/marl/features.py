# src/marl/features.py
"""
Features of the learned task-selection policy (the communication-free MARL baseline, see .ignore/revision/marl_baseline_design.md).

One function builds them for training and for execution, so a trained policy sees exactly what it was trained on. Everything comes from
what the drone itself holds: its own state, its memory of the parcels (location, weight, steps since it last confirmed the parcel in
place), what it last sensed of each peer, the declared specs and start cell of every drone (shared start information), and its own
sensor characteristics. No message, no heuristic given as a feature (no territory label, no cost-based proposal).

Three sets, all normalised to roughly [-1, 1]:
  self   (SELF_DIM)  the drone
  tasks  (n, TASK_DIM)  one row per believed-open parcel the drone can afford (feasibility is the shared trip-energy contract)
  peers  (m, PEER_DIM)  one row per peer
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SELF_DIM = 14
TASK_DIM = 11
PEER_DIM = 9
STEP_REF = 2000.0        # steps, the scale of times and ages when no ideal time is given (version 1)


def ideal_time(parcels, station_center, speeds, distance, steps_per_cell: float) -> float:
    """The ideal finish time of the mission from the declared start information: all flight work (station to pickup and back) over the fleet's total declared speed. Every drone computes the same value."""
    work = sum(2.0 * float(distance(station_center, (p["col"], p["row"]))) for p in parcels) * steps_per_cell
    return max(1.0, work / max(1e-9, sum(speeds)))


def build(snapshot: Dict, *, distance: Callable, trip_energy: Callable, battery: float, floor: float, capacity: float,
          unloaded_per_cell: float, weight_sensitivity: float, grid: Tuple[int, int], peer_memory: Dict[str, Tuple],
          sensing_limited: bool, sensing_noisy: bool, station_center: Tuple[int, int],
          own_speed: float = 1.0, peer_speed: Optional[Dict[str, float]] = None, steps_per_cell: float = 14.29,
          t_ideal: Optional[float] = None, n_total: Optional[int] = None
          ) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], np.ndarray]:
    """
    Returns (self vector, task ids, task matrix, peer ids, peer matrix).

    peer_memory: peer id -> (col, row, step last sensed, start col, start row); the position is the start cell until the
    peer is first sensed.
    """
    agent = snapshot["agent"]
    me = (agent["col"], agent["row"])
    step = float(snapshot.get("step", 0))
    cols, rows = float(grid[0]), float(grid[1])
    span = cols + rows
    usable = battery - floor
    t_ref = float(t_ideal) if t_ideal else STEP_REF                 # times and ages are read against the mission's own ideal time, so the same value means the same in every configuration
    n_open = sum(1 for p in snapshot.get("all_parcels", []) if not (p.get("picked") or p.get("delivered")))
    n_fleet = 1 + len(peer_memory)

    x_self = np.array([
        battery / max(1e-9, capacity),
        usable / max(1e-9, capacity),
        1.0 if snapshot.get("carrying", {}).get("has") else 0.0,
        me[0] / cols,
        me[1] / rows,
        unloaded_per_cell / 0.2 - 1.0,
        weight_sensitivity - 0.5,
        1.0 if sensing_limited else 0.0,
        1.0 if sensing_noisy else 0.0,
        min(2.0, step / t_ref),
        own_speed - 1.0,                                    # declared flight speed relative to the nominal
        own_speed / max(1e-9, float(np.mean([own_speed] + list((peer_speed or {}).values())))) - 1.0,     # relative to the fleet's mean speed
        min(2.0, n_open / max(1, n_fleet) / 5.0),                                                          # open parcels per drone
        (1.0 - n_open / max(1, n_total)) if n_total else 0.0,                                              # share of the parcels already taken
    ], dtype=np.float32)

    peer_ids = sorted(peer_memory)
    peer_pos = {pid: (peer_memory[pid][0], peer_memory[pid][1]) for pid in peer_ids}

    task_ids: List[str] = []
    rows_t: List[List[float]] = []
    for p in snapshot.get("all_parcels", []):
        if p.get("picked") or p.get("delivered"):
            continue
        pick = (p["col"], p["row"])
        weight = float(p.get("weight", 1.0))
        trip = trip_energy(me, pick, station_center, weight)
        if usable < trip:
            continue                                   # not affordable: not offered
        d = float(distance(me, pick))
        nearest_peer = min((float(distance(pos, pick)) for pos in peer_pos.values()), default=span)
        age = max(0.0, step - float(p.get("seen_step", 0)))
        task_ids.append(p["id"])
        rows_t.append([
            (pick[0] - me[0]) / cols,
            (pick[1] - me[1]) / rows,
            d / span,
            weight / 2.0 - 0.5,
            trip / max(1e-9, capacity),
            (usable - trip) / max(1e-9, capacity),
            float(distance(pick, station_center)) / span,
            min(2.0, age / t_ref),
            nearest_peer / span,
            1.0 if d <= nearest_peer else 0.0,         # no peer is believed nearer to it
            (d + float(distance(pick, station_center))) * steps_per_cell / max(1e-9, own_speed) / t_ref,    # steps this whole trip takes the drone at its own speed
        ])
    tasks = np.array(rows_t, dtype=np.float32) if rows_t else np.zeros((0, TASK_DIM), dtype=np.float32)

    rows_p: List[List[float]] = []
    for pid in peer_ids:
        col, row, seen, scol, srow = peer_memory[pid]
        rows_p.append([
            (col - me[0]) / cols,
            (row - me[1]) / rows,
            float(distance(me, (col, row))) / span,
            1.0 if seen >= step else 0.0,              # sensed right now
            min(2.0, max(0.0, step - seen) / t_ref),
            (scol - me[0]) / cols,
            (srow - me[1]) / rows,
            (peer_speed or {}).get(pid, 1.0) - 1.0,                                        # the peer's declared speed relative to the nominal
            (peer_speed or {}).get(pid, 1.0) / max(1e-9, own_speed) - 1.0,                # relative to this drone's
        ])
    peers = np.array(rows_p, dtype=np.float32) if rows_p else np.zeros((0, PEER_DIM), dtype=np.float32)
    return x_self, task_ids, tasks, peer_ids, peers
