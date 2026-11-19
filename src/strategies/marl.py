# src/strategies/marl.py
"""
Communication-free MARL baseline: a learned task-selection policy (IPPO with a shared attention actor, see
.ignore/revision/marl_baseline_design.md). Every drone runs the same network on its own features (src/marl/features.py), chooses
one of the tasks it can afford, and the shared controller flies it, with the shared trip-energy contract for feasibility and the
shared drop-cell rule for the delivery cell. No message, no coordinator.

Training: the strategy samples its choice and appends one record per decision (features, choice, log-probability, value) to a file
per drone and run; the trainer (src/marl/train.py) turns these into returns from the mission outcome. Evaluation: the choice is the
policy's argmax, so a run is reproducible.
"""
import os
import pickle
import random
from typing import Dict, List, Optional

import numpy as np

from src.marl import features, policy_np
from .base import BaseStrategy

_PARAM_CACHE: Dict[str, Dict[str, np.ndarray]] = {}

# Analysis only (environment variable MARL_ABLATE, a comma-separated list; empty by default): at run time the named feature group of the trained policy is
# replaced by an uninformative value, to measure how much the policy relies on it. Column numbers are those of src/marl/features.py.
ABLATIONS = {
    "age":            lambda x, t, p: (t.__setitem__((slice(None), 7), 0.0)),                      # every parcel looks freshly confirmed
    "peer_proximity": lambda x, t, p: (t.__setitem__((slice(None), 8), 1.0), t.__setitem__((slice(None), 9), 1.0)),   # no peer is near any parcel
    "peers":          lambda x, t, p: p.__imul__(0.0),                                            # the peer features read zero
    "peer_seen":      lambda x, t, p: (p.__setitem__((slice(None), 3), 0.0), p.__setitem__((slice(None), 4), 0.0)),   # whether and when a peer was sensed
    "peer_start":     lambda x, t, p: (p.__setitem__((slice(None), 5), 0.0), p.__setitem__((slice(None), 6), 0.0)),   # the peers' start cells
    "parcel_position": lambda x, t, p: (t.__setitem__((slice(None), 0), 0.0), t.__setitem__((slice(None), 1), 0.0)),  # where the parcel lies relative to the drone
    "distance":       lambda x, t, p: (t.__setitem__((slice(None), 2), 0.0), t.__setitem__((slice(None), 6), 0.0)),    # the distance to the parcel and from it to the station
    "energy":         lambda x, t, p: (t.__setitem__((slice(None), 4), 0.0), t.__setitem__((slice(None), 5), 0.0)),    # trip energy and margin
    "weight":         lambda x, t, p: t.__setitem__((slice(None), 3), 0.0),
    "step":           lambda x, t, p: x.__setitem__(9, 0.0),                                       # the time since the start of the mission
    "sensing_flags":  lambda x, t, p: (x.__setitem__(7, 0.0), x.__setitem__(8, 0.0)),
}
_ABLATE = [g for g in os.environ.get("MARL_ABLATE", "").split(",") if g]


def _params(path: str) -> Dict[str, np.ndarray]:
    if path not in _PARAM_CACHE:
        _PARAM_CACHE[path] = policy_np.load(path)
    return _PARAM_CACHE[path]


class MARLStrategy(BaseStrategy):
    enable_reactive_fallback = False
    requires_plan_completion_before_requery = False

    def __init__(self, physics, weights: str, sensing: Optional[Dict] = None, trace_dir: Optional[str] = None,
                 greedy: bool = True):
        super().__init__()
        self.physics = physics
        self.weights_path = weights
        sensing = sensing or {}
        self.sensing_limited = float(sensing.get("radius", float("inf"))) != float("inf") \
            or float(sensing.get("parcel_radius", float("inf"))) != float("inf")
        self.sensing_noisy = float(sensing.get("noise_std", 0.0)) > 0.0
        self.trace_dir = trace_dir
        self.greedy = bool(greedy) and not trace_dir          # a run that records a trace samples its choices
        self._peer_memory: Optional[Dict[str, tuple]] = None
        self._rng: Optional[random.Random] = None
        self._trace_path: Optional[str] = None

    def set_drone(self, drone):
        super().set_drone(drone)
        self._rng = random.Random(f"marl-{self.drone.run_seed}-{self.drone.agent_id}")
        if self.trace_dir:
            os.makedirs(self.trace_dir, exist_ok=True)
            self._trace_path = os.path.join(self.trace_dir, f"seed{self.drone.run_seed}_{self.drone.agent_id}.pkl")

    def _t_ideal(self, info, station_center) -> float:
        if getattr(self, "_t_ideal_cache", None) is None:
            self._t_ideal_cache = features.ideal_time(info.parcels, station_center, [float(getattr(s, "speed", 1.0) or 1.0) for s in info.fleet.values()],
                                                      self.physics.distance, self.physics.steps_per_cell)
        return self._t_ideal_cache

    # ------------------------------------------------------------------
    def decide(self, snapshot: Dict, agent_id: str = None) -> Dict:
        info = snapshot.get("start_info")
        me_id = snapshot["agent"]["id"]
        if self._peer_memory is None:
            self._peer_memory = {pid: (spec.start_cell[0], spec.start_cell[1], 0, spec.start_cell[0], spec.start_cell[1])
                                 for pid, spec in info.fleet.items() if pid != me_id}
        step = snapshot.get("step", 0)
        for a in snapshot.get("other_agents", []):          # what the sensor reports now
            if a["id"] in self._peer_memory:
                old = self._peer_memory[a["id"]]
                self._peer_memory[a["id"]] = (a["col"], a["row"], step, old[3], old[4])

        station = snapshot.get("nearest_station")
        remaining = [p for p in snapshot.get("all_parcels", []) if not p.get("picked") and not p.get("delivered")]
        if not remaining:
            return {"mode": "idle", "reason": "no_remaining_parcels"}
        if not station:
            return {"mode": "wait"}
        station_center = (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
        spec = info.fleet[me_id]
        x_self, task_ids, tasks, peer_ids, peers = features.build(
            snapshot, distance=self.physics.distance, trip_energy=self.trip_energy,
            battery=float(self.drone.power.level), floor=float(self.drone.power.floor),
            capacity=float(self.drone.power.capacity), unloaded_per_cell=spec.unloaded_per_cell,
            weight_sensitivity=spec.weight_sensitivity, grid=info.grid, peer_memory=self._peer_memory,
            sensing_limited=self.sensing_limited, sensing_noisy=self.sensing_noisy, station_center=station_center,
            own_speed=float(getattr(spec, "speed", 1.0) or 1.0), peer_speed={pid: float(getattr(s, "speed", 1.0) or 1.0) for pid, s in info.fleet.items() if pid != me_id},
            steps_per_cell=self.physics.steps_per_cell, t_ideal=self._t_ideal(info, station_center), n_total=info.n_orders)
        if not task_ids:
            return {"mode": "idle", "reason": "nothing_affordable"}     # nothing the drone can afford (or see): idle, so that the controller parks it, as for every other strategy
        for group in _ABLATE:
            ABLATIONS[group](x_self, tasks, peers)

        logits, value = policy_np.forward(_params(self.weights_path), x_self, tasks, peers)
        logp = policy_np.log_softmax(logits)
        if self.greedy:
            choice = int(np.argmax(logits))
        else:
            u, acc, choice = self._rng.random(), 0.0, len(task_ids) - 1
            for i, lp in enumerate(logp):
                acc += float(np.exp(lp))
                if u <= acc:
                    choice = i
                    break
        if self._trace_path:
            with open(self._trace_path, "ab") as fh:
                pickle.dump({"step": step, "agent": me_id, "x_self": x_self, "tasks": tasks, "peers": peers,
                             "action": choice, "logp": float(logp[choice]), "value": float(value)}, fh)

        parcel = next(p for p in remaining if p["id"] == task_ids[choice])
        pick = (parcel["col"], parcel["row"])
        drop = self._choose_drop_cell(pick, station)
        return {
            "mode": "plan",
            "plan": {
                "plan": [{"pickup": pick, "dropoff": drop, "weight": float(parcel.get("weight", 1.0))}],
                "confidence": float(np.exp(logp[choice])),
                "strategy": "marl-ippo",
                "narration": "learned task choice",
            },
        }
