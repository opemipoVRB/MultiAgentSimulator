# src/marl/global_state.py
"""
Global state of a mission at a decision step, reconstructed from the mission's logs. Used ONLY as the input of the centralised critic during
training (MAPPO); the deployed policy never sees it (see .ignore/revision/marl_baseline_design.md, "Training-time information").

From agents.json (each drone's cell at every step, its pick and drop events) and experiment.json (the parcels' positions and weights):
at step t the drones' positions and whether each carries a parcel, and the parcels not yet picked. Everything is expressed relative to the deciding drone.
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.marl.features import STEP_REF

DRONE_DIM = 4      # dcol, drow, distance, carrying
PARCEL_DIM = 4     # dcol, drow, distance, weight
SCALAR_DIM = 2     # step, share of parcels still open


class MissionLog:
    def __init__(self, agents_path: str, experiment_path: str, grid: Tuple[int, int] = (25, 16)):
        agents = json.load(open(agents_path))
        agents = list(agents.values()) if isinstance(agents, dict) else agents
        exp = json.load(open(experiment_path))
        self.cols, self.rows = float(grid[0]), float(grid[1])
        self.cells: Dict[str, List[Tuple[int, int]]] = {}
        self.carry: Dict[str, List[Tuple[int, int]]] = {}       # per drone: [(pick step, drop step)]
        picks = []                                              # (step, cell)
        for ag in agents:
            aid = ag["agent_id"]
            self.cells[aid] = [tuple(t["cell"]) for t in ag["trajectory"]]
            events = sorted(ag.get("actions", []), key=lambda e: e["time"]["step"])
            open_pick = None
            intervals = []
            for e in events:
                if e["type"] == "pick":
                    open_pick = e["time"]["step"]
                    picks.append((e["time"]["step"], tuple(e["cell"])))
                elif e["type"] == "drop" and open_pick is not None:
                    intervals.append((open_pick, e["time"]["step"]))
                    open_pick = None
            if open_pick is not None:
                intervals.append((open_pick, 10 ** 9))
            self.carry[aid] = intervals
        self.parcels = [(p["col"], p["row"], float(p.get("weight", 1.0))) for p in exp["parcels"]]
        self.picked_step: List[Optional[int]] = [None] * len(self.parcels)
        for step, cell in sorted(picks):                        # a pick takes the nearest not yet taken parcel at that cell
            best = None
            for i, (c, r, _) in enumerate(self.parcels):
                if self.picked_step[i] is None and (c, r) == cell:
                    best = i
                    break
            if best is not None:
                self.picked_step[best] = step

    def _cell(self, aid: str, step: int) -> Tuple[int, int]:
        traj = self.cells[aid]
        return traj[max(0, min(step, len(traj)) - 1)]

    def at(self, step: int, agent: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(other drones (m, DRONE_DIM), open parcels (k, PARCEL_DIM), scalars (SCALAR_DIM,)) at `step`, relative to `agent`."""
        me = self._cell(agent, step)
        span = self.cols + self.rows
        drones = []
        for aid in sorted(self.cells):
            if aid == agent:
                continue
            c = self._cell(aid, step)
            carrying = any(a <= step < b for a, b in self.carry[aid])
            drones.append([(c[0] - me[0]) / self.cols, (c[1] - me[1]) / self.rows,
                           (abs(c[0] - me[0]) + abs(c[1] - me[1])) / span, 1.0 if carrying else 0.0])
        parcels = []
        for i, (c, r, w) in enumerate(self.parcels):
            if self.picked_step[i] is not None and self.picked_step[i] <= step:
                continue
            parcels.append([(c - me[0]) / self.cols, (r - me[1]) / self.rows,
                            (abs(c - me[0]) + abs(r - me[1])) / span, w / 2.0 - 0.5])
        open_share = len(parcels) / max(1, len(self.parcels))
        return (np.array(drones, dtype=np.float32).reshape(-1, DRONE_DIM),
                np.array(parcels, dtype=np.float32).reshape(-1, PARCEL_DIM),
                np.array([min(1.0, step / STEP_REF), open_share], dtype=np.float32))
