# src/strategies/centralized_greedy.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Any, Dict, List
import math
from .base import BaseStrategy

class AgentPowerStub:
    def __init__(self):
        self.level = 0.0
        self.percent = lambda: 0

class CentralizedGreedyStrategy(BaseStrategy):
    strategy_name = "CentralizedGreedyStrategy"

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        assignments: Dict[Any, deque] = defaultdict(deque)
        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        agents = sorted(controllers, key=lambda c: getattr(getattr(c, "drone", None), "power", AgentPowerStub()).level,
                        reverse=True)
        remaining = list(parcels)
        for agent in agents:
            attempts = 0
            try:
                cx, cy = int(agent.drone.col), int(agent.drone.row)
            except Exception:
                continue
            while len(assignments[agent]) < self.max_assign_per_agent and remaining and attempts < 200:
                best = None
                best_d = None
                for p in remaining:
                    try:
                        if getattr(agent, "_can_attempt_parcel", lambda parcel: True)(p) is False:
                            continue
                        d = getattr(agent, "_euclid_dist_cells", lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]))(
                            (cx, cy), (p.col, p.row)
                        )
                        if best is None or d < best_d:
                            best = p
                            best_d = d
                    except Exception:
                        continue
                if best is None:
                    break
                assignments[agent].append(best)
                remaining.remove(best)
                attempts += 1
        self.mark_ran()
        return assignments
