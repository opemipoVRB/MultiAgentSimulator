# src/strategies/auction.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple
import math
from .base import BaseStrategy

class AuctionStrategy(BaseStrategy):
    strategy_name = "AuctionStrategy"

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        assignments: Dict[Any, deque] = defaultdict(deque)
        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        candidates: List[Tuple[float, Any, Any]] = []

        for p in parcels:
            for c in controllers:
                try:
                    if getattr(c, "_can_attempt_parcel", lambda parcel: True)(p) is False:
                        continue
                    d = getattr(c, "_euclid_dist_cells", lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]))(
                        (int(c.drone.col), int(c.drone.row)), (p.col, p.row)
                    )
                    batt = getattr(getattr(c, "drone", None), "power", None)
                    batt_val = batt.percent() if batt is not None else 100.0
                    score = d * (1.0 - min(0.4, batt_val / 250.0))
                    candidates.append((score, c, p))
                except Exception:
                    continue

        candidates.sort(key=lambda t: t[0])
        for score, c, p in candidates:
            if len(assignments[c]) < self.max_assign_per_agent:
                if p not in assignments[c]:
                    assignments[c].append(p)
        self.mark_ran()
        return assignments
