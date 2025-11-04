# src/strategies/reservation.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Any, Dict, List
import math
from .base import BaseStrategy

# If you migrate controllers to use reservations.ReservationManager,
# the strategy will respect that API. For now we try to import the module
# but gracefully fallback to an empty dict (legacy compatibility).
try:
    from reservations import RESERVATION_MANAGER as _RESV_MGR  # type: ignore
except Exception:
    _RESV_MGR = None  # legacy callers may still rely on controllers.RESERVATIONS

class ReservationStrategy(BaseStrategy):
    strategy_name = "ReservationStrategy"

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        assignments: Dict[Any, deque] = defaultdict(deque)
        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        # Snapshot reservations in a simple form for filtering
        reserved_keys = {}
        if _RESV_MGR is not None:
            for k, (owner, ts) in _RESV_MGR.list_reservations().items():
                reserved_keys[k] = owner
        else:
            # fallback: try controllers.RESERVATIONS if present (best-effort)
            try:
                from controllers import RESERVATIONS  # type: ignore
                reserved_keys = {k: v[0] for k, v in RESERVATIONS.items()}
            except Exception:
                reserved_keys = {}

        for c in controllers:
            if len(assignments[c]) >= self.max_assign_per_agent:
                continue
            try:
                cx, cy = int(c.drone.col), int(c.drone.row)
            except Exception:
                continue

            best = None
            best_d = None
            for p in parcels:
                key = (int(p.col), int(p.row))
                if key in reserved_keys and reserved_keys[key] is not c:
                    continue
                if getattr(c, "_can_attempt_parcel", lambda parcel: True)(p) is False:
                    continue
                d = math.hypot(cx - p.col, cy - p.row)
                if best is None or d < best_d:
                    best = p
                    best_d = d
            if best is not None:
                assignments[c].append(best)
        self.mark_ran()
        return assignments
