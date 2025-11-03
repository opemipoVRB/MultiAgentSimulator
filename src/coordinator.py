# src/coordinator.py
"""
Coordinator abstraction and strategy implementations for assigning parcels to agents.

Design:
 - BaseStrategy defines the API strategies must implement.
 - Concrete strategies implement assignment policies.
 - Coordinator holds the active strategy and delegates tick/assignment operations.
 - A singleton COORD is provided for convenience.

Usage (summary):
  from coordinator import COORD
  COORD.register_controllers(ai_controllers)
  COORD.tick(terrain)        # call every frame (tick respects tick_interval)
  assigned = COORD.pop_assignment_for(controller)  # called by controller when idle/looking for work
"""

from __future__ import annotations
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
import math


# ---------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------
class BaseStrategy:
    """
    Base class / interface for coordination strategies.
    Subclasses should implement `compute_assignments`.
    """

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        self.tick_interval = float(tick_interval)
        self.max_assign_per_agent = int(max_assign_per_agent)
        self._last_tick = 0.0

    def should_run(self) -> bool:
        return (time.time() - self._last_tick) >= self.tick_interval

    def mark_ran(self):
        self._last_tick = time.time()

    def compute_assignments(self,
                            controllers: List[Any],
                            terrain: Any) -> Dict[Any, deque]:
        """
        Compute and return a dict mapping controller -> deque(parcels).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("BaseStrategy.compute_assignments must be implemented.")


# ---------------------------------------------------------------------
# Reservation strategy (keeps behavior similar to your RESERVATIONS mapping)
# ---------------------------------------------------------------------
class ReservationStrategy(BaseStrategy):
    """
    Advisory-only strategy: suggest nearest unreserved parcel for each controller.
    Does not override reservations — controllers should still call _try_reserve / _reserved_by_me.
    """

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        super().__init__(tick_interval, max_assign_per_agent)

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        from collections import deque
        assignments: Dict[Any, deque] = defaultdict(deque)

        # get all available parcels
        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        # local import to respect the existing reservation table when making suggestions
        try:
            from controllers import RESERVATIONS  # type: ignore
        except Exception:
            RESERVATIONS = {}

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
                if getattr(p, "picked", False) or getattr(p, "delivered", False):
                    continue
                key = (int(p.col), int(p.row))
                # skip parcels reserved by other controllers
                if key in RESERVATIONS and RESERVATIONS[key][0] is not c:
                    continue
                # optional capability filter
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


# ---------------------------------------------------------------------
# Auction strategy (single-round deterministic auction)
# ---------------------------------------------------------------------
class AuctionStrategy(BaseStrategy):
    """
    For each parcel, compute a score per capable agent and assign it to the
    best-scoring agent. Agents are limited by max_assign_per_agent.
    """

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        super().__init__(tick_interval, max_assign_per_agent)

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        from collections import deque
        assignments: Dict[Any, deque] = defaultdict(deque)

        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        candidates: List[Tuple[float, Any, Any]] = []  # (score, controller, parcel)

        for p in parcels:
            for c in controllers:
                # skip controllers that can't attempt parcel
                try:
                    if getattr(c, "_can_attempt_parcel", lambda parcel: True)(p) is False:
                        continue
                    # distance heuristic (use controller helper if available)
                    try:
                        d = getattr(c, "_euclid_dist_cells", lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]))(
                            (int(c.drone.col), int(c.drone.row)), (p.col, p.row)
                        )
                    except Exception:
                        d = math.hypot(int(c.drone.col) - p.col, int(c.drone.row) - p.row)

                    batt_pct = getattr(getattr(c, "drone", None), "power", None)
                    batt_val = batt_pct.percent() if batt_pct is not None else 100.0
                    # lower score is better. Advantaged by battery (small bias)
                    score = d * (1.0 - min(0.4, batt_val / 250.0))
                    candidates.append((score, c, p))
                except Exception:
                    continue

        # sort by score and greedily assign respecting per-agent capacity
        candidates.sort(key=lambda t: t[0])
        for score, c, p in candidates:
            if len(assignments[c]) < self.max_assign_per_agent:
                # do not append duplicates
                if p not in assignments[c]:
                    assignments[c].append(p)

        self.mark_ran()
        return assignments


# ---------------------------------------------------------------------
# Centralized greedy strategy
# ---------------------------------------------------------------------
class CentralizedGreedyStrategy(BaseStrategy):
    """
    Repeatedly assign highest-energy agents their nearest-capable parcel.
    """

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        super().__init__(tick_interval, max_assign_per_agent)

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        from collections import deque
        assignments: Dict[Any, deque] = defaultdict(deque)

        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        # sort agents by battery level desc (best agents first)
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


# small helper stub used for sorting when an agent lacks power attribute
class AgentPowerStub:
    def __init__(self):
        self.level = 0.0
        self.percent = lambda: 0


# ---------------------------------------------------------------------
# Coordinator wrapper (holds strategy instance and delegates)
# ---------------------------------------------------------------------
class Coordinator:
    def __init__(self,
                 strategy_name: str = None,
                 tick_interval: float = 1.0,
                 max_assign_per_agent: int = 2):
        self._controllers: List[Any] = []
        self._assignments: Dict[Any, deque] = defaultdict(deque)
        self.set_strategy(strategy_name, tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)

    def set_strategy(self, strategy_name: str, tick_interval: Optional[float] = None,
                     max_assign_per_agent: Optional[int] = None):
        if tick_interval is None:
            tick_interval = 1.0
        if max_assign_per_agent is None:
            max_assign_per_agent = 2

        strategy_name = (strategy_name or "reservation").lower()
        if strategy_name == "reservation":
            self.strategy = ReservationStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
        elif strategy_name == "auction":
            self.strategy = AuctionStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
        elif strategy_name == "centralized_greedy":
            self.strategy = CentralizedGreedyStrategy(tick_interval=tick_interval,
                                                      max_assign_per_agent=max_assign_per_agent)
        else:
            # default fallback
            self.strategy = ReservationStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)

        # clear assignments when switching strategy
        self.clear_assignments()

    def register_controllers(self, controllers: List[Any]):
        self._controllers = list(controllers)
        self.clear_assignments()

    def clear_assignments(self):
        from collections import deque
        self._assignments = defaultdict(deque)

    def peek_assignment_for(self, controller: Any) -> Optional[Any]:
        dq = self._assignments.get(controller)
        if dq and len(dq) > 0:
            return dq[0]
        return None

    def pop_assignment_for(self, controller: Any) -> Optional[Any]:
        dq = self._assignments.get(controller)
        if dq and len(dq) > 0:
            return dq.popleft()
        return None

    def assign_direct(self, controller: Any, parcel: Any):
        dq = self._assignments[controller]
        if len(dq) < self.strategy.max_assign_per_agent:
            dq.append(parcel)

    def tick(self, terrain: Any, force: bool = False):
        """
        Recompute assignments if the strategy interval elapsed or force=True.
        """
        if not self._controllers:
            return
        if not force and not self.strategy.should_run():
            return

        try:
            new_assignments = self.strategy.compute_assignments(self._controllers, terrain)
            # Accept the new assignments as override (simple approach). Strategies return mapping controller->deque
            self._assignments = new_assignments
        except Exception:
            # on error, keep existing assignments
            pass
        finally:
            self.strategy.mark_ran()


# ---------------------------------------------------------------------
# Singleton convenience instance
# ---------------------------------------------------------------------
# COORD = Coordinator(strategy_name="reservation", tick_interval=1.0, max_assign_per_agent=2)
COORD = Coordinator(strategy_name="auction", tick_interval=1.0, max_assign_per_agent=2)
