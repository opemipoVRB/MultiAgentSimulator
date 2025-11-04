
"""
Coordinator abstraction and strategy implementations for assigning parcels to agents.

Extended features:
 - Accepts runtime options auto_reserve and use_leader in constructor.
 - Supports an optional "llm" strategy (if strategies/llm_strategy.py is present).
 - If auto_reserve=True, Coordinator will create reservations on behalf of controllers
   for assigned pickups (same shape as controllers.RESERVATIONS).
 - If use_leader=True, Coordinator defers action when not the elected leader (leader_coordinator
   must expose `am_i_leader()` or `is_leader()` function; otherwise Coordinator assumes leadership).
"""

from __future__ import annotations
import time
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# Try to import optional LLM strategy from strategies package
try:
    from strategies.multi_agent_llm_strategy import MultiAgentLLMStrategy
except Exception:
    MultiAgentLLMStrategy = None  # type: ignore

# Keep default deterministic strategies local for fallbacks
class BaseStrategy:
    strategy_name = None

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        self.tick_interval = float(tick_interval)
        self.max_assign_per_agent = int(max_assign_per_agent)
        self._last_tick = 0.0
        self.display_strategy_name()

    def display_strategy_name(self):
        # lightweight print so test runs show which strategy was created
        try:
            print("strategy ->", self.strategy_name)
        except Exception:
            pass

    def should_run(self) -> bool:
        return (time.time() - self._last_tick) >= self.tick_interval

    def mark_ran(self):
        self._last_tick = time.time()

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        raise NotImplementedError()


class ReservationStrategy(BaseStrategy):
    strategy_name = "ReservationStrategy"

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        super().__init__(tick_interval, max_assign_per_agent)

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        assignments: Dict[Any, deque] = defaultdict(deque)
        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        # local import to consult the same RESERVATIONS map used by controllers
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
                if key in RESERVATIONS and RESERVATIONS[key][0] is not c:
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


class AuctionStrategy(BaseStrategy):
    strategy_name = "AuctionStrategy"

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        super().__init__(tick_interval, max_assign_per_agent)

    def compute_assignments(self, controllers: List[Any], terrain: Any) -> Dict[Any, deque]:
        assignments: Dict[Any, deque] = defaultdict(deque)
        parcels = [p for p in terrain.parcels if not getattr(p, "picked", False) and not getattr(p, "delivered", False)]
        if not parcels:
            return assignments

        candidates = []  # (score, controller, parcel)
        for p in parcels:
            for c in controllers:
                try:
                    if getattr(c, "_can_attempt_parcel", lambda parcel: True)(p) is False:
                        continue
                    try:
                        d = getattr(c, "_euclid_dist_cells", lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]))(
                            (int(c.drone.col), int(c.drone.row)), (p.col, p.row)
                        )
                    except Exception:
                        d = math.hypot(int(c.drone.col) - p.col, int(c.drone.row) - p.row)
                    batt_pct = getattr(getattr(c, "drone", None), "power", None)
                    batt_val = batt_pct.percent() if batt_pct is not None else 100.0
                    # lower score is better; battery gives small bias
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


class CentralizedGreedyStrategy(BaseStrategy):
    strategy_name = "CentralizedGreedyStrategy"

    def __init__(self, tick_interval: float = 1.0, max_assign_per_agent: int = 2):
        super().__init__(tick_interval, max_assign_per_agent)

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


class AgentPowerStub:
    def __init__(self):
        self.level = 0.0
        self.percent = lambda: 0


# Coordinator wrapper
class Coordinator:
    def __init__(self,
                 strategy_name: str = None,
                 tick_interval: float = 1.0,
                 max_assign_per_agent: int = 2,
                 auto_reserve: bool = False,
                 use_leader: bool = False):
        """
        strategy_name: 'reservation'|'auction'|'centralized_greedy'|'llm'
        auto_reserve: if True, Coordinator will place reservations on assigned pickups
                      using controllers.RESERVATIONS (single-process in-memory mapping).
        use_leader: if True, Coordinator will only compute/overwrite assignments if it's the elected leader
                    according to leader_coordinator (if present). Otherwise assumes it's leader.
        """
        self._controllers: List[Any] = []
        self._assignments: Dict[Any, deque] = defaultdict(deque)
        self.auto_reserve = bool(auto_reserve)
        self.use_leader = bool(use_leader)
        self._leader_check = None  # optional callable to check leadership
        # discover optional leader_coordinator interface at runtime to avoid cycles
        try:
            import leader_coordinator as _lc  # type: ignore
            # leader_coordinator should expose `am_i_leader()` or `is_leader()` or similar
            if hasattr(_lc, "am_i_leader"):
                self._leader_check = getattr(_lc, "am_i_leader")
            elif hasattr(_lc, "is_leader"):
                self._leader_check = getattr(_lc, "is_leader")
            else:
                self._leader_check = None
        except Exception:
            self._leader_check = None

        self.set_strategy(strategy_name, tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)

    def _am_i_leader(self) -> bool:
        if not self.use_leader:
            return True
        if self._leader_check:
            try:
                return bool(self._leader_check())
            except Exception:
                return False
        # if use_leader requested but no leader_coordinator available, be conservative: assume True
        return True

    def set_strategy(self, strategy_name: str, tick_interval: Optional[float] = None,
                     max_assign_per_agent: Optional[int] = None):
        if tick_interval is None:
            tick_interval = 1.0
        if max_assign_per_agent is None:
            max_assign_per_agent = 2

        name = (strategy_name or "reservation").lower()
        if name == "reservation":
            self.strategy = ReservationStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
        elif name == "auction":
            self.strategy = AuctionStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
        elif name == "centralized_greedy":
            self.strategy = CentralizedGreedyStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
        elif strategy_name == "multi_agent_llm":
            if MultiAgentLLMStrategy is not None:
                self.strategy = MultiAgentLLMStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
            else:
                # fallback to auction if no LLM strategy is present
                self.strategy = AuctionStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)
        else:
            # default
            self.strategy = ReservationStrategy(tick_interval=tick_interval, max_assign_per_agent=max_assign_per_agent)

        # clear assignments when strategy changes
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

    def _maybe_auto_reserve(self, assignments: Dict[Any, deque]):
        """
        If auto_reserve True, create reservation entries in controllers.RESERVATIONS for the assigned pickup cells.
        Shape: RESERVATIONS[(col,row)] = (owner_controller, timestamp)
        """
        if not self.auto_reserve:
            return
        # import on demand to avoid top-level cycles
        try:
            from controllers import RESERVATIONS  # type: ignore
        except Exception:
            return

        now = time.time()
        for ctrl, dq in assignments.items():
            for p in list(dq)[: self.strategy.max_assign_per_agent]:
                try:
                    key = (int(p.col), int(p.row))
                except Exception:
                    continue
                # do not overwrite reservations held by others
                if key in RESERVATIONS:
                    # if already owned by this controller, refresh timestamp
                    if RESERVATIONS[key][0] is ctrl:
                        RESERVATIONS[key] = (ctrl, now)
                    else:
                        # skip, do not steal
                        continue
                else:
                    RESERVATIONS[key] = (ctrl, now)

    def tick(self, terrain: Any, force: bool = False):
        """
        Recompute assignments using strategy if should_run or force=True.
        Respects leadership (use_leader) when configured.
        """
        if not self._controllers:
            return
        if not self._am_i_leader():
            # when not leader, do nothing — leader will publish assignments if this is a multi-node system
            return
        if not force and not self.strategy.should_run():
            return

        try:
            new_assignments = self.strategy.compute_assignments(self._controllers, terrain)
            # Optionally auto-reserve assigned pickups so controllers don't race
            try:
                self._maybe_auto_reserve(new_assignments)
            except Exception:
                pass
            # Accept the new assignments as override
            self._assignments = new_assignments
        except Exception:
            # on error do not clobber existing assignments
            pass
        finally:
            try:
                self.strategy.mark_ran()
            except Exception:
                pass


# convenience singleton (change default flags here)
COORD = Coordinator(strategy_name="centralized_greedy", tick_interval=1.0,
                    max_assign_per_agent=2, auto_reserve=False, use_leader=False)
