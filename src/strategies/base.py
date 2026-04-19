# src/strategies/base.py
from typing import Dict, Any, List


class BaseStrategy:
    """
    Base interface for all delivery strategies.

    A strategy is a decision-making policy that produces high-level
    directives (plans, tasks, wait, idle) given a snapshot of the world.

    A strategy:
    - DOES decide what an agent should attempt next
    - DOES express planning authority and autonomy constraints
    - MAY be stateless or stateful across decisions

    A strategy explicitly DOES NOT:
    - execute movement or actions
    - mutate the world or terrain
    - own agents, parcels, or stations
    - perform physics, energy consumption, or timing

    Execution responsibility:
    -------------------------
    The AIAgentController is the sole executor of all plans.
    The strategy never issues commands directly to the world.

    Reactive autonomy contract:
    ---------------------------
    `enable_reactive_fallback` declares whether the controller is allowed
    to locally substitute or repair plan steps when execution constraints
    invalidate the original plan.

    - If True:
        The controller MAY perform local, opportunistic replanning
        (for example, choosing an alternative feasible parcel).

    - If False:
        The controller MUST NOT alter the plan.
        All replanning authority remains external to the agent
        (for example, via a CommandCenter).

    This flag is authoritative and must be respected by the controller.

    requires_plan_completion_before_requery:
        If True, the controller MUST NOT request a new directive
        until the current plan has fully completed or been invalidated.
    """
    enable_reactive_fallback: bool = False
    requires_plan_completion_before_requery: bool = False
    use_async_worker = False
    globally_infeasible = False
    FEASIBILITY_MARGIN = 3.0
    NOMINAL_WEIGHT = 1.0

    def __init__(self):
        self.physics = None
        self.drone = None

    def set_drone(self, drone):
        """
        Optional hook for strategies that need direct drone access
        (e.g. SPPDEC with onboard SLM).
        Default implementation is no-op.
        """
        self.drone = drone
        if hasattr(drone, "physics"):
            self.physics = drone.physics

    def decide(self, snapshot) -> Dict:
        """
        Returns an ExecutionDirective
        """
        raise NotImplementedError

    def request_plan(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a snapshot of the world, return a plan suggestion.

        Must return a dict containing at least:
          - plan: list
          - confidence: float
          - narration: str
          - strategy: str
        """
        raise NotImplementedError

    def is_globally_feasible(self, snapshot: Dict) -> bool:
        """
        Strategy-specific epistemic feasibility.
        Return False when the strategy believes the mission can no longer progress.
        """
        parcels = [
            p for p in snapshot["all_parcels"]
            if not p.get("picked") and not p.get("delivered")
        ]
        return self._any_parcel_feasible(snapshot, parcels)

    def _any_parcel_feasible(self, snapshot: Dict[str, Any], parcels: List[Dict[str, Any]]) -> bool:
        """
        Quick check if ANY parcel can be delivered with current battery.

        Parameters:
        -----------
        snapshot : Dict[str, Any]
            World snapshot
        parcels : List[Dict[str, Any]]
            Available parcels to check

        Returns:
        --------
        bool
            True if at least one parcel is feasible, False otherwise
        """
        if not parcels:
            return False

        if not self.physics or not self.drone:
            raise RuntimeError("Strategy must be initialized with drone and physics before feasibility checks")

        agent = snapshot["agent"]
        current_battery = float(self.drone.power.percent())

        cur_pos = (agent["col"], agent["row"])
        station = snapshot.get("nearest_station")

        # Quick drop estimate (station center)
        def quick_drop(pick_pos):
            if not station:
                return pick_pos
            center_col = station["col"] + station["w"] // 2
            center_row = station["row"] + station["h"] // 2
            return (center_col, center_row)

        # Check first few parcels (avoid O(n^2) in planning phase)
        for p in parcels[:min(10, len(parcels))]:
            pick_pos = (p["col"], p["row"])
            drop_pos = quick_drop(pick_pos)

            # Conservative energy estimate
            pickup_dist = self.physics.distance(a=cur_pos, b=pick_pos)
            delivery_dist = self.physics.distance(a=pick_pos, b=drop_pos)
            weight = float(p.get("weight", self.NOMINAL_WEIGHT))

            pickup_cost = self.physics.energy_cost(distance=pickup_dist, weight=0.0)
            delivery_cost = self.physics.energy_cost(distance=delivery_dist, weight=weight)
            total_cost = pickup_cost + delivery_cost

            if current_battery >= total_cost + self.FEASIBILITY_MARGIN:
                return True

        return False



