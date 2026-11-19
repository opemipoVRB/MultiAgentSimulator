# src/strategies/base.py
from typing import Dict, Any, List, Optional


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

    Directive modes a strategy may return from `decide`: "plan", "wait" (ask again next step), "idle" (nothing to do: park), "abort", and "look"
    (fly to the station centre to see every cell, then decide again). Every strategy has the same modes; the controller treats them alike.
    """

    # Per-cycle hook. A strategy that sets this True is called through `step` every simulation
    # step, not only when its agent needs a decision, and may return a directive, which the
    # controller executes like any other (for example {"mode": "interrupt"}). Default: not called.
    steps_every_cycle = False

    def on_perception_events(self, events):
        """Called when the drone's senses notice a change (parcel_seen, parcel_taken, peer_sighted, peer_lost).
        Default: ignore. Strategies that react to what the drone sees override this."""
        return None


    @staticmethod
    def _choose_drop_cell(pick, station):
        """
        The delivery cell for a parcel picked at `pick`: the free station cell nearest to the pickup, as the drone senses
        it on site (the station dict carries `free_cells`); with none free, any station cell. Without a station the
        parcel is dropped where it was picked.
        """
        if not station:
            return pick
        free = station.get("free_cells")
        if not free:
            free = [(c, r)
                    for r in range(station["row"], station["row"] + station["h"])
                    for c in range(station["col"], station["col"] + station["w"])]
        return min(free, key=lambda c: abs(c[0] - pick[0]) + abs(c[1] - pick[1]))

    def step(self, snapshot: Dict) -> Optional[Dict]:
        """Optional per-cycle call. Return None, or a directive for the controller to execute."""
        return None

    # Information model, set from the arm's spec by the factory (see src/perception.py): how the station is known, and whether the
    # drone stops when the station holds every parcel.
    station_memory: str = "observed"      # what the drone knows of the station cells: only what it has perceived and remembers (the live free-cell list is an unfair advantage)
    ledger_stop: bool = False
    # station_bids: the delivery leg of a bid or a feasibility check is priced to the cell the drone would drop at (the free station cell nearest to
    # the pickup, from the station the drone knows) instead of the station centre. The cost model is the strategy's own; only its input is better.
    station_bids: bool = False
    declared_weights: bool = True       # parcels carry the weight declared with the order from the start (shared start information), not the unseen default
    # ledger_verify: before parking on a prediction alone (the memory does not yet say the station is full), the drone flies to the station centre, where every
    # cell is in view, and parks only if the exact count agrees.
    ledger_verify: bool = False

    def predicts_job_done(self, snapshot) -> bool:
        """A strategy that predicts the station says here that, by its own prediction, every parcel is delivered (used with ledger_stop)."""
        return False

    def _bid_drop(self, pick, station):
        if self.station_bids:
            return self._choose_drop_cell(pick, station)
        return (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
    enable_reactive_fallback: bool = False
    requires_plan_completion_before_requery: bool = False
    use_async_worker = False
    globally_infeasible = False
    NOMINAL_WEIGHT = 1.0
    # False for strategies that apply the declared spec themselves (SPPDEC), so it is not applied twice.
    uses_declared_energy_model = True

    @property
    def FEASIBILITY_MARGIN(self) -> float:
        """The energy a drone keeps in reserve: the hardware floor of the drone (the same for every strategy)."""
        drone = getattr(self, "drone", None)
        return float(drone.power.floor) if drone is not None else 3.0

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
        nominal = getattr(self, "_nominal_physics", None) or self.physics
        self._nominal_physics = nominal
        spec = drone.declared_spec() if hasattr(drone, "declared_spec") else None
        if nominal is not None and spec is not None and self.uses_declared_energy_model:
            # Plan with the drone's declared consumption (measured before the mission), not the nominal constants.
            from src.energy import DeclaredPhysics
            self.physics = DeclaredPhysics(nominal, spec.unloaded_per_cell, spec.weight_sensitivity)

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

    def trip_energy(self, cur, pick, drop, weight: float) -> float:
        """
        The energy of one delivery trip, the same definition for every strategy: the pickup leg flown empty, the
        delivery leg flown loaded, and the energy of the pick and of the drop.
        """
        phys = self.physics
        return (phys.energy_cost(distance=phys.distance(a=cur, b=pick), weight=0.0)
                + phys.energy_cost(distance=phys.distance(a=pick, b=drop), weight=weight)
                + 2.0 * phys.pick_drop_cost(weight))

    # The energy-equivalent of one step of flight, added to a bid (0: bids are energy alone, the published baselines). Feasibility always uses energy alone.
    time_weight: float = 0.0

    def own_speed(self) -> float:
        """The flying drone's declared speed relative to the nominal (1.0 when nothing is declared): the same declared start information SPPDEC reads."""
        spec = self.drone.declared_spec() if self.drone is not None and hasattr(self.drone, "declared_spec") else None
        return float(getattr(spec, "speed", 1.0) or 1.0)

    def trip_time(self, cur, pick, drop, speed=None) -> float:
        """Steps one delivery trip takes (the empty leg to the pickup plus the loaded leg to the drop) at the flying agent's declared speed (its own unless `speed` is given); the
        nominal speed is the simulator's, the same for every strategy."""
        phys = self.physics
        s = self.own_speed() if speed is None else (float(speed) or 1.0)
        return phys.travel_steps(phys.distance(a=cur, b=pick) + phys.distance(a=pick, b=drop)) / s

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
        current_battery = float(self.drone.power.level)

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
            weight = float(p.get("weight", self.NOMINAL_WEIGHT))

            total_cost = self.trip_energy(cur_pos, pick_pos, drop_pos, weight)

            if current_battery >= total_cost + self.FEASIBILITY_MARGIN:
                return True

        return False



