from typing import Dict, List, Optional, Tuple
import time

from artifacts import Drone, Parcel, Terrain
from strategies.utils import manhattan, est_cost_cells


class CentralisedStrategy:
    """
    Centralised task allocation strategy.

    This strategy represents a command-and-control planning model
    in which individual agents do not possess local replanning authority.

    This class DOES:
      - assign parcels to agents using a global snapshot
      - enforce exclusivity of assignments across agents
      - determine feasibility using geometry and energy estimates
      - detect invalid or stalled assignments and release them

    This class explicitly DOES NOT:
      - move drones or execute actions
      - issue low-level control commands
      - allow agents to substitute tasks independently
      - assume continuous connectivity or perfect communication
      - permit reactive, agent-side replanning

    Authority model:
    ----------------
    All decision authority is centralized.
    Agents must execute exactly what is assigned.

    If execution fails or conditions change:
      - the agent must wait
      - the CommandCenter must reissue guidance

    Reactive fallback is therefore DISABLED by design.
    """

    enable_reactive_fallback = False
    requires_plan_completion_before_requery = True

    def __init__(self, stall_seconds: float = 6.0):
        self.stall_seconds = stall_seconds
        # agent_id -> assignment record
        self.assignments: Dict[str, Dict] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def decide(self, snapshot: Dict, agent_id: str) -> Dict:
        """
        Decide an executable directive for a specific agent based on a
        global snapshot of the world.

        This method adheres strictly to the planner → controller contract:
        - It NEVER returns raw execution steps.
        - It ALWAYS wraps plans inside a plan envelope.
        - The controller remains the sole executor.

        Parameters
        ----------
        snapshot : Dict
            Global snapshot built by CommandCenter.
            Expected keys: agents, parcels, stations.
        agent_id : str
            ID of the requesting agent.

        Returns
        -------
        Dict
            Directive dictionary with a `mode` key.
            Supported modes: plan, wait, idle
        """

        # --------------------------------------------------
        # 1. Resolve agent
        # --------------------------------------------------
        agents = snapshot.get("agents", [])
        parcels = snapshot.get("parcels", [])
        stations = snapshot.get("stations", [])

        agent = next(
            (a for a in agents if a.get("id") == agent_id),
            None,
        )

        if agent is None:
            print(f"[CENTRAL][DECIDE] agent_id={agent_id} UNKNOWN → WAIT")
            return {"mode": "wait"}

        # --------------------------------------------------
        # 2. Resolve station (single-station assumption)
        # --------------------------------------------------
        if not stations:
            print(f"[CENTRAL][DECIDE] agent_id={agent_id} NO_STATION → IDLE")
            return {"mode": "idle"}

        station = stations[0]

        # --------------------------------------------------
        # 3. Reuse existing assignment if still valid
        # --------------------------------------------------
        existing = self.assignments.get(agent_id)
        if existing is not None:
            parcel = next(
                (p for p in parcels if p["id"] == existing["parcel_id"]),
                None,
            )

            # VALID ASSIGNMENT CONDITIONS
            if parcel is not None and not parcel.get("delivered", False):
                drop_cell = self._best_station_entry_cell(parcel, station)

                # Do NOT release just because it's not picked yet
                return {
                    "mode": "plan",
                    "plan": [
                        {
                            "pickup": (parcel["col"], parcel["row"]),
                            "dropoff": drop_cell,
                            "weight": parcel.get("weight", 1.0),
                        }
                    ],
                    "confidence": 1.0,
                    "strategy": "centralised-single-step",
                    "narration": "Centralised assignment (authoritative)",
                }

            # Truly invalid → release
            del self.assignments[agent_id]
            print(
                f"[CENTRAL][DECIDE] agent_id={agent_id} released invalid assignment"
            )

        # --------------------------------------------------
        # 4. Filter available parcels
        # --------------------------------------------------
        reserved = {
            rec["parcel_id"]
            for rec in self.assignments.values()
        }

        available = [
            p for p in parcels
            if not p.get("picked")
               and not p.get("delivered")
               and p["id"] not in reserved
        ]

        if not available:
            print(
                f"[CENTRAL][DECIDE] agent_id={agent_id} "
                f"NO_AVAILABLE_PARCELS → WAIT"
            )
            return {"mode": "wait"}

        # --------------------------------------------------
        # 5. Select nearest parcel (agent → pickup cost)
        # --------------------------------------------------
        ax, ay = agent["col"], agent["row"]

        target = min(
            available,
            key=lambda p: abs(p["col"] - ax) + abs(p["row"] - ay),
        )

        # --------------------------------------------------
        # 6. Reserve assignment
        # --------------------------------------------------
        self.assignments[agent_id] = {
            "parcel_id": target["id"],
            "assigned_at": time.time(),
        }

        print(
            f"[CENTRAL][DECIDE] agent_id={agent_id} "
            f"assigned parcel={target['id']}"
        )

        # --------------------------------------------------
        # 7. Emit PLAN ENVELOPE (controller-compatible)
        # --------------------------------------------------
        drop_cell = self._best_station_entry_cell(target, station)

        return {
            "mode": "plan",
            "plan": {
                "plan": [
                    {
                        "pickup": (target["col"], target["row"]),
                        "dropoff": drop_cell,
                        "weight": target.get("weight", 1.0),
                    }
                ],
                "confidence": 1.0,
                "narration": "Centralised assignment: nearest available parcel",
                "strategy": "centralised-single-step",
            },
        }

    def step(
            self,
            *,
            terrain,
            agents: List,
            parcels: List,
            now: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Command-center-facing API.

        Performs global coordination and returns
        the current assignment map:
            agent_id -> parcel_id
        """
        if now is None:
            now = time.time()

        self._prune_invalid_assignments(agents, parcels, now)
        self._allocate_new_tasks(terrain, agents, parcels, now)

        return {
            agent_id: rec["parcel_id"]
            for agent_id, rec in self.assignments.items()
        }

    # ---------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------

    def _allocate_new_tasks(
            self,
            agents: List[Drone],
            parcels: List[Parcel],
            now: float,
    ):
        free_agents = self._free_agents(agents)
        free_parcels = self._free_parcels(parcels)

        if not free_agents or not free_parcels:
            return

        station = self._get_station_center()
        if station is None:
            return

        for parcel in free_parcels:
            best_agent = None
            best_cost = float("inf")

            for agent in free_agents:
                if not self._feasible(agent, parcel, station):
                    continue

                cost = manhattan(
                    (agent.col, agent.row),
                    (parcel.col, parcel.row),
                )

                if cost < best_cost:
                    best_cost = cost
                    best_agent = agent

            if best_agent is not None:
                self.assignments[best_agent.agent_id] = {
                    "parcel_id": parcel.id,
                    "assigned_at": now,
                    "last_progress_at": now,
                    "last_pos": (best_agent.col, best_agent.row),
                }

                free_agents.remove(best_agent)

                if not free_agents:
                    break

    # ---------------------------------------------------------
    # Assignment validity and failure detection
    # ---------------------------------------------------------

    def _prune_invalid_assignments(
            self,
            agents: List[Drone],
            parcels: List[Parcel],
            now: float,
    ):
        agent_map = {a.agent_id: a for a in agents}
        parcel_map = {p.id: p for p in parcels}

        for agent_id in list(self.assignments.keys()):
            rec = self.assignments[agent_id]

            agent = agent_map.get(agent_id)
            parcel = parcel_map.get(rec["parcel_id"])

            # Agent gone or dead
            if agent is None or agent.power.level <= 0.0:
                del self.assignments[agent_id]
                continue

            # Parcel gone or already delivered
            if parcel is None or parcel.delivered:
                del self.assignments[agent_id]
                continue

            # Progress check
            current_pos = (agent.col, agent.row)
            if current_pos != rec["last_pos"]:
                rec["last_pos"] = current_pos
                rec["last_progress_at"] = now
                continue

            # Stall detection
            if now - rec["last_progress_at"] > self.stall_seconds:
                del self.assignments[agent_id]

    # ---------------------------------------------------------
    # Feasibility logic
    # ---------------------------------------------------------

    def _feasible(
            self,
            agent: Drone,
            parcel: Parcel,
            station: Tuple[int, int],
    ) -> bool:
        """
        Conservative feasibility test.
        """
        to_parcel = manhattan(
            (agent.col, agent.row),
            (parcel.col, parcel.row),
        )

        to_station = manhattan(
            (parcel.col, parcel.row),
            station,
        )

        total_cells = to_parcel + to_station

        weight = getattr(parcel, "weight", 1.0)

        est_cost = est_cost_cells(total_cells, weight)

        return agent.power.level > est_cost

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _best_station_entry_cell(parcel: Dict, station: Dict) -> Tuple[int, int]:
        """
        Choose the station cell that minimizes delivery distance
        from the parcel location.

        This represents an optimal entry point, NOT a mandatory
        interior destination.
        """
        cells = [
            (c, r)
            for r in range(station["row"], station["row"] + station["h"])
            for c in range(station["col"], station["col"] + station["w"])
        ]

        return min(
            cells,
            key=lambda cell: abs(cell[0] - parcel["col"]) + abs(cell[1] - parcel["row"]),
        )

    def _free_agents(self, agents: List[Drone]) -> List[Drone]:
        assigned = set(self.assignments.keys())
        return [
            a for a in agents
            if a.agent_id not in assigned
               and a.power.level > 0.0
               and getattr(a, "carrying", None) is None
        ]

    def _free_parcels(self, parcels: List[Parcel]) -> List[Parcel]:
        assigned_parcels = {
            rec["parcel_id"]
            for rec in self.assignments.values()
        }

        return [
            p for p in parcels
            if not p.delivered
               and not p.picked
               and p.id not in assigned_parcels
        ]

    def _get_station_center(self) -> Optional[Tuple[int, int]]:
        if not self.terrain.stations:
            return None

        s = self.terrain.stations[0]
        return (
            s.col + s.w // 2,
            s.row + s.h // 2,
        )
