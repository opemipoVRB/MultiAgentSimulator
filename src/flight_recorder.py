from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time


class FlightRecorder:
    """
    FlightRecorder is a passive observability layer for a running multi-agent
    simulation.

    It continuously records fine-grained, time-aligned data about individual
    agents and their collective behavior, without influencing the simulation
    itself. The recorder exists to make coordination dynamics, interference,
    inefficiencies, and emergent fleet-level behavior explicit and measurable.

    What it records:

    Per-agent:
    - Full trajectories (grid cell, continuous position, distance traveled)
    - Battery and energy evolution over time
    - Discrete actions with precise timing (pick, drop, failures)
    - Parcel assignments and completed tasks
    - Collisions and co-occupancy with other agents

    Per-parcel:
    - Spawn location
    - Pick attempts and competing agents
    - Who picked, who delivered, and when
    - Aborted and wasted pursuit attempts

    Fleet-level:
    - Collisions and simultaneous cell occupancy
    - Parcel contention and resolution events
    - Aborted chases and wasted motion
    - Coordination inefficiencies and interference patterns

    Design principles:

    - Passive observer:
      The simulation drives the recorder. No control logic or feedback is added.

    - Agent-centric first, fleet-level second:
      Individual agent timelines are the primary truth.
      Fleet-level events are derived from agent interactions.

    - Explicit time semantics:
      Every recorded event includes:
        * simulation step (discrete)
        * simulation time (continuous)
        * wall-clock time (real-world)

    - Analysis-ready:
      Data is structured for post-run inspection, replay,
      plotting, and quantitative analysis of coordination quality.

    FlightRecorder does not attempt to optimize behavior.
    Its sole purpose is to make system-level coordination,
    interference, and inefficiencies observable.
    """

    # ======================================================
    # Initialization
    # ======================================================
    def __init__(
        self,
        run_id: str,
        planner: str,
        agents: List[Dict],
        parcels: List[Dict],
        grid_size: int,
        dt: float = 1.0,
        sim_dt: Optional[float] = None,
    ):
        """
                Initialize a FlightRecorder instance for a single experiment run.

                Args:
                    run_id:
                        Unique identifier for the experiment or run.
                        Used to associate flight data with experiment metadata.

                    planner:
                        Name of the planner or coordination strategy being evaluated
                        (e.g. "local", "auction", "llm", "centralized").

                    agents:
                        Agent definitions from the experiment schema.
                        Each agent entry must include a stable agent ID and start position.

                    parcels:
                        Parcel definitions from the experiment schema.
                        Each parcel must have a stable ID and spawn location so that
                        contention and delivery can be tracked consistently.

                    grid_size:
                        Size of one grid cell in pixels.
                        Used for distance normalization and heatmap generation.

                    dt:
                        Nominal simulation timestep in seconds.
                        Used when explicit simulation time is not provided.

                    sim_dt:
                        Explicit simulation timestep override (preferred).
                        Allows precise alignment with the simulator clock when available.
                """
        self.run_id = run_id
        self.planner = planner
        self.grid_size = grid_size
        self.dt = sim_dt if sim_dt is not None else dt
        self.started_at = time.time()

        # ==================================================
        # Agent registry
        # ==================================================
        self.agents: Dict[str, Dict] = {}
        for a in agents:
            self.agents[a["id"]] = {
                "agent_id": a["id"],
                "planner": planner,
                "start_cell": (a.get("start_col"), a.get("start_row")),

                # existing
                "trajectory": [],
                "actions": [],
                "energy": [],
                "collisions": [],
                "tasks": [],
                "final_battery": None,
                "distance_cells": 0.0,

                "planning": {
                    "initial_plan": None,
                    "plan_events": [],
                    "confidence": [],
                },
            }

        # ==================================================
        # Parcel registry
        # ==================================================
        self.parcels: Dict[int, Dict] = {}
        for p in parcels:
            self.parcels[p["id"]] = {
                "parcel_id": p["id"],
                "spawn_cell": (p["col"], p["row"]),
                "picked_by": None,
                "picked_at": None,
                "delivered_by": None,
                "delivered_at": None,
                "pick_attempts": [],
            }

        # ==================================================
        # Fleet-level logs
        # ==================================================
        self.fleet_events: List[Dict] = []

        # ==================================================
        # Internal trackers
        # ==================================================
        self._last_cell: Dict[str, Tuple[int, int]] = {}
        self._parcel_targets = defaultdict(set)
        self._parcel_chase_start = defaultdict(dict)
        self._cell_visits = defaultdict(int)

    # ======================================================
    # Per-step observation (preferred entry point)
    # ======================================================
    def tick(self, step: int, sim_time: float, agents: List):
        """
        Observe the world once per simulation step after all agents update.

        Args:
            step: Simulation step
            sim_time: Simulation time in seconds
            agents: Live Drone objects
        """
        now = time.time()
        cell_occupancy = defaultdict(list)

        for drone in agents:
            agent_id = drone.agent_id
            col, row = int(drone.col), int(drone.row)
            pos = (float(drone.pos.x), float(drone.pos.y))

            # ---- trajectory ----
            self.agents[agent_id]["trajectory"].append({
                "time": {
                    "step": step,
                    "sim_time": sim_time,
                    "world_time": now,
                },
                "cell": (col, row),
                "pos": pos,
            })

            # ---- heatmap ----
            self._cell_visits[(col, row)] += 1

            # ---- distance ----
            if agent_id in self._last_cell:
                if self._last_cell[agent_id] != (col, row):
                    self.agents[agent_id]["distance_cells"] += 1.0
            self._last_cell[agent_id] = (col, row)

            # ---- energy ----
            if hasattr(drone, "power"):
                self.agents[agent_id]["energy"].append({
                    "time": {
                        "step": step,
                        "sim_time": sim_time,
                        "world_time": now,
                    },
                    "battery": float(drone.power.level),
                    "battery_pct": float(drone.power.percent()),
                })

            cell_occupancy[(col, row)].append(agent_id)

        self._detect_collisions(step, sim_time, cell_occupancy)

    # ======================================================
    # Action-level logging
    # ======================================================
    def record_action(
        self,
        agent_id: str,
        kind: str,
        cell: Tuple[int, int],
        parcel,
        step: int,
        sim_time: float,
        world_time: Optional[float] = None,
    ):
        """
        Record a discrete agent action.
        """
        now = world_time if world_time is not None else time.time()

        entry = {
            "type": kind,
            "cell": tuple(cell),
            "time": {
                "step": step,
                "sim_time": sim_time,
                "world_time": now,
            },
        }

        parcel_id = getattr(parcel, "id", None)
        if parcel_id is not None:
            entry["parcel_id"] = parcel_id

        self.agents[agent_id]["actions"].append(entry)

        # ---- parcel contention ----
        if kind == "pick" and parcel_id is not None:
            self._parcel_targets[parcel_id].add(agent_id)
            self._parcel_chase_start[parcel_id][agent_id] = step
            self.parcels[parcel_id]["pick_attempts"].append(entry)

        if kind == "drop" and parcel_id is not None:
            self.agents[agent_id]["tasks"].append({
                "parcel_id": parcel_id,
                "time": entry["time"],
            })
            self.parcels[parcel_id]["delivered_by"] = agent_id
            self.parcels[parcel_id]["delivered_at"] = sim_time
            self._resolve_parcel(parcel_id, agent_id, step, sim_time)

        if kind in ("pick_failed", "drop_failed") and parcel_id is not None:
            self._abort_chase(parcel_id, agent_id, step, sim_time)

    # ======================================================
    # Collision detection
    # ======================================================
    def _detect_collisions(self, step: int, sim_time: float, cell_map: Dict):
        for cell, occupants in cell_map.items():
            if len(occupants) > 1:
                event = {
                    "type": "collision",
                    "cell": cell,
                    "agents": list(occupants),
                    "time": {
                        "step": step,
                        "sim_time": sim_time,
                        "world_time": time.time(),
                    },
                }
                self.fleet_events.append(event)

                for a in occupants:
                    if a in self.agents:
                        self.agents[a]["collisions"].append(event)

    # ======================================================
    # Contention and inefficiency detection
    # ======================================================
    def _resolve_parcel(self, parcel_id: int, winner: str, step: int, sim_time: float):
        contenders = self._parcel_targets.get(parcel_id, set())
        if len(contenders) > 1:
            losers = [a for a in contenders if a != winner]
            self.fleet_events.append({
                "type": "contention_resolved",
                "parcel_id": parcel_id,
                "winner": winner,
                "losers": losers,
                "time": {"step": step, "sim_time": sim_time},
            })
            for a in losers:
                start = self._parcel_chase_start[parcel_id].get(a)
                if start is not None:
                    self.fleet_events.append({
                        "type": "wasted_trajectory",
                        "agent": a,
                        "parcel_id": parcel_id,
                        "steps_wasted": step - start,
                        "time": {"step": step, "sim_time": sim_time},
                    })

        self._parcel_targets.pop(parcel_id, None)
        self._parcel_chase_start.pop(parcel_id, None)

    def _abort_chase(self, parcel_id: int, agent_id: str, step: int, sim_time: float):
        if parcel_id in self._parcel_targets:
            self._parcel_targets[parcel_id].discard(agent_id)
            start = self._parcel_chase_start.get(parcel_id, {}).get(agent_id)
            if start is not None:
                self.fleet_events.append({
                    "type": "aborted_chase",
                    "agent": agent_id,
                    "parcel_id": parcel_id,
                    "steps_spent": step - start,
                    "time": {"step": step, "sim_time": sim_time},
                })

    # ======================================================
    # Finalization and export
    # ======================================================
    def finalize(self):
        for agent_id, data in self.agents.items():
            if data["energy"]:
                data["final_battery"] = data["energy"][-1]["battery_pct"]

    def export(self) -> Dict:
        """
        Export a fully self-contained flight manifest.
        """
        self.finalize()
        return {
            "run_id": self.run_id,
            "planner": self.planner,
            "started_at": self.started_at,
            "agents": list(self.agents.values()),
            "parcels": list(self.parcels.values()),
            "fleet_events": self.fleet_events,
            "heatmap": self.export_heatmap(),
        }

    def export_heatmap(self) -> Dict:
        return {
            "cell_visits": [
                {"cell": cell, "count": count}
                for cell, count in self._cell_visits.items()
            ]
        }

    # ======================================================
    # Plot-ready helpers
    # ======================================================
    def get_agent_paths(self):
        return {
            aid: [t["cell"] for t in data["trajectory"]]
            for aid, data in self.agents.items()
        }

    def get_energy_curves(self):
        return {
            aid: [(e["time"]["step"], e["battery_pct"]) for e in data["energy"]]
            for aid, data in self.agents.items()
        }

    def get_collision_points(self):
        return [
            (e["cell"], e["time"]["step"])
            for e in self.fleet_events
            if e["type"] == "collision"
        ]

    def tick_agent(
            self,
            agent_id: str,
            drone,
            step: int,
            sim_time: float,
    ):
        """
        Record per-agent state for a single simulation step.

        This method logs trajectory, distance, energy, and heatmap data.
        It is intentionally agent-scoped and does NOT perform fleet checks.
        """
        now = time.time()

        col, row = int(drone.col), int(drone.row)
        pos = (float(drone.pos.x), float(drone.pos.y))

        agent = self.agents[agent_id]

        # ---- trajectory ----
        agent["trajectory"].append({
            "time": {
                "step": step,
                "sim_time": sim_time,
                "world_time": now,
            },
            "cell": (col, row),
            "pos": pos,
        })

        # ---- heatmap ----
        self._cell_visits[(col, row)] += 1

        # ---- distance ----
        last = self._last_cell.get(agent_id)
        if last is not None and last != (col, row):
            agent["distance_cells"] += 1.0
        self._last_cell[agent_id] = (col, row)

        # ---- energy ----
        if hasattr(drone, "power"):
            agent["energy"].append({
                "time": {
                    "step": step,
                    "sim_time": sim_time,
                    "world_time": now,
                },
                "battery": float(drone.power.level),
                "battery_pct": float(drone.power.percent()),
            })

    def tick_fleet(
            self,
            agents: list,
            step: int,
            sim_time: float,
    ):
        """
        Perform fleet-level checks AFTER all agents have been updated
        for the current step.

        Detects:
        - collisions
        - co-occupancy
        """

        from collections import defaultdict

        cell_occupancy = defaultdict(list)

        # Use recorder-tracked agent positions, NOT drone internals
        for agent_id, cell in self._last_cell.items():
            if cell is None:
                continue
            cell_occupancy[cell].append(agent_id)

        self._detect_collisions(step, sim_time, cell_occupancy)

    def record_initial_plan(
            self,
            agent_id: str,
            plan: list,
            confidence: float,
            projected_battery: Optional[float],
            step: int,
            sim_time: float,
    ):
        self.agents[agent_id]["planning"]["initial_plan"] = {
            "time": {
                "step": step,
                "sim_time": sim_time,
                "world_time": time.time(),
            },
            "plan": plan,
            "confidence": confidence,
            "projected_battery": projected_battery,
        }

        self.agents[agent_id]["planning"]["confidence"].append({
            "time": {
                "step": step,
                "sim_time": sim_time,
            },
            "value": confidence,
            "kind": "prior",
        })

    def record_plan_change(
            self,
            agent_id: str,
            old_step: dict,
            new_step: dict,
            estimated_cost: float,
            battery_before: float,
            battery_after: float,
            prior_confidence: float,
            posterior_confidence: float,
            reason: str,
            step: int,
            sim_time: float,
    ):
        event = {
            "type": "plan_replacement",
            "time": {
                "step": step,
                "sim_time": sim_time,
                "world_time": time.time(),
            },
            "old_step": old_step,
            "new_step": new_step,
            "energy": {
                "estimated_cost": estimated_cost,
                "battery_before": battery_before,
                "battery_after": battery_after,
            },
            "confidence": {
                "prior": prior_confidence,
                "posterior": posterior_confidence,
            },
            "reason": reason,
        }

        self.agents[agent_id]["planning"]["plan_events"].append(event)

        self.agents[agent_id]["planning"]["confidence"].append({
            "time": {
                "step": step,
                "sim_time": sim_time,
            },
            "value": posterior_confidence,
            "kind": "posterior",
        })

    def record_confidence_update(
            self,
            agent_id: str,
            confidence: float,
            modifier: float,
            note: str,
            step: int,
            sim_time: float,
    ):
        self.agents[agent_id]["planning"]["confidence"].append({
            "time": {
                "step": step,
                "sim_time": sim_time,
            },
            "value": confidence,
            "modifier": modifier,
            "note": note,
        })
