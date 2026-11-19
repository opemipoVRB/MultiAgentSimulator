from typing import Dict, List, Optional, Tuple
from artifacts import Drone, Parcel
from strategies.base import BaseStrategy
from src.system.abort import AbortReason
from src import sim_clock


class CentralisedStrategy(BaseStrategy):
    """
    Authoritative centralized scheduler with explicit task ledger.

    This class owns the global truth about:
    - Which parcel belongs to whom
    - Which agents are idle or busy
    - What is feasible
    - Whether the mission is still solvable

    Agents do not infer anything. They only execute instructions.
    """

    enable_reactive_fallback = False
    requires_plan_completion_before_requery = True

    def __init__(self, physics, variant="hungarian"):
        """
        Authoritative centralized scheduler with explicit task ledger.

        Parameters
        ----------
        physics : Physics
            Physics engine for energy and distance calculations.
        assignment_mode : str, optional
            Assignment algorithm to use. Options:
            - "hungarian" : Optimal min-cost bipartite matching (default)
            - "greedy" : First-feasible greedy assignment
        """
        self.physics = physics
        self.terrain = None
        self.assignment_mode = variant

        # Task ledger: parcel_id -> record
        self.tasks: Dict[str, Dict] = {}

        # Agent registry: agent_id -> record
        self.agents: Dict[str, Dict] = {}

        self.abort_reason = None
        self._abort_emitted = False
        self.mission_complete = False

        self.last_progress_time = sim_clock.now()
        self.task_assignment_times = {}  # task_id -> timestamp

        print(f"[CENTRAL][INIT] Assignment mode: {self.assignment_mode}")

    # ============================================================
    # Public interface
    # ============================================================

    def step(self, *, terrain, agents: List[Drone], parcels: List[Parcel], now=None):
        """
        Perform one authoritative coordination cycle.
        """
        if now is None:
            now = sim_clock.now()

        self.terrain = terrain

        self._sync_agents(agents)
        self._sync_tasks(parcels)
        self._detect_stuck_agents(now)
        self._assign_tasks(now)
        self._repair_ledger()

        self._detect_global_failure()

    def decide(self, snapshot: Dict, agent_id: str) -> Dict:
        """
        Translate ledger state into a concrete directive.
        """
        # Check mission completion
        all_completed = all(
            t["state"] == "completed"
            for t in self.tasks.values()
        )

        if all_completed:
            self.mission_complete = True
            agent_snapshot = next(
                (a for a in snapshot["agents"] if a["id"] == agent_id),
                None
            )

            if agent_snapshot and snapshot.get("stations"):
                station = snapshot["stations"][0]
                agent_col = agent_snapshot["col"]
                agent_row = agent_snapshot["row"]

                # Check if agent is inside station
                in_station = (
                        station["col"] <= agent_col < station["col"] + station["w"] and
                        station["row"] <= agent_row < station["row"] + station["h"]
                )

                if in_station:
                    return {"mode": "idle"}

            return {"mode": "idle"}

        # Check abort condition
        if self.abort_reason is not None:
            return {"mode": "idle"}

        # Get agent record
        agent = self.agents.get(agent_id)
        if agent is None:
            return {"mode": "wait"}

        # Check if agent has a task assigned
        task_id = agent.get("task")
        if not task_id:
            return {"mode": "wait"}

        task = self.tasks.get(task_id)
        if not task:
            return {"mode": "wait"}

        parcel = next((p for p in snapshot["parcels"] if p["id"] == task["parcel_id"]), None)
        if not parcel:
            return {"mode": "wait"}

        station = snapshot["stations"][0] if snapshot.get("stations") else None
        if not station:
            return {"mode": "wait"}

        station_obj = self.terrain.stations[0]

        drop = station_obj.least_used_free_cell(
            terrain=self.terrain,
            ref_col=parcel["col"],
            ref_row=parcel["row"],
        )

        if drop is None:
            self.abort_reason = AbortReason.GLOBALLY_INFEASIBLE
            return {"mode": "idle"}

        return {
            "mode": "plan",
            "plan": {
                "plan": [
                    {
                        "pickup": (parcel["col"], parcel["row"]),
                        "dropoff": drop,
                        "weight": parcel["weight"],
                    }
                ],
                "confidence": 1.0,
                "strategy": "centralised-ledger",
                "narration": f"Authoritative task {parcel['id']} assigned",
            },
        }

    # ============================================================
    # Ledger synchronization
    # ============================================================

    def _sync_agents(self, drones: List[Drone]):
        for d in drones:
            if d.agent_id not in self.agents:
                self.agents[d.agent_id] = {
                    "state": "idle",
                    "task": None,
                    "last_seen": sim_clock.now(),
                }

            if d.power.level <= 0:
                self.agents[d.agent_id]["state"] = "dead"

    def _sync_tasks(self, parcels: List[Parcel]):
        for p in parcels:
            if p.id not in self.tasks:
                self.tasks[p.id] = {
                    "parcel_id": p.id,
                    "state": "unassigned",
                    "agent": None,
                }

            if p.delivered:
                if self.tasks[p.id]["state"] != "completed":
                    self.last_progress_time = sim_clock.now()
                self.tasks[p.id]["state"] = "completed"

    # ============================================================
    # Scheduling logic
    # ============================================================

    def _assign_tasks(self, now):
        """
        Dispatch tasks to idle agents using the configured assignment algorithm.

        This is the main task assignment entry point. It delegates to either:
        - _hungarian_assign_tasks: Optimal min-cost bipartite matching
        - _greedy_assign_tasks: First-feasible greedy assignment

        The assignment mode is set during initialization via the `assignment_mode`
        parameter.

        Parameters
        ----------
        now : float
            Current timestamp for progress tracking.

        Notes
        -----
        Both algorithms respect the same constraints:
        - Only assigns to agents in "idle" state
        - Only assigns tasks in "unassigned" state
        - Validates energy feasibility via _agent_can_complete
        - Updates task ledger and agent registry atomically
        """
        if self.assignment_mode == "hungarian":
            self._hungarian_assign_tasks(now)
        elif self.assignment_mode == "greedy":
            self._greedy_assign_tasks(now)
        elif self.assignment_mode == "auction":
            self._auction_assign_tasks(now)
        else:
            raise ValueError(
                f"Unknown assignment mode: {self.assignment_mode}. "
                f"Must be 'hungarian' or 'greedy'."
            )

    def _greedy_assign_tasks(self, now):
        """
         Greedy task assignment: assigns nearest feasible parcel to each idle agent.

         Algorithm
         ---------
         1. Iterate through unassigned tasks in order
         2. For each task, find all feasible agents
         3. Select the first feasible agent (no optimization)
         4. Remove agent from idle pool
         5. Repeat until no tasks or agents remain

         Complexity
         ----------
         O(T × A) where T = tasks, A = agents

         Characteristics
         ---------------
         - Fast: No optimization overhead
         - Deterministic: First-feasible selection
         - Suboptimal: Does not minimize global cost
         - Sequential: Tasks processed in ledger order

         Use Cases
         ---------
         - Baseline comparison
         - Low-latency requirements
         - Small-scale scenarios where optimality is not critical

         Parameters
         ----------
         now : float
             Current timestamp for progress tracking.
         """
        idle_agents = [
            aid for aid, a in self.agents.items()
            if a["state"] == "idle" and a.get("task") is None  # 🔥 ADD: and no pending task
        ]

        unassigned_tasks = [
            t for t in self.tasks.values()
            if t["state"] == "unassigned"
        ]

        if not idle_agents or not unassigned_tasks:
            return

        station = self._get_station_center()
        if not station:
            return

        assignments = []

        for task in unassigned_tasks:
            parcel = next(p for p in self.terrain.parcels if p.id == task["parcel_id"])

            feasible_agents = []
            for aid in idle_agents:
                drone = self.terrain.agents[aid]
                if self._agent_can_complete(drone, parcel, station):
                    feasible_agents.append(aid)

            if feasible_agents:
                chosen = feasible_agents[0]
                assignments.append((chosen, task["parcel_id"]))
                idle_agents.remove(chosen)  # Already does this, but task is assigned AFTER loop

        self.last_progress_time = sim_clock.now()

        # 🔥 CRITICAL: Mark tasks as assigned BEFORE updating agent records
        # This prevents double-assignment if _assign_tasks runs again before messages are sent
        for agent_id, parcel_id in assignments:
            self.tasks[parcel_id]["state"] = "assigned"
            self.tasks[parcel_id]["agent"] = agent_id
            self.task_assignment_times[parcel_id] = now

        # THEN update agent records
        for agent_id, parcel_id in assignments:
            self.agents[agent_id]["task"] = parcel_id
            print(f"[CENTRAL][GREEDY] agent={agent_id} task={parcel_id}")

    def _hungarian_assign_tasks(self, now):
        """
        Optimal task assignment via Hungarian algorithm (min-cost bipartite matching).

        Algorithm
        ---------
        1. Build cost matrix C where C[i,j] = energy cost for agent i to complete task j
        2. Solve min-cost assignment using Kuhn-Munkres algorithm
        3. Apply all feasible matches atomically

        Cost Function
        -------------
        cost(agent, task) = E_pickup + E_delivery
        where:
            E_pickup = energy_cost(agent → parcel, weight=0)
            E_delivery = energy_cost(parcel → station, weight=parcel.weight)

        Infeasible assignments are marked with penalty cost 1e9.

        Complexity
        ----------
        O(n³) where n = max(agents, tasks)
        Implemented via scipy.optimize.linear_sum_assignment

        Characteristics
        ---------------
        - Optimal: Minimizes total system energy cost
        - Global: Considers all agent-task pairs simultaneously
        - Deterministic: Unique solution for given cost matrix
        - Batch: All assignments computed before application

        Use Cases
        ---------
        - Scientific baseline for comparison with decentralized strategies
        - Scenarios where global optimality justifies computational cost
        - Benchmark for measuring coordination quality

        Parameters
        ----------
        now : float
            Current timestamp for progress tracking.

        Raises
        ------
        ImportError
            If scipy is not available.

        References
        ----------
        Kuhn, H. W. (1955). "The Hungarian method for the assignment problem."
        Naval Research Logistics Quarterly, 2(1-2), 83-97.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError(
                "Hungarian assignment requires scipy. "
                "Install with: pip install scipy"
            )

        idle_agents = [
            aid for aid, a in self.agents.items()
            if a["state"] == "idle"
        ]

        unassigned_tasks = [
            t for t in self.tasks.values()
            if t["state"] == "unassigned"
        ]

        if not idle_agents or not unassigned_tasks:
            return

        station = self._get_station_center()
        if not station:
            return

        # Build cost matrix: rows = agents, cols = tasks
        cost_matrix = []

        for aid in idle_agents:
            drone = self.terrain.agents[aid]
            row = []

            for task in unassigned_tasks:
                parcel = next(p for p in self.terrain.parcels if p.id == task["parcel_id"])

                if self._agent_can_complete(drone, parcel, station):
                    # Compute energy cost for pickup + delivery
                    phys = self._physics_for(drone)
                    cost = (
                            phys.energy_cost(
                                self.physics.distance((drone.col, drone.row), (parcel.col, parcel.row)),
                                0.0
                            ) +
                            phys.energy_cost(
                                self.physics.distance((parcel.col, parcel.row), station),
                                parcel.weight
                            ) +
                            2.0 * phys.pick_drop_cost(parcel.weight)
                    )
                    cost += self.time_weight * self.trip_time((drone.col, drone.row), (parcel.col, parcel.row), station, speed=getattr(drone, "speed_scale", 1.0))
                    row.append(cost)
                else:
                    # Infeasible assignment gets penalty cost
                    row.append(1e9)

            cost_matrix.append(row)

        # Solve optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Apply assignments
        self.last_progress_time = sim_clock.now()

        for agent_idx, task_idx in zip(row_ind, col_ind):
            agent_id = idle_agents[agent_idx]
            task = unassigned_tasks[task_idx]

            # Skip if infeasible
            if cost_matrix[agent_idx][task_idx] >= 1e9:
                continue

            parcel_id = task["parcel_id"]

            self.tasks[parcel_id]["state"] = "assigned"
            self.tasks[parcel_id]["agent"] = agent_id
            self.agents[agent_id]["task"] = parcel_id
            self.task_assignment_times[parcel_id] = now

            print(
                f"[CENTRAL][HUNGARIAN] agent={agent_id} task={parcel_id} "
                f"cost={cost_matrix[agent_idx][task_idx]:.1f}"
            )

    def report_task_completed(self, agent_id: str, task_id: str):
        """
        Authoritative completion event from execution layer.
        """

        task = self.tasks.get(task_id)
        agent = self.agents.get(agent_id)

        if not task or not agent:
            print(f"[CENTRAL][COMPLETE-IGNORED] agent={agent_id} task={task_id} unknown")
            return

        # Validate consistency
        if task["agent"] != agent_id:
            print(
                f"[CENTRAL][COMPLETE-MISMATCH] agent={agent_id} "
                f"task={task_id} ledger_agent={task['agent']}"
            )
            return

        # Apply state change atomically
        task["state"] = "completed"
        task["agent"] = None

        agent["state"] = "idle"
        agent["task"] = None

        # Clear assignment time
        if task_id in self.task_assignment_times:
            del self.task_assignment_times[task_id]

        self.last_progress_time = sim_clock.now()

        print(f"[CENTRAL][COMPLETE] agent={agent_id} task={task_id}")

    def _repair_ledger(self):
        for tid, task in self.tasks.items():
            if task["state"] == "assigned":
                aid = task["agent"]

                if not aid or aid not in self.agents:
                    print(f"[CENTRAL][REPAIR] Task {tid} orphaned, unassigning")
                    task["state"] = "unassigned"
                    task["agent"] = None
                    continue

                agent = self.agents[aid]
                if agent.get("task") != tid:
                    print(f"[CENTRAL][REPAIR] Mismatch: task {tid} vs agent {aid}, unassigning")
                    task["state"] = "unassigned"
                    task["agent"] = None

    def _auction_assign_tasks(self, now):
        idle_agents = [
            aid for aid, a in self.agents.items()
            if a["state"] == "idle" and a.get("task") is None
        ]

        unassigned_tasks = [
            t for t in self.tasks.values()
            if t["state"] == "unassigned"
        ]

        if not idle_agents or not unassigned_tasks:
            return

        station = self._get_station_center()
        if not station:
            return

        bids = []  # (cost, agent_id, parcel_id)

        for task in unassigned_tasks:
            parcel = next(p for p in self.terrain.parcels if p.id == task["parcel_id"])

            for aid in idle_agents:
                drone = self.terrain.agents[aid]

                if not self._agent_can_complete(drone, parcel, station):
                    continue

                dist1 = self.physics.distance((drone.col, drone.row), (parcel.col, parcel.row))
                dist2 = self.physics.distance((parcel.col, parcel.row), station)

                phys = self._physics_for(drone)
                cost = (
                        phys.energy_cost(dist1, 0.0)
                        + phys.energy_cost(dist2, parcel.weight)
                        + 2.0 * phys.pick_drop_cost(parcel.weight)
                )

                bids.append((cost, aid, parcel.id))

        if not bids:
            return

        bids.sort(key=lambda x: x[0])

        assigned_agents = set()
        assigned_tasks = set()

        for cost, aid, pid in bids:
            if aid in assigned_agents or pid in assigned_tasks:
                continue

            self.tasks[pid]["state"] = "assigned"
            self.tasks[pid]["agent"] = aid
            self.agents[aid]["task"] = pid
            self.task_assignment_times[pid] = now

            assigned_agents.add(aid)
            assigned_tasks.add(pid)

            print(f"[CENTRAL][AUCTION] agent={aid} task={pid} cost={cost:.2f}")

        self.last_progress_time = sim_clock.now()

    # ============================================================
    # Failure detection
    # ============================================================
    def _detect_stuck_agents(self, now):
        """
        Detect agents that have been executing the same task for too long
        without completing it, and unassign them.

        This prevents deadlock when an agent gets stuck on a task due to:
        - Navigation failure
        - Parcel disappeared
        - Station unreachable
        - State machine issue

        Parameters
        ----------
        now : float
            Current timestamp for timeout calculation.
        """
        STUCK_TIMEOUT = 30.0  # 30 seconds

        for task_id, task_rec in list(self.tasks.items()):
            if task_rec["state"] != "assigned":
                continue

            assignment_time = self.task_assignment_times.get(task_id)
            if assignment_time is None:
                continue

            elapsed = now - assignment_time
            if elapsed > STUCK_TIMEOUT:
                agent_id = task_rec["agent"]
                print(
                    f"[CENTRAL][STUCK] agent={agent_id} task={task_id} "
                    f"stuck for {elapsed:.1f}s - unassigning"
                )

                # Unassign the stuck task
                task_rec["state"] = "unassigned"
                task_rec["agent"] = None

                if agent_id in self.agents:
                    agent_rec = self.agents[agent_id]
                    agent_rec["state"] = "idle"
                    agent_rec["task"] = None

                # Clear assignment time
                del self.task_assignment_times[task_id]

    def debug_state(self):
        """Print current ledger state for debugging"""
        print("\n[CENTRAL][DEBUG STATE]")
        print(f"  Agents: {len(self.agents)}")
        for aid, rec in self.agents.items():
            print(f"    {aid}: state={rec['state']} task={rec.get('task')}")

        print(f"  Tasks: {len(self.tasks)}")
        task_states = {}
        for tid, trec in self.tasks.items():
            state = trec['state']
            task_states[state] = task_states.get(state, 0) + 1
        for state, count in task_states.items():
            print(f"    {state}: {count}")

        print("\n[CENTRAL][INCOMPLETE TASKS]")
        for tid, trec in self.tasks.items():
            if trec['state'] != 'completed':
                print(f"    {tid}: state={trec['state']} assigned_to={trec.get('agent')}")
        print()

    def is_globally_feasible(self, snapshot=None):
        return self._globally_feasible_authoritative()

    def _globally_feasible_authoritative(self) -> bool:
        if not self.terrain:
            return True

        station = self._get_station_center()
        if not station:
            return False

        remaining_tasks = [
            t for t in self.tasks.values()
            if t["state"] != "completed"
        ]

        if not remaining_tasks:
            return True

        alive_agents = [
            aid for aid, a in self.agents.items()
            if a["state"] != "dead"
        ]

        for task in remaining_tasks:
            parcel = next(p for p in self.terrain.parcels if p.id == task["parcel_id"])

            for aid in alive_agents:
                drone = self.terrain.agents[aid]
                if self._agent_can_complete(drone, parcel, station):
                    return True

        return False

    def _detect_global_failure(self):
        unfinished = [t for t in self.tasks.values() if t["state"] != "completed"]
        agents_exist = any(a["state"] != "dead" for a in self.agents.values())

        if not unfinished:
            if not self.mission_complete:
                print("[CENTRAL][MISSION COMPLETE] All tasks successfully delivered")
                self.mission_complete = True
            return

        if not self.is_globally_feasible():
            if not self.globally_infeasible:
                print("[CENTRAL][ABORT] No agent can complete any remaining task")
            self.abort_reason = AbortReason.GLOBALLY_INFEASIBLE
            self.globally_infeasible = True
            return

        if not agents_exist:
            print("[CENTRAL][ABORT] No agents remain alive")
            self.abort_reason = AbortReason.GLOBALLY_INFEASIBLE
            self.globally_infeasible = True
            return

        # Deadlock only if nothing has progressed for too long
        elapsed = sim_clock.now() - self.last_progress_time
        if elapsed > 10.0:
            self.debug_state()
            print(f"[CENTRAL][ABORT] Global deadlock: no progress for {elapsed:.1f}s")
            self.abort_reason = AbortReason.GLOBALLY_INFEASIBLE

    # ============================================================
    # Feasibility
    # ============================================================

    def _physics_for(self, drone):
        """The energy model of one drone: its declared spec, measured in the pre-mission calibration."""
        cache = self.__dict__.setdefault("_declared_physics", {})
        if drone.agent_id not in cache:
            from src.calibration import calibrate
            from src.energy import DeclaredPhysics
            cal = calibrate(drone)
            cache[drone.agent_id] = DeclaredPhysics(self.physics, cal["unloaded_per_cell"], cal["weight_sensitivity"])
        return cache[drone.agent_id]

    def _agent_can_complete(self, agent: Drone, parcel: Parcel, station: Tuple[int, int]):
        dist_to_parcel = self.physics.distance(
            a=(agent.col, agent.row),
            b=(parcel.col, parcel.row),
        )
        dist_to_station = self.physics.distance(
            a=(parcel.col, parcel.row),
            b=station,
        )

        phys = self._physics_for(agent)
        cost = (
                phys.energy_cost(dist_to_parcel, 0.0)
                + phys.energy_cost(dist_to_station, parcel.weight)
                + 2.0 * phys.pick_drop_cost(parcel.weight)
        )

        return agent.power.level > cost + agent.reserve_floor

    # ============================================================
    # Utilities
    # ============================================================

    def _get_station_center(self):
        if not self.terrain.stations:
            return None
        s = self.terrain.stations[0]
        return s.col + s.w // 2, s.row + s.h // 2

