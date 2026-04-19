# src/strategies/single_task_bundle_cbba_decentralized_auction.py

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .base import BaseStrategy


class SingleTaskBundlesCBBADecentralizedStrategy(BaseStrategy):
    """
    Single-task CBBA decentralized auction strategy.

    Properties:
    - Exactly one task per agent at a time (MAX_BUNDLE_SIZE = 1)
    - Fully decentralized
    - Robust to message delay
    - No per-agent clocks
    - Uses simulation step as global time
    - No winner announcements
    - Convergent under asynchronous delivery
    """

    topology = "decentralized"
    enable_reactive_fallback = False
    requires_plan_completion_before_requery = False

    BID_INTERVAL = 5
    BID_EXPIRY = 20
    FEASIBILITY_MARGIN = 3.0

    def __init__(self, physics=None):
        super().__init__()
        self.agent_id = None
        self.physics = physics

        self.y = {}  # task_id -> winning agent
        self.z = {}  # task_id -> winning bid
        self.t = {}  # task_id -> timestamp
        self.committed_tasks = {}  # task_id -> agent_id
        self.last_y_snapshot = None
        self.y_stable_steps = 0

        # cache
        self.last_bid_step = -self.BID_INTERVAL
        self.last_processed_step = -1
        self.last_decision_step = -1
        self.pending_messages = []
        # task_id -> last winner
        self.last_winner = {}
        # task_id -> last change step
        self.last_change_step = {}
        self._last_decision = {
            "mode": "idle",
            "agent_id": None,
            "plan": [],
            "confidence": 0.0,
            "strategy": "single-task-cbba",
        }
        self.last_decision_step = None
        self.active_task = None
        self.bundle = []  # Ordered list of tasks you believe you own
        self.path = []  # Execution order (usually same as bundle)
        self.MAX_BUNDLE_SIZE = 1  # Single task only!

    def set_drone(self, drone):
        super().set_drone(drone)
        self.agent_id = drone.agent_id

        if self.physics is None and hasattr(drone, "physics"):
            self.physics = drone.physics

    # -----------------------------
    # Messaging
    # -----------------------------
    def _broadcast_cbba_state(self, step: int):
        payload = {
            "type": "cbba",
            "agent_id": self.agent_id,
            "y": dict(self.y),
            "z": dict(self.z),
            "t": dict(self.t),
        }

        for agent in self.drone.terrain.agents.values():
            if agent.agent_id != self.agent_id:
                self.drone.send_message(agent, payload, step)

    def _process_messages(self, step: int):
        if step == self.last_processed_step:
            return

        self.last_processed_step = step

        if self.drone.inbox:
            self.pending_messages.extend(self.drone.inbox)
            self.drone.inbox.clear()

        inbox = list(self.pending_messages)
        self.pending_messages.clear()

        for sender, msg in inbox:
            if not isinstance(msg, dict):
                continue
            if msg.get("type") != "cbba":
                continue

            yj = msg.get("y", {})
            zj = msg.get("z", {})
            tj = msg.get("t", {})

            all_tasks = set(self.y.keys()) | set(yj.keys())

            for task in all_tasks:
                yi = self.y.get(task)
                zi = self.z.get(task, float("-inf"))
                ti = self.t.get(task, -1)

                yj_task = yj.get(task)
                zj_task = zj.get(task, float("-inf"))
                tj_task = tj.get(task, -1)

                update = False

                if tj_task > ti:
                    update = True
                elif tj_task == ti:
                    if zj_task > zi:
                        update = True
                    elif zj_task == zi:
                        if (yj_task or "") > (yi or ""):
                            update = True

                if update:
                    self.y[task] = yj_task
                    self.z[task] = zj_task
                    self.t[task] = tj_task

    def _rebuild_bundle_from_memory(self):
        """Simple rebuild for single-task bundles."""
        agent_id = self.agent_id
        owned = [t for t, owner in self.y.items() if owner == agent_id]

        # For single-task, bundle is just the owned tasks (max 1)
        self.bundle = owned[:self.MAX_BUNDLE_SIZE]
        self.path = list(self.bundle)

    def _reconcile_bundle(self):
        """Simple reconciliation for single-task bundles."""
        if not self.bundle:
            return

        agent_id = self.agent_id

        for i, task in enumerate(self.bundle):
            owner = self.y.get(task)
            if owner != agent_id:
                # Lost this task. For single-task, just clear the bundle.
                removed = self.bundle[i:]
                self.bundle = self.bundle[:i]
                self.path = list(self.bundle)

                print(f"[SINGLE-TASK-CBBA-DEBUG][{agent_id}] Lost {task}, clearing bundle {removed}")
                return

    # -----------------------------
    # Core logic
    # -----------------------------

    def decide(self, snapshot: Dict) -> Dict:
        plan_obj = self.request_plan(snapshot)
        print(f"[SINGLE-TASK-CBBA][{self.agent_id}] request_plan returned:", plan_obj)

        mode = plan_obj.get("mode")

        # Pass through semantic modes unchanged
        if mode in ("idle", "wait", "abort"):
            return plan_obj

        # Otherwise treat as valid plan
        return {
            "mode": "plan",
            "plan": plan_obj,
        }

    def _extract_step(self, snapshot):
        return snapshot.get("step", 0)

    def request_plan(self, snapshot):
        agent_id = snapshot["agent"]["id"]
        step = self._extract_step(snapshot)

        def dbg(msg):
            print(f"[SINGLE-TASK-CBBA-DEBUG][{agent_id}][step={step}] {msg}")

        # -------------------------
        # Safe init
        # -------------------------
        if not hasattr(self, "last_y_snapshot"):
            self.last_y_snapshot = None
        if not hasattr(self, "y_stable_steps"):
            self.y_stable_steps = 0
        if not hasattr(self, "last_decision_step"):
            self.last_decision_step = None
        if not hasattr(self, "active_task"):
            self.active_task = None

        if step == self.last_decision_step:
            return self._last_decision
        self.last_decision_step = step

        # -------------------------
        # Always run CBBA logic
        # -------------------------
        self._process_messages(step)
        self._reconcile_bundle()
        self._rebuild_bundle_from_memory()

        dbg(f"Y table after inbox: {self.y}")
        dbg(f"Bundle after rebuild: {self.bundle}")

        # -------------------------
        # Active parcels
        # -------------------------
        parcels = [
            p for p in snapshot["all_parcels"]
            if not p.get("picked") and not p.get("delivered")
        ]

        if not parcels:
            result = {
                "mode": "idle",
                "agent_id": agent_id,
                "plan": [],
                "confidence": 0.0,
                "strategy": "single-task-cbba",
            }
            self._last_decision = result
            return result

        active_ids = {p["id"] for p in parcels}

        # -------------------------
        # Prune dead tasks globally
        # -------------------------
        for task in list(self.y.keys()):
            if task not in active_ids:
                dbg(f"Pruning dead task {task}")
                self.y.pop(task, None)
                self.z.pop(task, None)
                self.t.pop(task, None)

        # -------------------------
        # Single-task bidding (MAX_BUNDLE_SIZE = 1)
        # -------------------------
        if len(self.bundle) < self.MAX_BUNDLE_SIZE:
            scored = []

            for p in parcels:
                task_id = p["id"]

                owner = self.y.get(task_id)
                if owner is not None and owner != agent_id:
                    continue

                if task_id in self.bundle:
                    continue

                gain = self._marginal_gain(snapshot, p)
                if gain is None:
                    continue

                scored.append((task_id, gain))
            dbg(f"Scores: {dict(scored)}")

            if scored:
                best_task, best_score = max(scored, key=lambda x: x[1])

                zi = self.z.get(best_task, float("-inf"))
                yi = self.y.get(best_task)

                if yi is None or yi == agent_id or best_score > zi:
                    self.y[best_task] = agent_id
                    self.z[best_task] = best_score
                    self.t[best_task] = step

                    dbg(f"Bidding on {best_task} score={best_score:.3f}")
                    self._broadcast_cbba_state(step)

        # -------------------------
        # Convergence detection
        # -------------------------
        filtered_y = {k: v for k, v in self.y.items() if k in active_ids}
        y_snapshot = tuple(sorted(filtered_y.items()))

        if y_snapshot == self.last_y_snapshot:
            self.y_stable_steps += 1
        else:
            self.y_stable_steps = 0

        dbg(f"Y snapshot: {filtered_y}, stable_steps={self.y_stable_steps}")
        self.last_y_snapshot = y_snapshot

        if self.y_stable_steps < 2:
            result = {
                "mode": "wait",
                "agent_id": agent_id,
                "reason": "cbba_not_converged",
            }
            self._last_decision = result
            return result

        # -------------------------
        # Execute single task from bundle
        # -------------------------
        if not self.bundle:
            result = {
                "mode": "idle",
                "agent_id": agent_id,
                "reason": "empty_bundle",
            }
            self._last_decision = result
            return result

        plan = []
        station = snapshot["nearest_station"]

        for task_id in self.bundle:
            parcel = next((p for p in parcels if p["id"] == task_id), None)
            if parcel is None:
                continue

            pickup = [parcel["col"], parcel["row"]]

            drop_cell = self._choose_drop_cell(
                pick=(pickup[0], pickup[1]),
                station=station,
            )

            drop = [drop_cell[0], drop_cell[1]]

            plan.append({
                "pickup": pickup,
                "dropoff": drop,
                "weight": parcel.get("weight", 1.0),
            })

        if not plan:
            result = {
                "mode": "wait",
                "agent_id": agent_id,
                "reason": "no_valid_tasks",
            }
            self._last_decision = result
            return result

        dbg(f"EXECUTING single-task bundle {self.bundle}")
        self.active_task = self.bundle[0]

        result = {
            "agent_id": agent_id,
            "plan": plan,
            "confidence": 0.9,
            "strategy": "single-task-cbba",
        }

        self._last_decision = result
        return result

    # -----------------------------
    # Cost model (simplified for single-task)
    # -----------------------------

    def _marginal_gain(self, snapshot, parcel):
        """Simple gain calculation for single-task CBBA."""
        cost = self._estimate_cost(snapshot, parcel)
        if cost is None:
            return None

        # CBBA maximizes utility, so convert cost to utility
        return -cost

    def _estimate_cost(self, snapshot: Dict, parcel: Dict) -> Optional[float]:
        """Estimate cost for executing a single task."""
        agent = snapshot["agent"]
        station = snapshot.get("nearest_station")
        if not station:
            print(f"[SINGLE-TASK-CBBA][{self.agent_id}] No station in snapshot")
            return None

        cur = (agent["col"], agent["row"])
        pick = (parcel["col"], parcel["row"])
        drop = (
            station["col"] + station["w"] // 2,
            station["row"] + station["h"] // 2,
        )

        d1 = self.physics.distance(cur, pick)
        d2 = self.physics.distance(pick, drop)

        cost = (
                self.physics.energy_cost(d1, 0.0) +
                self.physics.energy_cost(d2, parcel["weight"]) +
                2 * self.physics.energy_cost(0.0, parcel["weight"])
        )

        battery = float(self.drone.power.level)
        feasible = battery >= cost + self.FEASIBILITY_MARGIN

        print(
            f"[SINGLE-TASK-CBBA][{self.agent_id}] "
            f"parcel={parcel['id']} "
            f"battery={battery:.2f} "
            f"cost={cost:.2f} "
            f"margin={self.FEASIBILITY_MARGIN} "
            f"feasible={feasible}"
        )

        if not feasible:
            return None

        return cost

    def _choose_drop_cell(
            self,
            pick: tuple[int, int],
            station: Optional[Dict[str, Any]],
    ) -> tuple[int, int]:
        if not station:
            return pick

        terrain = self.drone.terrain if hasattr(self.drone, "terrain") else None
        if terrain is None:
            return (
                station["col"] + station["w"] // 2,
                station["row"] + station["h"] // 2,
            )

        real_station = terrain.get_station_at(station["col"], station["row"])
        if real_station is None:
            return pick

        free = real_station.free_cells(terrain)

        if not free:
            free = real_station.parking_cells()

        chosen = min(
            free,
            key=lambda c: abs(c[0] - pick[0]) + abs(c[1] - pick[1])
        )

        return chosen