# src/strategies/cbba_decentralized_auction.py

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .base import BaseStrategy


class LoadBalancedCBBADecentralizedStrategy(BaseStrategy):
    """
    Correct decentralized auction-based load balancing.

    Properties:
    - One task per agent at a time
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


    def __init__(self, physics=None):
        super().__init__()
        self.agent_id = None
        self.physics = physics

        # # task_id -> {agent_id: bid_value}
        # self.bid_table: Dict[str, Dict[str, float]] = defaultdict(dict)
        # # task_id -> last update step
        # self.bid_timestamp: Dict[str, int] = {}
        # self.first_seen_step = {}
        # self.committed_tasks = {}  # task_id -> agent_id

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
            "strategy": "lb-cbba",
        }
        self.last_decision_step = None
        self.active_task = None
        self.bundle = []  # Ordered list of tasks you believe you own
        self.path = []  # Execution order (usually same as bundle)
        self.c = {}  # Marginal gains for tasks: task_id -> score
        self.MAX_BUNDLE_SIZE = 3  # or 5, configurable
        # False: the published fixed bundle length. True: this agent's bundle length is its share of the open tasks by capability (speed and battery of the
        # whole declared fleet), the same rule SPPDEC uses (`capability_caps`); an arm sets it, the published arm leaves it off.
        self.capability_bundle = False

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

        self.drone.broadcast(payload, step)

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

                # Conflict resolution of published CBBA (Choi, Brunet, How 2009): the higher winning bid wins, a tie goes to the lower
                # agent id; the timestamp only orders two reports about the same winner (the newer one replaces the older).
                if yj_task == yi:
                    update = tj_task > ti
                elif zj_task != zi:
                    update = zj_task > zi
                else:
                    update = (yj_task or "") < (yi or "") if yi is not None and yj_task is not None else yj_task is not None

                if update:
                    self.y[task] = yj_task
                    self.z[task] = zj_task
                    self.t[task] = tj_task

    def _rebuild_bundle_from_memory(self):
        agent_id = self.agent_id

        owned = [t for t, owner in self.y.items() if owner == agent_id]

        with_pos = [t for t in owned if t in self.c]
        without_pos = [t for t in owned if t not in self.c]

        # Sort by stored insertion index
        with_pos_sorted = sorted(with_pos, key=lambda t: self.c[t][1])

        self.bundle = with_pos_sorted + without_pos
        self.path = list(self.bundle)

    def _reconcile_bundle(self):
        if not self.bundle:
            return

        agent_id = self.agent_id

        for i, task in enumerate(self.bundle):
            owner = self.y.get(task)
            if owner != agent_id:
                # Lost this task. Must prune suffix.
                removed = self.bundle[i:]

                for t in removed:
                    self.c.pop(t, None)

                self.bundle = self.bundle[:i]
                self.path = list(self.bundle)

                print(f"[CBBA-DEBUG][{agent_id}] Lost {task}, pruning suffix {removed}")
                return

    # -----------------------------
    # Core logic
    # -----------------------------

    def decide(self, snapshot: Dict) -> Dict:
        plan_obj = self.request_plan(snapshot)
        print(f"[CBBA][{self.agent_id}] request_plan returned:", plan_obj)

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
            print(f"[CBBA-DEBUG][{agent_id}][step={step}] {msg}")

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
                "strategy": "lb-cbba",
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
                self.c.pop(task, None)

        # -------------------------
        # Multi-task bidding (true CBBA)
        # -------------------------
        if len(self.bundle) < self._bundle_limit(snapshot, parcels):
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

                if not self._can_add_to_bundle(snapshot, task_id):
                    dbg(f"Skipping {task_id}, infeasible under battery constraint")
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
        # Execute full bundle
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

        dbg(f"EXECUTING bundle {self.bundle}")
        self.active_task = self.bundle[0]

        result = {
            "agent_id": agent_id,
            "plan": plan,
            "confidence": 0.9,
            "strategy": "lb-cbba",
        }

        self._last_decision = result
        return result

    # -----------------------------
    # Cost model
    # -----------------------------

    def _bundle_limit(self, snapshot, parcels):
        """Bundle length of this agent: the fixed length, or with `capability_bundle` its capability share of the open tasks (declared start information only:
        speeds, consumption rates and starting batteries of the fleet, own current battery)."""
        if not self.capability_bundle:
            return self.MAX_BUNDLE_SIZE
        from src.strategies.sppdec.cbba_assumed_bids import capability_caps
        station = snapshot.get("nearest_station")
        if not station or not parcels or not hasattr(self.drone, "fleet_ids"):
            return self.MAX_BUNDLE_SIZE
        drop = (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
        agents = []
        for aid in self.drone.fleet_ids():
            spec = self.drone.declared_spec(aid)
            if spec is None:
                continue
            me = aid == self.agent_id
            agents.append({"id": aid, "speed": float(getattr(spec, "speed", 1.0) or 1.0), "rate": self.drone._declared_rate(aid),
                           "remaining": float(self.drone.power.level) if me else float(spec.capacity)})
        tasks = [{"id": p["id"], "pickup": (p["col"], p["row"]), "weight": p.get("weight", 1.0)} for p in parcels]
        caps = capability_caps(agents, tasks, drop, lambda a, b: self.physics.distance(a, b), lambda d, w: self.physics.energy_cost(distance=d, weight=w),
                               lambda w: self.physics.pick_drop_cost(w), float(self.drone.power.floor))
        return caps.get(self.agent_id, self.MAX_BUNDLE_SIZE)

    def _marginal_gain(self, snapshot, parcel):
        task_id = parcel["id"]

        # Try inserting task into every possible position in current path
        best_delta = None
        best_pos = None

        for i in range(len(self.path) + 1):
            new_path = self.path[:i] + [task_id] + self.path[i:]
            new_cost = self._estimate_bundle_cost(snapshot, new_path)

            old_cost = self._estimate_bundle_cost(snapshot, self.path)

            delta = new_cost - old_cost

            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_pos = i

        if best_delta is None:
            return None

        # Store insertion decision
        self.c[task_id] = (-best_delta, best_pos)

        return -best_delta

    def _can_add_to_bundle(self, snapshot, task_id):
        if task_id not in self.c:
            return True  # no marginal info yet, do not block bidding

        _, pos = self.c[task_id]

        candidate_path = (
                self.path[:pos] +
                [task_id] +
                self.path[pos:]
        )

        cost = self._estimate_bundle_cost(snapshot, candidate_path, with_time=False)
        battery = float(self.drone.power.level)

        return battery >= cost + self.FEASIBILITY_MARGIN

    def _estimate_bundle_cost(self, snapshot, path, with_time=True):
        if not path:
            return 0.0

        agent = snapshot["agent"]
        cur = (agent["col"], agent["row"])
        station = snapshot["nearest_station"]

        total = 0.0
        clock = 0.0          # steps flown along the path so far: time is charged at each task's completion, so a later task in a bundle costs more

        for task_id in path:
            parcel = next(p for p in snapshot["all_parcels"] if p["id"] == task_id)

            pick = (parcel["col"], parcel["row"])
            drop = self._bid_drop(pick, station)

            total += self.trip_energy(cur, pick, drop, parcel["weight"])
            if with_time:
                clock += self.trip_time(cur, pick, drop)
                total += self.time_weight * clock

            cur = drop

        return total

    def _estimate_cost(self, snapshot: Dict, parcel: Dict) -> Optional[float]:
        agent = snapshot["agent"]
        station = snapshot.get("nearest_station")
        if not station:
            print(f"[CBBA][{self.agent_id}] No station in snapshot")
            return None

        cur = (agent["col"], agent["row"])
        pick = (parcel["col"], parcel["row"])
        drop = self._bid_drop(pick, station)

        cost = self.trip_energy(cur, pick, drop, parcel["weight"])

        battery = float(self.drone.power.level)
        feasible = battery >= cost + self.FEASIBILITY_MARGIN

        print(
            f"[CBBA][{self.agent_id}] "
            f"parcel={parcel['id']} "
            f"battery={battery:.2f} "
            f"cost={cost:.2f} "
            f"margin={self.FEASIBILITY_MARGIN} "
            f"feasible={feasible}"
        )

        if not feasible:
            return None

        return cost + self.time_weight * self.trip_time(cur, pick, drop)       # the bid also prices the trip's time when the arm asks (time_weight > 0)
