# src/strategies/decentralized_auction.py

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .base import BaseStrategy


class DecentralizedAuctionStrategy(BaseStrategy):
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

        # task_id -> {agent_id: bid_value}
        self.bid_table: Dict[str, Dict[str, float]] = defaultdict(dict)

        # task_id -> last update step
        self.bid_timestamp: Dict[str, int] = {}

        # cache
        self.last_bid_step = -self.BID_INTERVAL
        self.first_seen_step = {}
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
        self.committed_tasks = {}  # task_id -> agent_id

    def set_drone(self, drone):
        super().set_drone(drone)
        self.agent_id = drone.agent_id

        if self.physics is None and hasattr(drone, "physics"):
            self.physics = drone.physics

    # -----------------------------
    # Messaging
    # -----------------------------

    def _broadcast_bid(self, task_id: str, bid: float, step: int):
        payload = {
            "type": "bid",
            "task_id": task_id,
            "agent_id": self.agent_id,
            "bid": bid,
            "step": step,
        }

        print(f"[CBBA][{self.agent_id}] SEND bid task={task_id} -> all")
        self.drone.broadcast(payload, step)

    def _broadcast_commit(self, task_id: str, step: int):
        payload = {
            "type": "commit",
            "task_id": task_id,
            "agent_id": self.agent_id,
            "step": step,
        }

        print(f"[CBBA][{self.agent_id}] SEND COMMIT task={task_id} -> all")
        self.drone.broadcast(payload, step)

    def _process_messages(self, step: int):
        # Prevent double-processing within same simulation step
        if step == self.last_processed_step:
            return

        self.last_processed_step = step

        if self.drone.inbox:
            self.pending_messages.extend(self.drone.inbox)
            self.drone.inbox.clear()

        inbox = list(self.pending_messages)
        self.pending_messages.clear()

        if inbox:
            self.drone.inbox.clear()

        for sender, msg in inbox:
            if not isinstance(msg, dict):
                continue

            mtype = msg.get("type")
            print("mtype", mtype)

            if mtype == "bid":
                task = msg["task_id"]
                agent = msg["agent_id"]
                bid = float(msg["bid"])
                ts = int(msg["step"])

                self.bid_table[task][agent] = bid
                self.bid_timestamp[task] = ts

            elif mtype == "commit":
                task = msg["task_id"]
                agent = msg["agent_id"]

                print(f"[CBBA][{self.agent_id}] RECEIVED COMMIT task={task} by {agent}")

                self.committed_tasks[task] = agent

                # Remove from local auction state
                self.bid_table.pop(task, None)
                self.bid_timestamp.pop(task, None)
                self.first_seen_step.pop(task, None)

            else:
                # Unknown message types should be ignored safely
                continue

        print(f"[CBBA][{self.agent_id}] processed messages:", len(inbox))

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

        if step == self.last_decision_step:
            return self._last_decision
        self.last_decision_step = step

        # Process incoming bids
        self._process_messages(step)

        # Prune expired bids
        for task, ts in list(self.bid_timestamp.items()):
            if step - ts > self.BID_EXPIRY:
                del self.bid_table[task]
                del self.bid_timestamp[task]
                self.first_seen_step.pop(task, None)

        # Get available parcels
        parcels = [p for p in snapshot["all_parcels"] if
                   not p.get("picked") and not p.get("delivered") and p["id"] not in self.committed_tasks]

        if not parcels:
            print(f"[CBBA][{self.agent_id}] GLOBAL DONE -> IDLE")
            result = {
                "mode": "idle",
                "agent_id": agent_id,
                "plan": [],
                "confidence": 0.0,
                "strategy": "lb-cbba",
            }
            self._last_decision = result
            return result

        # Compute own bids
        own_bids = {}
        for p in parcels:
            cost = self._estimate_cost(snapshot, p)
            if cost is not None:
                own_bids[p["id"]] = cost

        if not own_bids:
            print(f"[CBBA][{self.agent_id}] NO FEASIBLE BIDS -> WAIT")
            result = {
                "mode": "idle",
                "agent_id": agent_id,
                "reason": "no_feasible_bids",
            }
            self._last_decision = result
            return result

        # Broadcast bids
        if self.last_bid_step < 0 or step - self.last_bid_step >= self.BID_INTERVAL:
            print(f"[CBBA][{self.agent_id}] Broadcasting bids at step {step}")
            for task, bid in own_bids.items():
                self._broadcast_bid(task, bid, step)
            self.last_bid_step = step

        # Update local bid table
        for task, bid in own_bids.items():
            self.bid_table[task][self.agent_id] = bid

        # Determine winners
        winners = {}
        for task, bids in self.bid_table.items():
            if task not in self.first_seen_step:
                self.first_seen_step[task] = step

            if step - self.first_seen_step[task] < 2:
                continue

            winners[task] = min(bids.items(), key=lambda x: (x[1], x[0]))[0]

        my_tasks = [
            t for t, w in winners.items()
            if w == self.agent_id and t in own_bids
        ]

        if not my_tasks:
            result = {
                "mode": "wait",
                "agent_id": agent_id,
                "reason": "lost_auction",
            }
            self._last_decision = result
            return result

        # Choose best task
        chosen_task = min(my_tasks, key=lambda t: own_bids[t])

        self.committed_tasks[chosen_task] = self.agent_id
        self._broadcast_commit(chosen_task, step)

        parcel = next(p for p in parcels if p["id"] == chosen_task)

        pickup = [parcel["col"], parcel["row"]]
        station = snapshot["nearest_station"]

        drop_cell = self._choose_drop_cell(
            pick=(pickup[0], pickup[1]),
            station=station,
        )

        drop = [drop_cell[0], drop_cell[1]]

        result = {
            "agent_id": agent_id,
            "plan": [{"pickup": pickup, "dropoff": drop, "weight": parcel.get("weight", 1.0)}],
            "confidence": 0.9,
            "strategy": "lb-cbba",
        }

        self._last_decision = result
        return result

    # -----------------------------
    # Cost model
    # -----------------------------

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
