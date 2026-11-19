# src/perception.py
"""
Onboard perception: what the drone knows about the world, built only from what it senses.

Every strategy acts on the snapshot this produces. It holds the drone's memory of the parcels it has seen: a parcel
inside the sensor radius is read as it is now, and one outside keeps what the drone last saw. A parcel the drone has
never seen counts as available, with `unseen_weight` as its weight. Parcel locations are known from the start
information. Other drones appear only while sensed.

Perception is event driven: each sensor reading updates the memory at once and what changed comes out as events
(`update` returns them) that the strategy can react to instead of re-checking the state every step:
  parcel_seen      a parcel is sensed for the first time
  parcel_taken     a parcel the drone believed available is now known to be picked, delivered, or gone from its place
  peer_sighted     a drone comes into sensor range (or is sensed for the first time)
  peer_lost        a drone that was sensed is no longer sensed
Each event is a dict with a "type", the drone or parcel "id", and the "step".
"""
import time
from typing import Tuple, Any, Dict, List, Optional

from src.interface import Observation


class Perception:
    def __init__(self, unseen_weight: float = 1.0, station_memory: str = "observed", ledger_stop: bool = False, declared_weights: bool = True):
        self.unseen_weight = float(unseen_weight)
        # declared_weights: a parcel not yet sensed carries the weight declared with its order (shared start information) instead of the unseen default
        self.declared_weights = bool(declared_weights)
        # station_memory "live": the station's free cells are read as they are now (the simulator's default view).
        # "observed": the drone keeps a memory of the station; a cell is added as occupied when the drone senses it so (inside its sensor
        # radius), never removed (a delivered parcel stays), so the memory is as old as the drone's last visit.
        self.station_memory = station_memory
        # ledger_stop: the snapshot says whether the station holds every parcel (occupied cells >= parcels in the start information).
        self.ledger_stop = bool(ledger_stop)
        self._occupied_seen: set = set()
        self._full_view_now = False                  # every station cell is inside the sensor radius now
        self._cell_seen: Dict[tuple, int] = {}       # the last step each station cell was in the sensor radius
        # "predictive" adds to the observed memory an estimate of the count now: the drone's own full views of the station give (step, count)
        # points; between views the count is extrapolated at the delivery rate over the last points. Nothing is assumed about the scenario.
        self._full_views: List[tuple] = [(0, 0)]
        self._count_hat = 0.0
        self._known: List[Dict[str, Any]] = []      # parcels from the start information, in order
        self._memory: Dict[Any, Dict[str, Any]] = {}  # parcel id -> last seen state
        self._start_info = None
        self._peers_seen: set = set()

    def _start_weight(self, k) -> float:
        """The weight assumed for a parcel before it has been sensed."""
        if self.declared_weights and k.get("declared_weight") is not None:
            return float(k["declared_weight"])
        return self.unseen_weight

    def update(self, obs: Observation) -> List[Dict[str, Any]]:
        """Take in one reading. Called every step: the drone senses continuously, whether or not it is deciding."""
        if obs.start_info is not None and self._start_info is None:
            self._start_info = obs.start_info
            self._known = [dict(p) for p in obs.start_info.parcels]
        for o in getattr(obs, "new_orders", None) or []:
            if all(k["id"] != o["id"] for k in self._known):
                self._known.append(dict(o))
        events: List[Dict[str, Any]] = []
        for p in obs.parcels:
            before = self._memory.get(p["id"])
            if before is None:
                events.append({"type": "parcel_seen", "id": p["id"], "step": obs.step})
            if (before is None or (not before["picked"] and not before["delivered"])) and \
                    (p.get("picked") or p.get("delivered")):
                events.append({"type": "parcel_taken", "id": p["id"], "step": obs.step})
            was_there = not p.get("picked") and not p.get("delivered")
            taken_now = bool(p.get("picked") or p.get("delivered"))
            self._memory[p["id"]] = {"col": p["col"], "row": p["row"], "weight": p.get("weight", 1.0),
                                     "picked": bool(p.get("picked")), "delivered": bool(p.get("delivered")),
                                     # the step the drone learned the parcel was taken (an upper bound on when it was picked)
                                     "taken_step": ((before or {}).get("taken_step") if (before or {}).get("taken_step") is not None
                                                    else (obs.step if taken_now else None)),
                                     # the last step the drone confirmed the parcel in place (0: told at the start)
                                     "seen_step": obs.step if was_there else (before or {}).get("seen_step", 0)}
        # An empty place is information too: a parcel the drone remembers (or was told about) at a spot inside its
        # sensor range, and does not sense now, is no longer there (picked, or moved by a delivery).
        here = self._cell_now(obs)
        if obs.sensor_radius != float("inf"):
            sensed = {p["id"] for p in obs.parcels}
            for k in self._known:
                pid = k["id"]
                if pid in sensed:
                    continue
                m = self._memory.get(pid) or {"col": k["col"], "row": k["row"], "weight": self._start_weight(k),
                                              "picked": False, "delivered": False}
                if not m["picked"] and not m["delivered"] and \
                        abs(here[0] - m["col"]) + abs(here[1] - m["row"]) <= obs.sensor_radius:
                    self._memory[pid] = dict(m, picked=True, taken_step=obs.step)
                    events.append({"type": "parcel_taken", "id": pid, "step": obs.step})
        now = {a["id"] for a in obs.peers}
        events += [{"type": "peer_sighted", "id": i, "step": obs.step} for i in sorted(now - self._peers_seen)]
        events += [{"type": "peer_lost", "id": i, "step": obs.step} for i in sorted(self._peers_seen - now)]
        self._peers_seen = now
        return events

    @staticmethod
    def _cell_now(obs) -> Tuple[int, int]:
        """The cell the drone is in right now, from its own odometry (its committed cell only changes when a move ends, so in flight it is the cell it left)."""
        if obs.cell_px and obs.cell_px > 0:
            return int(obs.pos_px[0] // obs.cell_px), int(obs.pos_px[1] // obs.cell_px)
        return obs.col, obs.row

    def _estimate_count(self, step: int, full_view: bool, exact: Optional[int]) -> float:
        """Occupied station cells now: the exact count while the whole station is in view, otherwise the last exact count plus the delivery rate
        (over the last few full views) times the steps since."""
        if full_view:
            if step > self._full_views[-1][0]:
                if exact != self._full_views[-1][1] or len(self._full_views) == 1:
                    self._full_views.append((step, exact))
                else:
                    self._full_views[-1] = (step, exact)
            return float(exact)
        t_last, c_last = self._full_views[-1]
        info = self._start_info
        if info is None:
            return float(c_last)
        if self.station_memory == "predictive":         # the delivery rate over the last four full views
            window = self._full_views[-4:]
            span = window[-1][0] - window[0][0]
            rate = (window[-1][1] - window[0][1]) / span if span > 0 else 0.0
            return c_last + max(rate, 0.0) * (step - t_last)
        if self.station_memory in ("predictive_avg", "observed"):     # the average delivery rate since the start ("observed" only logs it)
            return c_last + (c_last / t_last if t_last > 0 else 0.0) * (step - t_last)
        total, fleet = info.n_orders, len(info.fleet)
        window = self._full_views[-4:]
        # The rate of one working drone: deliveries in the window per drone-step. Only as many drones as there are parcels left can be
        # working, so the number of working drones over an interval is min(fleet, parcels left at its start).
        drone_steps = sum((t1 - t0) * min(fleet, max(total - c0, 1)) for (t0, c0), (t1, _) in zip(window, window[1:]))
        rate = (window[-1][1] - window[0][1]) / drone_steps if drone_steps > 0 else 0.0
        if rate <= 0.0:
            return float(c_last)
        # Expected time of the following deliveries: the j-th of the R still to come takes 1 / (rate x working drones) after the one before it.
        elapsed, delivered = float(step - t_last), 0
        left = total - c_last
        t_next = 0.0
        while delivered < left:
            t_next += 1.0 / (rate * min(fleet, left - delivered))
            if t_next > elapsed:
                break
            delivered += 1
        return float(c_last + delivered)

    def snapshot(self, obs: Observation) -> Dict[str, Any]:
        """What the drone knows now, in the dict strategies consume."""
        self.update(obs)
        parcels = []
        for k in self._known:
            m = self._memory.get(k["id"])
            if m is None:
                parcels.append({"id": k["id"], "col": k["col"], "row": k["row"], "weight": self._start_weight(k),
                                "picked": False, "delivered": False, "seen_step": 0})
            else:
                parcels.append({"id": k["id"], **m})
        station = dict(obs.station, free_cells=[tuple(c) for c in obs.station_free_cells]) if obs.station else None
        full_view = False
        if station is not None and self.station_memory.split("_")[0] in ("observed", "predictive"):
            cells = [(c, r) for r in range(station["row"], station["row"] + station["h"])
                     for c in range(station["col"], station["col"] + station["w"])]
            live_free = set(station["free_cells"])
            in_view = [c for c in cells if abs(self._cell_now(obs)[0] - c[0]) + abs(self._cell_now(obs)[1] - c[1]) <= obs.sensor_radius]
            for c in in_view:
                self._cell_seen[c] = obs.step
                if c not in live_free:
                    self._occupied_seen.add(c)
            full_view = len(in_view) == len(cells)
            self._full_view_now = full_view
            station["free_cells"] = [c for c in cells if c not in self._occupied_seen]
            if self.station_memory != "live":
                self._count_hat = self._estimate_count(obs.step, full_view, len(cells) - len(live_free) if full_view else None)
        snap = {
            "agent": {"id": obs.id, "col": obs.col, "row": obs.row},
            "carrying": {"has": obs.carrying, "weight": obs.carrying_weight},
            "nearest_station": station,
            "all_parcels": parcels,
            "other_agents": [dict(a) for a in obs.peers],
            "step": obs.step,
            "timestamp": time.time(),
            "own": {"pos_px": obs.pos_px, "cell_px": obs.cell_px, "battery": obs.battery, "capacity": obs.capacity,
                    "carrying_weight": obs.carrying_weight},
        }
        if self._start_info is not None:
            snap["start_info"] = self._start_info
        if station is not None and self.station_memory != "live":
            snap["station_seen_step"] = dict(self._cell_seen)       # for a strategy that predicts the station
            snap["station_in_full_view"] = self._full_view_now
            snap["station_count_hat"] = self._count_hat             # the rate-extrapolated count (used by the predictive levels)
        if self.ledger_stop and station is not None and self._start_info is not None:
            occupied = station["w"] * station["h"] - len(station["free_cells"])
            if self.station_memory.startswith("predictive"):
                occupied = max(occupied, self._count_hat)
            snap["ledger_done"] = occupied >= self._start_info.n_orders
        return snap
