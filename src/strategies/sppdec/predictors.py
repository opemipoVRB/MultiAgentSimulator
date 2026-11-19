# src/strategies/sppdec/predictors.py
"""
Peer model: what an unseen peer has probably done since the drone last sensed it ("the peer acts as I would"; design:
.ignore/revision/marl_baseline_design.md section 8).

From the last sighting (position, whether it carried a parcel, step) the model simulates the peer forward to now:
  carrying  the peer flies to the station and drops at the free station cell nearest to where it is; the parcel is delivered
  empty     the peer flies to the open parcel nearest to it (the drone's own greedy rule) and picks it up; that parcel is taken,
            then the peer carries it to the station, and so on until the time since the sighting is used up
The speed is the drone's own, measured from the distance and the steps of its own observed moves (steps per cell), so nothing about the scenario is assumed.
Shadow mode only: predictions are logged with each decision and nothing reads them yet.
"""
import statistics
from typing import Callable, Dict, Iterable, List, Optional, Tuple

Cell = Tuple[int, int]


class PeerModel:
    KEEP = 20                   # own moves kept for the speed estimate
    MIN_MOVES = 3

    def __init__(self, distance: Callable):
        self.distance = distance
        self._last: Dict[str, Tuple[int, int, bool, int]] = {}      # peer id -> (col, row, carrying, step sensed)
        self._me: Optional[Cell] = None
        self._me_step: Optional[int] = None
        self._ratios: List[float] = []          # steps per cell of each observed move of this drone
        self._started = False
        self._depart: Dict[str, Cell] = {}                           # peer id -> its last position inside the station (its start cell until it has been seen there)
        self._sightings: Dict[str, List[Tuple[int, Cell]]] = {}      # peer id -> (step, cell) each time the drone sensed it (what the peer could have seen on its way)
        self.version = 0                        # grows when a peer is sighted; a cached prediction older than that is stale

    # ------------------------------------------------------------------
    def observe(self, step: int, me: Cell, visible: Iterable[Dict], start_info=None, me_id: Optional[str] = None, station: Optional[Dict] = None) -> None:
        if not self._started and start_info is not None:
            for pid, spec in start_info.fleet.items():
                if pid == me_id:
                    continue                                # the drone itself is not a peer
                self._last[pid] = (spec.start_cell[0], spec.start_cell[1], False, 0)
                self._depart[pid] = (spec.start_cell[0], spec.start_cell[1])
            self._started = True
        if self._me is not None and me != self._me and self._me_step is not None:
            cells = abs(me[0] - self._me[0]) + abs(me[1] - self._me[1])
            if cells >= 1 and step > self._me_step:
                self._ratios = (self._ratios + [(step - self._me_step) / cells])[-self.KEEP:]
        if self._me is None or me != self._me:
            self._me, self._me_step = me, step
        for a in visible:
            if a["id"] == me_id:
                continue
            self._last[a["id"]] = (int(a["col"]), int(a["row"]), bool(a.get("carrying")), step)
            self._sightings.setdefault(a["id"], []).append((step, (int(a["col"]), int(a["row"]))))
            if station and station["col"] <= a["col"] < station["col"] + station["w"] and station["row"] <= a["row"] < station["row"] + station["h"]:
                self._depart[a["id"]] = (int(a["col"]), int(a["row"]))          # where a peer decides: right after it delivers, inside the station
            if len(self._sightings[a["id"]]) > 600:
                del self._sightings[a["id"]][:100]
            self.version += 1

    def saw_after(self, pid: str, cell: Cell, since: int, radius: float) -> bool:
        """Was the peer sensed at or after `since` within `radius` of `cell`? Then it could have seen that cell (an empty place tells it a parcel is gone)."""
        return any(t >= since and abs(c[0] - cell[0]) + abs(c[1] - cell[1]) <= radius for t, c in reversed(self._sightings.get(pid, ())))

    def steps_per_cell(self) -> Optional[float]:
        """The drone's own speed: steps per cell over its recent observed moves. Waiting at a pickup or the station only lengthens a move, so
        the lower quartile of the recent ratios is taken."""
        if len(self._ratios) < self.MIN_MOVES:
            return None
        ordered = sorted(self._ratios)
        return ordered[max(0, (len(ordered) - 1) // 4)]

    # ------------------------------------------------------------------
    def _toward(self, a: Cell, b: Cell, cells: float) -> Cell:
        """The cell reached after `cells` cells of a Manhattan path from a to b (columns first)."""
        k = int(cells)
        dx, dy = b[0] - a[0], b[1] - a[1]
        mx = min(abs(dx), k)
        col = a[0] + (mx if dx >= 0 else -mx)
        my = min(abs(dy), k - mx)
        return (col, a[1] + (my if dy >= 0 else -my))

    def predict(self, step: int, open_parcels: List[Dict], station_free: List[Cell], station_cells: List[Cell],
                taken: Iterable[Tuple[str, Cell, int]] = (), cost: Optional[Callable] = None) -> Optional[Dict]:
        """
        Simulate every peer forward from its last sighting to `step`, on one shared timeline so that two peers never take the same parcel or fill the
        same cell (a parcel or a cell is reserved when a peer sets out for it). Returns None before the drone has measured its own speed, else
          {"peers": {peer id: {"pos": predicted cell, "taken": [parcel ids probably picked], "delivered": n drops}},
           "drops": [(step of the drop, station cell it fills, peer or parcel that made it)]}.
        """
        spc = self.steps_per_cell()
        if spc is None:
            return None
        # cost(peer id, from, to): the peer's own adjusted cost of flying a leg (its beta, w-bar and the energy it has used); distance when absent.
        # Peers that choose at about the same time are matched greedily by it, so a contested parcel goes to the peer that would win it.
        cost = cost or (lambda pid_, a_, b_: float(self.distance(a_, b_)))
        free = list(station_free) or list(station_cells)
        left = {p["id"]: (int(p["col"]), int(p["row"])) for p in open_parcels}
        self._parcel_cells = dict(left)
        state = {}
        for pid in sorted(self._last):
            col, row, carrying, t0 = self._last[pid]
            state[pid] = {"pos": (col, row), "carrying": carrying, "t": float(t0), "taken": [], "delivered": 0, "done": False, "dest": None, "plan": None}
        # A parcel the drone knows was picked (but has not seen delivered) is flying to the station: a carrier that set out from the parcel's place by the
        # step the drone learned of the pick at the latest.
        for pid_, cell_, learned in taken:
            state["parcel:" + str(pid_)] = {"pos": (int(cell_[0]), int(cell_[1])), "carrying": True, "t": float(learned), "taken": [], "delivered": 0,
                                           "done": False, "dest": None, "plan": None}
        drops: List[Tuple[int, Cell, str]] = []
        for _ in range(64 * max(1, len(state))):
            live = [pid for pid, st in state.items() if not st["done"]]
            if not live:
                break
            pid = min(live, key=lambda i: (state[i]["t"], i))          # the peer whose next leg starts first
            st = state[pid]
            if st["t"] >= step:
                st["done"] = True
                continue
            if st["carrying"]:
                if not free:
                    st["done"] = True
                    continue
                drop = min(free, key=lambda c: (self.distance(st["pos"], c), c))
                free.remove(drop)                                      # reserved: no other simulated drop uses it
                end = st["t"] + self.distance(st["pos"], drop) * spc
                if end > step:
                    st["pos"], st["done"] = self._toward(st["pos"], drop, (step - st["t"]) / spc), True     # still flying: the cell stays reserved, not filled
                    st["dest"] = drop
                    continue
                st["t"], st["pos"], st["carrying"] = end, drop, False
                st["delivered"] += 1
                drops.append((int(round(end)), drop, pid))
            else:
                if not left:
                    st["done"] = True
                    continue
                if st.get("plan") is None:
                    group = [q for q, sq in state.items() if not sq["done"] and not sq["carrying"] and not q.startswith("parcel:")
                             and sq.get("plan") is None and abs(sq["t"] - st["t"]) <= spc]
                    pairs = sorted((cost(q, state[q]["pos"], left[i]), q, i) for q in group for i in left)
                    taken_q, taken_i = set(), set()
                    for _c, q, i in pairs:
                        if q in taken_q or i in taken_i:
                            continue
                        taken_q.add(q)
                        taken_i.add(i)
                        state[q]["plan"] = i
                    for i in taken_i:
                        left.pop(i, None)                              # reserved for the peer that won it
                nxt = st.get("plan")
                if nxt is None:
                    st["done"] = True
                    continue
                st["plan"] = None
                target = self._parcel_cells[nxt]
                end = st["t"] + self.distance(st["pos"], target) * spc
                if end > step:
                    st["pos"], st["done"] = self._toward(st["pos"], target, (step - st["t"]) / spc), True
                    st["dest"] = target
                    continue
                st["t"], st["pos"], st["carrying"] = end, target, True
                st["taken"].append(nxt)
        return {"peers": {pid: {"pos": st["pos"], "taken": st["taken"], "delivered": st["delivered"], "carrying": st["carrying"], "dest": st["dest"]}
                          for pid, st in state.items() if not pid.startswith("parcel:")},
                "drops": drops}
