# src/world_port.py
"""
World side of the boundary for one drone: turns the world into the drone's Observation and applies its Commands.

The drone's onboard logic (controller and strategy) never touches the terrain or another drone; it receives what
`observe` returns and hands back Commands. This is the only place that reads the world on the drone's behalf.
Per step: every drone is observed, every drone thinks, then the world applies all commands (`apply_all`, in drone
order) and runs the physics. The answer to a command comes back with the drone's next observation.
"""
import time
from typing import Tuple, Any, Dict, List, Optional

from src.calibration import calibrate
from src.interface import Command, DeclaredSpec, Observation, Result, StartInfo


class WorldPort:
    def __init__(self, drone, terrain, sensor=None, start_info: Optional[StartInfo] = None, run_seed: int = 0):
        self.drone = drone
        self.terrain = terrain
        self.sensor = sensor     # world-side sensor model; None reports everything as it is
        self.run_seed = int(run_seed)
        self._start_info: Optional[StartInfo] = start_info
        self._last_result: Optional[Result] = None
        self._announced: set = set()      # late orders already announced to this drone (dynamic arrival)

    @staticmethod
    def build_start_info(terrain) -> StartInfo:
        """What the whole fleet knows before the first step: parcel locations, the map, and every drone's declared spec."""
        fleet = {}
        for a in terrain.agents.values():
            cal = calibrate(a, noise_std=getattr(a, "calibration_noise", 0.0),
                            resolution=getattr(a, "calibration_resolution", 0.0))
            # A declared value set by the scenario replaces the measured one (declared differs from actual)
            if getattr(a, "declared_rate", None) is not None:
                cal["unloaded_per_cell"] = round(float(a.declared_rate) * type(a).BASE_COST_PER_CELL, 9)
            if getattr(a, "declared_weight_factor", None) is not None:
                cal["weight_sensitivity"] = round(float(a.declared_weight_factor), 9)
            fleet[a.agent_id] = DeclaredSpec(unloaded_per_cell=cal["unloaded_per_cell"],
                                             weight_sensitivity=cal["weight_sensitivity"],
                                             capacity=float(a.power.capacity), start_cell=(int(a.col), int(a.row)),
                                             personality=getattr(a, "personality", None),
                                             speed=float(getattr(a, "speed_scale", 1.0)))
        parcels = [{"id": getattr(p, "id", None), "col": int(p.col), "row": int(p.row),
                    # the declared weight of the order (shared with the fleet; used only by an arm with declared_weights, otherwise weights are learned by sensing)
                    "declared_weight": float(getattr(p, "weight", 1.0))} for p in terrain.parcels
                   if int(getattr(p, "release_step", 0)) <= 0]
        grid = (terrain.screen_size[0] // terrain.grid_size, terrain.screen_size[1] // terrain.grid_size)
        late = len(terrain.parcels) - len(parcels)
        from src import obstacles
        return StartInfo(parcels=parcels, fleet=fleet, grid=grid, total_orders=(len(terrain.parcels) if late else None),
                         obstacles=(obstacles.cells() if obstacles.active() else None))

    # -- observe ------------------------------------------------------------------------------------------
    def _station_dict(self, station) -> Optional[Dict[str, int]]:
        if station is None:
            return None
        return {"col": int(station.col), "row": int(station.row), "w": int(station.w), "h": int(station.h)}

    def _parcels(self, step: int = 10 ** 9) -> List[Dict[str, Any]]:
        # an order not yet released (dynamic arrival) does not exist for the drones
        return [{"id": getattr(p, "id", None), "col": int(p.col), "row": int(p.row),
                 "weight": getattr(p, "weight", 1.0), "picked": bool(getattr(p, "picked", False)),
                 "delivered": bool(getattr(p, "delivered", False))} for p in self.terrain.parcels
                if int(getattr(p, "release_step", 0)) <= step]

    def _new_orders(self, step: int) -> List[Dict[str, Any]]:
        """Orders released since the drone last looked, as the shared order feed announces them: id, cell and declared weight (the same for every drone)."""
        out = []
        for p in self.terrain.parcels:
            r = int(getattr(p, "release_step", 0))
            if 0 < r <= step and p.id not in self._announced:
                self._announced.add(p.id)
                out.append({"id": p.id, "col": int(p.col), "row": int(p.row), "declared_weight": float(getattr(p, "weight", 1.0))})
        return out

    @staticmethod
    def _cell_in_flight(a) -> Tuple[int, int]:
        """The cell a drone is in right now. `col`/`row` are committed only when a move ends; during a flight the drone is at its continuous position."""
        return int(a.pos.x // a.grid_size), int(a.pos.y // a.grid_size)

    def _peers(self) -> List[Dict[str, Any]]:
        out = []
        for a in self.terrain.agents.values():
            if a is self.drone:
                continue
            col, row = self._cell_in_flight(a)
            out.append({"id": a.agent_id, "col": col, "row": row, "carrying": bool(getattr(a, "carrying", None))})
        return out

    def _drain_inbox(self):
        """Radio messages delivered to the drone since its last observation (each is delivered once)."""
        msgs = list(self.drone.inbox)
        self.drone.inbox.clear()
        return msgs

    def _take_result(self) -> Optional[Result]:
        r, self._last_result = self._last_result, None
        return r

    def observe(self, step: int, time_s: float = 0.0) -> Observation:
        d, t = self.drone, self.terrain
        station = t.nearest_station(d.col, d.row)
        here = t.parcel_at_cell(d.col, d.row)
        if here is not None and int(getattr(here, "release_step", 0)) > step:
            here = None
        parcels, peers = self._parcels(step), self._peers()
        new_orders = self._new_orders(step)
        if self.sensor is not None and self.sensor.active:
            parcels, peers, _ = self.sensor.sense(d.agent_id, step, self._cell_in_flight(d), parcels, peers)
        if self._start_info is None:
            self._start_info = self.build_start_info(self.terrain)
        return Observation(
            step=step, time=time_s, id=d.agent_id, col=int(d.col), row=int(d.row),
            battery=float(d.power.level), capacity=float(d.power.capacity), moving=bool(d.moving),
            carrying=bool(d.carrying), carrying_weight=(getattr(d.carrying, "weight", 0.0) if d.carrying else 0.0),
            lost=bool(d.lost),
            here_parcel_weight=None if here is None else getattr(here, "weight", 1.0),
            inside_station=bool(station is not None and station.contains_cell(d.col, d.row)),
            station=self._station_dict(station),
            station_free_cells=[] if station is None else list(station.free_cells(t)),
            parcels=parcels, new_orders=new_orders, peers=peers, start_info=self._start_info,
            inbox=self._drain_inbox(), reserve_floor=float(d.reserve_floor), last_result=self._take_result(),
            sensor_radius=(self.sensor.radius if self.sensor is not None else float("inf")),
            pos_px=(float(d.pos.x), float(d.pos.y)), cell_px=float(d.grid_size))

    # -- act ----------------------------------------------------------------------------------------------
    def apply_all(self, commands: List[Command], step: int) -> None:
        """Carry out a drone's commands for this step, in order. The answer arrives with its next observation."""
        for cmd in commands:
            result = self.apply(cmd, step)
            if cmd.kind != "none":
                self._last_result = result

    def apply(self, cmd: Command, step: int = 0) -> Result:
        """Carry out one command on the drone."""
        d, t = self.drone, self.terrain
        for recipient_id, payload in cmd.messages:
            self.send(recipient_id, payload, step)
        if cmd.kind == "none":
            return Result("none", step)
        if cmd.kind == "move_to":
            d.set_target_cell(*cmd.cell)
            return Result("move_to", step, True, tuple(cmd.cell))
        if cmd.kind == "stop":
            d.clear_path()
            d.hover_here()      # the drone hovers where it is: commit the cell it is in without moving it (otherwise its cell stays the one it left, and the world and the drone disagree about where it is)
            return Result("stop", step)
        if cmd.kind == "pick":
            parcel = t.parcel_at_cell(d.col, d.row)
            if parcel is not None and int(getattr(parcel, "release_step", 0)) > step:
                parcel = None
            success = bool(parcel is not None and d.perform_pick(parcel))
            if not success:
                d.last_action = ("pick_failed", (d.col, d.row), None)
            return Result("pick", step, success, (int(d.col), int(d.row)))
        if cmd.kind == "drop":
            station = t.nearest_station(d.col, d.row)
            success = bool(d.perform_drop(d.carrying))
            if not success and station is not None:
                # the station redirects a drone whose cell is blocked to its least used free cell
                alt = station.least_used_free_cell(t, ref_col=d.col, ref_row=d.row)
                if alt:
                    d.set_target_cell(*alt)
            return Result("drop", step, success, (int(d.col), int(d.row)))
        if cmd.kind == "request_parking":
            station = t.nearest_station(d.col, d.row)
            cell = station.park_drone(d) if station else None
            if cell:
                d.set_target_cell(*cell)
            return Result("request_parking", step, bool(cell), cell)
        raise ValueError(cmd.kind)

    def send(self, recipient_id: str, payload, step: int) -> bool:
        """Hand a message for `recipient_id` to the world's radio channel (loss and delay apply there)."""
        if recipient_id == "*":
            receivers = [a for a in self.terrain.agents.values() if a is not self.drone]
            return self.drone.broadcast_message(receivers, payload, step)
        receiver = (self.terrain.command_center if recipient_id == "command_center"
                    else self.terrain.agents[recipient_id])
        return self.drone.send_message(receiver, payload, step)
