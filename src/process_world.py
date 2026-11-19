# src/process_world.py
"""
World side of the drone process: a controller stand-in that hands each observation to the drone's own process and
returns its commands. It has the same observe / think / act interface as the in-process controller.

The drone process (src/drone_process.py) is pinned to one core. The world keeps the physics, the sensors, the radio
channel and the flight recorder; the drone's records come back with its answer and are written here, in drone order.
"""
import multiprocessing as mp
from typing import Any, Dict, List, Optional

from src.world_port import WorldPort

_CTX = None


def _context():
    """A fork server with the heavy modules preloaded, so starting a drone process is quick."""
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context("forkserver")
        _CTX.set_forkserver_preload(["src.controllers", "src.strategies.factory", "src.physics.physics_2d"])
    return _CTX


def allocate_cores(n_drones: int) -> List[Optional[int]]:
    """One physical core per drone (its first logical CPU), leaving the first physical core to the world. If there are
    more drones than cores the later drones wrap around and share."""
    from src.system.topology import physical_core_groups
    groups = physical_core_groups()
    drone_groups = groups[1:] if len(groups) > 1 else groups
    return [drone_groups[i % len(drone_groups)][0] for i in range(n_drones)]


class RemoteController:
    def __init__(self, drone, terrain, strategy_spec, topology, sensor, start_info, run_seed, core=None):
        from src.drone_process import SLMClient  # noqa: F401  (import check: the same module the child uses)
        self.agent_id = drone.agent_id
        self.port = WorldPort(drone, terrain, sensor, start_info, run_seed)
        self._telemetry: List[tuple] = []
        self.parked = False                    # as last reported by the drone
        ctx = _context()
        self._conn, child = ctx.Pipe()
        init = {
            "agent_id": self.agent_id, "strategy_spec": strategy_spec, "topology": topology, "run_seed": run_seed,
            "start_info": start_info, "personality": getattr(drone, "personality", None),
            "slm": {"enabled": bool(getattr(drone, "slm_enabled", False)),
                    "base_url": getattr(drone, "slm_base_url", None), "model": getattr(drone, "slm_model", None)},
            "base_cost_per_cell": type(drone).BASE_COST_PER_CELL, "core": core,
        }
        from src.drone_process import drone_main
        self._proc = ctx.Process(target=drone_main, args=(child, init), daemon=True)
        self._proc.start()
        child.close()
        msg = self._conn.recv()
        if msg[0] != "ready":
            raise RuntimeError(f"drone process for {self.agent_id} did not start: {msg}")
        self.pid = msg[1]

    # -- the interface the runner uses --------------------------------------------------------------------------
    def observe(self, step: int, sim_time: float):
        return self.port.observe(step, sim_time)

    def think(self, obs, dt: float, step: int, sim_time: float):
        self._conn.send(("think", obs, dt, step, sim_time))
        msg = self._conn.recv()
        if msg[0] == "error":
            raise RuntimeError(f"drone process {self.agent_id} failed:\n{msg[1]}")
        _, commands, signals, telemetry = msg
        self._telemetry = telemetry
        self.parked = bool(signals.get("parked", False))
        return commands, signals

    def act(self, commands, step: int) -> None:
        self.port.apply_all(commands, step)

    def replay(self, recorder) -> None:
        """Write the records the drone sent with its last answer to the flight recorder."""
        for name, args, kwargs in self._telemetry:
            getattr(recorder, name)(*args, **kwargs)
        self._telemetry = []

    def close(self) -> None:
        try:
            self._conn.send(("stop",))
            self._conn.recv()
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
