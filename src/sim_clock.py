# src/sim_clock.py
"""
Simulation clock shared by the experiment runner and time-dependent logic.

With a fixed timestep (default), every simulation step advances time by exactly
SIM_DT seconds, independent of how long the frame took on the host machine.
Movement, cooldowns and timeouts then depend only on the step count, so
makespan measures coordination rather than machine speed or SLM inference time.

Set SIM_FIXED_TIMESTEP=0 to reproduce the submitted (v1.0-iotj) behaviour,
where dt was the real wall-clock time between frames.
"""
import os
import time

FIXED_TIMESTEP = os.getenv("SIM_FIXED_TIMESTEP", "1") != "0"
SIM_DT = 1.0 / 60.0
CELL_PX = 100                      # pixels per grid cell
DRONE_SPEED_PX_S = 420             # drone speed, pixels per second
STEPS_PER_CELL = CELL_PX / (DRONE_SPEED_PX_S * SIM_DT)       # steps a drone needs to fly one cell (14.29)

_sim_time = 0.0


def set_time(sim_time: float) -> None:
    """Called by the runner once per step."""
    global _sim_time
    _sim_time = float(sim_time)


def now() -> float:
    """Current time for simulation logic (timeouts, assignment timestamps)."""
    return _sim_time if FIXED_TIMESTEP else time.time()


def step_dt(wall_dt: float) -> float:
    """Timestep used to advance physics for one simulation step."""
    return SIM_DT if FIXED_TIMESTEP else wall_dt
