# src/drone_process.py
"""
The drone as a process: the drone's own computer.

One process per drone, pinned to one core. Inside it run the strategy, the perception layer, the controller (the
driver software) and the onboard language model client. It knows only what the world tells it: the start information
once, then one Observation per step; it answers each with its Commands, any signals (for example a request to abort)
and the records it would like written to the flight recorder. Nothing else crosses the boundary.

World side counterpart: src/process_world.py.
"""
import json
import os
import random
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.config import SLM_NUM_CTX

# One generation at a time per drone for its own model client (the model server queues the rest).
_SLM_LOCK = threading.Semaphore(4)


class RecorderProxy:
    """Stands in for the flight recorder inside the drone process: records are buffered and shipped to the world with
    the drone's answer to each step, where they are written in drone order."""

    def __init__(self):
        self._buffer: List[tuple] = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self._buffer.append((name, args, kwargs))
        return record

    def drain(self) -> List[tuple]:
        out, self._buffer = self._buffer, []
        return out


class SLMClient:
    """The drone's onboard language model client (a request to the model server)."""

    def __init__(self, agent_id: str, enabled: bool, base_url: Optional[str], model: Optional[str]):
        self.agent_id = agent_id
        self.slm_enabled = bool(enabled)
        self.slm_base_url = base_url
        self.slm_model = model
        self.last_usage = None

    def slm_generate(self, prompt: str, timeout: int = 60, retries: int = 2, output_schema=None,
                     num_predict: int = 256, think: bool = False) -> str:
        if not self.slm_enabled or not self.slm_base_url:
            raise RuntimeError(f"SLM not active for drone {self.agent_id}")
        import requests
        base = self.slm_base_url.rstrip("/")
        url = base if base.endswith("/api/generate") else f"{base}/api/generate"
        payload = {
            "model": self.slm_model, "prompt": prompt, "stream": False,
            **({"think": True} if think else {}),
            **({"format": output_schema} if output_schema and not think else {}),
            "options": {"num_ctx": SLM_NUM_CTX, "num_predict": int(num_predict), "temperature": 0.0, "top_p": 0.1},
        }
        last_error = None
        for attempt in range(retries + 1):
            try:
                with _SLM_LOCK:
                    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                                             timeout=timeout)
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                data = response.json()
                text = data.get("response")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"Bad response: {text}")
                self.last_usage = {"prompt_tokens": data.get("prompt_eval_count"), "gen_tokens": data.get("eval_count")}
                if (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0) >= SLM_NUM_CTX - 16:
                    raise RuntimeError(f"context window ({SLM_NUM_CTX} tokens) full: the prompt may have been cut")
                return text.strip()
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"[SLM FAILURE][{self.agent_id}] after {retries + 1} attempts: {last_error}")


class DroneIdentity:
    """What the drone knows about itself that is configuration, not world: its id, personality, physics and model
    client. (BASE_COST_PER_CELL is the nominal model constant its declared rate is measured against.)"""
    BASE_COST_PER_CELL = 0.2

    def __init__(self, agent_id, personality, physics, slm: SLMClient, base_cost_per_cell: float):
        self.agent_id = agent_id
        self.personality = personality
        self.physics = physics
        self.BASE_COST_PER_CELL = base_cost_per_cell
        self.slm_enabled = slm.slm_enabled
        self.slm_base_url = slm.slm_base_url
        self.slm_generate = slm.slm_generate
        self._slm = slm

    @property
    def slm_model(self):
        return self._slm.slm_model

    @slm_model.setter
    def slm_model(self, model):
        self._slm.slm_model = model

    @property
    def slm_last_usage(self):
        return self._slm.last_usage


def drone_main(conn, init: Dict[str, Any]) -> None:
    """Entry point of a drone process. `init`: agent_id, strategy_spec, topology, run_seed, start_info, personality,
    slm {enabled, base_url, model}, base_cost_per_cell, core (or None)."""
    if init.get("core") is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(init["core"])})
    agent_id = init["agent_id"]
    random.seed(f"{init['run_seed']}-{agent_id}")

    from src.controllers import AIAgentController
    from src.physics.physics_2d import Grid2DPhysics
    from src.strategies.factory import build_strategy

    physics = Grid2DPhysics()
    if getattr(init["start_info"], "obstacles", None):
        from src import obstacles          # the declared map of no-fly blocks
        obstacles.set_obstacles(init["start_info"].obstacles, init["start_info"].grid)
    if init["strategy_spec"]["name"].startswith("centralized_cc"):
        # The ground station plans; the drone only receives directives over the radio and carries them out.
        from src.strategies.base import BaseStrategy

        class DirectiveFollower(BaseStrategy):
            def decide(self, snapshot):
                return {"mode": "wait"}
        strategy = DirectiveFollower()
        strategy.physics = physics
    else:
        strategy = build_strategy(strategy_spec=init["strategy_spec"],
                                  terrain=SimpleNamespace(command_center=None, stations=[]),
                                  agents=[], physics=physics, graphics_dir=None, grid_size=0)
    recorder = RecorderProxy()
    slm = SLMClient(agent_id, **{"enabled": init["slm"]["enabled"], "base_url": init["slm"]["base_url"],
                                 "model": init["slm"]["model"]})
    identity = DroneIdentity(agent_id, init.get("personality"), physics, slm, init["base_cost_per_cell"])
    controller = AIAgentController(drone=identity, terrain=None, strategy=strategy, topology=init["topology"],
                                   flight_recorder=recorder, sensor=None, start_info=init["start_info"],
                                   run_seed=init["run_seed"], port=None)
    strategy.set_drone(controller.services)
    if hasattr(strategy, "flight_recorder"):
        strategy.flight_recorder = recorder
    conn.send(("ready", os.getpid()))
    _think_cpu, _think_calls = 0.0, 0
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            if os.environ.get("DRONE_CPU_LOG"):      # optional measurement of the CPU time this drone spent thinking; changes nothing else
                with open(os.environ["DRONE_CPU_LOG"], "a") as fh:
                    fh.write(json.dumps({"agent": agent_id, "strategy": init["strategy_spec"]["name"], "think_cpu_s": _think_cpu, "calls": _think_calls}) + "\n")
            conn.send(("stopped",))
            return
        _, obs, dt, step, sim_time = msg
        try:
            _t0 = time.process_time()
            commands, signals = controller.think(obs, dt, step, sim_time)
            _think_cpu += time.process_time() - _t0
            _think_calls += 1
            conn.send(("thought", commands, signals, recorder.drain()))
        except Exception as exc:  # report the failure to the world instead of dying silently
            import traceback
            conn.send(("error", traceback.format_exc()))
            return
