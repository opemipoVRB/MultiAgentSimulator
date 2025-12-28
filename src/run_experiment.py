# src/run_experiment.py
"""
GUI-only experiment runner for the Drone Agent simulator.

Usage:
  python src/run_experiment.py

Behavior:
 - Everything driven via GUI screens.
 - Mirrors visuals from games.py to keep look and feel identical.
 - Experiment JSONs: created as temp_<run_id>.json while running, renamed to experiment_<run_id>.json on success.
 - Resumes the most recent unfinished temp_*.json or experiment_*.json when possible.
 - Supports per-strategy snapshots so an aborted run can resume from the exact world state.
 - Stores agents and comm_config; snapshot contains agents_state list.
"""
import pygame
import json
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from flight_recorder import FlightRecorder
from artifacts import Terrain, Drone, Parcel
from controllers import HumanAgentController, AIAgentController, ControllerSwitcher
from utils import load_image, scale_to_cell

# Visual parity settings from games.py
GRID_SIZE = 100
SPEED = 420
DRONE_SCALE = 1.25
PARCEL_SCALE = 0.70
ANIM_FPS = 12
FLASH_DURATION = 0.15

BASE_DIR = Path(__file__).parent
GRAPHICS_DIR = BASE_DIR / "graphics"
EXPERIMENTS_DIR = BASE_DIR.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PREFIX = "temp_"
EXPERIMENT_PREFIX = "experiment_"

# metric headers kept for reference; results are stored in JSON
METRIC_HEADERS = [
    "run_id",
    "planner",
    "seed",
    "n_drones",
    "n_parcels",
    "comm_reliability",
    "battery_capacity",
    "episodic_interval",
    "total_energy_consumed",
    "total_distance_traveled",
    "tasks_completed",
    "mission_makespan_steps",
    "unsafe_failures",
    "messages_sent",
    "runtime_seconds",
    "timestamp",
]


class MetricsAccumulator:
    def __init__(self, run_id: str, planner: str, seed: int, n_drones: int, n_parcels: int,
                 comm_reliability: float = 1.0, battery_capacity: float = 100.0,
                 episodic_interval: float = 1.0):
        self.fields: Dict[str, float] = {h: 0.0 for h in METRIC_HEADERS}
        self.fields["run_id"] = run_id
        self.fields["planner"] = planner
        self.fields["seed"] = int(seed)
        self.fields["n_drones"] = int(n_drones)
        self.fields["n_parcels"] = int(n_parcels)
        self.fields["comm_reliability"] = float(comm_reliability)
        self.fields["battery_capacity"] = float(battery_capacity)
        self.fields["episodic_interval"] = float(episodic_interval)
        self.fields["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.total_energy_consumed = 0.0
        self.total_distance_traveled = 0.0
        self.tasks_completed = 0
        self.mission_makespan_steps = 0
        self.unsafe_failures = 0
        self.messages_sent = 0

        self.start_time = time.time()

    def record_distance(self, pixels: float, grid_size: int):
        cells = float(pixels) / float(grid_size)
        self.total_distance_traveled += cells

    def record_task_complete(self, n: int = 1):
        self.tasks_completed += int(n)

    def record_unsafe_failure(self, n: int = 1):
        self.unsafe_failures += int(n)

    def record_messages_sent(self, n: int = 1):
        self.messages_sent += int(n)

    def finalize(self, steps: int, battery_capacity: float, battery_level: float):
        self.mission_makespan_steps = int(steps)
        self.total_energy_consumed = max(0.0, battery_capacity - battery_level)

        self.fields["total_energy_consumed"] = round(self.total_energy_consumed, 3)
        self.fields["total_distance_traveled"] = round(self.total_distance_traveled, 3)
        self.fields["tasks_completed"] = int(self.tasks_completed)
        self.fields["mission_makespan_steps"] = int(self.mission_makespan_steps)
        self.fields["unsafe_failures"] = int(self.unsafe_failures)
        self.fields["messages_sent"] = int(self.messages_sent)

        runtime = time.time() - self.start_time
        self.fields["runtime_seconds"] = round(runtime, 3)
        self.fields["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def as_dict(self):
        # convenience for storing in JSON results
        self.fields["total_energy_consumed"] = round(self.total_energy_consumed, 3)
        self.fields["total_distance_traveled"] = round(self.total_distance_traveled, 3)
        self.fields["tasks_completed"] = int(self.tasks_completed)
        self.fields["mission_makespan_steps"] = int(self.mission_makespan_steps)
        self.fields["unsafe_failures"] = int(self.unsafe_failures)
        self.fields["messages_sent"] = int(self.messages_sent)
        runtime = time.time() - self.start_time
        self.fields["runtime_seconds"] = round(runtime, 3)
        self.fields["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return dict(self.fields)


def deterministic_parcel_positions(seed: Optional[int], n_parcels: int, cols: int, rows: int,
                                   forbidden_cells: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    rng = random.Random(seed if seed is not None else int(time.time()))
    all_cells = [(c, r) for r in range(rows) for c in range(cols)]
    if forbidden_cells:
        forbidden = set(forbidden_cells)
        all_cells = [c for c in all_cells if c not in forbidden]
    rng.shuffle(all_cells)
    return all_cells[:max(0, min(n_parcels, len(all_cells)))]


class AgentSpawnError(Exception):
    pass


class ExperimentRunnerGUI:
    SCHEMA_VERSION = 1

    def __init__(self):
        # UI display init
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_size = pygame.display.get_surface().get_size()
        pygame.display.set_caption("Drone Agent - Experiment Runner")
        self.clock = pygame.time.Clock()

        # fonts to match games.py
        self.font = pygame.font.SysFont("Consolas", 20)
        self.narration_font = pygame.font.SysFont("Consolas", 22)
        self.title_font = pygame.font.SysFont("Consolas", 72)
        self.large_font = pygame.font.SysFont("Consolas", 48)

        # load graphics same way as games.py
        self.images = self._load_graphics()

        # experiment metadata
        self.seed = int(time.time()) % (2 ** 31)
        self.parcels = 15
        self.planners = ["local"]
        self.run_id = str(uuid.uuid4())[:8]
        self.experiment: Dict = {}
        self.experiment_json_path: Optional[Path] = None

        # communication config (placeholder)
        self.default_comm_config = {"reliability": 1.0, "delay_mean": 0.0, "delay_std": 0.0}

        # behavior controls
        self.max_steps = 60 * 60 * 5
        self.stall_seconds = 6.0  # if idle longer than this considered a stall when parcels remain

    def _show_modal(
            self,
            *,
            title: str,
            message: str,
            background=(30, 30, 30),
            title_color=(255, 80, 80),
            hint: str = "Press any key to continue",
            exit_keys=None,
    ):
        """
        Generic blocking modal renderer.

        Parameters
        ----------
        title : str
            Modal title text.
        message : str
            Message body. Newlines are respected.
        background : tuple
            RGB background color.
        title_color : tuple
            RGB title color.
        hint : str
            Hint text displayed at the bottom.
        exit_keys : set[int] | None
            Specific pygame keys that close the modal.
            If None, any key or mouse press closes it.
        """
        if exit_keys is not None:
            exit_keys = set(exit_keys)

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()

                if e.type == pygame.KEYDOWN:
                    if exit_keys is None or e.key in exit_keys:
                        return

                if exit_keys is None and e.type == pygame.MOUSEBUTTONDOWN:
                    return

            self.screen.fill(background)

            title_surf = self.large_font.render(title, True, title_color)
            self.screen.blit(
                title_surf,
                title_surf.get_rect(center=(self.screen_size[0] // 2, 180)),
            )

            y = 260
            for line in message.split("\n"):
                txt = self.font.render(line, True, (230, 230, 230))
                self.screen.blit(
                    txt,
                    txt.get_rect(center=(self.screen_size[0] // 2, y)),
                )
                y += 34

            hint_surf = self.font.render(hint, True, (180, 180, 180))
            self.screen.blit(
                hint_surf,
                hint_surf.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 120)),
            )

            pygame.display.flip()
            self.clock.tick(30)

    def show_error(self, title: str, message: str):
        """
        Blocking modal error dialog.

        Used for fatal or informational errors.
        Returns after user acknowledgement.
        """
        self._show_modal(
            title=title,
            message=message,
            background=(30, 30, 30),
            title_color=(255, 80, 80),
            hint="Press any key to continue",
            exit_keys=None,  # any key or mouse
        )

    def _show_error_and_restart_setup(self, message: str):
        """
        Show a configuration error and restart setup flow.
        """
        self._show_modal(
            title="Invalid Configuration",
            message=message,
            background=(40, 44, 52),
            title_color=(240, 100, 100),
            hint="Press ENTER to return to setup",
            exit_keys={pygame.K_RETURN, pygame.K_ESCAPE},
        )

    def show_experiment_setup(
            self,
            initial_agents: int = 1,
            initial_parcels: int = 15,
            min_agents: int = 1,
            max_agents: int = 50,
            min_parcels: int = 0,
            max_parcels: int = 1000,
    ):
        """
        GUI screen to choose:
          - number of agents
          - number of parcels

        Controls:
          - TAB switches between fields
          - UP / DOWN increment selected field
          - Type digits to enter value
          - ENTER to accept
          - ESC to abort
        Returns:
          (n_agents, n_parcels)
        """

        agents = initial_agents
        parcels = initial_parcels
        typing = ""
        active_field = "agents"  # or "parcels"

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()

                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        raise SystemExit()

                    elif e.key == pygame.K_TAB:
                        active_field = "parcels" if active_field == "agents" else "agents"
                        typing = ""

                    elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return agents, parcels

                    elif e.key == pygame.K_BACKSPACE:
                        typing = typing[:-1]

                    elif e.key == pygame.K_UP:
                        if active_field == "agents":
                            agents = min(max_agents, agents + 1)
                        else:
                            parcels = min(max_parcels, parcels + 1)
                        typing = ""

                    elif e.key == pygame.K_DOWN:
                        if active_field == "agents":
                            agents = max(min_agents, agents - 1)
                        else:
                            parcels = max(min_parcels, parcels - 1)
                        typing = ""

                    elif e.unicode and e.unicode.isdigit():
                        typing += e.unicode
                        try:
                            v = int(typing)
                            if active_field == "agents":
                                agents = max(min_agents, min(max_agents, v))
                            else:
                                parcels = max(min_parcels, min(max_parcels, v))
                        except ValueError:
                            typing = ""

            # -------- render --------
            self.screen.fill((40, 44, 52))

            header = self.large_font.render("Experiment Setup", True, (230, 230, 230))
            self.screen.blit(header, header.get_rect(center=(self.screen_size[0] // 2, 140)))

            agents_color = (240, 240, 200) if active_field == "agents" else (180, 180, 180)
            parcels_color = (240, 240, 200) if active_field == "parcels" else (180, 180, 180)

            agents_txt = self.title_font.render(str(agents), True, agents_color)
            parcels_txt = self.title_font.render(str(parcels), True, parcels_color)

            self.screen.blit(
                self.font.render("Agents", True, agents_color),
                (self.screen_size[0] // 2 - 260, 300),
            )
            self.screen.blit(agents_txt, agents_txt.get_rect(center=(self.screen_size[0] // 2 - 200, 360)))

            self.screen.blit(
                self.font.render("Parcels", True, parcels_color),
                (self.screen_size[0] // 2 + 100, 300),
            )
            self.screen.blit(parcels_txt, parcels_txt.get_rect(center=(self.screen_size[0] // 2 + 200, 360)))

            hint = self.font.render(
                "TAB switch field | UP/DOWN adjust | Type digits | ENTER start | ESC cancel",
                True,
                (200, 200, 200),
            )
            self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 120)))

            pygame.display.flip()
            self.clock.tick(30)

    def _load_graphics(self):
        def _load_and_scale(name, scale_factor):
            p = GRAPHICS_DIR / name
            img = load_image(p)
            return scale_to_cell(img, GRID_SIZE, scale_factor) if img else None

        drone_static = _load_and_scale("drone_static.png", DRONE_SCALE)

        drone_rot_frames = []
        i = 0
        while True:
            p = GRAPHICS_DIR / f"drone_rotating_{i}.png"
            if p.exists():
                try:
                    img = pygame.image.load(str(p)).convert_alpha()
                    drone_rot_frames.append(scale_to_cell(img, GRID_SIZE, DRONE_SCALE))
                except Exception:
                    pass
                i += 1
            else:
                break
        if not drone_rot_frames:
            r = load_image(GRAPHICS_DIR / "drone_rotating.png")
            if r:
                drone_rot_frames = [scale_to_cell(r, GRID_SIZE, DRONE_SCALE)]

        drone_static_with_parcel = _load_and_scale("drone_static_with_parcel.png", DRONE_SCALE)

        drone_rot_with_parcel_frames = []
        i = 0
        while True:
            p = GRAPHICS_DIR / f"drone_rotating_with_parcel_{i}.png"
            if p.exists():
                try:
                    img = pygame.image.load(str(p)).convert_alpha()
                    drone_rot_with_parcel_frames.append(scale_to_cell(img, GRID_SIZE, DRONE_SCALE))
                except Exception:
                    pass
                i += 1
            else:
                break
        if not drone_rot_with_parcel_frames:
            r = load_image(GRAPHICS_DIR / "drone_rotating_with_parcel.png")
            if r:
                drone_rot_with_parcel_frames = [scale_to_cell(r, GRID_SIZE, DRONE_SCALE)]

        parcel_img = None
        try:
            parcel_img_raw = load_image(GRAPHICS_DIR / "parcel.png")
            parcel_img = scale_to_cell(parcel_img_raw, GRID_SIZE, PARCEL_SCALE) if parcel_img_raw else None
        except Exception:
            parcel_img = None

        return {
            "drone_static": drone_static,
            "drone_rot_frames": drone_rot_frames,
            "drone_static_with_parcel": drone_static_with_parcel,
            "drone_rot_with_parcel_frames": drone_rot_with_parcel_frames,
            "parcel_img": parcel_img,
        }

    def _find_recent_experiment(self) -> Optional[Path]:
        # prefer unfinished temp_* first that match seed/parcels, else most recent experiment_*
        candidates = sorted(EXPERIMENTS_DIR.glob("temp_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
        candidates2 = sorted(EXPERIMENTS_DIR.glob(f"{EXPERIMENT_PREFIX}*.json"), key=lambda p: p.stat().st_mtime,
                             reverse=True)
        if candidates2:
            return candidates2[0]
        return None

    def _load_or_create_experiment(
            self,
            n_agents: int,
            n_parcels: int,
            selected_planner: str,
    ):
        """
        Load the most recent unfinished experiment if one exists.
        Otherwise, create a new experiment using GUI-authoritative values.

        RULE: When resuming, we MUST ensure the experiment matches GUI parameters.
        If parameters differ, we create a new experiment instead.
        """

        # -------------------------------------------------
        # 1) Look for most recent unfinished experiment
        # -------------------------------------------------
        candidate = self._find_recent_experiment()
        if candidate:
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))

                if data.get("status") != "completed":
                    # Check if stored parameters match GUI parameters
                    stored_agents = len(data.get("agents", []))
                    stored_parcels = len(data.get("parcels", []))

                    # CRITICAL: Only resume if parameters match
                    if stored_agents == n_agents and stored_parcels == n_parcels:
                        # Resume existing experiment
                        self.experiment = data
                        self.experiment.setdefault("current_index", 0)

                        # Backward compatibility: strategies
                        if "strategies" not in self.experiment:
                            planners = self.experiment.get("planners", ["local"])
                            self.experiment["strategies"] = [
                                {"planner": p, "snapshot": None, "status": "pending"}
                                for p in planners
                            ]

                        # Schema evolution defaults
                        self.experiment.setdefault("schema_version", self.SCHEMA_VERSION)

                        # Agents (default single agent if missing)
                        if "agents" not in self.experiment:
                            planners = self.experiment.get("planners", ["local"])
                            self.experiment["agents"] = [
                                {
                                    "id": "agent_0",
                                    "type": "drone",
                                    "planner": planners[0],
                                }
                            ]

                        # Communication config
                        self.experiment.setdefault(
                            "comm_config",
                            self.default_comm_config.copy()
                        )

                        self.experiment_json_path = candidate
                        self.seed = self.experiment.get("seed", self.seed)

                        # IMPORTANT: Use GUI value, not stored value
                        self.parcels = n_parcels  # ⬅️ FIX: Use GUI value

                        print(f"[EXPERIMENT] Resuming unfinished experiment {candidate}")
                        return
                    else:
                        print(f"[EXPERIMENT] Parameters mismatch: "
                              f"stored agents={stored_agents} vs GUI={n_agents}, "
                              f"stored parcels={stored_parcels} vs GUI={n_parcels}")
                        # Fall through to create new experiment

                else:
                    print(
                        f"[EXPERIMENT] Most recent experiment {candidate} already completed. "
                        "Creating new experiment."
                    )

            except Exception as e:
                print(f"[EXPERIMENT] Failed to resume {candidate}: {e}")

        # -------------------------------------------------
        # 2) Create new experiment (GUI IS AUTHORITATIVE)
        # -------------------------------------------------
        self.parcels = n_parcels
        self.seed = int(time.time()) % (2 ** 31)

        cols = self.screen_size[0] // GRID_SIZE
        rows = self.screen_size[1] // GRID_SIZE

        center_col = max(2, cols // 2 - 2)
        center_row = max(2, rows // 2 - 2)

        station_cells = [
            (c, r)
            for r in range(center_row, center_row + 4)
            for c in range(center_col, center_col + 4)
        ]

        parcel_positions = deterministic_parcel_positions(
            self.seed,
            n_parcels,  # ✅ GUI VALUE USED HERE
            cols,
            rows,
            forbidden_cells=station_cells,
        )

        self.run_id = str(uuid.uuid4())[:8]
        temp_name = f"{TEMP_PREFIX}{self.run_id}.json"
        self.experiment_json_path = EXPERIMENTS_DIR / temp_name

        planners = [selected_planner]
        strategies = [
            {"planner": p, "snapshot": None, "status": "pending"}
            for p in planners
        ]

        # Parcels with stable IDs
        parcels_with_id = []
        for i, (c, r) in enumerate(parcel_positions):
            parcels_with_id.append(
                {
                    "id": i,
                    "col": int(c),
                    "row": int(r),
                    "picked": False,
                    "delivered": False,
                }
            )

        # Agents (GUI-controlled)
        start_col = cols // 2
        start_row = rows // 2

        agents = [
            {
                "id": f"agent_{i}",
                "type": "drone",
                "planner": selected_planner,
                "start_col": start_col + i * 2,
                "start_row": start_row,
            }
            for i in range(n_agents)
        ]

        self.experiment = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "seed": self.seed,
            "planners": planners,
            "parcels": parcels_with_id,
            "strategies": strategies,
            "agents": agents,
            "comm_config": self.default_comm_config.copy(),
            "current_index": 0,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [],
        }

        self._write_experiment_json()
        print(f"[EXPERIMENT] Created new experiment {self.experiment_json_path}")

    def _write_experiment_json(self):
        if not self.experiment_json_path:
            return
        # persist current experiment state
        self.experiment_json_path.write_text(json.dumps(self.experiment, indent=2), encoding="utf-8")

    def _finalize_experiment_file(self):
        # rename temp_ to experiment_ when finished successfully
        if not self.experiment_json_path:
            return
        p = self.experiment_json_path
        if p.name.startswith(TEMP_PREFIX):
            final = EXPERIMENTS_DIR / f"{EXPERIMENT_PREFIX}{p.name[len(TEMP_PREFIX):]}"
            try:
                p.rename(final)
                self.experiment_json_path = final
                print(f"[EXPERIMENT] Renamed {p} -> {final}")
            except Exception:
                # fallback to keeping existing path
                pass

    # -----------------------
    # Snapshot and logging helpers
    # -----------------------
    def _ensure_snapshot_dict(self, strategy_idx: int) -> Dict:
        """
        Ensure that strategies[strategy_idx]['snapshot'] is a dict. Return the snapshot dict.
        This prevents NoneType errors when appending events/messages.
        """
        strategies = self.experiment.setdefault("strategies", [])
        if strategy_idx < 0 or strategy_idx >= len(strategies):
            raise IndexError("strategy_idx out of range")
        strat = strategies[strategy_idx]
        if strat.get("snapshot") is None:
            strat["snapshot"] = {}
        return strat["snapshot"]

    def _append_message(self, strategy_idx: int, msg: Dict):
        """
        Append an opaque message dict to the current strategy's snapshot.message_log.
        Messages remain freeform. The helper ensures snapshot exists.
        """
        try:
            snap = self._ensure_snapshot_dict(strategy_idx)
        except IndexError:
            return
        log = snap.setdefault("message_log", [])
        log.append(msg)
        # checkpoint occasionally
        if len(log) % 200 == 0:
            self._write_experiment_json()

    def _append_event(self, strategy_idx: int, ev: Dict):
        """
        Append an event dict to the current strategy's snapshot.events list.
        Useful for moves, pick/drop attempts, unsafe events, etc.
        """
        try:
            snap = self._ensure_snapshot_dict(strategy_idx)
        except IndexError:
            return
        evs = snap.setdefault("events", [])
        evs.append(ev)
        if len(evs) % 500 == 0:
            self._write_experiment_json()

    def _make_snapshot(self, planner_name: str, agents: List[Drone], terrain: Terrain, steps: int) -> Dict:
        """
        Create a compact snapshot capturing parcels, agents' states, and steps elapsed.
        agents is a list of Drone instances in the same order as self.experiment['agents'].
        The snapshot contains:
          - agents_state: list of per-agent dicts
          - parcels: list of parcel dicts (id, col, row, weight, picked, delivered)
          - steps, planner, time
        """
        # map existing experiment parcels (by position) to their ids if they exist
        id_map = {}
        for pdef in self.experiment.get("parcels", []):
            key = (int(pdef["col"]), int(pdef["row"]))
            id_map[key] = pdef.get("id")

        parcels_snap = []
        for p in terrain.parcels:
            pid = id_map.get((int(p.col), int(p.row)))
            parcels_snap.append({
                "id": pid if pid is not None else None,
                "col": int(p.col),
                "row": int(p.row),
                "weight": getattr(p, "weight", 1.0),
                "picked": bool(getattr(p, "picked", False)),
                "delivered": bool(getattr(p, "delivered", False)),
            })

        agents_state = []
        for idx, agent in enumerate(agents):
            carried_id = None
            if getattr(agent, "carrying", None):
                carried = agent.carrying
                # try to locate carried parcel id by matching position
                for ps in parcels_snap:
                    if ps["col"] == int(carried.col) and ps["row"] == int(carried.row):
                        carried_id = ps.get("id")
                        break
            agent_state = {
                "agent_id": self.experiment.get("agents", [{}])[idx].get("id", f"agent_{idx}"),
                "type": "drone",
                "col": int(agent.col),
                "row": int(agent.row),
                "pos_x": float(agent.pos.x),
                "pos_y": float(agent.pos.y),
                "battery_level": float(getattr(agent.power, "level", getattr(agent, "battery_capacity", 100.0))),
                "battery_capacity": float(getattr(agent.power, "capacity", getattr(agent, "battery_capacity", 100.0))),
                "carrying_id": carried_id,
            }
            agents_state.append(agent_state)

        snap = {
            "planner": planner_name,
            "steps": int(steps),
            "time": time.time(),
            "agents_state": agents_state,
            "parcels": parcels_snap,
        }
        return snap

    def _restore_snapshot(self, snapshot: Dict) -> Tuple[Terrain, List[Drone], int]:
        """
        Restore world state from a snapshot and return (terrain, agents_list, steps).

        Contract:
        - Agent identity MUST come from snapshot["agents_state"][*]["agent_id"]
        - No agent IDs are invented, reordered, or renamed
        - Terrain is authoritative for spatial occupancy
        """

        # ----------------------------
        # Recreate terrain
        # ----------------------------
        terrain = Terrain(
            GRID_SIZE,
            self.screen_size,
            parcel_img=self.images.get("parcel_img"),
            parcel_scale=PARCEL_SCALE,
        )

        cols = self.screen_size[0] // GRID_SIZE
        rows = self.screen_size[1] // GRID_SIZE
        center_col = max(2, cols // 2 - 2)
        center_row = max(2, rows // 2 - 2)
        terrain.add_station(center_col, center_row, w=4, h=4)

        # ----------------------------
        # Recreate parcels
        # ----------------------------
        parcels_snap = snapshot.get("parcels", [])
        recreated: List[Parcel] = []

        for ps in parcels_snap:
            p = Parcel(
                int(ps["col"]),
                int(ps["row"]),
                terrain.grid_size,
                weight=float(ps.get("weight", 1.0)),
            )
            p.picked = bool(ps.get("picked", False))
            p.delivered = bool(ps.get("delivered", False))
            recreated.append(p)

        terrain.parcels = recreated

        # ----------------------------
        # Recreate agents (STRICT)
        # ----------------------------
        agents_state = snapshot.get("agents_state")
        if not agents_state:
            raise ValueError("Snapshot missing agents_state; cannot restore agents")

        agents: List[Drone] = []

        for astate in agents_state:
            if "agent_id" not in astate:
                raise ValueError("Snapshot agent missing agent_id; refusing to restore")

            agent_id = astate["agent_id"]

            start_col = int(astate["col"])
            start_row = int(astate["row"])

            drone = Drone(
                start_cell=(start_col, start_row),
                grid_size=GRID_SIZE,
                screen_size=self.screen_size,
                terrain=terrain,
                agent_id=agent_id,
            )

            # --- restore precise position ---
            drone.pos = pygame.Vector2(
                float(astate.get("pos_x", drone.pos.x)),
                float(astate.get("pos_y", drone.pos.y)),
            )
            drone.col = start_col
            drone.row = start_row
            drone.last_cell = (start_col, start_row)

            # --- restore battery ---
            cap = float(astate.get("battery_capacity", drone.power.capacity))
            lvl = float(astate.get("battery_level", drone.power.level))
            drone.power.capacity = cap
            drone.power.level = min(lvl, cap)

            # --- restore carried parcel ---
            carrying_id = astate.get("carrying_id")
            if carrying_id is not None:
                id_to_pos = {
                    orig["id"]: (int(orig["col"]), int(orig["row"]))
                    for orig in self.experiment.get("parcels", [])
                }
                target_pos = id_to_pos.get(carrying_id)

                if target_pos:
                    for p in recreated:
                        if (p.col, p.row) == target_pos:
                            drone.carrying = p
                            break

            agents.append(drone)

            # ----------------------------
            # Register occupancy explicitly
            # ----------------------------
            terrain.register_agent_cell(drone, drone.col, drone.row)

        steps = int(snapshot.get("steps", 0))
        return terrain, agents, steps

    # -----------------------
    # Existing UI helpers
    # -----------------------
    def show_splash(self, timeout=2.0):
        # same look and feel as games.py splash
        splash_img = None
        p = GRAPHICS_DIR / "drone_static.png"
        if p.exists():
            splash_img = load_image(p)
            if splash_img:
                max_w = int(self.screen_size[0] * 0.6)
                max_h = int(self.screen_size[1] * 0.4)
                w, h = splash_img.get_size()
                scale = min(1.0, max_w / w, max_h / h)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                splash_img = pygame.transform.smoothscale(splash_img, new_size)

        start = time.time()
        while True:
            now = time.time()
            elapsed = now - start
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN or e.type == pygame.MOUSEBUTTONDOWN:
                    return
            self.screen.fill((24, 32, 48))
            if splash_img:
                img_rect = splash_img.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2 - 80))
                self.screen.blit(splash_img, img_rect)
                text_y = img_rect.bottom + 28
            else:
                text_y = self.screen_size[1] // 2 - 60

            title = self.title_font.render("Agentic Swarm Lab", True, (220, 230, 240))
            subtitle = self.large_font.render("Episodic LLM Guidance", True, (180, 200, 220))
            hint = self.font.render("Press any key to continue or wait...", True, (180, 180, 200))

            self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, text_y)))
            self.screen.blit(subtitle, subtitle.get_rect(center=(self.screen_size[0] // 2, text_y + 64)))
            self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, text_y + 140)))

            pygame.display.flip()
            self.clock.tick(60)
            if elapsed >= timeout:
                return

    def show_setup_and_planner_select(self):
        """
        Show a GUI screen that displays:
         - current experiment info (seed, parcels, planners)
         - allows choosing planner index with left/right or TAB
         - Enter to start
         - ESC to quit (abort)
        """
        planners = list(self.planners)
        planner_index = 0
        typing = ""
        info_lines = [
            f"Seed: {self.seed}",
            f"Parcels: {self.parcels}",
        ]

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        print("[EXPERIMENT] Setup aborted by user")
                        raise SystemExit()
                    elif e.key in (pygame.K_RIGHT, pygame.K_TAB):
                        planner_index = (planner_index + 1) % max(1, len(planners))
                    elif e.key == pygame.K_LEFT:
                        planner_index = (planner_index - 1) % max(1, len(planners))
                    elif e.key == pygame.K_RETURN or e.key == pygame.K_KP_ENTER:
                        return planners[planner_index]
                    elif e.key == pygame.K_BACKSPACE:
                        typing = typing[:-1]
                    elif e.unicode and e.unicode.isdigit():
                        typing += e.unicode
                        try:
                            v = int(typing)
                            if v > 1000:
                                typing = str(1000)
                        except ValueError:
                            typing = ""
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    planner_index = (planner_index + 1) % max(1, len(planners))

            self.screen.fill((40, 44, 52))
            header = self.large_font.render("Experiment Setup", True, (230, 230, 230))
            self.screen.blit(header, header.get_rect(center=(self.screen_size[0] // 2, 120)))

            for i, ln in enumerate(info_lines):
                text_surf = self.font.render(ln, True, (210, 210, 210))
                self.screen.blit(text_surf, (self.screen_size[0] // 2 - 220, 220 + i * 30))

            left = self.screen_size[0] // 2 - 220
            top = 340
            for i, p in enumerate(planners):
                txt = self.font.render(f"[{i + 1}] {p}", True,
                                       (240, 240, 200) if i == planner_index else (180, 180, 180))
                self.screen.blit(txt, (left + (i % 3) * 220, top + (i // 3) * 36))

            bottom_hint = self.font.render(f"Selected planner: {planners[planner_index]}   Press Enter to start", True,
                                           (230, 230, 230))
            self.screen.blit(bottom_hint, (self.screen_size[0] // 2 - 320, self.screen_size[1] - 120))

            pygame.display.flip()
            self.clock.tick(30)

    def _configure_planner(self, name: str):
        """
        Planner configuration hook.

        For snapshot-based planners (e.g., NaiveStrategy),
        planner selection is structural rather than parametric.
        No runtime configuration is performed.
        """
        print(f"[PLANNER] using planner '{name}' (no runtime configuration)")

    def _setup_agents(self, terrain: Terrain):
        agents = {}

        station = terrain.nearest_station(
            self.screen_size[0] // GRID_SIZE // 2,
            self.screen_size[1] // GRID_SIZE // 2,
        )

        if not station:
            raise AgentSpawnError(
                "No station found. Cannot spawn agents."
            )

        station_cells = [
            (c, r)
            for r in range(station.row, station.row + station.h)
            for c in range(station.col, station.col + station.w)
        ]

        rng = random.Random(self.seed)
        rng.shuffle(station_cells)

        if len(station_cells) < len(self.experiment["agents"]):
            raise AgentSpawnError(
                f"Too many agents ({len(self.experiment['agents'])}) "
                f"for station capacity ({len(station_cells)}).\n\n"
                "Reduce the number of agents."
            )

        for agent_def, start_cell in zip(self.experiment["agents"], station_cells):
            drone = Drone(
                start_cell=start_cell,
                grid_size=GRID_SIZE,
                screen_size=self.screen_size,
                terrain=terrain,
                agent_id=agent_def["id"],
            )

            agents[agent_def["id"]] = drone
            terrain.register_agent_cell(drone, start_cell[0], start_cell[1])

        return agents, 0, None

    def _setup_scene(self, maybe_snapshot: Optional[Dict] = None) -> Tuple[Terrain, List[Drone], int]:
        """
        Create Terrain and agents.
        If maybe_snapshot provided, restore world state from it.
        Returns (terrain, agents_list, steps_already_taken)
        """

        # ---- Snapshot path (authoritative) ----
        if maybe_snapshot:
            try:
                terrain, agents, steps = self._restore_snapshot(maybe_snapshot)
                return terrain, agents, steps
            except Exception:
                # fall back to fresh scene if snapshot restore fails
                pass

        # ---- Fresh scene path (from experiment schema) ----

        # 1. Terrain
        terrain = Terrain(
            GRID_SIZE,
            self.screen_size,
            parcel_img=self.images.get("parcel_img"),
            parcel_scale=PARCEL_SCALE,
        )

        cols = self.screen_size[0] // GRID_SIZE
        rows = self.screen_size[1] // GRID_SIZE
        center_col = max(2, cols // 2 - 2)
        center_row = max(2, rows // 2 - 2)
        terrain.add_station(center_col, center_row, w=4, h=4)

        # 2. Parcels (schema-driven, deterministic)
        for pdef in self.experiment.get("parcels", []):
            if not isinstance(pdef, dict):
                continue

            terrain.add_parcel(
                int(pdef["col"]),
                int(pdef["row"]),
            )

            p = terrain.parcel_at_cell(
                int(pdef["col"]),
                int(pdef["row"]),
                include_delivered=True,
            )
            if p:
                p.picked = bool(pdef.get("picked", False))
                p.delivered = bool(pdef.get("delivered", False))

        # 3. Agents (delegated, future-proof)
        agents_dict, steps, error = self._setup_agents(terrain)

        if error:
            self._show_error_and_restart_setup(error)
            raise RuntimeError("Restart setup")  # intentional control break

        return terrain, list(agents_dict.values()), steps

    @staticmethod
    def all_parcels_delivered(terrain) -> bool:
        return not any(
            not getattr(p, "delivered", False)
            for p in terrain.parcels
        )

    @staticmethod
    def agent_at_base(drone, terrain) -> bool:
        return terrain.get_station_at(
            int(drone.col),
            int(drone.row),
        ) is not None

    @staticmethod
    def agent_dead(drone) -> bool:
        return (
                hasattr(drone, "power")
                and drone.power.level <= 0.0
        )

    def run(self):
        # -------------------------------------------------
        # Splash + setup phase (single abort boundary)
        # -------------------------------------------------
        self.show_splash(timeout=1.0)

        while True:
            try:
                # --- experiment scale setup (AUTHORITATIVE) ---
                n_agents, n_parcels = self.show_experiment_setup(
                    initial_agents=1,
                    initial_parcels=self.parcels,
                )

                self.parcels = n_parcels

                # --- planner selection ---
                selected_planner = self.show_setup_and_planner_select()

                # -------------------------------------------------
                # Load or create experiment AFTER GUI (SAFE)
                # -------------------------------------------------
                self._load_or_create_experiment(
                    n_agents=n_agents,
                    n_parcels=n_parcels,
                    selected_planner=selected_planner,
                )

                break  # only break once everything is valid

            except RuntimeError:
                # Recoverable configuration error → return to setup
                continue

            except SystemExit:
                print("[EXPERIMENT] User aborted at setup")
                return

        # -------------------------------------------------
        # Commit setup choices to experiment schema
        # -------------------------------------------------
        self.parcels = n_parcels

        planners = [selected_planner]
        self.experiment["planners"] = planners

        # rebuild agents deterministically
        start_col = (self.screen_size[0] // GRID_SIZE) // 2
        start_row = (self.screen_size[1] // GRID_SIZE) // 2

        self.experiment["agents"] = [
            {
                "id": f"agent_{i}",
                "type": "drone",
                "planner": selected_planner,
                "start_col": start_col + i * 2,
                "start_row": start_row,
            }
            for i in range(n_agents)
        ]

        if "strategies" not in self.experiment or len(self.experiment["strategies"]) != len(planners):
            self.experiment["strategies"] = [
                {"planner": p, "snapshot": None, "status": "pending"} for p in planners
            ]

        self.experiment["current_index"] = 0
        self.experiment.setdefault("schema_version", self.SCHEMA_VERSION)
        self.experiment.setdefault("comm_config", self.default_comm_config.copy())

        self._write_experiment_json()

        # -------------------------------------------------
        # STRATEGY LOOP
        # -------------------------------------------------
        for idx, strategy in enumerate(self.experiment["strategies"]):
            if idx < int(self.experiment.get("current_index", 0)):
                continue

            planner_name = strategy["planner"]
            print(f"[EXPERIMENT] Running planner '{planner_name}' ({idx + 1}/{len(self.experiment['strategies'])})")

            self._configure_planner(planner_name)

            snapshot = strategy.get("snapshot")
            try:
                terrain, agents, pre_steps = self._setup_scene(snapshot)
            except AgentSpawnError as e:
                self._show_error_and_restart_setup(str(e))
                return  # exit run(), caller restarts setup

            if terrain is None:
                return  # safely return to GUI loop

            # -------------------------------------------------
            # >>> FLIGHT RECORDER: instantiate once per strategy
            # -------------------------------------------------
            flight_recorder = FlightRecorder(
                run_id=self.experiment["run_id"],
                planner=planner_name,
                agents=self.experiment["agents"],
                parcels=self.experiment["parcels"],
                grid_size=GRID_SIZE,
                sim_dt=1.0 / 60.0,
            )

            # controllers (one per agent)
            controllers = {}
            for i, drone in enumerate(agents):
                agent_id = self.experiment["agents"][i]["id"]
                drone.agent_id = agent_id
                human = HumanAgentController(drone, terrain)
                ai = AIAgentController(
                    drone,
                    terrain,
                    enable_reactive_fallback=True,
                    flight_recorder=flight_recorder
                )
                print("Drone", drone)
                print("Drone ID", drone.agent_id)
                switcher = ControllerSwitcher([human, ai])
                switcher.index = 1
                controllers[agent_id] = switcher

            metrics = MetricsAccumulator(
                run_id=self.experiment["run_id"],
                planner=planner_name,
                seed=self.experiment["seed"],
                n_drones=len(agents),
                n_parcels=len(self.experiment.get("parcels", [])),
                comm_reliability=self.experiment["comm_config"].get("reliability", 1.0),
                battery_capacity=agents[0].power.capacity if agents else 100.0,
                episodic_interval=1.0,
            )

            steps = int(pre_steps or 0)
            last_move_time = time.time()
            last_action_time = time.time()
            aborted = False

            prev_positions = {i: (a.pos.x, a.pos.y) for i, a in enumerate(agents)}

            # -------------------------------------------------
            # MAIN SIMULATION LOOP
            # -------------------------------------------------

            while steps < self.max_steps:
                steps += 1
                sim_time = steps * (1.0 / 60.0)

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        aborted = True
                        break

                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        now = time.time()
                        active = ((now - last_move_time) < 2.0) or ((now - last_action_time) < 2.0)
                        if active:
                            aborted = True
                        else:
                            steps = self.max_steps
                        raise SystemExit()

                    for switcher in controllers.values():
                        switcher.handle_event(e)

                if aborted:
                    snap = self._make_snapshot(planner_name, agents, terrain, steps)
                    self.experiment["strategies"][idx]["snapshot"] = snap
                    self.experiment["strategies"][idx]["status"] = "aborted"
                    self.experiment["status"] = "aborted"
                    self.experiment["current_index"] = idx
                    self._write_experiment_json()
                    break

                dt = self.clock.tick(60) / 1000.0

                # ------------------------------
                # UPDATE AGENTS
                # ------------------------------
                for i, drone in enumerate(agents):
                    agent_id = self.experiment["agents"][i]["id"]
                    controllers[agent_id].update(dt, steps, sim_time)
                    drone.update(
                        dt,
                        SPEED,
                        ANIM_FPS,
                        rot_frames_with_parcel=self.images.get("drone_rot_with_parcel_frames"),
                        rot_frames=self.images.get("drone_rot_frames"),
                    )

                    px, py = prev_positions[i]
                    dx = drone.pos.x - px
                    dy = drone.pos.y - py
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > 0.001:
                        last_move_time = time.time()

                    metrics.record_distance(dist, GRID_SIZE)
                    prev_positions[i] = (drone.pos.x, drone.pos.y)

                    # -------------------------------------------------
                    # >>> FLIGHT RECORDER: per-agent tick (state)
                    # -------------------------------------------------
                    flight_recorder.tick_agent(
                        agent_id=agent_id,
                        drone=drone,
                        step=steps,
                        sim_time=sim_time,
                    )

                    if hasattr(drone, "_last_action") and drone._last_action:
                        kind, cell, parcel = drone._last_action
                        last_action_time = time.time()

                        # -------------------------------------------------
                        # >>> FLIGHT RECORDER: record action
                        # -------------------------------------------------
                        flight_recorder.record_action(
                            agent_id=agent_id,
                            kind=kind,
                            cell=cell,
                            parcel=parcel,
                            step=steps,
                            sim_time=sim_time,
                            world_time=time.time(),
                        )

                        if kind == "drop" and parcel and not getattr(parcel, "delivered", False):
                            parcel.delivered = True
                            parcel.picked = False
                            try:
                                station = terrain.get_station_at(cell[0], cell[1])
                                if station:
                                    try:
                                        station.register_delivery(cell)
                                    except TypeError:
                                        station.register_delivery()
                            except Exception:
                                pass
                            metrics.record_task_complete(1)

                        if kind in ("pick_failed", "drop_failed"):
                            metrics.record_unsafe_failure(1)

                        self._append_event(
                            idx,
                            {"step": steps, "type": kind, "agent": agent_id, "cell": cell},
                        )

                        drone._last_action = None

                # -------------------------------------------------
                # >>> FLIGHT RECORDER: fleet-level checks
                # -------------------------------------------------
                flight_recorder.tick_fleet(
                    agents=agents,
                    step=steps,
                    sim_time=sim_time,
                )

                # -------------------------------------------------
                # TERMINATION CONDITIONS
                # -------------------------------------------------

                # ---------------------------------------------
                # 1. Detect and log agent losses (NON-TERMINAL)
                # ---------------------------------------------
                dead_agents = [a for a in agents if self.agent_dead(a)]

                for d in dead_agents:
                    # Log each death only once
                    if not getattr(d, "_death_logged", False):
                        flight_recorder.fleet_events.append({
                            "type": "agent_lost",
                            "agent": getattr(d, "agent_id", None),
                            "cell": (int(d.col), int(d.row)),
                            "battery": getattr(d.power, "level", None),
                            "time": {
                                "step": steps,
                                "sim_time": sim_time,
                                "world_time": time.time(),
                            },
                        })
                        d._death_logged = True

                # ---------------------------------------------
                # 2. Determine remaining mission capability
                # ---------------------------------------------
                alive_agents = [a for a in agents if not self.agent_dead(a)]

                # ---------------------------------------------
                # 3. HARD FAILURE: mission impossible
                # ---------------------------------------------
                if not alive_agents and not self.all_parcels_delivered(terrain):
                    # No agents left and parcels remain
                    self.experiment["strategies"][idx]["status"] = "failed"
                    self.experiment["status"] = "failed"
                    self.experiment["current_index"] = idx + 1

                    self._write_experiment_json()
                    break

                # ---------------------------------------------
                # 4. SUCCESS CONDITION (INTENT-BASED)
                # ---------------------------------------------
                if self.all_parcels_delivered(terrain):

                    all_alive_parked = True

                    for i, drone in enumerate(agents):
                        if self.agent_dead(drone):
                            continue  # lost agents do not block completion

                        agent_id = self.experiment["agents"][i]["id"]
                        controller = controllers[agent_id].current

                        if not getattr(controller, "parked", False):
                            all_alive_parked = False
                            break

                    if all_alive_parked:
                        # SUCCESS: parcels delivered AND all alive agents parked
                        self.experiment["strategies"][idx]["status"] = "completed"
                        self.experiment["strategies"][idx]["snapshot"] = None
                        self.experiment["current_index"] = idx + 1

                        self.experiment["status"] = (
                            "completed"
                            if idx + 1 >= len(self.experiment["strategies"])
                            else "in_progress"
                        )

                        self._write_experiment_json()
                        break

                    # else:
                    # Parcels delivered but some agents still parking.
                    # Keep simulation running.

                # ---------------------------------------------
                # Otherwise: continue simulation
                # ---------------------------------------------

                # rendering
                self.screen.fill((192, 192, 192))
                for x in range(0, self.screen_size[0], GRID_SIZE):
                    pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
                for y in range(0, self.screen_size[1], GRID_SIZE):
                    pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

                terrain.draw(self.screen)
                for drone in agents:
                    drone.draw(self.screen, self.images)

                pygame.display.flip()

            if aborted:
                break

            metrics.finalize(steps, agents[0].power.capacity, agents[0].power.level)

            # -------------------------------------------------
            # >>> FLIGHT RECORDER: export + persist
            # -------------------------------------------------
            flight_data = flight_recorder.export()

            self.experiment["results"].append({
                "planner": planner_name,
                "seed": self.experiment["seed"],
                "metrics": metrics.as_dict(),
                "flight_recorder": flight_data,
            })

            self._write_experiment_json()

            if self.experiment["status"] == "completed":
                self._finalize_experiment_file()

        print("[EXPERIMENT] Runner finished. Experiment JSON files are in the experiments folder.")
        # -------------------------------------------------
        # POST-RUN INSPECTION LOOP
        # -------------------------------------------------
        print("[EXPERIMENT] Entering post-run inspection mode. Press ESC to exit.")

        inspecting = True
        while inspecting:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    inspecting = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    inspecting = False

            # redraw final world state
            self.screen.fill((192, 192, 192))
            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            pygame.display.flip()
            self.clock.tick(30)


def main():
    runner = ExperimentRunnerGUI()
    try:
        runner.run()
    except SystemExit:
        pygame.quit()


if __name__ == "__main__":
    main()
