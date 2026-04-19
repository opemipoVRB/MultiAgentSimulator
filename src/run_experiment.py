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
import json
import random
import time
import uuid
from collections import Counter

import pygame
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from src.artifacts import Terrain, Drone, Parcel
from src.config import OLLAMA_MODE
from src.controllers import HumanAgentController, AIAgentController, ControllerSwitcher
from src.observability import FlightRecorder, MetricsAccumulator
from src.physics.physics_2d import Grid2DPhysics
from src.strategies.sppdec.personalities import SPPDEC_PERSONALITIES
from src.system.abort import AbortReason
from src.utils import load_image, scale_to_cell
from src.system.capacity import SystemCapacityChecker, AgentCapacityError
from enum import Enum

# ---------------------------------------------------------------------------
#  Physics Coonfig
# ---------------------------------------------------------------------------

physics = Grid2DPhysics()
NAIVE_NUM_TRIALS = 10  # Monte Carlo greedy rollouts per agent

GRID_SIZE = 100
SPEED = 420
DRONE_SCALE = 1.5
PARCEL_SCALE = 0.70
ANIM_FPS = 12
FLASH_DURATION = 0.15

BASE_DIR = Path(__file__).parent
GRAPHICS_DIR = BASE_DIR / "graphics"
EXPERIMENTS_DIR = BASE_DIR.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_PREFIX = "temp_"
EXPERIMENT_PREFIX = "experiment_"

USE_SWEEP = True

try:
    from src.observability.configs import SCALING_CONFIG, generate_station_scaling
except ImportError:
    SCALING_CONFIG = None
    generate_station_scaling = None


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


class ExperimentState(Enum):
    NEW = "new"
    RESUMABLE = "resumable"
    COMPLETED = "completed"


class ExperimentRunnerGUI:
    SCHEMA_VERSION = 1

    def __init__(self):
        # experiment metadata
        self.seed = int(time.time()) % (2 ** 31)
        # GUI-controlled defaults
        self.parcels = 15
        self.planners = ["local"]
        self.run_id = str(uuid.uuid4())[:8]
        self.experiment: Dict = {}
        self.experiment_dir: Optional[Path] = None
        self.experiment_json_path: Optional[Path] = None
        self._sweep_queue = None
        self._sweep_initialized = False
        self._strategy_initialized = False

        # communication config (placeholder)
        self.default_comm_config = {"reliability": 1.0, "delay_mean": 0.0, "delay_std": 0.0}

        # behavior controls
        self.max_steps = 60 * 60 * 5
        self.stall_seconds = 6.0  # if idle longer than this considered a stall when parcels remain

        self.checker = SystemCapacityChecker()
        self.system_limit = self.checker.max_supported_agents()
        self.fast_mode = True  # New: fast execution with status display

        # Normal pygame initialization for GUI mode
        pygame.init()
        pygame.key.set_repeat(250, 35)
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_size = pygame.display.get_surface().get_size()
        pygame.display.set_caption("Drone Agent - Experiment Runner")
        self.clock = pygame.time.Clock()

        # fonts to match games.py
        self.font = pygame.font.SysFont("Consolas", 20)
        self.narration_font = pygame.font.SysFont("Consolas", 22)
        self.title_font = pygame.font.SysFont("Consolas", 72)
        self.large_font = pygame.font.SysFont("Consolas", 48)
        self.images = self._load_graphics()

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

    def _show_error_and_restart_setup(self, message: Optional[str] = None):
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

    def scan_sweep_config_state(self, sweep_name: str, agents: int, parcels: int):
        """
        Returns a dict describing the true state of a sweep configuration.
        """
        scale_dir = (
                EXPERIMENTS_DIR
                / sweep_name
                / f"A{agents}_P{parcels}"
        )

        if not scale_dir.exists():
            return {
                "exists": False,
                "completed": False,
                "pending_strategies": set(),
                "terminal_strategies": set(),
            }

        pending = set()
        terminal = set()
        experiment_completed = False  # Add this flag

        for exp_dir in scale_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
                continue

            exp_json = exp_dir / "experiment.json"
            if not exp_json.exists():
                continue

            try:
                data = json.loads(exp_json.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Check experiment-level completion first
            exp_status = data.get("status", "in_progress")
            current_index = data.get("current_index", 0)
            total_strategies = len(data.get("strategies", []))

            # Experiment is truly completed if:
            # 1. Status is "completed", OR
            # 2. current_index >= total_strategies (all strategies processed)
            if exp_status == "completed": # or current_index >= total_strategies:
                experiment_completed = True

            for s in data.get("strategies", []):
                name = s.get("family") or s.get("name")
                status = s.get("status")

                if status in {"completed", "aborted", "failed", "infeasible"}:
                    terminal.add(name)
                else:
                    pending.add(name)

        # A strategy is truly pending only if it never appears terminal anywhere
        truly_pending = pending - terminal

        return {
            "exists": True,
            "completed": experiment_completed,  # Use experiment-level completion
            "pending_strategies": truly_pending,
            "terminal_strategies": terminal,
        }

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
        max_agents = self.system_limit
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
                    elif e.type == pygame.KEYDOWN and e.unicode.isdigit():
                        typing += e.unicode
                        v = int(typing)

                        if active_field == "agents":
                            agents = max(min_agents, min(max_agents, v))
                        else:
                            parcels = max(min_parcels, min(max_parcels, v))

            # -------- render --------
            self.screen.fill((40, 44, 52))

            header = self.large_font.render("Experiment Setup", True, (230, 230, 230))
            self.screen.blit(header, header.get_rect(center=(self.screen_size[0] // 2, 140)))
            limit_txt = self.font.render(
                f"System agent limit: {max_agents}",
                True,
                (160, 160, 160),
            )
            self.screen.blit(
                limit_txt,
                limit_txt.get_rect(center=(self.screen_size[0] // 2, 190)),
            )
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

    def _find_recent_experiment(self, root: Path) -> Optional[Path]:
        candidates = []

        if not root.exists():
            return None

        for exp_json in root.glob("experiment_*/experiment.json"):
            try:
                data = json.loads(exp_json.read_text(encoding="utf-8"))
            except Exception:
                continue

            strategies = data.get("strategies", [])

            # resumable if ANY strategy is runnable
            if strategies and any(
                    s.get("status") in {"pending", "aborted", "running"}
                    for s in strategies
            ):
                candidates.append((exp_json, data))

        if not candidates:
            return None

        def resume_priority(item):
            _, data = item
            idx = int(data.get("current_index", 0))
            return idx

        candidates.sort(key=resume_priority, reverse=True)
        return candidates[0][0]

    def _show_resume_modal(self, exp_path: Path) -> bool:
        """
        Ask the user whether to resume an unfinished experiment.
        Returns True to resume, False to discard.
        """
        name = exp_path.parent.name

        self._show_modal(
            title="Resume Experiment?",
            message=(
                "An unfinished experiment was found:\n\n"
                f"{name}\n\n"
                "Press ENTER to resume\n"
                "Press ESC to start a new experiment"
            ),
            background=(28, 32, 40),
            title_color=(120, 200, 255),
            hint="ENTER = Resume | ESC = New",
            exit_keys={pygame.K_RETURN, pygame.K_ESCAPE},
        )

        # capture last key pressed
        keys = pygame.key.get_pressed()
        return keys[pygame.K_RETURN]

    def build_sweep_status(self, sweep_name: str, sweep_configs: list[dict]):
        rows = []
        for cfg in sweep_configs:
            agents = cfg["agents"]
            parcels = cfg["parcels"]

            scan = self.scan_sweep_config_state(
                sweep_name=sweep_name,
                agents=agents,
                parcels=parcels,
            )

            completed = scan["completed"]
            pending = len(scan["pending_strategies"])

            if completed:
                state = "completed"
            elif pending > 0:
                state = "partial"
            else:
                state = "new"

            rows.append({
                "agents": agents,
                "parcels": parcels,
                "state": state,
                "pending": pending,
                "terminal": len(scan["terminal_strategies"]),
            })

        return rows

    def _load_or_create_experiment(self, root: Path) -> ExperimentState:
        """
        Load or resume an experiment under a specific root directory.

        Rules:
        - If an experiment exists and has runnable strategies, let user choose which to run
        - Completed strategies are blocked at UI level
        - If all strategies are terminal, treat experiment as completed
        - Never create a second experiment under the same root
        """

        self._resumed = False
        self._experiment_completed = False

        TERMINAL = {"completed", "failed", "infeasible"}

        # -------------------------------------------------
        # 1) Find existing experiment under this root
        # -------------------------------------------------
        exp_jsons = sorted(
            root.glob("experiment_*/experiment.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not exp_jsons:
            return ExperimentState.NEW

        candidate = exp_jsons[0]

        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            strategies = data.get("strategies", [])

            if not strategies:
                print("[EXPERIMENT] No strategies found")
                self._experiment_completed = True
                return ExperimentState.COMPLETED

            # -------------------------------------------------
            # Normalize experiment status
            # -------------------------------------------------
            exp_status = data.get("status", "in_progress")

            if exp_status == "completed":
                if any(s.get("status") in {"pending", "aborted"} for s in strategies):
                    data["status"] = "in_progress"
                    exp_status = "in_progress"

            # Clear snapshots for terminal strategies only
            for s in strategies:
                if s.get("status") in TERMINAL:
                    s["snapshot"] = None

            # -------------------------------------------------
            # Check if anything is runnable at all
            # -------------------------------------------------
            runnable_exists = any(
                s.get("status") in {"pending", "aborted"}
                for s in strategies
            )

            if not runnable_exists:
                data["status"] = "completed"
                self._experiment_completed = True
                return ExperimentState.COMPLETED

            # -------------------------------------------------
            # Extract sweep context safely
            # -------------------------------------------------
            try:
                sweep_name = root.parents[1].name
            except IndexError:
                sweep_name = "manual"

            scale = root.name  # A{agents}_P{parcels}
            try:
                parts = scale.split("_")
                agents = int(parts[0][1:])
                parcels = int(parts[1][1:])
            except Exception:
                agents = parcels = -1

            # -------------------------------------------------
            # STRATEGY SELECTION MODAL (AUTHORITATIVE)
            # -------------------------------------------------
            if getattr(self, '_strategy_initialized', False):
                # Skip modal - use current_index from experiment (works for both sweep and manual)
                selected_index = data.get("current_index", 0)
                print(f"[AUTO] Auto-resuming at strategy index {selected_index}")
            elif USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False):
                # In sweep mode, automatically skip completed strategies and start from next pending
                pending_strategies = [i for i, s in enumerate(strategies) if s.get("status") in {"pending", "aborted"}]
                if pending_strategies:
                    selected_index = pending_strategies[0]  # Start from first pending strategy
                    print(f"[SWEEP] Auto-selecting first pending strategy at index {selected_index}")
                else:
                    # All strategies completed, mark experiment as completed
                    print(f"[SWEEP] All strategies completed, marking experiment as done")
                    data["status"] = "completed"
                    self._experiment_completed = True
                    return ExperimentState.COMPLETED
            else:
                # Show selection modal for manual mode only
                selected_index = self._show_experiment_status_modal(
                    sweep_name=sweep_name,
                    agents=agents,
                    parcels=parcels,
                    run_id=data.get("run_id", "unknown"),
                    current_index=data.get("current_index", 0),
                    strategies=strategies,
                )

                if selected_index is None:
                    print("[EXPERIMENT] Experiment discarded by user")
                    self._experiment_completed = True
                    return ExperimentState.COMPLETED

                # Mark strategy selection as initialized after first choice
                self._strategy_initialized = True
            # -------------------------------------------------
            # RESUME PATH (STRATEGY-LEVEL)
            # -------------------------------------------------
            self.experiment = data

            self.experiment.setdefault("run_id", str(uuid.uuid4())[:8])
            self.experiment.setdefault("schema_version", self.SCHEMA_VERSION)
            self.experiment.setdefault("comm_config", self.default_comm_config.copy())
            self.experiment.setdefault(
                "created_at",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            )
            self.experiment.setdefault("results", [])

            # User-selected strategy index is authoritative
            self.experiment["current_index"] = int(selected_index)

            # Seed consistency
            if "seed" in self.experiment:
                self.seed = self.experiment["seed"]
            else:
                self.experiment["seed"] = self.seed

            self.experiment["status"] = "in_progress"
            self.experiment_json_path = candidate
            self.experiment_dir = candidate.parent
            self.parcels = len(self.experiment.get("parcels", []))
            self.run_id = self.experiment["run_id"]

            self._resumed = True

            print(f"[EXPERIMENT] Resuming experiment {candidate}")
            print(f"[EXPERIMENT] Run ID: {self.run_id}")
            print(f"[EXPERIMENT] Starting at strategy index {selected_index}")

            return ExperimentState.RESUMABLE

        except Exception as e:
            print(f"[EXPERIMENT] Failed to load experiment {candidate}: {e}")
            self._experiment_completed = True
            return ExperimentState.COMPLETED

    def _write_flight_artifacts(self, run_id, planner, strategy_name, flight_data):
        """
        Persist flight recorder data into the current experiment directory.
        """

        if self.experiment_dir is None:
            raise RuntimeError("Experiment directory not initialized")

        # Create a subfolder per strategy
        strategy_dir = self.experiment_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        agents_path = strategy_dir / "agents.json"
        agents_path.write_text(json.dumps(flight_data["agents"], indent=2), encoding="utf-8")
        paths["agents"] = f"{strategy_name}/agents.json"

        fleet_path = strategy_dir / "fleet.json"
        fleet_path.write_text(
            json.dumps(
                {
                    "parcels": flight_data["parcels"],
                    "fleet_events": flight_data["fleet_events"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["fleet"] = f"{strategy_name}/fleet.json"

        heatmap_path = strategy_dir / "heatmap.json"
        heatmap_path.write_text(
            json.dumps(flight_data["heatmap"], indent=2),
            encoding="utf-8",
        )
        paths["heatmap"] = f"{strategy_name}/heatmap.json"

        meta_path = strategy_dir / "meta.json"
        meta_path.write_text(
            json.dumps(flight_data["meta"], indent=2),
            encoding="utf-8",
        )
        paths["meta"] = f"{strategy_name}/meta.json"

        return paths

    def _write_experiment_json(self):
        if not self.experiment_json_path:
            return

        # Ensure directory exists
        self.experiment_json_path.parent.mkdir(parents=True, exist_ok=True)

        self.experiment_json_path.write_text(
            json.dumps(self.experiment, indent=2),
            encoding="utf-8",
        )

    def _finalize_experiment_file(self):
        self.experiment["status"] = "completed"

        # Clear all snapshots when experiment is fully completed
        for strategy in self.experiment.get("strategies", []):
            strategy["snapshot"] = None

        self._write_experiment_json()

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

    def _make_terrain_snapshot(self, planner_name: str, agents: List[Drone], terrain: Terrain, steps: int) -> Dict:
        """
        Create a compact snapshot capturing parcels, agents' states, and steps elapsed.
        agents is a list of Drone instances in the same order as self.experiment['agents'].
        The snapshot contains:
          - agents_state: list of per-agent dicts
          - parcels: list of parcel dicts (id, col, row, weight, picked, delivered)
          - steps, planner, time
        """

        if steps <= 0:
            raise ValueError("Refusing to create snapshot with zero steps")

        # map existing experiment parce_make_snapshotls (by position) to their ids
        id_map = {}
        for pdef in self.experiment.get("parcels", []):
            key = (int(pdef["col"]), int(pdef["row"]))
            id_map[key] = pdef.get("id")

        parcels_snap = []
        for p in terrain.parcels:
            pid = id_map.get((int(p.col), int(p.row)))
            parcels_snap.append({
                "id": getattr(p, "id", None),
                "col": int(p.col),
                "row": int(p.row),
                "weight": getattr(p, "weight", 1.0),
                "picked": bool(getattr(p, "picked", False)),
                "delivered": bool(getattr(p, "delivered", False)),
            })

        agents_state = []
        # In _make_terrain_snapshot method, update the agents_state loop:
        for idx, agent in enumerate(agents):
            carried_id = None
            if getattr(agent, "carrying", None):
                # Get the ID directly from the carried parcel
                carried_id = getattr(agent.carrying, "id", None)

            agents_state.append({
                "agent_id": self.experiment["agents"][idx]["id"],
                "type": "drone",
                "col": int(agent.col),
                "row": int(agent.row),
                "pos_x": float(agent.pos.x),
                "pos_y": float(agent.pos.y),
                "battery_level": float(agent.power.level),
                "battery_capacity": float(agent.power.capacity),
                "carrying_id": carried_id,  # Now properly tracks parcel ID
            })

        return {
            "planner": planner_name,
            "steps": int(steps),
            "time": time.time(),
            "agents_state": agents_state,
            "parcels": parcels_snap,
        }

    def _restore_snapshot(self, snapshot: Dict, strategy_name: str, flight_recorder: FlightRecorder) -> Tuple[
        Terrain, List[Drone], int, List[Parcel]]:
        """
        Restore world state from a snapshot and return (terrain, agents_list, steps, parcels).
        Snapshot is authoritative.
        """

        # ----------------------------
        # Validate snapshot
        # ----------------------------
        agents_state = snapshot.get("agents_state")
        if not agents_state:
            raise ValueError("Snapshot missing agents_state")

        # ----------------------------
        # Lock experiment agents to snapshot
        # ----------------------------
        self.experiment["agents"] = [
            {
                "id": a["agent_id"],
                "type": "drone",
            }
            for a in agents_state
        ]

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
        terrain.add_station(center_col, center_row, w=5, h=5)

        # ----------------------------
        # Recreate parcels
        # ----------------------------
        recreated_parcels: List[Parcel] = []

        for ps in snapshot.get("parcels", []):
            snapshot_parcel_id = ps.get("id")

            p = Parcel(
                int(ps["col"]),
                int(ps["row"]),
                terrain.grid_size
            )

            if snapshot_parcel_id is not None:
                p.id = str(snapshot_parcel_id)

            p.picked = bool(ps.get("picked", False))
            p.delivered = bool(ps.get("delivered", False))

            if "weight" in ps:
                p.weight = float(ps["weight"])

            recreated_parcels.append(p)

            print(f"[RESTORE] Parcel at ({int(ps['col'])},{int(ps['row'])}) - "
                  f"ID: {p.id}, picked: {p.picked}, delivered: {p.delivered}")

        terrain.parcels = recreated_parcels

        # ----------------------------
        # Recreate agents (with onboard SLM)
        # ----------------------------
        agents: List[Drone] = []
        base_port = 11434

        for idx, astate in enumerate(agents_state):
            agent_id = astate["agent_id"]

            drone = Drone(
                start_cell=(int(astate["col"]), int(astate["row"])),
                grid_size=GRID_SIZE,
                screen_size=self.screen_size,
                terrain=terrain,
                agent_id=agent_id,
                flight_recorder=flight_recorder
            )

            # precise position
            drone.pos = pygame.Vector2(
                float(astate["pos_x"]),
                float(astate["pos_y"]),
            )
            drone.col = int(astate["col"])
            drone.row = int(astate["row"])
            drone.last_cell = (drone.col, drone.row)

            # battery restore
            cap = float(astate.get("battery_capacity", drone.power.capacity))
            lvl = float(astate.get("battery_level", drone.power.level))
            drone.power.capacity = cap
            drone.power.level = min(lvl, cap)

            # carried parcel restore
            carrying_id = astate.get("carrying_id")
            if carrying_id is not None:
                carrying_id_str = str(carrying_id)

                for p in recreated_parcels:
                    if getattr(p, "id", None) == carrying_id_str:
                        drone.carrying = p
                        print(f"[RESTORE] Agent {agent_id} is carrying parcel {carrying_id_str}")
                        break
                else:
                    print(
                        f"[RESTORE] WARNING: Agent {agent_id} should be carrying parcel {carrying_id_str} but it wasn't found"
                    )

            # ----------------------------
            # Restore embodied cognition
            # ----------------------------
            if OLLAMA_MODE == "external":
                drone.slm_base_url = "http://127.0.0.1:11434"
                drone.slm_enabled = True
                print(f"[RESTORE][SLM] Using external Ollama for {agent_id}")
            else:
                slm_port = base_port + idx + 1
                print(f"[RESTORE][SLM] Restarting {agent_id} on port {slm_port}")
                drone.start_slm(port=slm_port)

            terrain.register_agent_cell(drone, drone.col, drone.row)
            terrain.agents[drone.agent_id] = drone
            agents.append(drone)

            print(f"[RESTORE] Agent {agent_id} at ({drone.col},{drone.row}) - "
                  f"battery: {drone.power.level:.1f}%")

        steps = int(snapshot.get("steps", 0))

        print(f"[RESTORE] Snapshot restored: {len(agents)} agents, {len(recreated_parcels)} parcels, step {steps}")

        return terrain, agents, steps, recreated_parcels

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

    def _setup_agents(self, terrain: Terrain, *, strategy_name: str, strategy_spec: dict):
        agents = {}
        self._current_strategy = strategy_spec

        station = terrain.nearest_station(
            self.screen_size[0] // GRID_SIZE // 2,
            self.screen_size[1] // GRID_SIZE // 2,
        )

        if not station:
            raise AgentSpawnError("No station found. Cannot spawn agents.")

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

        # ----------------------------------------
        # Build personality assignment pool
        # ----------------------------------------
        expanded_personalities = None

        if "personality_distribution" in strategy_spec:
            expanded_personalities = []
            for p, count in strategy_spec["personality_distribution"].items():
                expanded_personalities.extend([p] * count)

        elif "agent_personalities" in strategy_spec:
            expanded_personalities = list(strategy_spec["agent_personalities"])

        # ----------------------------------------
        # SLM port allocation (infrastructure-level)
        # ----------------------------------------
        base_port = 11434
        used_ports = set()

        for idx, (agent_def, start_cell) in enumerate(
                zip(self.experiment["agents"], station_cells)
        ):
            drone = Drone(
                start_cell=start_cell,
                grid_size=GRID_SIZE,
                screen_size=self.screen_size,
                terrain=terrain,
                agent_id=agent_def["id"],
            )

            # ----------------------------------------
            # Personality assignment
            # ----------------------------------------
            if expanded_personalities:
                drone.personality = expanded_personalities[idx % len(expanded_personalities)]
            else:
                drone.personality = agent_def.get("personality", "rational")

            print(f"[PERSONALITY] {drone.agent_id} -> {drone.personality}")

            # ----------------------------------------
            # Assign SLM ports for SPPDEC
            # ----------------------------------------
            if strategy_name.startswith("sppdec"):
                if OLLAMA_MODE == "external":
                    drone.slm_base_url = "http://127.0.0.1:11434"
                    drone.slm_enabled = True
                    print(f"[SLM] Using external Ollama for {drone.agent_id}")
                else:
                    slm_port = base_port + idx + 1
                    used_ports.add(slm_port)
                    drone.start_slm(port=slm_port)
                    print(f"[SLM] Booting {drone.agent_id} on port {slm_port}")

            agents[agent_def["id"]] = drone
            terrain.register_agent_cell(drone, start_cell[0], start_cell[1])
            terrain.agents[drone.agent_id] = drone

        if used_ports:
            print(f"[SLM] Active ports: {sorted(used_ports)}")

        self._active_drones = list(agents.values())
        return agents, 0, None

    def _setup_scene(self, maybe_snapshot: Optional[Dict], strategy_name: str, strategy_spec: dict,
                     flight_recorder: FlightRecorder) -> \
            Tuple[
                Terrain, List[Drone], int, List[Parcel]]:
        """
        Create Terrain and agents.
        If maybe_snapshot provided, restore world state from it.
        Returns (terrain, agents_list, steps_already_taken, all_parcels)
        """

        # ---- Snapshot path (AUTHORITATIVE) ----
        if maybe_snapshot is not None:
            terrain, agents, steps, all_parcels = self._restore_snapshot(snapshot=maybe_snapshot,
                                                                         strategy_name=strategy_name,
                                                                         flight_recorder=flight_recorder
                                                                         )  # Changed to 4 values
            self._active_drones = agents
            return terrain, agents, steps, all_parcels

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
        s = self.experiment["station"]
        terrain.add_station(s["col"], s["row"], w=s["w"], h=s["h"])

        # 2. Parcels (schema-driven)
        for pdef in self.experiment.get("parcels", []):
            terrain.add_parcel(int(pdef["col"]), int(pdef["row"]))
            p = terrain.parcel_at_cell(
                int(pdef["col"]),
                int(pdef["row"]),
                include_delivered=True,
            )
            if p:
                p.picked = bool(pdef.get("picked", False))
                p.delivered = bool(pdef.get("delivered", False))

        # 3. Agents

        agents_dict, steps, error = self._setup_agents(
            terrain,
            strategy_name=strategy_name,
            strategy_spec=strategy_spec
        )

        if error:
            self._show_error_and_restart_setup(error)
            raise RuntimeError("Restart setup")

        # Collect all parcels (both newly created and existing)
        all_parcels = list(terrain.parcels)

        return terrain, list(agents_dict.values()), steps, all_parcels

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

    def agent_active(self, drone, terrain) -> bool:
        # dead drones never count
        if self.agent_dead(drone):
            return False

        # must have controller
        controller = getattr(drone, "controller", None)
        if controller is None:
            return False

        # if controller explicitly parked, not active
        if getattr(controller, "parked", False):
            return False

        # if at base and no pending work, not active
        if self.agent_at_base(drone, terrain) and not getattr(controller, "has_pending_task", False):
            return False

        return True

    def _debug_mission_state(self, terrain, agents, step):
        print(f"\n[DEBUG][Step {step}] Mission State Check:")
        print("Parcels:")
        for p in terrain.parcels:
            pid = getattr(p, "id", "unknown")
            picked = getattr(p, "picked", False)
            delivered = getattr(p, "delivered", False)
            col = getattr(p, "col", "?")
            row = getattr(p, "row", "?")
            print(f"  Parcel {pid} at ({col},{row}): picked={picked}, delivered={delivered}")

        print("\nAgents:")
        for a in agents:
            aid = getattr(a, "agent_id", "unknown")
            carrying = getattr(a, "carrying", None)
            if carrying:
                pid = getattr(carrying, "id", "unknown")
                print(f"  Agent {aid}: CARRYING parcel {pid}")
            else:
                print(f"  Agent {aid}: Not carrying")
        # Check mission_exhausted conditions
        undelivered = any(not getattr(p, "delivered", False) for p in terrain.parcels)
        carrying = any(getattr(a, "carrying", None) is not None for a in agents)
        print(f"\nConditions: undelivered={undelivered}, carrying={carrying}")
        print(f"mission_exhausted result: {self.mission_exhausted(terrain, agents)}")

    def all_alive_agents_parked(self, terrain, agents) -> bool:
        """Check if all alive agents are at the station."""
        for agent in agents:
            if not self.agent_dead(agent):
                if not self.agent_at_base(agent, terrain):
                    return False
        return True

    def mission_exhausted(self, terrain, agents) -> bool:
        # 1. Check all parcels - if any are not delivered, mission is not exhausted
        for p in terrain.parcels:
            if not getattr(p, "delivered", False):
                return False

        # 2. No agent is carrying a parcel
        for a in agents:
            if getattr(a, "carrying", None) is not None:
                return False

        # 3. No agent has a pending task
        for agent in agents:
            controller = getattr(agent, "controller", None)
            if controller and getattr(controller, "has_pending_task", False):
                return False

        # 4. ALL ALIVE AGENTS MUST BE AT THE STATION (PARKED)
        for agent in agents:
            if not self.agent_dead(agent):
                # Check if agent is at the station
                if not self.agent_at_base(agent, terrain):
                    return False
                # Also check if the controller recognizes it's parked
                controller = getattr(agent, "controller", None)
                if controller and hasattr(controller, "parked"):
                    if not controller.parked:
                        return False

        return True

    def _finalize_strategy(
            self,
            *,
            idx: int,
            status: str,
            planner_name: str,
            strategy_name: str,
            strategy_type: str,
            steps: int,
            flight_recorder: FlightRecorder,
    ):
        assert status in {"completed", "aborted", "failed", "infeasible"}, f"Invalid status: {status}"

        # Resolve authoritative strategy spec from experiment
        strategy = self.experiment["strategies"][idx]

        # Pull metrics from composition instead of param
        metrics = flight_recorder.metrics
        if metrics is None:
            raise RuntimeError("FlightRecorder.metrics is None. MetricsAccumulator was never attached.")

        print(
            f"[FINALIZE] Strategy {idx} '{strategy_name}' | "
            f"mode={strategy_type} | planner={planner_name} | "
            f"status={status} | steps={steps}"
        )

        # -------------------------------------------------
        # Export flight artifacts first
        # -------------------------------------------------
        flight_data = flight_recorder.export()

        # -----------------------------------
        # Capture personality composition
        # -----------------------------------
        active_drones = getattr(self, "_active_drones", [])

        personality_map = {
            d.agent_id: getattr(d, "personality", "unknown")
            for d in active_drones
        }
        personality_distribution = dict(Counter(personality_map.values()))

        artifact_paths = self._write_flight_artifacts(
            run_id=self.experiment["run_id"],
            planner=planner_name,
            strategy_name=strategy_name,
            flight_data=flight_data,
        )

        print(
            f"[FINALIZE] Artifacts written | "
            f"keys={list(artifact_paths.keys())}"
        )

        # -------------------------------------------------
        # Derive authoritative metrics from FlightRecorder
        # -------------------------------------------------
        agents = flight_data.get("agents", [])
        fleet_events = flight_data.get("fleet_events", [])

        total_distance = sum(a.get("distance_cells", 0.0) for a in agents)

        total_energy_consumed = 0.0
        for a in agents:
            energy_series = a.get("energy", [])
            if len(energy_series) >= 2:
                start = energy_series[0]["battery_pct"]
                end = energy_series[-1]["battery_pct"]
                total_energy_consumed += max(0.0, start - end)

        tasks_completed = sum(len(a.get("tasks", [])) for a in agents)
        unsafe_failures = len([e for e in fleet_events if e.get("type") == "collision"])

        runtime_seconds = round(time.time() - metrics.start_time, 3)

        derived_metrics = {
            "run_id": self.experiment["run_id"],
            "planner": planner_name,
            "seed": self.experiment["seed"],
            "n_drones": len(agents),
            "n_parcels": len(self.experiment.get("parcels", [])),
            "comm_reliability": self.experiment.get("comm_config", {}).get("reliability", 1.0),
            "battery_capacity": metrics.fields.get("battery_capacity", 100.0),
            "episodic_interval": 1.0,
            "total_energy_consumed": round(total_energy_consumed, 3),
            "total_distance_traveled": round(total_distance, 3),
            "tasks_completed": tasks_completed,
            "mission_makespan_steps": steps,
            "unsafe_failures": unsafe_failures,
            "messages_sent": metrics.messages_sent,
            "runtime_seconds": runtime_seconds,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(
            f"[FINALIZE] Derived metrics | "
            f"distance={derived_metrics['total_distance_traveled']:.2f} | "
            f"energy={derived_metrics['total_energy_consumed']:.3f} | "
            f"tasks={tasks_completed}"
        )
        # -------------------------------
        # Termination diagnostics
        # -------------------------------
        final_agents = flight_data.get("agents", [])

        dead_total = sum(1 for a in final_agents if a.get("energy", []) and a["energy"][-1]["battery_pct"] <= 0)
        dead_at_station = sum(
            1 for a in final_agents
            if a.get("energy", []) and a["energy"][-1]["battery_pct"] <= 0 and a.get("at_station", False)
        )
        lost_in_field = dead_total - dead_at_station

        termination_summary = {
            "dead_total": dead_total,
            "dead_at_station": dead_at_station,
            "lost_in_field": lost_in_field,
        }

        # -------------------------------------------------
        # Append result (rich identity preserved)
        # -------------------------------------------------
        self.experiment["results"].append({
            "strategy_index": idx,

            # Identity
            "strategy_name": strategy_name,  # e.g. sppdec_combined_instance_3
            "family": strategy.get("family"),  # e.g. sppdec_combined
            "base_name": strategy.get("name"),  # original base name
            "variant": strategy.get("variant"),  # combined, control, etc
            "instance_id": strategy.get("instance_id"),  # 0..N
            "description": strategy.get("description"),
            "personality_map": personality_map,
            "personality_distribution": personality_distribution,
            # Metadata
            "strategy_type": strategy_type,
            "planner": planner_name,
            "seed": self.experiment["seed"],
            "status": status,
            "steps_executed": steps,
            "termination": termination_summary,
            # Outputs
            "metrics": derived_metrics,
            "artifacts": artifact_paths,
        })

        print(
            f"[FINALIZE] Result appended | "
            f"results_count={len(self.experiment['results'])}"
        )

        # -------------------------------------------------
        # Update strategy bookkeeping
        # -------------------------------------------------
        self.experiment["strategies"][idx]["status"] = status
        self.experiment["strategies"][idx]["snapshot"] = None
        self.experiment["strategies"][idx]["personality_map"] = personality_map

        print("Finalized Strategy:", strategy)
        print(
            f"[FINALIZE] Strategy state updated | "
            f"strategy_status={status}"
        )

        # -------------------------------------------------
        # Advance experiment progress
        # -------------------------------------------------
        self.experiment["current_index"] = idx + 1

        if self.experiment["current_index"] >= len(self.experiment["strategies"]):
            self.experiment["status"] = "completed"
            self._experiment_completed = True
            print("[EXPERIMENT] All strategies completed")
        else:
            self.experiment["status"] = "in_progress"
            print(
                f"[EXPERIMENT] Moving to next strategy: "
                f"{self.experiment['current_index']}"
            )

        # -------------------------------------------------
        # Shutdown SLMs
        # -------------------------------------------------
        active_drones = getattr(self, "_active_drones", [])
        slm_users = [d for d in active_drones if getattr(d, "slm_enabled", False)]

        if slm_users:
            print("[SLM] Shutting down all drone SLM instances")

        for drone in slm_users:
            try:
                drone.stop_slm()
                print(f"[SLM] Stopped {drone.agent_id}")
            except Exception as e:
                print(f"[SLM] Failed to stop {drone.agent_id}: {e}")

        # -------------------------------------------------
        # Persist experiment state
        # -------------------------------------------------
        print("[FINALIZE] Persisting experiment state to disk")
        self._write_experiment_json()

    def _show_experiment_complete_screen(self, terrain, agents, config_name=None, duration: float = 3.0):
        """
        Final blocking screen shown once ALL strategies are completed.
        Shows current config name and next config info in sweep mode.
        """
        if self.fast_mode:
            print(f"[FAST] Config complete: {config_name or 'experiment'}")
            return

        start_time = time.time()

        # Determine if we're in sweep mode and get next config info
        in_sweep_mode = USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False)
        next_config_info = None

        if in_sweep_mode and self._sweep_queue:
            next_config = self._sweep_queue[0]  # Peek at next config without removing it
            next_agents = next_config.get("agents", "?")
            next_parcels = next_config.get("parcels", "?")
            next_config_info = f"A{next_agents}_P{next_parcels}"

        while time.time() - start_time < duration:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return

            self.screen.fill((192, 192, 192))

            # grid
            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            # Main completion message with config name
            if config_name:
                msg = self.large_font.render(f"EXPERIMENT {config_name} COMPLETE", True, (0, 180, 0))
            else:
                msg = self.large_font.render("EXPERIMENT COMPLETE", True, (0, 180, 0))
            self.screen.blit(msg, msg.get_rect(center=(self.screen_size[0] // 2, 120)))

            # Next action message with specific config
            if next_config_info:
                next_msg = self.font.render(
                    f"Starting next: {next_config_info} ({len(self._sweep_queue)} remaining)",
                    True,
                    (100, 100, 100)
                )
                self.screen.blit(next_msg, next_msg.get_rect(center=(self.screen_size[0] // 2, 170)))
            elif in_sweep_mode:
                final_msg = self.font.render("All sweep experiments complete", True, (60, 60, 60))
                self.screen.blit(final_msg, final_msg.get_rect(center=(self.screen_size[0] // 2, 170)))

            pygame.display.flip()
            self.clock.tick(30)

    def _show_post_run_inspection(
            self,
            *,
            terrain,
            agents,
            strategy_name: str,
            status: str,
            variant: Optional[str] = None,
            config: Optional[dict] = None,
            seconds: int = 5,
    ):
        """
        Show final strategy state briefly, then auto-advance to next strategy.
        """

        if self.fast_mode:
            print(f"[FAST] Finished: {strategy_name} → {status.upper()}")
            return

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            remaining = seconds - int(elapsed)

            if remaining <= 0:
                return  # auto-continue to next strategy

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return  # allow manual skip

            self.screen.fill((192, 192, 192))

            # grid
            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            title_text = strategy_name
            if variant:
                title_text = f"{strategy_name} ({variant})"

            title = self.large_font.render(
                f"Strategy Finished: {title_text}",
                True,
                (50, 50, 50),
            )

            status_text = self.font.render(
                f"Status: {status.upper()}",
                True,
                (80, 80, 80),
            )

            countdown_text = self.font.render(
                f"Next strategy in {remaining}s",
                True,
                (100, 100, 100),
            )

            self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, 80)))
            self.screen.blit(status_text, status_text.get_rect(center=(self.screen_size[0] // 2, 130)))
            self.screen.blit(countdown_text, countdown_text.get_rect(center=(self.screen_size[0] // 2, 160)))

            if config:
                mid_font = pygame.font.SysFont("Consolas", 32)
                y = 230
                for k, v in config.items():
                    txt = mid_font.render(f"{k}: {v}", True, (60, 60, 60))
                    self.screen.blit(txt, txt.get_rect(center=(self.screen_size[0] // 2, y)))
                    y += 50

            pygame.display.flip()
            self.clock.tick(30)

    def _show_strategy_countdown(
            self,
            *,
            terrain,
            agents,
            strategy_name: str,
            topology: str,
            seconds: int = 5,
            variant: Optional[str] = None,
            config: Optional[dict] = None
    ) -> bool:
        """
        Overlay countdown shown before a strategy starts.

        This screen is rendered directly on top of the active terrain
        to preserve spatial and experimental continuity.
        """

        if self.fast_mode:
            print(f"[FAST] Starting: {strategy_name}")
            return True

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            remaining = seconds - int(elapsed)

            if remaining <= 0:
                return True

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                    return False

            # -------------------------------------------------
            # Draw world
            # -------------------------------------------------
            self.screen.fill((192, 192, 192))

            for x in range(0, self.screen_size[0], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
            for y in range(0, self.screen_size[1], GRID_SIZE):
                pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

            terrain.draw(self.screen)
            for drone in agents:
                drone.draw(self.screen, self.images)

            # -------------------------------------------------
            # Overlay info
            # -------------------------------------------------
            overlay_y = 110

            title = self.large_font.render("NEXT STRATEGY", True, (80, 120, 200))
            self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, overlay_y)))

            overlay_y += 70

            name_text = strategy_name
            if variant:
                name_text = f"{strategy_name} ({variant})"

            name = self.title_font.render(name_text.upper(), True, (20, 20, 20))
            self.screen.blit(name, name.get_rect(center=(self.screen_size[0] // 2, overlay_y)))

            overlay_y += 50

            meta = [f"Topology: {topology}"]

            if config:
                for k, v in config.items():
                    meta.append(f"{k}: {v}")

            for line in meta:
                txt = self.font.render(line, True, (40, 40, 40))
                self.screen.blit(txt, txt.get_rect(center=(self.screen_size[0] // 2, overlay_y)))
                overlay_y += 34

            overlay_y += 30

            countdown = self.large_font.render(str(remaining), True, (200, 100, 40))
            self.screen.blit(countdown, countdown.get_rect(center=(self.screen_size[0] // 2, overlay_y)))

            hint = self.font.render("Press ESC to skip this strategy", True, (90, 90, 90))
            self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 100)))

            pygame.display.flip()
            self.clock.tick(30)

    def scan_completed_experiments(self, sweep_name: str) -> Set[Tuple[int, int, int]]:
        """Scan filesystem for sweep configs that should be skipped."""
        completed = set()

        if not EXPERIMENTS_DIR.exists():
            return completed

        sweep_dir = EXPERIMENTS_DIR / sweep_name
        if not sweep_dir.exists():
            return completed

        TERMINAL = {"completed", "aborted", "failed", "infeasible"}

        for scale_dir in sweep_dir.iterdir():
            if not scale_dir.is_dir() or not scale_dir.name.startswith("A"):
                continue

            try:
                parts = scale_dir.name.split("_")
                agents = int(parts[0][1:])  # A{agents}
                parcels = int(parts[1][1:])  # P{parcels}
            except (ValueError, IndexError):
                continue

            for exp_dir in scale_dir.iterdir():
                if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
                    continue

                exp_json = exp_dir / "experiment.json"
                if not exp_json.exists():
                    continue

                try:
                    exp_data = json.loads(exp_json.read_text(encoding="utf-8"))
                    strategies = exp_data.get("strategies", [])

                    # ---------------------------------------------
                    # Experiment is completed ONLY if all strategies
                    # are in terminal states
                    # ---------------------------------------------
                    if strategies and all(
                            s.get("status") in TERMINAL for s in strategies
                    ):
                        instance_id = strategies[0].get("instance_id", 0)
                        completed.add((agents, parcels, instance_id))

                except (json.JSONDecodeError, IOError):
                    continue

        print(f"[SWEEP] Found {len(completed)} completed experiment configs")
        return completed

    def prune_completed_sweeps(self, sweep_configs: List[Dict], completed: Set[Tuple[int, int, int]]) -> List[Dict]:
        """Remove completed configurations from sweep queue."""
        pruned = []

        for config in sweep_configs:
            agents = config.get("agents", 0)
            parcels = config.get("parcels", 0)
            instance_id = config.get("instance_id", 0)

            if (agents, parcels, instance_id) not in completed:
                pruned.append(config)
            else:
                print(f"[SWEEP] Skipping completed: A{agents}_P{parcels}_I{instance_id}")

        print(f"[SWEEP] Pruned {len(sweep_configs) - len(pruned)} completed configs")
        print(f"[SWEEP] Remaining: {len(pruned)} configs to run")
        return pruned

    def _wrap_text(self, text: str, font, max_width: int) -> List[str]:
        """
        Wrap text to fit within max_width pixels.

        Returns a list of lines that fit within the specified width.
        """
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            # Test if adding this word would exceed max_width
            test_line = current_line + (" " if current_line else "") + word
            test_width = font.size(test_line)[0]  # FIXED: use font.size() instead of font.get_rect()

            if test_width <= max_width:
                current_line = test_line
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(current_line)
                current_line = word

                # Handle case where single word is too long
                while font.size(current_line)[0] > max_width and len(current_line) > 1:  # FIXED: use font.size()
                    # Split the word
                    split_point = len(current_line) // 2
                    lines.append(current_line[:split_point] + "-")
                    current_line = current_line[split_point:]

        # Add the last line
        if current_line:
            lines.append(current_line)

        return lines

    def _show_experiment_status_modal(
            self,
            *,
            sweep_name: str,
            agents: int,
            parcels: int,
            run_id: str,
            current_index: int,
            strategies: list,
            max_visible_strategies: int = None,
    ):

        dialog_message = None
        selected_index = current_index
        total = len(strategies)

        lines = []
        for i, s in enumerate(strategies):
            status = s.get("status", "pending")

            status_icon = {
                "completed": "✓",
                "pending": "○",
                "infeasible": "◇",
                "failed": "✗",
                "aborted": "◇",
            }.get(status, "?")

            marker = "→" if i == selected_index else " "
            lines.append(f"{marker} {status_icon} [{status.upper():9}] {s.get('name')}")

        while True:
            key, selected_index = self._show_scroll_modal(
                title=f"Experiment Status | {agents}A {parcels}P | Run {run_id}",
                lines=lines,
                background=(28, 32, 40),
                title_color=(120, 200, 255),
                hint="↑↓ Select | ENTER: Run Strategy | ESC: Discard Experiment",
                exit_keys={pygame.K_RETURN, pygame.K_ESCAPE},
                max_visible_lines=max_visible_strategies,
                dialog_message=dialog_message,
            )

            if dialog_message:
                dialog_message = None
                continue

            if key == pygame.K_ESCAPE:
                return None

            if key == pygame.K_RETURN:
                status = strategies[selected_index].get("status")

                if status == "completed":
                    dialog_message = (
                        "This strategy has already been completed.\n\n"
                        "Select another strategy to execute."
                    )
                    continue

                return selected_index

    def _show_scroll_modal(
            self,
            *,
            title: str,
            lines: list[str],
            background=(30, 30, 30),
            title_color=(120, 200, 255),
            hint: str,
            exit_keys: set[int],
            max_visible_lines: int = None,
            dialog_message: Optional[str] = None,
    ):
        exit_keys = set(exit_keys)

        font_height = self.font.get_height()
        LINE_HEIGHT = font_height + 4

        TITLE_HEIGHT = 80
        HINT_HEIGHT = 60
        STATUS_HEIGHT = 30
        NAV_HEIGHT = 25
        PADDING = 40

        available_height = (
                self.screen_size[1]
                - TITLE_HEIGHT
                - HINT_HEIGHT
                - STATUS_HEIGHT
                - NAV_HEIGHT
                - PADDING
        )
        calculated_max_lines = max(1, available_height // LINE_HEIGHT)

        if max_visible_lines is None:
            MAX_VISIBLE_LINES = calculated_max_lines
        else:
            MAX_VISIBLE_LINES = min(max_visible_lines, calculated_max_lines)

        PANEL_WIDTH = int(self.screen_size[0] * 0.35)
        CONTENT_HEIGHT = MAX_VISIBLE_LINES * LINE_HEIGHT
        PANEL_HEIGHT = CONTENT_HEIGHT + 60

        available_vertical_space = (
                self.screen_size[1]
                - TITLE_HEIGHT
                - HINT_HEIGHT
                - STATUS_HEIGHT
                - NAV_HEIGHT
        )
        PANEL_Y = TITLE_HEIGHT + (available_vertical_space - PANEL_HEIGHT) // 2
        PANEL_X = (self.screen_size[0] - PANEL_WIDTH) // 2

        INNER_PADDING = 20
        CONTENT_X = PANEL_X + INNER_PADDING
        CONTENT_Y = PANEL_Y + 30
        CONTENT_WIDTH = PANEL_WIDTH - INNER_PADDING * 2

        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(self._wrap_text(line, self.font, CONTENT_WIDTH))

        total_lines = len(wrapped_lines)
        scroll = 0
        max_scroll = max(0, total_lines - MAX_VISIBLE_LINES)

        selected_index = 0
        max_selection = max(0, total_lines - 1)

        viewport_rect = pygame.Rect(CONTENT_X, CONTENT_Y, CONTENT_WIDTH, CONTENT_HEIGHT)
        panel_rect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT)

        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit()

                if e.type == pygame.KEYDOWN:
                    # Dialog has priority
                    if dialog_message and e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                        return e.key, selected_index

                    if e.key in exit_keys:
                        return e.key, selected_index

                    if dialog_message:
                        continue

                    if e.key == pygame.K_DOWN:
                        selected_index = min(selected_index + 1, max_selection)
                        if selected_index >= scroll + MAX_VISIBLE_LINES:
                            scroll = min(selected_index - MAX_VISIBLE_LINES + 1, max_scroll)

                    elif e.key == pygame.K_UP:
                        selected_index = max(selected_index - 1, 0)
                        if selected_index < scroll:
                            scroll = selected_index

            self.screen.fill(background)

            title_surf = self.large_font.render(title, True, title_color)
            self.screen.blit(title_surf, title_surf.get_rect(center=(self.screen_size[0] // 2, 40)))

            pygame.draw.rect(self.screen, (22, 26, 34), panel_rect, border_radius=12)
            pygame.draw.rect(self.screen, (90, 100, 120), panel_rect, 2, border_radius=12)

            self.screen.set_clip(viewport_rect)

            y = CONTENT_Y
            for i, line in enumerate(wrapped_lines[scroll:scroll + MAX_VISIBLE_LINES]):
                idx = scroll + i
                highlight = idx == selected_index

                if highlight:
                    pygame.draw.rect(
                        self.screen,
                        (60, 80, 120),
                        (CONTENT_X - 5, y - 2, CONTENT_WIDTH + 10, LINE_HEIGHT),
                        border_radius=4,
                    )

                txt = self.font.render(
                    ("→ " if highlight else "  ") + line,
                    True,
                    (255, 255, 255) if highlight else (230, 230, 230),
                )
                self.screen.blit(txt, (CONTENT_X, y))
                y += LINE_HEIGHT

            self.screen.set_clip(None)

            # -----------------------------
            # INLINE DIALOG OVERLAY
            # -----------------------------
            if dialog_message:
                overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (0, 0))

                popup_w = int(self.screen_size[0] * 0.4)
                popup_h = 160
                popup_x = (self.screen_size[0] - popup_w) // 2
                popup_y = (self.screen_size[1] - popup_h) // 2

                rect = pygame.Rect(popup_x, popup_y, popup_w, popup_h)
                pygame.draw.rect(self.screen, (32, 36, 46), rect, border_radius=10)
                pygame.draw.rect(self.screen, (200, 120, 80), rect, 2, border_radius=10)

                y = popup_y + 30
                for line in dialog_message.split("\n"):
                    txt = self.font.render(line, True, (230, 230, 230))
                    self.screen.blit(
                        txt,
                        txt.get_rect(center=(self.screen_size[0] // 2, y)),
                    )
                    y += 28

                hint_txt = self.font.render("ESC or ENTER to close", True, (160, 160, 160))
                self.screen.blit(
                    hint_txt,
                    hint_txt.get_rect(center=(self.screen_size[0] // 2, popup_y + popup_h - 28)),
                )

            hint_surf = self.font.render(hint, True, (180, 180, 180))
            self.screen.blit(
                hint_surf,
                hint_surf.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 30)),
            )

            pygame.display.flip()
            self.clock.tick(30)

    def show_sweep_selection_modal(self, configs, max_visible_configs: int = 20):
        lines = []
        dialog_message = None
        selected_index = 0

        for i, cfg in enumerate(configs):
            agents = cfg["agents"]
            parcels = cfg["parcels"]
            state = cfg["state"]

            status_icon = {
                "completed": "✓",
                "partial": "◯",
                "new": "○",
            }.get(state, "?")

            lines.append(
                f"[{i + 1:2d}] {status_icon} A{agents:2d} P{parcels:2d} [{state.upper():9}]"
            )

        while True:
            key, selected_index = self._show_scroll_modal(
                title=f"Sweep Configuration Selection ({len(configs)} total)",
                lines=lines,
                background=(30, 34, 42),
                title_color=(220, 220, 220),
                hint="↑↓ Select | ENTER: Choose | ESC: Cancel",
                exit_keys={pygame.K_RETURN, pygame.K_ESCAPE},
                max_visible_lines=max_visible_configs,
                dialog_message=dialog_message,
            )

            if dialog_message:
                dialog_message = None
                continue

            if key == pygame.K_ESCAPE:
                return None

            if key == pygame.K_RETURN:
                selected = configs[selected_index]

                if selected["state"] == "completed":
                    dialog_message = (
                        f"A{selected['agents']} P{selected['parcels']} is already completed.\n\n"
                        "All strategies for this configuration have been executed."
                    )
                    continue

                return selected

    def _show_experiment_status_screen(self, strategy_name, steps, agents, terrain):
        """Show experiment status during fast execution mode"""
        self.screen.fill((40, 44, 52))

        # Title with current mode
        title = self.large_font.render("EXPERIMENT RUNNING", True, (100, 255, 100))
        self.screen.blit(title, title.get_rect(center=(self.screen_size[0] // 2, 120)))

        # Current mode indicator (more prominent)
        mode_text = "FAST MODE" if self.fast_mode else "NORMAL MODE"
        mode_color = (255, 200, 100) if self.fast_mode else (100, 255, 100)
        mode_indicator = self.large_font.render(mode_text, True, mode_color)
        self.screen.blit(mode_indicator, mode_indicator.get_rect(center=(self.screen_size[0] // 2, 170)))

        # Strategy info
        strategy_text = self.font.render(f"Strategy: {strategy_name}", True, (230, 230, 230))
        self.screen.blit(strategy_text, strategy_text.get_rect(center=(self.screen_size[0] // 2, 220)))

        # Step counter
        step_text = self.font.render(f"Step: {steps:,}", True, (230, 230, 230))
        self.screen.blit(step_text, step_text.get_rect(center=(self.screen_size[0] // 2, 260)))

        # Agent status
        alive_count = len([a for a in agents if not self.agent_dead(a)])
        total_count = len(agents)
        agent_text = self.font.render(f"Agents: {alive_count}/{total_count} alive", True, (230, 230, 230))
        self.screen.blit(agent_text, agent_text.get_rect(center=(self.screen_size[0] // 2, 300)))

        # Parcel status
        remaining_parcels = sum(1 for p in terrain.parcels if not p.delivered)
        total_parcels = len(terrain.parcels)
        parcel_text = self.font.render(f"Parcels: {total_parcels - remaining_parcels}/{total_parcels} delivered", True,
                                       (230, 230, 230))
        self.screen.blit(parcel_text, parcel_text.get_rect(center=(self.screen_size[0] // 2, 340)))

        # Toggle instructions
        hint = self.font.render("Press H to toggle between NORMAL/FAST rendering", True, (180, 180, 180))
        self.screen.blit(hint, hint.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] - 100)))

    def run(self):
        # -------------------------------------------------
        # Splash + setup phase (single abort boundary)
        # -------------------------------------------------
        if not getattr(self, '_splash_shown', False) or not (USE_SWEEP and SCALING_CONFIG):
            self.show_splash(timeout=1.0)
            self._splash_shown = True

        # RESET RESUME FLAG FOR SWEEP PROCESSING (but preserve selection state)
        if USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False):
            self._resumed = False  # ← Only reset if sweep already initialized

        # Show sweep selection modal (GUI)
        if USE_SWEEP and SCALING_CONFIG and not getattr(self, '_sweep_initialized', False):
            # Generate sweeps if missing BEFORE building status
            if not SCALING_CONFIG.get("sweeps"):
                station = self.experiment.get("station")
                if station:
                    w, h = station["w"], station["h"]
                else:
                    w, h = 6, 6

                mode = SCALING_CONFIG.get("mode", "research")
                budget = SCALING_CONFIG.get("budget", 40)

                SCALING_CONFIG["sweeps"] = SCALING_CONFIG.get("sweeps") or generate_station_scaling(w, h, mode=mode, budget=budget)
                print(f"[SWEEP] Generated {len(SCALING_CONFIG['sweeps'])} configs")

            sweep_name = SCALING_CONFIG.get("name", "sweep")
            status_rows = self.build_sweep_status(sweep_name=sweep_name, sweep_configs=SCALING_CONFIG["sweeps"])

            selected = self.show_sweep_selection_modal(status_rows)
            if selected is None:
                print("[SWEEP] User aborted sweep selection")
                return False

            self._sweep_initialized = True
            print("[SWEEP] Selection made, proceeding with automated batch execution")

        if not getattr(self, "_resumed", False):
            self._experiment_completed = False

        # -------------------------------------------------
        # Initialize setup variables defensively
        # -------------------------------------------------
        n_agents = None
        n_parcels = None
        selected_planner = None
        # -------------------------------------------------
        # Fresh setup path (only if not resumed)
        # -------------------------------------------------
        if not getattr(self, "_resumed", False):

            # -------- SWEEP MODE --------
            if USE_SWEEP and SCALING_CONFIG:
                # Generate sweeps once if missing (existing logic)
                if not SCALING_CONFIG.get("sweeps"):
                    station = self.experiment.get("station")
                    if station:
                        w, h = station["w"], station["h"]
                    else:
                        w, h = 6, 6

                    mode = SCALING_CONFIG.get("mode", "research")
                    budget = SCALING_CONFIG.get("budget", 40)

                    SCALING_CONFIG["sweeps"] = generate_station_scaling(w, h, mode=mode, budget=budget)
                    print(f"[SWEEP] Generated {len(SCALING_CONFIG['sweeps'])} configs")

                # Initialize queue with persistence awareness
                if self._sweep_queue is None:
                    sweep_name = SCALING_CONFIG.get("name", "sweep")
                    completed = self.scan_completed_experiments(sweep_name)

                    # Prune completed configs from sweep
                    full_sweep = list(SCALING_CONFIG["sweeps"])
                    self._sweep_queue = self.prune_completed_sweeps(full_sweep, completed)

                # If empty after pruning, all done
                if not self._sweep_queue:
                    print("[SWEEP] All configurations completed.")
                    return

                # Show sweep selection modal only once at startup (not resumed)
                if not getattr(self, '_sweep_initialized', False):
                    sweep_name = SCALING_CONFIG.get("name", "sweep")
                    status_rows = self.build_sweep_status(
                        sweep_name=sweep_name,
                        sweep_configs=SCALING_CONFIG["sweeps"],
                    )

                    selected = self.show_sweep_selection_modal(status_rows)
                    if selected is None:
                        print("[SWEEP] User aborted sweep selection")
                        return False

                    # Optional: filter queue to start from selected config
                    # For now, we'll just mark as initialized and proceed with queue
                    self._sweep_initialized = True
                    print("[SWEEP] Selection made, proceeding with automated batch execution")

                # Process sweep queue automatically
                while self._sweep_queue:
                    cfg = self._sweep_queue[0]  # ← Peek at config, don't remove yet
                    # Reset flags for new config
                    self._strategy_initialized = False

                    n_agents = cfg["agents"]
                    n_parcels = cfg["parcels"]

                    sweep_name = SCALING_CONFIG.get("name", "sweep")

                    state = self.scan_sweep_config_state(
                        sweep_name=sweep_name,
                        agents=n_agents,
                        parcels=n_parcels,
                    )

                    if state["completed"]:
                        print(f"[SWEEP] A{n_agents}_P{n_parcels} already completed, skipping")
                        self._sweep_queue.pop(0)  # ← Remove the completed config
                        continue

                    scale_root = EXPERIMENTS_DIR / sweep_name / f"A{n_agents}_P{n_parcels}"

                    state = self._load_or_create_experiment(scale_root)

                    if state == ExperimentState.COMPLETED:
                        print(f"[SWEEP] A{n_agents}_P{n_parcels} completed during load, skipping")
                        self._sweep_queue.pop(0)  # ← Remove the completed config
                        continue

                    # If we reach here, we have a valid config to process
                    self._sweep_queue.pop(0)  # ← Remove the config we're about to process

                    if state == ExperimentState.RESUMABLE:
                        break

                    if state == ExperimentState.NEW:
                        break

                    break
                else:
                    print("[SWEEP] All configurations completed.")
                    return False

                if self._resumed:
                    print(f"[SWEEP] Resumed config: agents={n_agents}, parcels={n_parcels}")
                else:
                    print(f"[SWEEP] Running new config: agents={n_agents}, parcels={n_parcels}")

                # Enforce capacity
                try:
                    self.checker.enforce(n_agents)
                except AgentCapacityError as e:
                    print(f"[SWEEP] Skipping config {cfg}: {e}")
                    return

                if USE_SWEEP and SCALING_CONFIG:
                    selected_planner = self.planners[0]
                else:
                    selected_planner = self.show_setup_and_planner_select()
            # -------- GUI MODE --------
            else:
                # Only show GUI setup if not in an active sweep
                if not (USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False)):
                    while True:
                        try:
                            n_agents, n_parcels = self.show_experiment_setup(
                                initial_agents=1,
                                initial_parcels=self.parcels,
                                max_agents=self.system_limit,
                            )

                            self.parcels = n_parcels
                            selected_planner = self.show_setup_and_planner_select()
                            # Resume only after user input defines context
                            self._load_or_create_experiment(EXPERIMENTS_DIR)
                            try:
                                self.checker.enforce(n_agents)
                            except AgentCapacityError as e:
                                self._show_error_and_restart_setup(str(e))
                                continue
                            break

                        except RuntimeError:
                            continue

                        except SystemExit:
                            print("[EXPERIMENT] User aborted at setup")
                            return
                else:
                    # In sweep mode after initialization - values already set from sweep config
                    selected_planner = self.planners[0]  # Use default planner for sweep

        # -------------------------------------------------
        # Commit setup choices to experiment schema
        # -------------------------------------------------
        if not getattr(self, "_resumed", False):
            assert n_parcels is not None, "n_parcels must be set for new experiment"
            assert selected_planner is not None, "selected_planner must be set for new experiment"

            strategy_vault = [
                # -------------------------------
                # Decentralized Experimental Variants
                # -------------------------------
                {
                    "name": "naive_decentralized",
                    "strategy_type": "naive",
                    "variant": "reactive",
                    "topology": "decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Reactive decentralized baseline. Agents greedily select tasks using only local perception without coordination, prediction, or communication."
                },
                {
                    "name": "decentralized_auction",
                    "variant": "auction_baseline",
                    "strategy_type": "auction",
                    "topology": "decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Decentralized auction baseline with explicit message passing. Agents broadcast bids for tasks and resolve conflicts through communication."
                },
                {
                    "name": "cbba_single_task",
                    "variant": "cbba_single",
                    "strategy_type": "cbba",
                    "topology": "decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "CBBA variant limited to single task allocation. Tests consensus-based coordination under restricted bundle formation."
                },
                {
                    "name": "cbba_full",
                    "variant": "cbba_bundle",
                    "strategy_type": "cbba",
                    "topology": "decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Full CBBA implementation with bundle construction and consensus convergence. Represents structured decentralized coordination with communication."
                },

                # -------------------------------
                # Centralized Experimental Variants
                # -------------------------------
                {
                    "name": "centralized_cc_greedy",
                    "variant": "greedy",
                    "strategy_type": "centralized",
                    "topology": "centralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Centralized controller using greedy assignment. Global information available but allocation performed using local-minimum heuristic."
                },
                {
                    "name": "centralized_cc_hungarian",
                    "variant": "hungarian",
                    "strategy_type": "centralized",
                    "topology": "centralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Centralized optimal task allocation using the Hungarian algorithm. Provides upper-bound performance baseline under full observability."
                },
                {
                    "name": "centralized_cc",
                    "variant": "auction",
                    "strategy_type": "centralized",
                    "topology": "centralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Centralized auction mechanism. Tasks are globally coordinated via a single auctioneer that assigns parcels based on agent bids."
                },

                # -------------------------------
                # SPPDEC Experimental Variants
                # -------------------------------
                {
                    "name": "sppdec",
                    "variant": "control",  # baseline_payload=0.0, alpha=0.0
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "SPPDEC with SLM policy selection but without payload-aware self modeling and without peer energy penalties. Uses perception, feasibility, and learned uncertainty only."
                },
                {
                    "name": "sppdec",
                    "variant": "baseline_only",  # baseline_payload=0.15, alpha=0.0
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "SPPDEC with SLM policy selection and payload-aware self-cost modeling, but without fairness pressure. Tests the effect of improved self modeling alone."
                },
                {
                    "name": "sppdec",
                    "variant": "fairness_only",  # baseline_payload=0.0, alpha=0.02
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "SPPDEC with SLM policy selection and belief-based peer energy penalties, but without payload-aware self modeling. Tests fairness pressure from inferred peer workload."
                },
                {
                    "name": "sppdec",
                    "variant": "combined",  # baseline_payload=0.15, alpha=0.02
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Full SPPDEC Phase 1. Uses SLM policy selection with both payload-aware self modeling and belief-based peer energy penalties for adaptive decentralized coordination."
                },
                {
                    "name": "sppdec",
                    "variant": "personality_mix_balanced_quartet",
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Is complete social coverage valuable without extremes?",
                    "agent_personalities": [
                        "rational",
                        "selfish",
                        "altruistic",
                        "reckless"
                    ],
                    "personality_distribution": {
                        "rational": 1,
                        "selfish": 1,
                        "altruistic": 1,
                        "reckless": 1,
                    }
                },
                {
                    "name": "sppdec",
                    "variant": "personality_mix_no_reckless",
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Heterogeneous SPPDEC team with mixed personalities",
                    "agent_personalities": [
                        "rational",
                        "selfish",
                        "altruistic",
                        "reckless"
                    ],
                    "personality_distribution": {
                        "rational": 2,
                        "selfish": 1,
                        "altruistic": 1,
                        "reckless": 0,
                    }
                },
                {
                    "name": "sppdec",
                    "variant": "personality_mix_controlled_chaos",
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Find the sweet spot between too much and too little disruption. Double Rationals to mediate, Selfish for efficiency, Reckless as catalytic disruptor",
                    "agent_personalities": [
                        "rational",
                        "selfish",
                        "altruistic",
                        "reckless"
                    ],
                    "personality_distribution": {
                        "rational": 2,
                        "selfish": 1,
                        "altruistic": 0,
                        "reckless": 1,
                    }
                },
                {
                    "name": "sppdec",
                    "variant": "personality_mix_3_brains_1_wildcard",
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "Test if Rational majority can fully harness Reckless's catalytic power",
                    "agent_personalities": [
                        "rational",
                        "selfish",
                        "altruistic",
                        "reckless"
                    ],
                    "personality_distribution": {
                        "rational": 3,
                        "selfish": 0,
                        "altruistic": 0,
                        "reckless": 1,
                    }
                },
                {
                    "name": "sppdec",
                    "variant": "personality_mix_efficiency_overdrive",
                    "strategy_type": "sppdec",
                    "topology": "predictive_decentralized",
                    "planner": selected_planner,
                    "snapshot": None,
                    "status": "pending",
                    "description": "What happens when efficiency pressure meets disruption?",
                    "agent_personalities": [
                        "rational",
                        "selfish",
                        "altruistic",
                        "reckless"
                    ],
                    "personality_distribution": {
                        "rational": 1,
                        "selfish": 2,
                        "altruistic": 0,
                        "reckless": 1,
                    }
                },
            ]

            strategies_expanded = []
            num_instances = SCALING_CONFIG.get("num_instances", 1)

            for i in range(num_instances):
                for strategy in strategy_vault:
                    instance = strategy.copy()

                    variant = strategy.get("variant", "base")
                    family = f"{strategy['name']}_{variant}"

                    instance["name"] = f"{family}_instance_{i + 1}"
                    instance["instance_id"] = i
                    instance["variant"] = variant
                    instance["family"] = family

                    strategies_expanded.append(instance)

            strategies = strategies_expanded

            # Now use strategies_expanded instead of strategies
            self.experiment = {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "seed": self.seed,
                "scale": {
                    "n_agents": n_agents,
                    "n_parcels": n_parcels,
                },
                "parcels": [],
                "agents": [],
                "comm_config": self.default_comm_config.copy(),
                "current_index": 0,
                "status": "in_progress",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "results": [],
                "strategies": strategies,
            }

            # Generate parcel positions
            cols = self.screen_size[0] // GRID_SIZE
            rows = self.screen_size[1] // GRID_SIZE
            center_col = max(2, cols // 2 - 2)
            center_row = max(2, rows // 2 - 2)

            station_spec = {
                "col": center_col,
                "row": center_row,
                "w": 6,
                "h": 6,
            }
            self.experiment["station"] = station_spec

            s = self.experiment["station"]

            # Forbidden cells inside the station footprint
            forbidden_cells = []
            for r in range(s["row"], s["row"] + s["h"]):
                for c in range(s["col"], s["col"] + s["w"]):
                    forbidden_cells.append((c, r))

            parcel_positions = deterministic_parcel_positions(
                self.seed, n_parcels, cols, rows, forbidden_cells
            )

            # Add parcels to experiment
            for i, (col, row) in enumerate(parcel_positions):
                # Generate a string ID for each parcel
                parcel_id = f"parcel_{i:03d}"  # e.g., "parcel_000", "parcel_001"
                # OR use UUID: parcel_id = str(uuid.uuid4())[:8]

                # Update the parcel creation to be more explicit:
                parcel_data = {
                    "id": str(parcel_id),  # Explicitly cast to string
                    "col": int(col),
                    "row": int(row),
                    "picked": bool(False),
                    "delivered": bool(False)
                }
                self.experiment["parcels"].append(parcel_data)
            # rebuild agents deterministically
            start_col = (self.screen_size[0] // GRID_SIZE) // 2
            start_row = (self.screen_size[1] // GRID_SIZE) // 2

            personality_cycle = list(SPPDEC_PERSONALITIES.keys())
            self.experiment["agents"] = [
                {
                    "id": f"agent_{i}",
                    "type": "drone",
                    "planner": selected_planner,
                    "start_col": start_col + i * 2,
                    "start_row": start_row,
                    "personality": personality_cycle[i % len(personality_cycle)],
                }
                for i in range(n_agents)
            ]

            # Create experiment directory
            if USE_SWEEP and SCALING_CONFIG:
                sweep_name = SCALING_CONFIG.get("name", "sweep")
                scale_dir = f"A{n_agents}_P{n_parcels}"
                self.experiment_dir = (
                        EXPERIMENTS_DIR
                        / sweep_name
                        / scale_dir
                        / f"experiment_{self.run_id}"
                )
            else:
                self.experiment_dir = EXPERIMENTS_DIR / f"experiment_{self.run_id}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self.experiment_json_path = self.experiment_dir / "experiment.json"

            self._write_experiment_json()

        # -------------------------------------------------
        # STRATEGY LOOP
        # -------------------------------------------------
        assert not (
                USE_SWEEP
                and getattr(self, "_experiment_completed", False)
                and int(self.experiment.get("current_index", 0)) < len(self.experiment["strategies"])
        ), "Experiment marked completed but strategies remain"

        # ONLY THEN check if current experiment has strategies
        current_index = int(self.experiment.get("current_index", 0))
        if current_index >= len(self.experiment["strategies"]):
            if USE_SWEEP and SCALING_CONFIG and self._sweep_queue:
                # More sweep configs exist, continue processing
                return True
            else:
                # No more sweep configs, exit
                print(
                    f"[EXPERIMENT] No remaining strategies to run ({current_index}/{len(self.experiment['strategies'])} processed)")
                return False

        remaining = len(self.experiment["strategies"]) - current_index
        print(
            f"[EXPERIMENT] Starting strategy {current_index + 1}/{len(self.experiment['strategies'])} ({remaining} remaining)")

        for idx, strategy in enumerate(self.experiment["strategies"]):
            # print("Strategies", self.experiment["strategies"])
            if idx < int(self.experiment.get("current_index", 0)):
                continue
            # print(f"self.experiment[\"strategies\"] {self.experiment['strategies']}")

            # Reset completion wait counter for this strategy
            if hasattr(self, '_completion_wait_counter'):
                del self._completion_wait_counter
            base_name = strategy.get("name", f"strategy_{idx}")
            variant = strategy.get("variant")

            if base_name == "sppdec" and variant:
                strategy_name = f"sppdec_{variant}"
            elif base_name == "centralized_cc" and variant:
                print("BASE_NAME", base_name)
                strategy_name = f"centralized_cc_{variant}"
            else:
                strategy_name = base_name

            strategy_type = strategy.get("strategy_type", "unknown")
            topology = strategy.get("topology", "unknown")
            planner_name = strategy["planner"]
            print(f"[EXPERIMENT] Running planner '{planner_name}' ({idx + 1}/{len(self.experiment['strategies'])})"
                  f"Current Strategy {strategy_name}"
                  f"Strategy mode {topology}")

            self._configure_planner(planner_name)

            snapshot = strategy.get("snapshot")
            # -------------------------------------------------
            # >>> FLIGHT RECORDER: instantiate once per strategy
            # -------------------------------------------------
            run_id = self.experiment["run_id"]

            metrics = MetricsAccumulator(
                run_id=self.experiment["run_id"],
                planner=planner_name,
                seed=self.experiment["seed"],
                n_drones=len(self.experiment["agents"]),
                n_parcels=len(self.experiment.get("parcels", [])),
                comm_reliability=self.experiment["comm_config"].get("reliability", 1.0),
                battery_capacity=100.0,  # safe default before agents exist
                episodic_interval=1.0,
            )
            flight_recorder = FlightRecorder(
                run_id=run_id,
                planner=planner_name,
                strategy_name=strategy_name,
                agents=self.experiment["agents"],
                parcels=self.experiment["parcels"],  # Still pass initial parcel definitions
                grid_size=GRID_SIZE,
                sim_dt=1.0 / 60.0,
                metrics=metrics
            )
            try:
                terrain, agents, pre_steps, all_parcels = self._setup_scene(maybe_snapshot=snapshot,
                                                                            strategy_name=strategy_name,
                                                                            strategy_spec=strategy,
                                                                            flight_recorder=flight_recorder
                                                                            )

                proceed = self._show_strategy_countdown(
                    terrain=terrain,
                    agents=agents,
                    strategy_name=strategy_name,
                    topology=topology,
                    variant=variant,
                    config=strategy.get("config")
                )

                if not proceed:
                    print("[EXPERIMENT] Strategy skipped by user")
                    continue

                from src.strategies.factory import build_strategy
                strategies = {}
                for drone in agents:
                    import copy
                    strategies[drone.agent_id] = build_strategy(
                        strategy_spec=copy.deepcopy(strategy),
                        terrain=terrain,
                        agents=agents,
                        physics=physics,
                        graphics_dir=GRAPHICS_DIR,
                        grid_size=GRID_SIZE,
                    )

            except AgentSpawnError as e:
                self._show_error_and_restart_setup(str(e))
                break
            # exit run(), caller restarts setup

            if terrain is None:
                return  # safely return to GUI loop

            # After creating flight_recorder:
            if snapshot:
                # Reconstruct history from snapshot
                flight_recorder.reconstruct_from_snapshot(snapshot)

                # Then register current parcels (they'll be skipped if already registered)
                for parcel in all_parcels:
                    flight_recorder.register_parcel(parcel)
            else:
                # New simulation
                for parcel in all_parcels:
                    flight_recorder.register_parcel(parcel)
            try:
                # controllers (one per agent)
                controllers = {}
                for i, drone in enumerate(agents):
                    agent_id = self.experiment["agents"][i]["id"]
                    drone.agent_id = agent_id
                    if terrain.command_center:
                        terrain.command_center.network.flight_recorder = flight_recorder
                        terrain.command_center.register_agent(drone)

                    ai = AIAgentController(
                        drone=drone,
                        terrain=terrain,
                        strategy=strategies[agent_id],
                        flight_recorder=flight_recorder,
                        topology=topology
                    )
                    drone.controller = ai
                    if hasattr(drone, "network"):
                        drone.network.flight_recorder = flight_recorder
                    strategies[agent_id].set_drone(drone)
                    if hasattr(strategies[agent_id], "flight_recorder"):
                        strategies[agent_id].flight_recorder = flight_recorder

                    print("Drone", drone)
                    print("Drone ID", drone.agent_id)

                    # Only create human controller and switcher in GUI mode
                    human = HumanAgentController(drone, terrain)
                    switcher = ControllerSwitcher([human, ai])
                    switcher.index = 1  # Start with AI controller
                    controllers[agent_id] = switcher

                from concurrent.futures import ThreadPoolExecutor
                executor = ThreadPoolExecutor(max_workers=len(agents))

                steps = int(pre_steps or 0)
                last_move_time = time.time()
                last_action_time = time.time()
                aborted = False
                strategy_finished = False

                prev_positions = {i: (a.pos.x, a.pos.y) for i, a in enumerate(agents)}

                # -------------------------------------------------
                # MAIN SIMULATION LOOP
                # -------------------------------------------------

                while steps < self.max_steps:
                    steps += 1
                    # Keep physics clock in sync with the simulation
                    physics.current_step = steps
                    sim_time = steps * (1.0 / 60.0)

                    # CENTRALIZED COGNITION TICK
                    if terrain.command_center:
                        terrain.command_center.update(
                            agents=agents,
                            parcels=terrain.parcels,
                            now=time.time(),
                        )
                        if hasattr(terrain.command_center,
                                   "globally_infeasible") and terrain.command_center.globally_infeasible:
                            print("[EXPERIMENT] CommandCenter signaled global infeasibility")

                            self._finalize_strategy(
                                idx=idx,
                                status="aborted",
                                planner_name=planner_name,
                                strategy_name=strategy_name,
                                strategy_type=strategy_type,
                                steps=steps,
                                flight_recorder=flight_recorder,
                            )

                            aborted = True
                            break

                    # -------------------------------------------------
                    # RESET NETWORK BANDWIDTH FOR NEW STEP
                    # -------------------------------------------------
                    for drone in agents:
                        if hasattr(drone, "network"):
                            drone.network.reset_step()

                    if terrain.command_center and hasattr(terrain.command_center, "network"):
                        terrain.command_center.network.reset_step()

                    # Single event handling section
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            aborted = True
                            break

                        if e.type == pygame.KEYDOWN:
                            if e.key == pygame.K_ESCAPE:
                                aborted = True
                                break
                            elif e.key == pygame.K_h:  # H to toggle fast mode
                                self.fast_mode = not self.fast_mode
                                mode_text = "FAST" if self.fast_mode else "NORMAL"
                                print(f"[TOGGLE] Switched to {mode_text} mode")
                                continue

                        # Controller events
                        for switcher in controllers.values():
                            if hasattr(switcher, 'handle_event'):
                                switcher.handle_event(e)

                    dt = self.clock.tick(60) / 1000.0

                    # Rendering section (no more event handling here)
                    if not self.fast_mode:
                        # FULL RENDERING MODE
                        self.screen.fill((192, 192, 192))
                        for x in range(0, self.screen_size[0], GRID_SIZE):
                            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.screen_size[1]))
                        for y in range(0, self.screen_size[1], GRID_SIZE):
                            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.screen_size[0], y))

                        terrain.draw(self.screen)
                        for drone in agents:
                            drone.draw(self.screen, self.images)
                            if topology in ("decentralized", "predictive_decentralized", "centralized"):
                                drone.draw_network_pulse(self.screen, min_cells=1, max_cells=3)
                    else:
                        # FAST MODE: Status display only
                        self._show_experiment_status_screen(strategy_name, steps, agents, terrain)

                    pygame.display.flip()
                    # ------------------------------
                    # UPDATE AGENTS
                    # ------------------------------
                    # ==============================
                    # PHASE 1: Collect decisions in parallel
                    # ==============================
                    from concurrent.futures import ThreadPoolExecutor

                    if not hasattr(self, "_executor"):
                        self._executor = ThreadPoolExecutor(max_workers=len(agents))

                    executor = self._executor

                    futures = {}

                    for i, drone in enumerate(agents):
                        agent_id = self.experiment["agents"][i]["id"]
                        controller = controllers[agent_id]

                        # If centralized, decisions flow through network only
                        if terrain.command_center:
                            continue
                        if controller.needs_decision():
                            futures[agent_id] = executor.submit(controller.compute_decision)

                    # Barrier: wait for all to finish
                    decisions = {aid: fut.result() for aid, fut in futures.items()}
                    # ==============================
                    # ABORT PROPAGATION CHECK
                    # ==============================
                    for agent_id, decision in decisions.items():
                        if isinstance(decision, dict) and decision.get("mode") == "abort":
                            reason = decision.get("reason", AbortReason.AGENT_ENERGY_INFEASIBLE)

                            print(f"[EXPERIMENT] Strategy requested abort (agent={agent_id}, reason={reason})")

                            # Resolve the actual agent object
                            agent = next(a for a in agents if a.agent_id == agent_id)

                            # Record abort inside FlightRecorder
                            flight_recorder.record_abort(
                                reason=reason,
                                agent_id=agent_id,
                                step=steps,
                                sim_time=sim_time,
                                metadata={
                                    "battery_pct": float(agent.power.percent()) if hasattr(agent, "power") else None,
                                    "battery": float(agent.power.level) if hasattr(agent, "power") else None,
                                    "position": (int(agent.col), int(agent.row)),
                                    "carrying": getattr(agent.carrying, "id", None),
                                    "topology": topology,
                                    "strategy": strategy_name,
                                }
                            )

                            # Then finalize as usual
                            self._finalize_strategy(
                                idx=idx,
                                status="aborted",
                                planner_name=planner_name,
                                strategy_name=strategy_name,
                                strategy_type=strategy_type,
                                steps=steps,
                                flight_recorder=flight_recorder,
                            )

                            strategy_finished = True
                            aborted = True
                            break

                    if aborted:
                        break

                    # Commit all decisions at the same logical time
                    if not terrain.command_center:
                        decisions = {aid: fut.result() for aid, fut in futures.items()}
                        for agent_id, decision in decisions.items():
                            controllers[agent_id].apply_decision(decision)

                    # ==============================
                    # PHASE 2: Physics + execution
                    # ==============================

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

                        flight_recorder.record_distance(dist, GRID_SIZE)
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

                        if hasattr(drone, "last_action") and drone.last_action:
                            kind, cell, parcel = drone.last_action
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
                                flight_recorder.record_task_complete(1)

                            if kind in ("pick_failed", "drop_failed"):
                                flight_recorder.record_unsafe_failure(1)

                            self._append_event(
                                idx,
                                {"step": steps, "type": kind, "agent": agent_id, "cell": cell},
                            )

                            drone.last_action = None

                    # -------------------------------------------------
                    # >>> FLIGHT RECORDER: fleet-level checks
                    # -------------------------------------------------
                    flight_recorder.tick_fleet(
                        agents=agents,
                        step=steps,
                        sim_time=sim_time,
                    )

                    # -------------------------------------------------
                    # NETWORK DELIVERY PHASE
                    # -------------------------------------------------

                    # Deliver messages sent by drones
                    for drone in agents:
                        if hasattr(drone, "process_network"):
                            drone.process_network(steps)

                    # Deliver messages sent by CommandCenter
                    if terrain.command_center and hasattr(terrain.command_center, "process_network"):
                        terrain.command_center.process_network(steps)

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
                    active_controllers = [
                        a.controller for a in alive_agents
                        if a.controller is not None
                    ]

                    parked_controllers = [
                        c for c in active_controllers
                        if getattr(c, "parked", False)
                    ]

                    remaining = sum(1 for p in terrain.parcels if not p.delivered)
                    # -------------------------------------------------
                    # RUNNER DIAGNOSTIC TRACE (publication-grade)
                    # -------------------------------------------------
                    agent_modes = {}
                    for a in alive_agents:
                        controller = getattr(a, "controller", None)
                        if controller is None:
                            mode = "no_controller"
                        elif getattr(controller, "parked", False):
                            mode = "idle"
                        elif getattr(controller, "has_pending_task", False):
                            mode = "plan"
                        else:
                            mode = "uncertain"
                        agent_modes[a.agent_id] = mode

                    decision = "continue"
                    if alive_agents and len(parked_controllers) == len(active_controllers) and remaining > 0:
                        decision = "infeasible"
                    elif self.mission_exhausted(terrain, agents):
                        decision = "completed"
                    flight_recorder.fleet_events.append({
                        "type": "runner_state",
                        "step": steps,
                        "agent_modes": agent_modes,
                        "remaining_parcels": remaining,
                        "decision": decision,
                    })

                    # ---------------------------------------------
                    # DEBUG: Mission classification visibility
                    # ---------------------------------------------
                    agent_states = {
                        a.agent_id: {
                            "dead": self.agent_dead(a),
                            "at_station": self.agent_at_base(a, terrain),
                            "lost": self.agent_dead(a) and not self.agent_at_base(a, terrain),
                            "parked": getattr(a.controller, "parked", None) if a.controller else None,
                            "has_pending_task": getattr(a.controller, "has_pending_task",
                                                        None) if a.controller else None,
                            "battery": round(a.power.level, 2) if hasattr(a, "power") else None,
                        }
                        for a in agents
                    }
                    dead_total = sum(1 for s in agent_states.values() if s["dead"])
                    dead_at_station = sum(1 for s in agent_states.values() if s["dead"] and s["at_station"])
                    lost_in_field = sum(1 for s in agent_states.values() if s["lost"])
                    # ---------------------------------------------
                    # 3. HARD FAILURE: mission infeasible (behavioral)
                    # ---------------------------------------------
                    if alive_agents and len(parked_controllers) == len(active_controllers) and remaining > 0:
                        print("[EXPERIMENT] Mission infeasible: all agents exhausted and parked with remaining parcels")

                        flight_recorder.fleet_events.append({
                            "type": "mission_infeasible",
                            "step": steps,
                            "alive_agents": len(alive_agents),
                            "parked_agents": len(parked_controllers),
                            "remaining_parcels": remaining,
                            "agent_modes": agent_modes,
                            "decision": decision,
                        })

                        self._finalize_strategy(
                            idx=idx,
                            status="infeasible",
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )

                        self._show_post_run_inspection(
                            terrain=terrain,
                            agents=agents,
                            strategy_name=strategy_name,
                            status="infeasible",
                            variant=variant,
                            config={
                                "alive_agents": len(alive_agents),
                                "parked_agents": len(parked_controllers),
                                "remaining_parcels": remaining,
                                "reason": "All agents aborted planning due to insufficient battery and entered parking state"
                            }
                        )
                        print(
                            "[EXPERIMENT] Mission infeasible. "
                            "All remaining agents voluntarily entered idle with parcels remaining."
                        )

                        break

                    # ---------------------------------------------
                    # 4. HARD FAILURE: all agents lost
                    # ---------------------------------------------
                    if not alive_agents and remaining > 0:
                        self._finalize_strategy(
                            idx=idx,
                            status="failed",
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )

                        self._show_post_run_inspection(
                            terrain=terrain,
                            agents=agents,
                            strategy_name=strategy_name,
                            status="failed",
                            variant=variant,
                            config={
                                "alive_agents": 0,
                                "remaining_parcels": remaining,
                                "reason": "All drones lost before completing deliveries"
                            }
                        )
                        print(
                            "[EXPERIMENT] Mission failed. "
                            f"All agents exhausted. Lost in field={lost_in_field}, "
                            f"dead at station={dead_at_station}, remaining parcels={remaining}."
                        )

                        break

                    # ---------------------------------------------
                    # 5. SUCCESS CONDITION - NEW PARKING-AWARE CHECK
                    # ---------------------------------------------
                    if self.mission_exhausted(terrain, agents):
                        status = "completed"
                        # print(f"[DEBUG] Mission exhausted condition met at step {steps}")
                        # Add a brief wait to ensure all agents are settled at the station
                        if not hasattr(self, '_completion_wait_counter'):
                            self._completion_wait_counter = 0
                            print(f"[DEBUG] Starting completion wait...")

                        self._completion_wait_counter += 1

                        # Wait for 1 second (60 frames at 60 FPS) before finalizing
                        if self._completion_wait_counter < 60:
                            # print(f"[DEBUG] Waiting for agents to settle ({self._completion_wait_counter}/60)...")
                            continue  # Continue simulation for a bit longer

                        print(f"[DEBUG] All conditions confirmed. Marking strategy as completed.")

                        # Export flight data
                        self._finalize_strategy(
                            idx=idx,
                            status=status,
                            planner_name=planner_name,
                            strategy_name=strategy_name,
                            strategy_type=strategy_type,
                            steps=steps,
                            flight_recorder=flight_recorder,
                        )

                        # BREAK OUT OF SIMULATION LOOP IMMEDIATELY
                        strategy_finished = True
                        break
                if aborted:
                    print("[EXPERIMENT] Simulation loop exited due to abort")
                    break

                # -------------------------------------------------
                # POST-RUN UI (per strategy)
                # -------------------------------------------------
                if idx < len(self.experiment["strategies"]) - 1:
                    # There are more strategies to run
                    if not (USE_SWEEP and SCALING_CONFIG and getattr(self, '_sweep_initialized', False)):
                        self._show_post_run_inspection(
                            terrain=terrain,
                            agents=agents,
                            strategy_name=strategy_name,
                            status=self.experiment["strategies"][idx]["status"],
                            variant=variant,
                            config=strategy.get("config")
                        )
                    continue
                else:
                    # This was the last strategy
                    self._show_experiment_complete_screen(terrain, agents)
                    break

            finally:
                try:
                    flight_recorder.finalize()
                except Exception:
                    pass

        # -------------------------------------------------
        # SIGNAL TO CALLER WHETHER SWEEP SHOULD CONTINUE
        # -------------------------------------------------
        # At the very end of run() method, replace the final return logic:
        if USE_SWEEP and SCALING_CONFIG:
            if self._sweep_queue:
                print(f"[SWEEP] Moving to next config, {len(self._sweep_queue)} remaining")
                return True  # Continue to next sweep config
            else:
                print("[SWEEP] All sweep configs completed")
                return False
        else:
            return False


def main():
    runner = ExperimentRunnerGUI()
    try:
        while True:
            should_continue = runner.run()
            if not should_continue:
                break
    except SystemExit:
        pygame.quit()


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     import cProfile
#
#     profiler = cProfile.Profile()
#     profiler.enable()
#
#     main()
#
#     profiler.disable()
#     profiler.dump_stats("profile_results.prof")
