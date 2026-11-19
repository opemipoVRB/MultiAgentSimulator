# src/artifacts.py
import heapq
import math
import os
import random
import time
import hashlib
import uuid
import pygame
import requests
from collections import defaultdict
from typing import List, Tuple, Optional, Iterable, Dict
from src.config import OLLAMA_MODELS_DIR, OLLAMA_MODEL, OLLAMA_MODE, SLM_NUM_CTX
from src.resources import PowerResource, NetworkResource
import threading
from collections import deque

SLM_SEMAPHORE = threading.Semaphore(4)  # allow only 2 concurrent generations


# -------------------------
# World artifacts
# -------------------------


class Drone:
    """
    Drone supporting 8-neighbour A* planning (energy-aware) but moving in smooth
    Euclidean straight-line motion to the chosen world-space target when allow_direct=True.

    - A* used to compute energy-aware path when allow_direct=False (or obstacles added later).
    - Direct mode: drone moves straight to final cell center, producing smooth diagonals.
    - Energy accounting is based on Euclidean distance moved (cells = pixels / grid_size).
    """

    # energy cost constants (tuneable)
    BASE_COST_PER_CELL = 0.2  # energy units per cell travelled (empty)
    WEIGHT_FACTOR = 0.5  # additional factor per unit weight when carrying
    PICK_DROP_COST = 0.7  # energy units for a pick/drop (base) multiplied by weight factor

    def __init__(self,
                 start_cell: Tuple[int, int],
                 grid_size: int,
                 screen_size: Tuple[int, int],
                 terrain,
                 battery_capacity: float = 100.0,
                 allow_diagonal: bool = True,
                 allow_direct: bool = True,
                 agent_id: Optional[str] = None,
                 flight_recorder=None,
                 ):
        """
        :param start_cell: (col,row)
        :param grid_size: pixels per cell
        :param screen_size: (width, height) in pixels
        :param terrain: shared world state used to record this drone’s physical occupancy of grid cells.
        :param battery_capacity: initial battery energy units
        :param allow_diagonal: permit diagonal neighbour steps in A*
        :param allow_direct: if True prefer direct straight-line moves to final cell center
        :param agent_id: optional identifier string for this agent (useful for logs/planner snapshots)

        """
        self.inbox = []
        self.terrain = terrain
        self.agent_id = agent_id or None
        self.col, self.row = start_cell
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.pos = pygame.Vector2(self.col * grid_size + grid_size / 2,
                                  self.row * grid_size + grid_size / 2)

        # world-space movement
        self.target: Optional[pygame.Vector2] = None  # current world-space target (center of a cell)
        self.moving = False

        # carried parcel
        self.carrying = None  # Parcel reference

        # animation
        self.anim_t = 0.0
        self.anim_frame = 0

        # last action for UI
        self.last_action = None

        # resources (PowerResource and NetworkResource must exist in file)
        self.power = PowerResource(capacity=battery_capacity)
        # Consumption-rate multiplier for heterogeneous fleets (1.0 = the nominal model).
        self.energy_rate = 1.0
        # Flight speed relative to the nominal speed (1.0 = sim_clock.DRONE_SPEED_PX_S) for heterogeneous fleets; energy per cell does not depend on it.
        self.speed_scale = 1.0
        # How strongly carried weight raises consumption on this drone (the nominal model is WEIGHT_FACTOR).
        self.weight_factor = self.WEIGHT_FACTOR
        # Hardware reserve: energy the drone's pack is not meant to be drained below. A drone profile setting, the same
        # for every strategy; it is not enforced (nothing stops the drone) but defines what counts as usable energy.
        self.reserve_floor = 3.0
        self.network = NetworkResource(
            capacity=100,
            range_cells=1000,
            reliability=1.0,
            delay_range=(0, 0),
            flight_recorder=flight_recorder,  # ← PASS THROUGH
        )
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_failed = 0

        # Rolling window for traffic visualization
        self.sent_history = deque(maxlen=50)  # store timesteps of recent sends
        self.recv_history = deque(maxlen=50)

        # -------------------------
        # Onboard SLM (optional but embodied)
        # -------------------------
        self.slm_host = "localhost"  # always defined
        self.ollama_port = None  # assigned when simulation boots
        self.slm_base_url = None  # derived once port is known
        self.slm_model = OLLAMA_MODEL
        self.ollama_process = None
        self.slm_enabled = False

        # lost flag when battery drained
        self.lost = False

        # path / planning storage (adjacent cells)
        self.path: List[Tuple[int, int]] = []
        self._path_step_idx = 0

        # behavior flags
        self.allow_diagonal = bool(allow_diagonal)
        self.allow_direct = bool(allow_direct)
        self.last_cell = (self.col, self.row)
        self.network_active = False
        self.controller = None  # will be injected after construction

    # -------------------------
    # A* (energy-aware) helpers
    # -------------------------
    def _neighbours(self, c: int, r: int) -> Iterable[Tuple[int, int, float]]:
        """Yield valid neighbour cells and step distance in cell-units (1.0 cardinal, sqrt(2) diagonal)."""
        base = [
            (c - 1, r, 1.0), (c + 1, r, 1.0), (c, r - 1, 1.0), (c, r + 1, 1.0)
        ]
        if self.allow_diagonal:
            diag = math.sqrt(2.0)
            base += [(c - 1, r - 1, diag), (c - 1, r + 1, diag), (c + 1, r - 1, diag), (c + 1, r + 1, diag)]

        max_col = (self.screen_size[0] // self.grid_size) - 1
        max_row = (self.screen_size[1] // self.grid_size) - 1
        for nc, nr, step_cost in base:
            if 0 <= nc <= max_col and 0 <= nr <= max_row:
                yield nc, nr, step_cost

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int], euclidean: bool = True) -> float:
        """Admissible heuristic: Euclidean (when diagonal allowed) or Manhattan otherwise."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if euclidean and self.allow_diagonal:
            return math.hypot(dx, dy)
        return abs(dx) + abs(dy)

    def _astar_path(self, start: Tuple[int, int], goal: Tuple[int, int], carry_weight: float) -> List[Tuple[int, int]]:
        """
        A* minimizing energy (not raw steps). Returns list of adjacent cells from start(exclusive) to goal(inclusive).
        """
        if start == goal:
            return []

        weight_mul = (1.0 + carry_weight * self.WEIGHT_FACTOR)
        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        gscore: Dict[Tuple[int, int], float] = {start: 0.0}
        visited = set()

        while open_heap:
            f, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # reconstruct path (exclude start)
                rev = []
                cur = current
                while cur is not None and cur != start:
                    rev.append(cur)
                    cur = came_from.get(cur)
                rev.reverse()
                return rev

            for nc, nr, step_cell_dist in self._neighbours(current[0], current[1]):
                neighbor = (nc, nr)
                step_energy = step_cell_dist * self.BASE_COST_PER_CELL * weight_mul
                tentative_g = gscore[current] + step_energy

                if neighbor not in gscore or tentative_g + 1e-9 < gscore[neighbor]:
                    gscore[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = self._heuristic(neighbor, goal, euclidean=True) * self.BASE_COST_PER_CELL * weight_mul
                    heapq.heappush(open_heap, (tentative_g + h, neighbor))

        # unreachable (shouldn't happen on open grid)
        return []

    # -------------------------
    # Public movement API
    # -------------------------
    def clear_path(self):
        """Stop moving and clear any stored plan."""
        self.path = []
        self._path_step_idx = 0
        self.target = None
        self.moving = False

    def set_target_cell(self, col: int, row: int):
        """
        Compute a planned route and begin moving.
        - If allow_direct=True we prefer moving straight to the final goal center (smooth diagonal).
        - Otherwise compute A* adjacent-step path that accounts for carry weight.
        """
        if self.lost:
            return

        max_col = (self.screen_size[0] // self.grid_size) - 1
        max_row = (self.screen_size[1] // self.grid_size) - 1
        col = max(0, min(col, max_col))
        row = max(0, min(row, max_row))

        if (col, row) == (self.col, self.row):
            self.clear_path()
            return

        start = (int(self.col), int(self.row))
        goal = (col, row)
        carry_weight = self.carrying.weight if self.carrying else 0.0

        # Map with no-fly blocks (revision): fly the route the planners priced (straight when clear, else around the blocks)
        from src import obstacles
        if obstacles.active():
            here = (int(self.pos.x // self.grid_size), int(self.pos.y // self.grid_size))
            _, way = obstacles.route(here, goal)
            if way:
                self.path = list(way)
                self._path_step_idx = 0
                self.target = pygame.Vector2(self.path[0][0] * self.grid_size + self.grid_size / 2,
                                             self.path[0][1] * self.grid_size + self.grid_size / 2)
                self.moving = True
                return

        # Prefer direct straight-line move if enabled (smooth diagonal)
        if self.allow_direct:
            self.path = [goal]
            self._path_step_idx = 0
            self.target = pygame.Vector2(goal[0] * self.grid_size + self.grid_size / 2,
                                         goal[1] * self.grid_size + self.grid_size / 2)
            self.moving = True
            return

        # Otherwise compute energy-aware A* adjacent steps
        new_path = self._astar_path(start, goal, carry_weight)
        if not new_path:
            self.clear_path()
            return

        self.path = new_path
        self._path_step_idx = 0
        step_cell = self.path[self._path_step_idx]
        self.target = pygame.Vector2(step_cell[0] * self.grid_size + self.grid_size / 2,
                                     step_cell[1] * self.grid_size + self.grid_size / 2)
        self.moving = True

    def _advance_path_step(self):
        """Advance to next step (used when A* path is active)."""
        self._path_step_idx += 1
        if self._path_step_idx >= len(self.path):
            self.clear_path()
            return
        step_cell = self.path[self._path_step_idx]
        self.target = pygame.Vector2(step_cell[0] * self.grid_size + self.grid_size / 2,
                                     step_cell[1] * self.grid_size + self.grid_size / 2)
        self.moving = True

    # -------------------------
    # Movement + energy update
    # -------------------------
    def update(self, dt: float, speed: float, anim_fps: float,
               rot_frames_with_parcel: Optional[List[pygame.Surface]],
               rot_frames: Optional[List[pygame.Surface]]):
        """
        Called each frame by the game loop.
        - speed is pixels per second.
        - consumes energy proportional to Euclidean distance moved (cells = pixels / grid_size).
        """
        if self.lost:
            self.moving = False
            return

        if self.moving and self.target:
            direction = self.target - self.pos
            dist_pixels = direction.length()

            if dist_pixels == 0:
                self.arrive()
            else:
                step_pixels = speed * dt
                move = min(step_pixels, dist_pixels)
                cells_moved = (move / float(self.grid_size))
                carrying_weight = self.carrying.weight if self.carrying else 0.0
                energy_cost = self.energy_needed_for_cells(cells_moved, carrying_weight)
                self.power.consume(energy_cost)

                # if battery drains mid-move, set lost and place where it ran out
                if self.power.is_depleted():
                    if move > 0 and dist_pixels > 0:
                        self.pos += direction.normalize() * move
                    self.lost = True
                    self.moving = False
                    self.target = None
                    return

                # actually move
                if move >= dist_pixels - 1e-9:
                    self.pos = self.target
                    self.arrive()
                else:
                    self.pos += direction.normalize() * move

            # animate rotors while moving
            self.anim_t += dt
            if anim_fps > 0 and self.anim_t >= 1.0 / anim_fps:
                self.anim_t = 0.0
                if self.carrying and rot_frames_with_parcel:
                    self.anim_frame = (self.anim_frame + 1) % len(rot_frames_with_parcel)
                elif rot_frames:
                    self.anim_frame = (self.anim_frame + 1) % len(rot_frames)
                else:
                    self.anim_frame = 0
        else:
            self.anim_frame = 0

    def consume_hover(self, ratio: float, dt: float, speed_px_s: float):
        """
        Energy a drone that is in the air but not moving draws for one step: `ratio` times the power of flying empty at the nominal speed
        (BASE_COST_PER_CELL per cell at `speed_px_s`), times the same payload factor as flight. A multirotor holds itself up with its rotors, so
        waiting in the air is not free. `ratio` 0 (the default of the runner) reproduces the earlier model, in which waiting cost nothing.
        """
        if ratio <= 0.0 or self.lost:
            return 0.0
        carrying_weight = self.carrying.weight if self.carrying else 0.0
        cost = float(ratio) * self.energy_needed_for_cells(speed_px_s * dt / float(self.grid_size), carrying_weight)
        self.power.consume(cost)
        self.hover_energy_used = getattr(self, "hover_energy_used", 0.0) + cost
        if self.power.is_depleted():
            self.lost = True
            self.moving = False
            self.target = None
        return cost

    def hover_here(self):
        """
        A stop: the drone stays exactly where it is and commits the cell its position lies in (no snap to the cell centre, so a stop moves the drone by nothing).
        Its committed cell and the terrain occupancy then agree with its position.
        """
        self.col = int(self.pos.x // self.grid_size)
        self.row = int(self.pos.y // self.grid_size)
        self.last_cell = (self.col, self.row)
        self.target = None
        self.moving = False
        self.terrain.clear_agent_cell(self)
        self.terrain.register_agent_cell(self, self.col, self.row)

    def arrive(self):
        """
        Commit movement by snapping the drone to a grid cell and updating
        authoritative world occupancy.

        This method is the *only* place where physical presence is finalized:
        planners express intent, movement interpolates in continuous space,
        and Terrain records the committed cell on arrival.
        """

        # --------------------------------------------------
        # Remember previous cell for terrain bookkeeping
        # --------------------------------------------------
        prev_cell = self.last_cell

        # --------------------------------------------------
        # Snap continuous position to integer grid cell
        # --------------------------------------------------
        self.col = int(self.pos.x // self.grid_size)
        self.row = int(self.pos.y // self.grid_size)
        self.last_cell = (self.col, self.row)

        # Re-center world-space position exactly on the cell center
        self.pos = pygame.Vector2(
            self.col * self.grid_size + self.grid_size / 2,
            self.row * self.grid_size + self.grid_size / 2,
        )

        # Movement is complete at this point
        self.target = None
        self.moving = False

        # --------------------------------------------------
        # Authoritative terrain occupancy update
        # --------------------------------------------------
        # The drone informs the Terrain that it has physically left the
        # previous cell and now occupies the new one. This enforces
        # global spatial invariants (e.g., no two drones in one cell).
        self.terrain.clear_agent_cell(self)
        self.terrain.register_agent_cell(self, self.col, self.row)

        # --------------------------------------------------
        # A* adjacent-path progression (unchanged behavior)
        # --------------------------------------------------
        # If this arrival satisfies the next expected path step, advance;
        # otherwise, clear the path defensively to avoid desynchronization.
        if self.path and self._path_step_idx < len(self.path):
            expected = self.path[self._path_step_idx]
            if (self.col, self.row) == expected:
                self._advance_path_step()
            else:
                self.clear_path()


    # -------------------------
    # Pick / Drop (energy aware)
    # -------------------------
    def perform_pick(self, parcel: 'Parcel') -> bool:
        """
        Pick up a parcel from the current cell.

        This operation:
        - Consumes energy proportional to parcel weight
        - Removes the parcel from physical world occupancy
        - Marks the parcel as being carried by this drone

        Returns True on success, False if the pick fails.
        """
        if parcel is None:
            return False

        cost = self.energy_needed_for_pick_drop(parcel.weight)
        self.power.consume(cost)

        if self.power.is_depleted():
            self.lost = True
            return False

        # Parcel is no longer physically present in the world
        parcel.remove_from_world()

        # Drone now carries the parcel
        self.carrying = parcel
        self.last_action = ("pick", (self.col, self.row), parcel)

        return True

    def perform_drop(self, parcel: 'Parcel') -> bool:
        """
        Drop the carried parcel at the drone's current grid cell.

        This operation enforces physical world constraints:
        - A parcel cannot be dropped into an occupied cell
        - Energy is consumed only for valid physical actions
        - The parcel re-enters the world as a physical object
        - Battery depletion may still cause the drone to be lost

        Returns:
            True if the drop succeeds,
            False if the drop is blocked by physics or invalid input.
        """
        if parcel is None:
            return False

        # --------------------------------------------------
        # PHYSICS CHECK (authoritative)
        # --------------------------------------------------
        # A parcel cannot be placed into an occupied cell.
        # Occupancy includes delivered and undelivered parcels.
        if self.terrain.occupied_cell(self.col, self.row):
            self.last_action = ("drop_blocked", (self.col, self.row), parcel)
            return False

        # --------------------------------------------------
        # Energy cost for dropping the parcel
        # --------------------------------------------------
        cost = self.energy_needed_for_pick_drop(parcel.weight)
        self.power.consume(cost)

        # --------------------------------------------------
        # Commit parcel back into the world
        # --------------------------------------------------
        parcel.picked = False
        parcel.delivered = True
        parcel.place_at(self.col, self.row)

        # Drone no longer carries the parcel
        self.carrying = None
        self.last_action = ("drop", (self.col, self.row), parcel)

        # --------------------------------------------------
        # Battery depletion after action
        # --------------------------------------------------
        if self.power.is_depleted():
            self.lost = True

        return True

    # -------------------------
    # Distance & energy helpers
    # -------------------------
    def distance_cells_to(self, col: int, row: int, metric: str = "manhattan") -> float:
        """Distance between current integer cell and target in cell-units."""
        cur_col = int(self.col)
        cur_row = int(self.row)
        dx = abs(col - cur_col)
        dy = abs(row - cur_row)
        if metric == "chebyshev":
            return float(max(dx, dy))
        if metric == "euclidean":
            return math.hypot(dx, dy)
        return float(dx + dy)

    def to_snapshot(self) -> dict:
        """
        Return a compact planner-friendly snapshot for this drone.
        Mirrors the fields used by AIAgentController._make_terrain_snapshot.
        """
        return {
            "agent": {
                "id": self.agent_id,
                "col": int(self.col),
                "row": int(self.row),
                "battery_pct": int(self.power.percent()) if hasattr(self, "power") else 0,
            },
            "carrying": {
                "has": bool(self.carrying),
                "weight": float(self.carrying.weight) if self.carrying else 0.0,
            },
            "timestamp": time.time(),
        }

    def energy_needed_for_cells(self, n_cells: float, weight: float = 0.0) -> float:
        """Energy to travel `n_cells` cells while carrying `weight`."""
        return self.energy_rate * float(n_cells) * self.BASE_COST_PER_CELL * (1.0 + weight * self.weight_factor)

    def energy_needed_for_pick_drop(self, weight: float = 0.0) -> float:
        return self.energy_rate * float(self.PICK_DROP_COST) * (1.0 + weight * self.weight_factor)

    def can_reach_and_return(self, target_col: int, target_row: int, home_col: int, home_row: int) -> bool:
        """
        Conservative feasibility check using Euclidean distances when diagonal allowed.
        Adds small margin for pick/drop.
        """
        if self.allow_diagonal:
            dist_to_target = self.distance_cells_to(target_col, target_row, metric="euclidean")
            dist_target_to_home = self.distance_cells_to(home_col, home_row, metric="euclidean")
        else:
            dist_to_target = self.distance_cells_to(target_col, target_row, metric="manhattan")
            dist_target_to_home = self.distance_cells_to(home_col, home_row, metric="manhattan")

        carrying_weight = self.carrying.weight if self.carrying else 0.0
        needed = self.energy_needed_for_cells(dist_to_target, carrying_weight) + \
                 self.energy_needed_for_cells(dist_target_to_home, 0.0) + \
                 self.energy_needed_for_pick_drop(0.0)  # small margin
        return self.power.level >= needed

    # -------------------------
    # Drawing
    # -------------------------
    def draw(self, surf, images: dict):
        """
        images keys:
          - drone_static
          - drone_rot_frames (list)
          - drone_static_with_parcel
          - drone_rot_with_parcel_frames (list)
          - parcel_img
        """
        x, y = int(self.pos.x), int(self.pos.y)
        img = None

        if self.carrying:
            if self.moving and images.get("drone_rot_with_parcel_frames"):
                frames = images["drone_rot_with_parcel_frames"]
                img = frames[self.anim_frame % len(frames)]
            elif not self.moving and images.get("drone_static_with_parcel"):
                img = images["drone_static_with_parcel"]
            elif self.moving and images.get("drone_rot_frames"):
                frames = images["drone_rot_frames"]
                img = frames[self.anim_frame % len(frames)]
            else:
                img = images.get("drone_static")
        else:
            if self.moving and images.get("drone_rot_frames"):
                frames = images["drone_rot_frames"]
                img = frames[self.anim_frame % len(frames)]
            else:
                img = images.get("drone_static")

        if img:
            rect = img.get_rect(center=(x, y))
            surf.blit(img, rect)
        else:
            pygame.draw.circle(surf, (3, 54, 96), (x, y), int(self.grid_size * 0.35))

        # overlay parcel if carrying and no dedicated with-parcel image
        if self.carrying and not images.get("drone_static_with_parcel") and not images.get(
                "drone_rot_with_parcel_frames"):
            parcel_img = images.get("parcel_img")
            if parcel_img:
                rect = parcel_img.get_rect(center=(x, y + int(self.grid_size * 0.18)))
                surf.blit(parcel_img, rect)
            else:
                s = int(self.grid_size * 0.7 * 0.6)
                pygame.draw.rect(surf, (200, 160, 60),
                                 (x - s // 2, y + int(self.grid_size * 0.18) - s // 2, s, s))

        # ----------------------------
        # Battery icon and percent
        # ----------------------------
        try:
            pct = int(self.power.percent())
        except Exception:
            pct = 0

        # dimensions relative to grid size
        bar_w = max(28, int(self.grid_size * 0.28))
        bar_h = max(8, int(self.grid_size * 0.08))
        corner = 3

        # position the battery just above the drone image
        bar_x = x - bar_w // 2
        bar_y = y - int(self.grid_size * 0.45) - bar_h

        # outline
        outline_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
        try:
            pygame.draw.rect(surf, (30, 30, 30), outline_rect, border_radius=corner)
        except TypeError:
            pygame.draw.rect(surf, (30, 30, 30), outline_rect)
        # inner fill background (dark)
        inner_rect = pygame.Rect(bar_x + 2, bar_y + 2, bar_w - 4, bar_h - 4)
        try:
            pygame.draw.rect(surf, (60, 60, 60), inner_rect, border_radius=corner)
        except TypeError:
            pygame.draw.rect(surf, (60, 60, 60), inner_rect)

        # fill width based on pct
        fill_w = max(1, int((inner_rect.width) * max(0.0, pct) / 100.0))
        fill_rect = pygame.Rect(inner_rect.x, inner_rect.y, fill_w, inner_rect.height)

        if pct >= 60:
            fill_color = (80, 200, 80)  # green
        elif pct >= 30:
            fill_color = (240, 200, 80)  # yellow
        else:
            fill_color = (220, 80, 80)  # red

        try:
            pygame.draw.rect(surf, fill_color, fill_rect, border_radius=corner)
        except TypeError:
            pygame.draw.rect(surf, fill_color, fill_rect)

        # small percent text to the right of the battery
        try:
            fnt = pygame.font.SysFont("Consolas", max(10, int(self.grid_size * 0.08)))
            txt = fnt.render(f"{pct}%", True, (20, 20, 20))
            txt_pos = (bar_x + bar_w + 6, bar_y - 1)
            surf.blit(txt, txt_pos)
        except Exception:
            pass

        # If drone is lost, draw a small red X over it
        if getattr(self, "lost", False):
            sz = max(8, int(self.grid_size * 0.15))
            pygame.draw.line(surf, (200, 40, 40), (x - sz, y - sz), (x + sz, y + sz), 3)
            pygame.draw.line(surf, (200, 40, 40), (x + sz, y - sz), (x - sz, y + sz), 3)

    def draw_network_pulse(self, surf, min_cells: int = None, max_cells: int = None):
        """
        Visual-only network activity indicator.

        Draws a pulsing ring around the drone to represent
        communication activity and traffic intensity.

        Parameters
        ----------
        surf : pygame.Surface
            Target surface to draw on.
        min_cells : int | None
            Minimum pulse radius in grid cells. If None, derived from activity.
        max_cells : int | None
            Maximum pulse radius in grid cells. If None, derived from activity.
        """

        if not getattr(self, "network_active", False):
            return

        if not getattr(self, "network", None):
            return

        if self.power.is_depleted():
            return

        # ---------------------------------------
        # Activity-driven intensity
        # ---------------------------------------
        rate = self.recent_message_rate()

        # Normalize into 0.0 to 1.0
        intensity = min(1.0, rate / 0.2)  # 0.2 msgs/step = very busy

        # ---------------------------------------
        # Dynamic defaults if not explicitly given
        # ---------------------------------------
        if min_cells is None:
            min_cells = 1

        if max_cells is None:
            max_cells = int(2 + intensity * 6)  # busy drone gets wider pulse

        pulse_speed = 1.5 + intensity * 4.0

        cx, cy = int(self.pos.x), int(self.pos.y)

        # ---------------------------------------
        # Pulse animation
        # ---------------------------------------
        t = time.time()
        phase = 0.5 + 0.5 * math.sin(t * pulse_speed)

        inner_radius = min_cells * self.grid_size
        outer_radius = max_cells * self.grid_size
        radius = int(inner_radius + phase * (outer_radius - inner_radius))

        alpha = int(60 + 120 * intensity)

        overlay = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

        pygame.draw.circle(
            overlay,
            (120, 200, 255, alpha),
            (radius, radius),
            radius,
        )

        surf.blit(overlay, (cx - radius, cy - radius))

    def start_slm(self, port: int = 11434):
        import subprocess, os, time, requests, socket

        if OLLAMA_MODE == "external":
            self.slm_base_url = f"http://{self.slm_host}:11434"
            self.slm_enabled = True
            return

        self.slm_base_url = f"http://{self.slm_host}:{port}"

        def server_healthy():
            try:
                r = requests.get(f"{self.slm_base_url}/api/version", timeout=0.5)
                return r.status_code == 200
            except Exception:
                return False

        # If already running and healthy, reuse
        if server_healthy():
            print(f"[SLM] Reusing healthy server on port {port}")
            self.slm_enabled = True
            return

        # Start fresh server, no killing
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.slm_host}:{port}"
        env["OLLAMA_MODELS"] = os.path.abspath(os.path.expanduser(OLLAMA_MODELS_DIR))
        print(f"[SLM] Using model dir: {env['OLLAMA_MODELS']}")

        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Wait until healthy
        for _ in range(60):
            if server_healthy():
                self.slm_enabled = True
                return
            time.sleep(0.2)

        raise RuntimeError(f"Ollama failed to boot on port {port}")

    def stop_slm(self):
        """
        Cleanly shutdown the onboard SLM instance.
        """
        import os
        import signal

        if not self.slm_enabled or not self.ollama_process:
            return

        try:
            # Kill the whole process group to avoid orphaned children
            os.killpg(os.getpgid(self.ollama_process.pid), signal.SIGTERM)
            self.ollama_process.wait(timeout=5)

        except Exception:
            try:
                self.ollama_process.kill()
            except Exception:
                pass

        self.ollama_process = None
        self.ollama_port = None
        self.slm_enabled = False

    def slm_generate(self, prompt: str, timeout: int = 60, retries: int = 2, output_schema=None, num_predict: int = 256,
                     think: bool = False) -> str:
        if not self.slm_enabled or not self.slm_base_url:
            raise RuntimeError(f"SLM not active for drone {self.agent_id}")

        import time
        import requests

        base = self.slm_base_url.rstrip("/")
        if base.endswith("/api/generate"):
            url = base
        else:
            url = f"{base}/api/generate"

        print(f"[DEBUG] URL = {url}")

        payload = {
            "model": self.slm_model,
            "prompt": prompt,
            "stream": False,  # CRITICAL
            # think: the model reasons before it answers (a hybrid model such as qwen3:8b); the answer is then free text, because
            # constrained decoding would suppress the reasoning
            **({"think": True} if think else {}),
            **({"format": output_schema} if output_schema and not think else {}),  # constrained decoding (JSON schema)
            "options": {
                "num_ctx": SLM_NUM_CTX,  # the prompt and the reply must fit (see src/config.py)
                "num_predict": int(num_predict),  # bound the reply; the arbitration answer needs ~120 tokens
                "temperature": 0.0,
                "top_p": 0.1,
            },
        }

        headers = {
            "Content-Type": "application/json"
        }

        last_error = None

        for attempt in range(retries + 1):
            try:
                with SLM_SEMAPHORE:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=timeout,
                    )

                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

                try:
                    data = response.json()
                except Exception:
                    raise RuntimeError(f"Invalid JSON: {response.text[:200]}")

                if "response" not in data:
                    raise RuntimeError(f"Missing 'response' key: {data}")

                text = data["response"]

                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"Bad response: {text}")

                self.slm_last_usage = {"prompt_tokens": data.get("prompt_eval_count"), "gen_tokens": data.get("eval_count")}
                if (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0) >= SLM_NUM_CTX - 16:
                    raise RuntimeError(f"context window ({SLM_NUM_CTX} tokens) full: the prompt may have been cut")
                return text.strip()

            except Exception as e:
                last_error = e
                if attempt < retries:
                    backoff = 0.5 * (attempt + 1)
                    print(f"[SLM][{self.agent_id}] attempt {attempt + 1} failed: {e}. Retrying in {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    raise RuntimeError(
                        f"[SLM FAILURE][{self.agent_id}] after {retries + 1} attempts: {last_error}"
                    )

    def send_message(self, receiver, payload, current_step: int) -> bool:
        # print(f"[SEND] {self.agent_id} -> {receiver.agent_id} @ step {current_step}")
        success, _ = self.network.send(self, receiver, payload, current_step)
        if success:
            self.messages_sent += 1
            self.sent_history.append(current_step)
            # Instant visual feedback on send
            self.network_active = True
        else:
            self.messages_failed += 1

        return success

    def broadcast_message(self, receivers, payload, current_step: int) -> bool:
        """One radio transmission heard by every receiver in range."""
        success, _ = self.network.send_broadcast(self, receivers, payload, current_step)
        if success:
            self.messages_sent += 1
            self.sent_history.append(current_step)
            self.network_active = True
        else:
            self.messages_failed += 1
        return success

    def process_network(self, current_step: int):
        self.network.current_step = current_step
        # print(f"[NET-TICK] {self.agent_id} step={current_step}")
        deliveries = self.network.deliver(current_step)

        # Update visual state based on real network usage
        self.network_active = self.recent_message_rate() > 0

        for receiver, payload in deliveries:
            receiver.on_message(payload, sender=self)
        # print(f"[NETWORK] {self.agent_id} process_network called at step {current_step}")

    def on_message(self, payload, sender=None):
        # print(f"[NETWORK] {self.agent_id} received from {sender.agent_id if sender else None}: {payload}")
        self.inbox.append((sender.agent_id if sender else None, payload))
        self.messages_received += 1
        # print(f"[RECV] {self.agent_id} got from {sender.agent_id if sender else None}: {payload}")
        # Mark recent receive activity for visualization
        self.network_active = True  # instant pulse
        current_step = getattr(self.network, "current_step", None)
        if current_step is not None:
            self.recv_history.append(current_step)

    def recent_message_rate(self, window=50) -> float:
        events = list(self.sent_history) + list(self.recv_history)

        if not events:
            return 0.0

        events.sort()
        oldest = events[0]
        newest = events[-1]
        span = max(1, newest - oldest)

        return len(events) / span


class Parcel:
    def __init__(self, col: int, row: int, grid_size: int):
        """
         Inanimate world object representing a deliverable parcel.

        Parcels do not have agency or control logic. They only expose
        physical state (position, picked, delivered) which is interpreted
        and enforced by the Terrain.

        weight: relative weight scalar that increases energy cost when carried.
        """
        self.col = int(col)
        self.row = int(row)
        self.grid_size = int(grid_size)
        self.pos = pygame.Vector2(self.col * grid_size + grid_size / 2, self.row * grid_size + grid_size / 2)
        self.picked = False
        self.delivered = False
        self.weight = self._sample_weight()
        # Deterministic id (cell and weight): ids appear in strategy logs and model prompts, so a run must give the same
        # ids every time; they only need to be unique within a run (the terrain makes them so).
        self.id = hashlib.sha1(f"{self.col},{self.row},{self.weight!r}".encode()).hexdigest()[:8]
    # -------------------------
    # Physical presence helpers
    # -------------------------

    @staticmethod
    def _sample_weight() -> float:
        return random.uniform(0.5, 2.0)

    def place_at(self, col: int, row: int):
        """
        Commit this parcel to a physical grid cell.
        This is used when a drone drops the parcel back into the world.
        """
        self.col = int(col)
        self.row = int(row)
        self.pos = pygame.Vector2(
            self.col * self.grid_size + self.grid_size / 2,
            self.row * self.grid_size + self.grid_size / 2
        )

    def remove_from_world(self):
        """
        Mark this parcel as no longer physically present in the world.

        This corresponds to the parcel being picked up and carried
        by a drone. Terrain is expected to unregister the parcel
        separately.
        :return:
        """
        self.picked = True

    def draw(self, surf, parcel_img=None, parcel_scale=0.7):
        # Always draw delivered parcels, but visually distinct (they are not pickable)
        if self.delivered:
            if parcel_img:
                img = parcel_img.copy()
                img.set_alpha(180)
                rect = img.get_rect(center=(int(self.pos.x), int(self.pos.y)))
                surf.blit(img, rect)
            else:
                s = int(self.grid_size * parcel_scale * 0.5)
                rect = (int(self.pos.x) - s // 2, int(self.pos.y) - s // 2, s, s)
                pygame.draw.rect(surf, (140, 120, 60), rect)
                pygame.draw.rect(surf, (100, 100, 100), rect, 2)
            return

        # do not draw picked parcels (they are carried by drone)
        if self.picked:
            return

        if parcel_img:
            rect = parcel_img.get_rect(center=(int(self.pos.x), int(self.pos.y)))
            surf.blit(parcel_img, rect)
        else:
            s = int(self.grid_size * parcel_scale * 0.6)
            pygame.draw.rect(surf, (200, 160, 60),
                             (int(self.pos.x) - s // 2, int(self.pos.y) - s // 2, s, s))


class DeliveryStation:
    """Delivery station occupying a rectangular block of grid cells.

    Tracks per-cell usage counts in `usage` to allow controllers to select
    least-used cells and to avoid repeating the same cell every time.
    """

    def __init__(self, col: int, row: int, grid_size: int, w: int = 2, h: int = 2):
        self.col = int(col)
        self.row = int(row)
        self.w = max(1, int(w))
        self.h = max(1, int(h))
        self.grid_size = int(grid_size)
        center_col = col + (self.w - 1) / 2
        center_row = row + (self.h - 1) / 2
        self.pos = pygame.Vector2(center_col * grid_size + grid_size / 2,
                                  center_row * grid_size + grid_size / 2)
        self.delivered = 0  # counter for deliveries to this station
        # usage count per cell (col,row) => number of deliveries there
        self.usage = defaultdict(int)
        self.parked_drones = {}

    # ----------------------------
    # DRONE PARKING (AUTHORITATIVE)
    # ----------------------------

    def debug_state(self):
        return {
            "parked": dict(self.parked_drones),
            "usage": dict(self.usage),
        }

    def parking_cells(self):
        return [
            (c, r)
            for r in range(self.row, self.row + self.h)
            for c in range(self.col, self.col + self.w)
        ]

    def occupied_parking_cells(self):
        return set(self.parked_drones.values())

    def find_free_parking_cell(self):
        for cell in self.parking_cells():
            if cell not in self.occupied_parking_cells():
                return cell
        return None

    def park_drone(self, drone):
        # already parked
        if drone.agent_id in self.parked_drones:
            return self.parked_drones[drone.agent_id]

        print("Parked Drones", self.parked_drones)

        free = self.find_free_parking_cell()
        if free is None:
            return None

        self.parked_drones[drone.agent_id] = free
        return free

    def register_delivery(self, cell: Tuple[int, int]):
        """Record a delivery into a cell inside the station. Caller should ensure the cell belongs to the station."""
        self.usage[(int(cell[0]), int(cell[1]))] += 1
        self.delivered += 1

    def free_cells(self, terrain) -> List[Tuple[int, int]]:
        """Return list of station cells that are currently free of *any* non-picked parcel (delivered parcels also occupy)."""
        # NOTE: we treat delivered parcels as occupying the cell for drop-placement decisions,
        # because you wanted delivered parcels to remain visible and not be pickable; they should
        # also block future drops unless explicitly allowed.
        cells = [(c, r) for r in range(self.row, self.row + self.h) for c in range(self.col, self.col + self.w)]
        free = [c for c in cells if not terrain.occupied_cell(c[0], c[1])]
        return free

    def least_used_free_cell(
            self,
            terrain,
            ref_col: Optional[int] = None,
            ref_row: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Return the nearest free station cell to (ref_col, ref_row).
        If ref_col/ref_row are not provided, use the station center.

        Tie break order:
          1. Smaller Manhattan distance to reference point
          2. Lower usage count
          3. Row, then column for deterministic ordering
        """
        free = self.free_cells(terrain)
        if not free:
            return None

        if ref_col is None or ref_row is None:
            ref_col = self.col + self.w // 2
            ref_row = self.row + self.h // 2

        def sort_key(c: Tuple[int, int]) -> Tuple[int, int, int, int]:
            dist = abs(c[0] - ref_col) + abs(c[1] - ref_row)
            usage = self.usage.get((c[0], c[1]), 0)
            return (dist, usage, c[1], c[0])

        free.sort(key=sort_key)
        return free[0]

    def contains_cell(self, col: int, row: int) -> bool:
        return (self.col <= col < self.col + self.w) and (self.row <= row < self.row + self.h)

    def draw(self, surf):
        x = self.col * self.grid_size
        y = self.row * self.grid_size
        width = self.w * self.grid_size
        height = self.h * self.grid_size

        s = pygame.Surface((width, height), pygame.SRCALPHA)
        s.fill((30, 110, 200, 60))  # translucent blue
        surf.blit(s, (x, y))

        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surf, (40, 120, 200), rect, 3)

        font = pygame.font.SysFont("Consolas", max(12, self.grid_size // 6))
        txt = font.render(str(self.delivered), True, (255, 255, 255))
        surf.blit(txt, (x + 6, y + 6))


class Terrain:
    def __init__(self, grid_size: int, screen_size: Tuple[int, int], parcel_img=None, parcel_scale: float = 0.7):
        self.grid_size = int(grid_size)
        self.screen_size = screen_size
        self.parcel_img = parcel_img
        self.parcel_scale = parcel_scale
        self.parcels: List[Parcel] = []
        self.stations: List[DeliveryStation] = []
        self.command_center = None
        self.agent_cells = {}  # agent_id -> (col, row)
        self.agents = {}  # agent_id -> Drone

    def spawn_random(self, n: int):
        cols = self.screen_size[0] // self.grid_size
        rows = self.screen_size[1] // self.grid_size
        for _ in range(n):
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)
            # random weight between 0.5 and 2.0
            w = random.uniform(0.5, 2.0)
            self.add_parcel(c, r)

    def add_parcel(self, col: int, row: int):
        # only add if no parcel at cell at all (including delivered), and not inside station
        if self.parcel_at_cell(col, row, include_delivered=True) is None and not self.is_station_cell(col, row):
            parcel = Parcel(col, row, self.grid_size)
            taken = {q.id for q in self.parcels}
            while parcel.id in taken:   # two parcels with the same cell and weight: derive a distinct id
                parcel.id = hashlib.sha1((parcel.id + "+").encode()).hexdigest()[:8]
            self.parcels.append(parcel)

    def parcel_at_cell(self, col: int, row: int, include_delivered: bool = False) -> Optional[Parcel]:
        """
        Return parcel at (col,row) that is pickable (not picked and not delivered), unless include_delivered=True.
        """
        for p in self.parcels:
            if p.col == col and p.row == row:
                if p.picked:
                    continue
                if p.delivered and not include_delivered:
                    continue
                return p
        return None

    def register_agent_cell(self, drone, col: int, row: int):
        self.agent_cells[drone.agent_id] = (col, row)

    def clear_agent_cell(self, drone):
        self.agent_cells.pop(drone.agent_id, None)

    def occupied_cell(self, col: int, row: int) -> bool:
        """
        Return True if the cell contains any parcel that is not currently picked (includes delivered).
        This is the occupancy definition used for choosing drop cells.
        """
        for p in self.parcels:
            if p.col == col and p.row == row:
                if p.picked:
                    continue
                # if parcel exists here and is not picked (delivered or undelivered), consider occupied
                return True
        return False

    def add_station(self, col: int, row: int, w: int = 2, h: int = 2):
        max_col = (self.screen_size[0] // self.grid_size) - 1
        max_row = (self.screen_size[1] // self.grid_size) - 1
        col = max(0, min(col, max_col))
        row = max(0, min(row, max_row))
        w = max(1, min(w, max_col - col + 1))
        h = max(1, min(h, max_row - row + 1))
        self.stations.append(DeliveryStation(col, row, self.grid_size, w=w, h=h))

    def is_station_cell(self, col: int, row: int) -> bool:
        for s in self.stations:
            if s.contains_cell(col, row):
                return True
        return False

    def get_station_at(self, col: int, row: int) -> Optional[DeliveryStation]:
        for s in self.stations:
            if s.contains_cell(col, row):
                return s
        return None

    def nearest_station(self, col: int, row: int) -> Optional[DeliveryStation]:
        if not self.stations:
            return None
        best = None
        best_dist = None
        for s in self.stations:
            dx = s.pos.x - (col * self.grid_size + self.grid_size / 2)
            dy = s.pos.y - (row * self.grid_size + self.grid_size / 2)
            d = dx * dx + dy * dy
            if best is None or d < best_dist:
                best = s
                best_dist = d
        return best

    def draw(self, surf):
        for p in self.parcels:
            p.draw(surf, parcel_img=self.parcel_img, parcel_scale=self.parcel_scale)

        for s in self.stations:
            s.draw(surf)

        if self.command_center:
            self.command_center.draw(surf)


class CommandCenter:
    """
    Non-mobile coordination artifact.

    - Lives spatially in the world
    - Owns a NetworkResource
    - Has a communication range (in grid cells)
    - Draws itself using an image and its network radius
    - Maintains authoritative coordination via CentralisedStrategy
    """

    def __init__(
            self,
            col: int,
            row: int,
            grid_size: int,
            physics,
            network_capacity: int = 100,
            network_range: int = 12,
            variant: str = "greedy",
            image_path: str = "assets/command_center.png",
            flight_recorder=None,
            power_capacity: float = 10000.0,
    ):
        self.col = int(col)
        self.row = int(row)
        self.grid_size = int(grid_size)

        self.pos = pygame.Vector2(
            self.col * grid_size + grid_size / 2,
            self.row * grid_size + grid_size / 2,
        )

        self.network = NetworkResource(
            capacity=network_capacity,
            range_cells=network_range,
            reliability=0.95,
            delay_range=(1, 3),
            flight_recorder=flight_recorder,
        )

        self.agent_id = "command_center"
        self.power = PowerResource(capacity=power_capacity)

        self.messages_sent = 0
        self.messages_received = 0
        self.sent_history = deque(maxlen=100)

        self.terrain = None
        self.agents = {}
        self.physics = physics
        self._no_progress_ticks = 0
        self._last_sent_mode = {}
        self._directive_in_flight = set()
        self.globally_infeasible = False

        from src.strategies.centralised import CentralisedStrategy
        self.strategy = CentralisedStrategy(physics=physics,
                                            variant=variant
                                            )

        # Load and scale icon
        try:
            raw = pygame.image.load(image_path).convert_alpha()
            size = int(self.grid_size * 0.9)
            self.image = pygame.transform.smoothscale(raw, (size, size))
        except Exception as e:
            print(f"[COMMAND CENTER] Failed to load image '{image_path}': {e}")
            self.image = None

        # For visibility in the terrain
        self.is_visible = True
        self.rect = pygame.Rect(
            self.col * grid_size,
            self.row * grid_size,
            grid_size,
            grid_size
        )

    # =====================================================
    # Rendering
    # =====================================================

    def _draw_network_pulse(self, surf):
        """Draw the pulsing network radius visualization"""
        if not hasattr(self.network, 'channel') or not self.network.channel:
            return

        cx, cy = int(self.pos.x), int(self.pos.y)

        rate = self.recent_message_rate()
        intensity = min(1.0, rate / 2.0)

        inner_cells = 2
        outer_cells = int(self.network.channel.range_cells * (0.4 + 0.6 * intensity))

        inner_radius = inner_cells * self.grid_size
        outer_radius = outer_cells * self.grid_size

        pulse_speed = 1.5 + 3.0 * intensity
        t = time.time()
        phase = 0.5 + 0.5 * math.sin(t * pulse_speed)

        radius = int(inner_radius + phase * (outer_radius - inner_radius))
        alpha = int(60 + 160 * intensity)

        overlay = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

        pygame.draw.circle(
            overlay,
            (120, 170, 255, alpha),
            (radius, radius),
            radius,
        )

        pygame.draw.circle(
            overlay,
            (180, 220, 255, min(255, alpha + 40)),
            (radius, radius),
            radius,
            width=2,
        )

        surf.blit(overlay, (cx - radius, cy - radius))

    def draw(self, surf, camera=None):
        """Draw the command center with network pulse and battery"""
        if not self.is_visible:
            return

        cx, cy = int(self.pos.x), int(self.pos.y)

        # Apply camera offset if provided
        if camera:
            cx -= camera.x
            cy -= camera.y

        # Network pulse FIRST (so icon sits on top)
        if not self.power.is_depleted():
            self._draw_network_pulse(surf)

        # Command center icon
        if self.image:
            rect = self.image.get_rect(center=(cx, cy))
            surf.blit(self.image, rect)
        else:
            r = int(self.grid_size * 0.4)
            pygame.draw.circle(surf, (90, 160, 255), (cx, cy), r, 3)

        # Battery bar above command center
        try:
            pct = int(self.power.percent())
            bar_w = int(self.grid_size * 0.8)
            bar_h = 6

            bar_x = cx - bar_w // 2
            bar_y = cy - int(self.grid_size * 0.6)

            pygame.draw.rect(surf, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
            fill_w = int(bar_w * pct / 100)
            pygame.draw.rect(surf, (100, 220, 100), (bar_x, bar_y, fill_w, bar_h))
        except Exception:
            pass

    # =====================================================
    # World binding
    # =====================================================

    def bind_terrain(self, terrain):
        """Connect to the terrain for global state access"""
        self.terrain = terrain
        print(f"[COMMAND CENTER] Bound to terrain at ({self.col}, {self.row})")

    def register_agent(self, drone):
        """Register a drone for centralized coordination"""
        self.agents[drone.agent_id] = drone
        print(f"[CC][REGISTER] agent_id={drone.agent_id}")

    # =====================================================
    # Snapshot
    # =====================================================

    def snapshot(self) -> dict:
        """Build an authoritative global snapshot of the world"""
        if not self.terrain:
            return {"agents": [], "parcels": [], "stations": []}

        t = self.terrain

        return {
            "agents": [
                {
                    "id": a.agent_id,
                    "col": a.col,
                    "row": a.row,
                    "battery_pct": a.power.percent(),
                }
                for a in t.agents.values()
            ],
            "parcels": [
                {
                    "id": p.id,
                    "col": p.col,
                    "row": p.row,
                    "weight": getattr(p, "weight", 1.0),
                    "picked": p.picked,
                    "delivered": p.delivered,
                }
                for p in t.parcels
            ],
            "stations": [
                {
                    "col": s.col,
                    "row": s.row,
                    "w": s.w,
                    "h": s.h,
                }
                for s in t.stations
            ],
        }

    # =====================================================
    # Core update loop
    # =====================================================

    def update(self, agents, parcels, now=None):
        """
        Update command center state and perform task assignment.

        This method:
        1. Consumes power for operation
        2. Runs centralized strategy step (ledger sync + assignment)
        3. Proactively sends directives to agents with newly assigned tasks
        """
        if self.strategy.mission_complete:
            return
        if self.terrain is None:
            return

        self.power.consume(0.01)

        self.strategy.step(
            terrain=self.terrain,
            agents=agents,
            parcels=parcels,
            now=now,
        )

        # Proactive send for agents with assigned tasks
        for aid, record in self.strategy.agents.items():
            if aid in self._directive_in_flight:
                continue

            if record["state"] == "dead":
                continue

            # Agent has a task assigned but is still idle (message not sent yet)
            if record["state"] == "idle" and record.get("task"):
                agent_obj = self.agents.get(aid)
                if agent_obj:
                    snapshot = self.snapshot()
                    decision = self._decide_timed(snapshot, aid)

                    if decision.get("mode") == "plan":
                        success, reason = self.network.send(
                            sender=self,
                            receiver=agent_obj,
                            payload=decision,
                            current_step=self.physics.current_step if hasattr(self.physics, 'current_step') else 0
                        )

                        if success:
                            self._directive_in_flight.add(aid)
                            self.messages_sent += 1
                            self.sent_history.append(time.time())

                            # NOW transition to executing
                            record["state"] = "executing"

                            print(f"[CC][PROACTIVE-SEND] agent_id={aid} queued_for_delivery mode=plan")
                        else:
                            print(f"[CC][PROACTIVE-SEND] agent_id={aid} FAILED reason={reason}")

                            # Unassign on failure - CRITICAL FIX
                            task_id = record["task"]
                            if task_id in self.strategy.tasks:
                                print(f"[CC][RECOVERY] Unassigning task {task_id} due to proactive send failure")
                                self.strategy.tasks[task_id]["state"] = "unassigned"
                                self.strategy.tasks[task_id]["agent"] = None

                                # Clear assignment time to prevent stuck agent timeout
                                if task_id in self.strategy.task_assignment_times:
                                    del self.strategy.task_assignment_times[task_id]

                            record["task"] = None

    # =====================================================
    # Messaging
    # =====================================================

    def _get_agent(self, agent_id: str):
        """Resolve an agent_id to a registered Drone instance"""
        return self.agents.get(agent_id)

    def on_message(self, payload, sender=None):
        """A radio message from a drone reaches the ground station."""
        self.messages_received += 1
        kind = payload.get("type") if isinstance(payload, dict) else None
        aid = getattr(sender, "agent_id", None)
        if aid is None:
            return
        if kind == "request":
            decision = self.request_directive(aid)
            # the end of the mission (or a depleted or infeasible station) is also told to the drone over the radio
            if decision.get("mode") == "idle":
                step = self.physics.current_step if hasattr(self.physics, "current_step") else 0
                agent = self._get_agent(aid)
                if agent is not None:
                    self.network.send(sender=self, receiver=agent, payload=decision, current_step=step)
        elif kind == "completed":
            self.report_completion(aid)

    def report_completion(self, agent_id: str) -> None:
        """
        A drone reports that it has delivered its parcel: the ledger records the task as completed and the directive
        lock on that drone is cleared so it can be given the next task at once.
        """
        record = self.strategy.agents.get(agent_id)
        if record and record["state"] == "executing":
            parcel_id = record["task"]
            self.strategy.report_task_completed(agent_id, parcel_id)
            self._directive_in_flight.discard(agent_id)
            print(f"[CENTRAL][COMPLETE] agent={agent_id} task={parcel_id}")
            return parcel_id
        return None

    def _decide_timed(self, snapshot, agent_id):
        """strategy.decide, with the CPU time of the planner added up when the environment names a log file (DRONE_CPU_LOG; a measurement only, nothing else changes)."""
        log = os.environ.get("DRONE_CPU_LOG")
        if not log:
            return self.strategy.decide(snapshot, agent_id)
        if not hasattr(self, "_plan_cpu"):
            self._plan_cpu, self._plan_calls = 0.0, 0
            import atexit
            atexit.register(self._write_plan_cpu, log)
        t0 = time.process_time()
        try:
            return self.strategy.decide(snapshot, agent_id)
        finally:
            self._plan_cpu += time.process_time() - t0
            self._plan_calls += 1

    def _write_plan_cpu(self, log):
        import json
        with open(log, "a") as fh:
            fh.write(json.dumps({"agent": "command_center", "strategy": type(self.strategy).__name__, "think_cpu_s": self._plan_cpu, "calls": self._plan_calls}) + "\n")

    def request_directive(self, agent_id: str) -> dict:
        """
        Handle a directive request from an agent.

        This is the synchronous query path used when agents explicitly
        request instructions. It complements the proactive send mechanism
        in update().

        Returns
        -------
        dict
            Decision directive with mode and optional plan payload
        """
        if self.power.is_depleted():
            print(f"[CC][POWER] CommandCenter depleted. Cannot send directives.")
            return {"mode": "wait"}

        if self.strategy.mission_complete:
            return {"mode": "idle"}

        self.messages_received += 1

        agent = self._get_agent(agent_id)
        if agent is None:
            print(f"[CC][REQUEST] agent_id={agent_id} status=UNKNOWN_AGENT → WAIT")
            return {"mode": "wait"}

        if agent_id in self._directive_in_flight:
            return {"mode": "wait"}

        if self.globally_infeasible:
            return {"mode": "idle"}

        snapshot = self.snapshot()
        decision = self._decide_timed(snapshot, agent_id)

        if decision.get("mode") == "abort":
            print(f"[CC][ABORT] Global infeasibility detected. Signaling abort.")
            # self.globally_infeasible = True
            decision = {"mode": "idle"}

        success, reason = self.network.send(
            sender=self,
            receiver=agent,
            payload=decision,
            current_step=self.physics.current_step if hasattr(self.physics, 'current_step') else 0
        )

        if success:
            self._directive_in_flight.add(agent_id)
            self.messages_sent += 1
            self.sent_history.append(time.time())
            if hasattr(agent, 'network_active'):
                agent.network_active = True

            # Mark as executing if plan was sent
            if decision.get("mode") == "plan":
                agent_record = self.strategy.agents.get(agent_id)
                if agent_record:
                    agent_record["state"] = "executing"

            print(f"[CC][SEND] agent_id={agent_id} queued_for_delivery mode={decision.get('mode')}")
        else:
            print(f"[CC][SEND] agent_id={agent_id} FAILED reason={reason}")

            # CRITICAL FIX: Clean up on network failure
            if decision.get("mode") == "plan":
                agent_record = self.strategy.agents.get(agent_id)
                if agent_record and agent_record.get("task"):
                    task_id = agent_record["task"]

                    if task_id in self.strategy.tasks:
                        print(f"[CC][RECOVERY] Unassigning task {task_id} due to network loss")
                        self.strategy.tasks[task_id]["state"] = "unassigned"
                        self.strategy.tasks[task_id]["agent"] = None

                        # Clear assignment time to prevent stuck agent timeout
                        if task_id in self.strategy.task_assignment_times:
                            del self.strategy.task_assignment_times[task_id]

                    agent_record["task"] = None

        return {"mode": "wait"}

    def process_network(self, current_step: int):
        """Deliver all pending network messages"""
        deliveries = self.network.deliver(current_step)

        for receiver, payload in deliveries:
            if hasattr(receiver, 'on_message'):
                receiver.on_message(payload, sender=self)

            # Clear directive lock when message is delivered
            if hasattr(receiver, 'agent_id'):
                self._directive_in_flight.discard(receiver.agent_id)

    # =====================================================
    # Diagnostics
    # =====================================================

    def recent_message_rate(self, window=50):
        """Calculate recent message sending rate"""
        now = time.time()
        recent = [t for t in self.sent_history if now - t <= window]
        return len(recent) / window if window > 0 else 0

    # =====================================================
    # Terrain integration methods
    # =====================================================

    def get_position(self):
        """Get grid position for terrain integration"""
        return (self.col, self.row)

    def get_rect(self):
        """Get bounding rectangle for collision detection"""
        return pygame.Rect(
            self.col * self.grid_size,
            self.row * self.grid_size,
            self.grid_size,
            self.grid_size
        )
