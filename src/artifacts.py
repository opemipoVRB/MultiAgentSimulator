# src/artifacts.py

import random
import math
import heapq
from typing import List, Tuple, Optional, Iterable, Dict

from collections import defaultdict
import pygame


# -------------------------
# Resource classes
# -------------------------
class AgentResource:
    """Base resource class for agents (Power, Network, ...)."""

    def __init__(self, capacity: float):
        self.capacity = float(capacity)
        self.level = float(capacity)

    def percent(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return max(0.0, min(100.0, (self.level / self.capacity) * 100.0))

    def is_depleted(self) -> bool:
        return self.level <= 0.0

    def recharge(self, amount: float):
        self.level = min(self.capacity, self.level + float(amount))

    def consume(self, amount: float):
        """Consume amount (not percent). Negative values will recharge."""
        self.level -= float(amount)
        if self.level < 0.0:
            self.level = 0.0


class PowerResource(AgentResource):
    """Power resource measured in 'energy units'. Default capacity = 100."""

    def __init__(self, capacity: float = 100.0):
        super().__init__(capacity)


class NetworkResource(AgentResource):
    """Placeholder for future network/signal modeling."""

    def __init__(self, capacity: float = 100.0):
        super().__init__(capacity)


# -------------------------
# World artifacts
# -------------------------
class Parcel:
    def __init__(self, col: int, row: int, grid_size: int, weight: float = 1.0):
        """
        weight: relative weight scalar that increases energy cost when carried.
        """
        self.col = int(col)
        self.row = int(row)
        self.grid_size = int(grid_size)
        self.pos = pygame.Vector2(self.col * grid_size + grid_size / 2, self.row * grid_size + grid_size / 2)
        self.picked = False
        self.delivered = False
        self.weight = float(weight)

    def draw(self, surf, parcel_img=None, parcel_scale=0.7):
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

        if self.picked:
            return

        if parcel_img:
            rect = parcel_img.get_rect(center=(int(self.pos.x), int(self.pos.y)))
            surf.blit(parcel_img, rect)
        else:
            s = int(self.grid_size * parcel_scale * 0.6)
            pygame.draw.rect(surf, (200, 160, 60),
                             (int(self.pos.x) - s // 2, int(self.pos.y) - s // 2, s, s))


class Drone:
    """
    Drone supporting 8-neighbour A* planning (energy-aware) but moving in smooth
    Euclidean straight-line motion to the chosen world-space target when allow_direct=True.
    """

    BASE_COST_PER_CELL = 0.4
    WEIGHT_FACTOR = 0.8
    PICK_DROP_COST = 0.4

    def __init__(self,
                 start_cell: Tuple[int, int],
                 grid_size: int,
                 screen_size: Tuple[int, int],
                 battery_capacity: float = 100.0,
                 allow_diagonal: bool = True,
                 allow_direct: bool = True):
        """
        :param start_cell: (col,row)
        :param grid_size: pixels per cell
        :param screen_size: (width, height) in pixels
        :param battery_capacity: initial battery energy units
        :param allow_diagonal: permit diagonal neighbour steps in A*
        :param allow_direct: if True prefer direct straight-line moves to final cell center
        """
        self.col, self.row = start_cell
        self.grid_size = grid_size
        self.screen_size = screen_size
        self.pos = pygame.Vector2(self.col * grid_size + grid_size / 2,
                                  self.row * grid_size + grid_size / 2)

        self.target: Optional[pygame.Vector2] = None
        self.moving = False
        self.carrying = None
        self.anim_t = 0.0
        self.anim_frame = 0
        self._last_action = None

        self.power = PowerResource(capacity=battery_capacity)
        self.network = NetworkResource(capacity=100.0)
        self.lost = False
        self.at_station = True

        self.path: List[Tuple[int, int]] = []
        self._path_step_idx = 0

        self.allow_diagonal = bool(allow_diagonal)
        self.allow_direct = bool(allow_direct)

    # -------------------------
    # A* helpers
    # -------------------------
    def _neighbours(self, c: int, r: int) -> Iterable[Tuple[int, int, float]]:
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
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if euclidean and self.allow_diagonal:
            return math.hypot(dx, dy)
        return abs(dx) + abs(dy)

    def _astar_path(self, start: Tuple[int, int], goal: Tuple[int, int], carry_weight: float) -> List[Tuple[int, int]]:
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

        return []

    # -------------------------
    # Movement API
    # -------------------------
    def clear_path(self):
        self.path = []
        self._path_step_idx = 0
        self.target = None
        self.moving = False

    def current_target_cell(self) -> Optional[Tuple[int, int]]:
        if 0 <= self._path_step_idx < len(self.path):
            return self.path[self._path_step_idx]
        return None

    def set_target_cell(self, col: int, row: int):
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

        if self.allow_direct:
            self.path = [goal]
            self._path_step_idx = 0
            self.target = pygame.Vector2(goal[0] * self.grid_size + self.grid_size / 2,
                                         goal[1] * self.grid_size + self.grid_size / 2)
            self.moving = True
            return

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
        self._path_step_idx += 1
        if self._path_step_idx >= len(self.path):
            self.clear_path()
            return
        step_cell = self.path[self._path_step_idx]
        self.target = pygame.Vector2(step_cell[0] * self.grid_size + self.grid_size / 2,
                                     step_cell[1] * self.grid_size + self.grid_size / 2)
        self.moving = True

    # -------------------------
    # Update (movement + energy)
    # -------------------------
    def update(self, dt: float, speed: float, anim_fps: float,
               rot_frames_with_parcel: Optional[List[pygame.Surface]],
               rot_frames: Optional[List[pygame.Surface]]):
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

                if self.power.is_depleted():
                    if move > 0 and dist_pixels > 0:
                        self.pos += direction.normalize() * move
                    self.lost = True
                    self.moving = False
                    self.target = None
                    return

                if move >= dist_pixels - 1e-9:
                    self.pos = self.target
                    self.arrive()
                else:
                    self.pos += direction.normalize() * move

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

    def arrive(self):
        self.col = int(self.pos.x // self.grid_size)
        self.row = int(self.pos.y // self.grid_size)
        self.pos = pygame.Vector2(self.col * self.grid_size + self.grid_size / 2,
                                  self.row * self.grid_size + self.grid_size / 2)
        self.target = None
        self.moving = False

        if self.path and self._path_step_idx < len(self.path):
            expected = self.path[self._path_step_idx]
            if (self.col, self.row) == expected:
                self._advance_path_step()
            else:
                self.clear_path()

    # -------------------------
    # Pick / Drop
    # -------------------------
    def perform_pick(self, parcel: 'Parcel') -> bool:
        if parcel is None:
            return False
        cost = self.energy_needed_for_pick_drop(parcel.weight)
        self.power.consume(cost)
        if self.power.is_depleted():
            self.lost = True
            return False
        parcel.picked = True
        self.carrying = parcel
        self._last_action = ("pick", (self.col, self.row), parcel)
        return True

    def perform_drop(self, parcel: 'Parcel') -> bool:
        if parcel is None:
            return False
        cost = self.energy_needed_for_pick_drop(parcel.weight)
        self.power.consume(cost)
        if self.power.is_depleted():
            parcel.picked = False
            parcel.col = self.col
            parcel.row = self.row
            parcel.pos = pygame.Vector2(self.col * self.grid_size + self.grid_size / 2,
                                        self.row * self.grid_size + self.grid_size / 2)
            self.carrying = None
            self.lost = True
            self._last_action = ("drop", (self.col, self.row), parcel)
            return True

        parcel.picked = False
        parcel.col = self.col
        parcel.row = self.row
        parcel.pos = pygame.Vector2(self.col * self.grid_size + self.grid_size / 2,
                                    self.row * self.grid_size + self.grid_size / 2)
        self.carrying = None
        self._last_action = ("drop", (self.col, self.row), parcel)
        return True

    # -------------------------
    # Distance & energy helpers
    # -------------------------
    def distance_cells_to(self, col: int, row: int, metric: str = "manhattan") -> float:
        cur_col = int(self.col)
        cur_row = int(self.row)
        dx = abs(col - cur_col)
        dy = abs(row - cur_row)
        if metric == "chebyshev":
            return float(max(dx, dy))
        if metric == "euclidean":
            return math.hypot(dx, dy)
        return float(dx + dy)

    def energy_needed_for_cells(self, n_cells: float, weight: float = 0.0) -> float:
        return float(n_cells) * self.BASE_COST_PER_CELL * (1.0 + weight * self.WEIGHT_FACTOR)

    def energy_needed_for_pick_drop(self, weight: float = 0.0) -> float:
        return float(self.PICK_DROP_COST) * (1.0 + weight * self.WEIGHT_FACTOR)

    def can_reach_and_return(self, target_col: int, target_row: int, home_col: int, home_row: int) -> bool:
        if self.allow_diagonal:
            dist_to_target = self.distance_cells_to(target_col, target_row, metric="euclidean")
            dist_target_to_home = self.distance_cells_to(home_col, home_row, metric="euclidean")
        else:
            dist_to_target = self.distance_cells_to(target_col, target_row, metric="manhattan")
            dist_target_to_home = self.distance_cells_to(home_col, home_row, metric="manhattan")

        carrying_weight = self.carrying.weight if self.carrying else 0.0
        needed = self.energy_needed_for_cells(dist_to_target, carrying_weight) + \
                 self.energy_needed_for_cells(dist_target_to_home, 0.0) + \
                 self.energy_needed_for_pick_drop(0.0)
        return self.power.level >= needed

    # -------------------------
    # Drawing
    # -------------------------
    def draw(self, surf, images: dict):
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

        if self.carrying and not images.get("drone_static_with_parcel") and not images.get("drone_rot_with_parcel_frames"):
            parcel_img = images.get("parcel_img")
            if parcel_img:
                rect = parcel_img.get_rect(center=(x, y + int(self.grid_size * 0.18)))
                surf.blit(parcel_img, rect)
            else:
                s = int(self.grid_size * 0.7 * 0.6)
                pygame.draw.rect(surf, (200, 160, 60),
                                 (x - s // 2, y + int(self.grid_size * 0.18) - s // 2, s, s))

        try:
            pct = int(self.power.percent())
        except Exception:
            pct = 0

        bar_w = max(28, int(self.grid_size * 0.28))
        bar_h = max(8, int(self.grid_size * 0.08))
        corner = 3
        bar_x = x - bar_w // 2
        bar_y = y - int(self.grid_size * 0.45) - bar_h
        outline_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
        try:
            pygame.draw.rect(surf, (30, 30, 30), outline_rect, border_radius=corner)
        except TypeError:
            pygame.draw.rect(surf, (30, 30, 30), outline_rect)
        inner_rect = pygame.Rect(bar_x + 2, bar_y + 2, bar_w - 4, bar_h - 4)
        try:
            pygame.draw.rect(surf, (60, 60, 60), inner_rect, border_radius=corner)
        except TypeError:
            pygame.draw.rect(surf, (60, 60, 60), inner_rect)

        fill_w = max(1, int((inner_rect.width) * max(0.0, pct) / 100.0))
        fill_rect = pygame.Rect(inner_rect.x, inner_rect.y, fill_w, inner_rect.height)
        if pct >= 60:
            fill_color = (80, 200, 80)
        elif pct >= 30:
            fill_color = (240, 200, 80)
        else:
            fill_color = (220, 80, 80)
        try:
            pygame.draw.rect(surf, fill_color, fill_rect, border_radius=corner)
        except TypeError:
            pygame.draw.rect(surf, fill_color, fill_rect)

        try:
            fnt = pygame.font.SysFont("Consolas", max(10, int(self.grid_size * 0.08)))
            txt = fnt.render(f"{pct}%", True, (20, 20, 20))
            txt_pos = (bar_x + bar_w + 6, bar_y - 1)
            surf.blit(txt, txt_pos)
        except Exception:
            pass

        if getattr(self, "lost", False):
            sz = max(8, int(self.grid_size * 0.15))
            pygame.draw.line(surf, (200, 40, 40), (x - sz, y - sz), (x + sz, y + sz), 3)
            pygame.draw.line(surf, (200, 40, 40), (x + sz, y - sz), (x - sz, y + sz), 3)


class DeliveryStation:
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
        self.delivered = 0
        self.usage = defaultdict(int)

    def register_delivery(self, cell: Tuple[int, int]):
        self.usage[(int(cell[0]), int(cell[1]))] += 1
        self.delivered += 1

    def free_cells(self, terrain) -> List[Tuple[int, int]]:
        cells = [(c, r) for r in range(self.row, self.row + self.h) for c in range(self.col, self.col + self.w)]
        free = [c for c in cells if not terrain.occupied_cell(c[0], c[1])]
        return free

    def least_used_free_cell(self, terrain) -> Optional[Tuple[int, int]]:
        free = self.free_cells(terrain)
        if not free:
            return None
        free.sort(key=lambda c: (self.usage.get((c[0], c[1]), 0), c[1], c[0]))
        return free[0]

    def contains_cell(self, col: int, row: int) -> bool:
        return (self.col <= col < self.col + self.w) and (self.row <= row < self.row + self.h)

    def draw(self, surf):
        x = self.col * self.grid_size
        y = self.row * self.grid_size
        width = self.w * self.grid_size
        height = self.h * self.grid_size
        s = pygame.Surface((width, height), pygame.SRCALPHA)
        s.fill((30, 110, 200, 60))
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

    def spawn_random(self, n: int):
        cols = self.screen_size[0] // self.grid_size
        rows = self.screen_size[1] // self.grid_size
        for _ in range(n):
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)
            w = random.uniform(0.5, 2.0)
            self.add_parcel(c, r, weight=w)

    def add_parcel(self, col: int, row: int, weight: float = 1.0):
        if self.parcel_at_cell(col, row) is None and not self.is_station_cell(col, row):
            self.parcels.append(Parcel(col, row, self.grid_size, weight=weight))

    def parcel_at_cell(self, col: int, row: int, include_delivered: bool = False) -> Optional[Parcel]:
        for p in self.parcels:
            if p.col == col and p.row == row:
                if p.picked:
                    continue
                if p.delivered and not include_delivered:
                    continue
                return p
        return None

    def occupied_cell(self, col: int, row: int) -> bool:
        for p in self.parcels:
            if p.col == col and p.row == row:
                if p.picked:
                    continue
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
