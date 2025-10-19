# src/controllers.py
import pygame
import random
import time
import math
from typing import Optional, Tuple, List, Dict

from llm_planner import PlannerClient

# create planner client: set use_llm=True if you installed LangChain and have credentials.
PLANNER = PlannerClient(use_llm=False)

# If True, AI will attempt trips even if it cannot guarantee a return-to-base.
# The drone will be allowed to attempt pickup/drop and may become lost if battery drains mid-route.
ALLOW_RISKY_TRIPS = True


class BaseAgentController:
    """Base controller API used by the game loop."""
    def __init__(self, drone, terrain):
        self.drone = drone
        self.terrain = terrain

    def handle_event(self, event):
        pass

    def update(self, dt: float):
        pass


class HumanAgentController(BaseAgentController):
    """Manual control for the drone (keyboard)."""
    def __init__(self, drone, terrain):
        super().__init__(drone, terrain)
        self._last_keys = pygame.key.get_pressed()

    def handle_event(self, event):
        pass

    def update(self, dt: float):
        keys = pygame.key.get_pressed()
        if self.drone.lost:
            return

        # Movement (only set a new target if drone isn't already moving)
        if not self.drone.moving:
            dx = 0
            dy = 0
            if keys[pygame.K_LEFT]:
                dx -= 1
            if keys[pygame.K_RIGHT]:
                dx += 1
            if keys[pygame.K_UP]:
                dy -= 1
            if keys[pygame.K_DOWN]:
                dy += 1

            if dx != 0 or dy != 0:
                new_col = self.drone.col + dx
                new_row = self.drone.row + dy
                # ensure energy to move and optionally return home
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    home_col = int(station.col + station.w // 2)
                    home_row = int(station.row + station.h // 2)
                else:
                    home_col, home_row = self.drone.col, self.drone.row

                # The human controller still uses a conservative check (so player doesn't instantly run out).
                if hasattr(self.drone, "can_reach_and_return") and not self.drone.can_reach_and_return(
                        new_col, new_row, home_col, home_row):
                    # insufficient energy for that move + return -> send home instead
                    self.drone.set_target_cell(home_col, home_row)
                else:
                    self.drone.set_target_cell(new_col, new_row)

        # Edge-detect SPACE for pick/drop (human triggered)
        if keys[pygame.K_SPACE] and not self._last_keys[pygame.K_SPACE]:
            if self.drone.carrying is None:
                p = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p:
                    success = self.drone.perform_pick(p)
                    if not success:
                        # pick failed due to battery -> pick_failed UI flash
                        self.drone._last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
            else:
                # drop only if target cell has no non-picked parcel (includes delivered)
                if self.terrain.occupied_cell(self.drone.col, self.drone.row):
                    self.drone._last_action = ("drop_failed", (self.drone.col, self.drone.row), None)
                else:
                    self.drone.perform_drop(self.drone.carrying)

        self._last_keys = keys


class AIAgentController(BaseAgentController):
    """
    Planner-driven AI controller.

    Behavior:
      - Maintains a plan (list of pickup/drop pairs) returned by PLANNER.
      - Periodically requests replans.
      - Revalidates pickups/dropoffs, checks battery feasibility for the *immediate* leg
        (if ALLOW_RISKY_TRIPS=True it will NOT require return-to-base energy).
      - Ensures immediate pickup/drop is attempted as soon as the drone reaches the cell.
      - Prefers free cells inside nearest station and avoids repeating the same station cell when possible.

    CHANGE: When planner returns no plan, controller will fall back to a simple local behavior:
      - If there are undelivered parcels in the field, pick the nearest parcel and head there.
      - `fallback_active` and `fallback_target` expose that state for HUD/debugging.
    """
    def __init__(self, drone, terrain, search_radius: Optional[int] = None, replanning_interval: float = 4.0):
        super().__init__(drone, terrain)
        self.state = "idle"
        self.plan: List[Dict] = []   # each item: {"pickup": (c,r), "dropoff": (c,r), "weight": float}
        self.plan_idx: int = 0
        self.cooldown = 0.0
        self.search_radius = search_radius
        self._last_plan_time = 0.0
        self._replanning_interval = replanning_interval
        self._last_action_time = 0.0
        self.last_narration: Optional[str] = None
        # used to bias away from repeating the very last chosen drop cell
        self._last_chosen_drop: Optional[Tuple[int, int]] = None

        # Fallback state exposed for HUD and debugging
        self.fallback_active: bool = False
        self.fallback_target: Optional[Tuple[int, int]] = None

    # -------------
    # Helpers
    # -------------
    def _field_parcels_remaining(self) -> bool:
        """Return True if any parcel in the terrain is not delivered (still in the field)."""
        for p in self.terrain.parcels:
            # treat undelivered parcels in the world as "in the field". Picked parcels are not in the field.
            if not getattr(p, "delivered", False) and not getattr(p, "picked", False):
                return True
        return False

    # ---------------------------
    # Planner interaction
    # ---------------------------
    def _make_snapshot(self) -> Dict:
        """
        Create a compact snapshot of the world state for the planner input.
        """
        snap = {
            "agent": {
                "col": int(self.drone.col),
                "row": int(self.drone.row),
                "battery_pct": int(self.drone.power.percent()) if hasattr(self.drone, "power") else 0,
                "battery_level": float(self.drone.power.level) if hasattr(self.drone, "power") else None,
            },
            "carrying": {
                "has": bool(self.drone.carrying),
                "weight": getattr(self.drone.carrying, "weight", 0.0) if self.drone.carrying else 0.0
            },
            "nearest_station": None,
            "all_parcels": [],
            "timestamp": time.time(),
            "energy_model": {
                "base_cost_per_cell": float(getattr(self.drone, "BASE_COST_PER_CELL", 0.2)),
                "weight_factor": float(getattr(self.drone, "WEIGHT_FACTOR", 0.5)),
                "pick_drop_cost": float(getattr(self.drone, "PICK_DROP_COST", 0.7)),
                "allow_diagonal": bool(getattr(self.drone, "allow_diagonal", True))
            },
            "grid": {
                "grid_size": int(self.terrain.grid_size),
                "cols": int(self.terrain.screen_size[0] // self.terrain.grid_size),
                "rows": int(self.terrain.screen_size[1] // self.terrain.grid_size),
            }
        }

        # nearest station as before
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if station:
            snap["nearest_station"] = {"col": int(station.col), "row": int(station.row),
                                       "w": int(station.w), "h": int(station.h),
                                       "center": (int(station.col + station.w // 2), int(station.row + station.h // 2))
                                       }

        # helper to compute euclidean distance in cell units
        def _euclid_cell_dist(a_col, a_row, b_col, b_row):
            dx = float(b_col - a_col)
            dy = float(b_row - a_row)
            return (dx * dx + dy * dy) ** 0.5

        base = snap["energy_model"]["base_cost_per_cell"]
        wfactor = snap["energy_model"]["weight_factor"]

        def est_energy_for_cells(n_cells, weight):
            return float(n_cells) * base * (1.0 + weight * wfactor)

        # For each parcel, include estimates so LLM doesn't have to guess mapping battery%->energy
        for p in self.terrain.parcels:
            parcel_info = {
                "col": int(p.col),
                "row": int(p.row),
                "weight": float(getattr(p, "weight", 1.0)),
                "picked": bool(getattr(p, "picked", False)),
                "delivered": bool(getattr(p, "delivered", False))
            }

            adist = _euclid_cell_dist(self.drone.col, self.drone.row, p.col, p.row)
            parcel_info["dist_agent_cells_euclidean"] = round(adist, 3)

            if station:
                sc = int(station.col + station.w // 2)
                sr = int(station.row + station.h // 2)
                sdist = _euclid_cell_dist(p.col, p.row, sc, sr)
                parcel_info["dist_to_station_cells_euclidean"] = round(sdist, 3)
                est = est_energy_for_cells(adist, 0.0) + est_energy_for_cells(sdist, parcel_info["weight"])
                parcel_info["est_energy_agent_pickup_and_deliver_to_station"] = round(est, 3)
            else:
                parcel_info["dist_to_station_cells_euclidean"] = None
                parcel_info["est_energy_agent_pickup_and_deliver_to_station"] = None

            md = abs(int(self.drone.col) - int(p.col)) + abs(int(self.drone.row) - int(p.row))
            parcel_info["dist_agent_cells_manhattan"] = int(md)

            snap["all_parcels"].append(parcel_info)

        return snap

    def _request_plan(self, force_refresh: bool = False):
        """Ask PLANNER for a plan given current snapshot. Handles parsing and some truncation for narration."""
        snap = self._make_snapshot()
        try:
            plan_obj = PLANNER.request_plan(snap, force_refresh=force_refresh)
        except Exception as ex:
            # Planner failed: log and keep empty plan
            print("[PLANNER] request failed:", ex)
            self.plan = []
            self.plan_idx = 0
            self.last_narration = None
            self._last_plan_time = time.time()
            return

        plan_list = plan_obj.get("plan", []) if isinstance(plan_obj, dict) else []
        parsed = []
        for step in plan_list:
            try:
                pickup = (int(step["pickup"][0]), int(step["pickup"][1]))
                dropoff = (int(step["dropoff"][0]), int(step["dropoff"][1]))
                weight = float(step.get("weight", 1.0))
                parsed.append({"pickup": pickup, "dropoff": dropoff, "weight": weight})
            except Exception:
                continue

        self.plan = parsed
        self.plan_idx = 0
        self._last_plan_time = time.time()

        # narration trimming - keep it short for UI
        raw_n = plan_obj.get("narration", "") if isinstance(plan_obj, dict) else ""
        sentences = [s.strip() for s in raw_n.replace("\n", " ").split(".") if s.strip()]
        truncated = ". ".join(sentences[:3])
        if len(truncated) > 300:
            truncated = truncated[:297].rstrip() + "..."
        self.last_narration = truncated

        try:
            print("[PLANNER] plan received:", self.plan)
            print("[PLANNER] narration:", self.last_narration)
        except Exception:
            pass

    # ---------------------------
    # Plan helpers
    # ---------------------------
    def _current_step(self) -> Optional[Dict]:
        if 0 <= self.plan_idx < len(self.plan):
            return self.plan[self.plan_idx]
        return None

    def _advance_plan(self):
        self.plan_idx += 1
        if self.plan_idx >= len(self.plan):
            # plan exhausted
            self.plan = []
            self.plan_idx = 0
            self.state = "idle"

    def _ensure_energy_for_route(self, from_cell: Tuple[int, int], to_cell: Tuple[int, int],
                                 carry_weight: float = 0.0, require_return: bool = False) -> bool:
        """
        Conservative energy check for the leg from_cell -> to_cell.
        If require_return=True it also checks ability to return home (not used when ALLOW_RISKY_TRIPS=True).
        When ALLOW_RISKY_TRIPS is True, the controller will allow immediate legs (bypass check).
        """
        # If risky mode and only checking immediate leg, allow the trip unconditionally.
        if ALLOW_RISKY_TRIPS and not require_return:
            return True

        # conservative Manhattan estimate for energy needed (controller-side)
        dist = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])
        needed = self.drone.energy_needed_for_cells(dist, carry_weight)

        if require_return:
            station = self.terrain.nearest_station(from_cell[0], from_cell[1])
            if station:
                home_col = int(station.col + station.w // 2)
                home_row = int(station.row + station.h // 2)
                dist_back = abs(to_cell[0] - home_col) + abs(to_cell[1] - home_row)
                needed += self.drone.energy_needed_for_cells(dist_back, 0.0)

        return (self.drone.power.level >= needed)

    # ---------------------------
    # Delivery cell chooser
    # ---------------------------
    def _choose_delivery_cell(self) -> Tuple[int, int]:
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)

        def _set_and_return(cell: Tuple[int, int]) -> Tuple[int, int]:
            self._last_chosen_drop = cell
            return cell

        if station:
            try:
                least_used = station.least_used_free_cell(self.terrain)
            except Exception:
                least_used = None

            if least_used:
                return _set_and_return(least_used)

            cells = [(c, r) for r in range(station.row, station.row + station.h)
                     for c in range(station.col, station.col + station.w)]
            free_cells = [c for c in cells if not self.terrain.occupied_cell(c[0], c[1])]
            if free_cells:
                choices = [c for c in free_cells if c != self._last_chosen_drop]
                if not choices:
                    choices = free_cells
                return _set_and_return(random.choice(choices))

            choices = [c for c in cells if c != self._last_chosen_drop]
            if choices:
                return _set_and_return(random.choice(choices))
            return _set_and_return(random.choice(cells))

        cols = self.terrain.screen_size[0] // self.terrain.grid_size
        rows = self.terrain.screen_size[1] // self.terrain.grid_size
        attempts = 0
        chosen = None
        while attempts < 400:
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)
            if (c, r) == (self.drone.col, self.drone.row):
                attempts += 1
                continue
            if self.terrain.is_station_cell(c, r):
                attempts += 1
                continue
            if not self.terrain.occupied_cell(c, r) and (c, r) != self._last_chosen_drop:
                chosen = (c, r)
                break
            if attempts > 50 and not self.terrain.occupied_cell(c, r):
                chosen = (c, r)
                break
            attempts += 1

        if chosen is None:
            chosen = (self.drone.col, self.drone.row)
        return _set_and_return(chosen)

    # ---------------------------
    # Main update loop
    # ---------------------------
    def update(self, dt: float):
        # cooldown timer
        if self.cooldown > 0:
            self.cooldown -= dt

        # reset fallback indicators at start of tick; set True only if we go into fallback branch
        self.fallback_active = False
        self.fallback_target = None

        if self.drone.lost:
            return

        now = time.time()

        # Immediate pick/drop when not moving
        if not self.drone.moving:
            # If not carrying, attempt local pickup if there is a parcel here
            if self.drone.carrying is None:
                p_here = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p_here:
                    print(f"[AI] arrived at parcel cell {(self.drone.col, self.drone.row)} - attempting pick (battery={int(self.drone.power.percent())}%)")
                    success = self.drone.perform_pick(p_here)
                    if not success:
                        self.drone._last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
                        # consider returning to station only when there are NO field parcels remaining
                        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                        if station and not self._field_parcels_remaining():
                            home = (station.col + station.w // 2, station.row + station.h // 2)
                            self.drone.set_target_cell(*home)
                            self.state = "returning"
                        return
                    else:
                        self.state = "carrying"
                        self._last_action_time = now
                        return

            # If carrying, attempt immediate drop if we are on a station / drop cell and cell is free
            else:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and station.contains_cell(self.drone.col, self.drone.row):
                    if not self.terrain.occupied_cell(self.drone.col, self.drone.row):
                        print(f"[AI] arrived at station cell {(self.drone.col, self.drone.row)} - attempting drop (battery={int(self.drone.power.percent())}%)")
                        parcel_ref = self.drone.carrying
                        self.drone.perform_drop(parcel_ref)
                        self._last_chosen_drop = (self.drone.col, self.drone.row)
                        step = self._current_step()
                        if step and tuple(step["dropoff"]) == (self.drone.col, self.drone.row):
                            self._advance_plan()
                        return

        # Request a plan if needed
        time_to_replan = (now - self._last_plan_time) > self._replanning_interval
        if (not self.plan) or time_to_replan:
            self._request_plan(force_refresh=False)

        # If drone currently moving wait for arrival (arrival handles pick/drop)
        if self.drone.moving:
            return

        # ---------- carrying branch ----------
        if self.drone.carrying:
            # clear fallback when following plan
            self.fallback_active = False
            self.fallback_target = None

            step = self._current_step()
            if step:
                planned_drop = tuple(step["dropoff"])
            else:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    try:
                        alt = station.least_used_free_cell(self.terrain)
                    except Exception:
                        alt = None
                    if alt:
                        planned_drop = alt
                    else:
                        planned_drop = (station.col + station.w // 2, station.row + station.h // 2) if station else (self.drone.col, self.drone.row)
                else:
                    planned_drop = (self.drone.col, self.drone.row)

            if self.terrain.occupied_cell(planned_drop[0], planned_drop[1]) or \
                    (self._last_chosen_drop is not None and planned_drop == self._last_chosen_drop):
                station_for_drop = self.terrain.nearest_station(planned_drop[0], planned_drop[1])
                if station_for_drop:
                    try:
                        alt = station_for_drop.least_used_free_cell(self.terrain)
                    except Exception:
                        alt = None
                    if alt:
                        planned_drop = alt
                    else:
                        planned_drop = self._choose_delivery_cell()
                else:
                    planned_drop = self._choose_delivery_cell()

            enough = self._ensure_energy_for_route(
                (self.drone.col, self.drone.row),
                planned_drop,
                carry_weight=getattr(self.drone.carrying, "weight", 0.0),
                require_return=(not ALLOW_RISKY_TRIPS)
            )

            if not enough:
                # Only auto-return to station if there are NO undelivered field parcels left.
                if not ALLOW_RISKY_TRIPS:
                    station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                    if station and not self._field_parcels_remaining():
                        home = (station.col + station.w // 2, station.row + station.h // 2)
                        self.drone.set_target_cell(*home)
                        self.state = "returning"
                        return
                    else:
                        if self._field_parcels_remaining():
                            if ALLOW_RISKY_TRIPS:
                                if not self.drone.moving:
                                    print(f"[PLANNER] attempting risky drop to {planned_drop} battery={int(self.drone.power.percent())}%")
                                    self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                                return
                            else:
                                self._request_plan(force_refresh=True)
                                return
                else:
                    if not self.drone.moving:
                        print(f"[PLANNER] attempting risky drop to {planned_drop} battery={int(self.drone.power.percent())}%")
                        self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                    return

            # at planned drop cell?
            if (self.drone.col, self.drone.row) == planned_drop:
                if not self.terrain.occupied_cell(planned_drop[0], planned_drop[1]):
                    parcel_ref = self.drone.carrying
                    self.drone.perform_drop(parcel_ref)
                    self._last_chosen_drop = planned_drop
                    if step and tuple(step["dropoff"]) == planned_drop:
                        self._advance_plan()
                    return
                else:
                    self._request_plan(force_refresh=True)
                    return

            # otherwise set target
            if not self.drone.moving:
                self.drone.set_target_cell(planned_drop[0], planned_drop[1])
            return

        # ---------- not carrying branch ----------
        step = self._current_step()

        # If planner gave no step, but parcels remain in the field, head to the nearest parcel (fallback).
        if not step:
            if self._field_parcels_remaining():
                best = None
                best_d = None
                for p in self.terrain.parcels:
                    if getattr(p, "delivered", False) or getattr(p, "picked", False):
                        continue
                    dx = (p.col - self.drone.col)
                    dy = (p.row - self.drone.row)
                    d = math.hypot(dx, dy)
                    if best is None or d < best_d:
                        best = p
                        best_d = d

                if best:
                    # set fallback state for HUD
                    self.fallback_active = True
                    self.fallback_target = (best.col, best.row)

                    # go get it (ALLOW_RISKY_TRIPS bypasses conservative blocking)
                    if not self.drone.moving:
                        print(f"[AI] no plan returned but field parcels remain -> heading to nearest parcel at ({best.col},{best.row}) battery={int(self.drone.power.percent())}%")
                        self.drone.set_target_cell(best.col, best.row)
                        self.state = "seeking"
                    return

            # truly idle (no plan and no parcels)
            self.state = "idle"
            return

        # If pickup planned, validate parcel still there
        pickup = tuple(step["pickup"])
        pobj = self.terrain.parcel_at_cell(pickup[0], pickup[1])
        if pobj is None or getattr(pobj, "delivered", False):
            self._advance_plan()
            if (now - self._last_plan_time) > self._replanning_interval:
                self._request_plan(force_refresh=True)
            return

        dropoff = tuple(step["dropoff"])
        can_reach_pick = self._ensure_energy_for_route((self.drone.col, self.drone.row), pickup,
                                                       carry_weight=0.0,
                                                       require_return=False)
        can_reach_after_pick = self._ensure_energy_for_route(pickup, dropoff,
                                                             carry_weight=step.get("weight", 1.0),
                                                             require_return=(not ALLOW_RISKY_TRIPS))
        if not (can_reach_pick and can_reach_after_pick):
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)
            if station and not ALLOW_RISKY_TRIPS and not self._field_parcels_remaining():
                home = (station.col + station.w // 2, station.row + station.h // 2)
                self.drone.set_target_cell(*home)
                self.state = "returning"
                return
            else:
                if ALLOW_RISKY_TRIPS:
                    if not self.drone.moving:
                        print(f"[PLANNER] attempting risky pickup at {pickup} battery={int(self.drone.power.percent())}%")
                        self.drone.set_target_cell(pickup[0], pickup[1])
                    return
                else:
                    self._request_plan(force_refresh=True)
                    return

        # if at pickup cell -> pick
        if (self.drone.col, self.drone.row) == pickup:
            success = self.drone.perform_pick(pobj)
            if success:
                self.state = "carrying"
                self._last_action_time = now
                return
            else:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and not self._field_parcels_remaining():
                    home = (station.col + station.w // 2, station.row + station.h // 2)
                    self.drone.set_target_cell(*home)
                    return
                else:
                    self.drone.lost = True
                    return

        # otherwise set target to pickup
        if not self.drone.moving:
            if pobj.picked or getattr(pobj, "delivered", False):
                self._advance_plan()
                return
            # no fallback in this branch (we follow the plan)
            self.drone.set_target_cell(pickup[0], pickup[1])
            self.state = "seeking"
            return


class ControllerSwitcher:
    """Simple switcher between controllers (Human/AI). TAB cycles."""
    def __init__(self, controllers: List[BaseAgentController]):
        assert controllers, "provide at least one controller"
        self.controllers = controllers
        self.index = 0

    @property
    def current(self) -> BaseAgentController:
        return self.controllers[self.index]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self.index = (self.index + 1) % len(self.controllers)
        else:
            self.current.handle_event(event)

    def update(self, dt: float):
        self.current.update(dt)
