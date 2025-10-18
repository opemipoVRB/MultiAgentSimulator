# src/controllers.py
import pygame
import random
import time
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
      - Ensures immediate pickup/drop is attempted as soon as the drone reaches the cell
        (fixes the "fly-by" without picking issue).
      - Prefers free cells inside nearest station and avoids repeating the same station cell when possible.
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

    # ---------------------------
    # Planner interaction
    # ---------------------------
    def _make_snapshot(self) -> Dict:
        """Create a compact snapshot of the world state for the planner input."""
        snap = {
            "agent": {"col": int(self.drone.col), "row": int(self.drone.row),
                      "battery_pct": int(self.drone.power.percent()) if hasattr(self.drone, "power") else 0},
            "carrying": {"has": bool(self.drone.carrying),
                         "weight": getattr(self.drone.carrying, "weight", 0.0) if self.drone.carrying else 0.0},
            "nearest_station": None,
            "all_parcels": [],
            "timestamp": time.time()
        }
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if station:
            snap["nearest_station"] = {"col": station.col, "row": station.row, "w": station.w, "h": station.h}

        for p in self.terrain.parcels:
            snap["all_parcels"].append({
                "col": p.col, "row": p.row, "weight": getattr(p, "weight", 1.0),
                "picked": getattr(p, "picked", False), "delivered": getattr(p, "delivered", False)
            })
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

        # debug log so you can inspect planner decisions
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
        When ALLOW_RISKY_TRIPS is True, require_return should be False for immediate legs.
        """
        # distance in cells (Manhattan) is conservative; Drone may use diagonal movement when moving.
        dist = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])

        needed = self.drone.energy_needed_for_cells(dist, carry_weight)

        if require_return:
            # determine nearest station as "home" for return calculation
            station = self.terrain.nearest_station(from_cell[0], from_cell[1])
            if station:
                home_col = int(station.col + station.w // 2)
                home_row = int(station.row + station.h // 2)
                dist_back = abs(to_cell[0] - home_col) + abs(to_cell[1] - home_row)
                needed += self.drone.energy_needed_for_cells(dist_back, 0.0)

        # If we allow risky trips, don't force a margin for return; just require energy >= needed (no extra margin).
        # If you want some margin, add a small constant here.
        return (self.drone.power.level >= needed)

    # ---------------------------
    # Delivery cell chooser
    # ---------------------------
    def _choose_delivery_cell(self) -> Tuple[int, int]:
        """
        Choose a delivery cell. Preference order:
         1. least-used free cell inside nearest station (if any)
         2. other free station cell avoiding repeating the last chosen
         3. random free cell on map avoiding last chosen if possible
         4. fallback to current drone cell
        Stores chosen into self._last_chosen_drop for biasing future choices.
        """
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)

        def _set_and_return(cell: Tuple[int, int]) -> Tuple[int, int]:
            self._last_chosen_drop = cell
            return cell

        if station:
            # prefer least-used free cell (station helper provided in artifacts if implemented)
            try:
                least_used = station.least_used_free_cell(self.terrain)
            except Exception:
                least_used = None

            if least_used:
                return _set_and_return(least_used)

            # otherwise pick a free cell but avoid repeating last chosen if possible
            cells = [(c, r) for r in range(station.row, station.row + station.h)
                     for c in range(station.col, station.col + station.w)]
            free_cells = [c for c in cells if not self.terrain.occupied_cell(c[0], c[1])]
            if free_cells:
                choices = [c for c in free_cells if c != self._last_chosen_drop]
                if not choices:
                    choices = free_cells
                return _set_and_return(random.choice(choices))

            # no free cells -> pick some station cell avoiding repeats
            choices = [c for c in cells if c != self._last_chosen_drop]
            if choices:
                return _set_and_return(random.choice(choices))
            return _set_and_return(random.choice(cells))

        # No station present: sample free cells on the map
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
            # relax after many attempts
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

        if self.drone.lost:
            return

        now = time.time()

        # --- IMPORTANT: If drone is not moving, attempt immediate pick/drop BEFORE any planning decisions.
        # This ensures a drone that just arrived on a parcel cell actually performs the pick,
        # and a drone that is carrying and on a valid drop cell performs the drop.
        if not self.drone.moving:
            # If not carrying, attempt local pickup if there is a parcel here
            if self.drone.carrying is None:
                p_here = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p_here:
                    # attempt pick immediately
                    print(f"[AI] arrived at parcel cell {(self.drone.col, self.drone.row)} - attempting pick (battery={int(self.drone.power.percent())}%)")
                    success = self.drone.perform_pick(p_here)
                    if not success:
                        # pick failed; mark UI flash via last_action is handled by perform_pick (returns False on lost)
                        self.drone._last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
                        # consider returning to station if possible
                        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                        if station:
                            home = (station.col + station.w // 2, station.row + station.h // 2)
                            self.drone.set_target_cell(*home)
                            self.state = "returning"
                        return
                    else:
                        # picked successfully; set controller state and exit so carrying branch handles drop next frame
                        self.state = "carrying"
                        self._last_action_time = now
                        return

            # If carrying, attempt immediate drop if we are on a station / drop cell and cell is free
            else:
                # if cell is within a station (or could be any free cell), drop now
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and station.contains_cell(self.drone.col, self.drone.row):
                    # only drop if cell is not occupied by another undelivered parcel
                    if not self.terrain.occupied_cell(self.drone.col, self.drone.row):
                        print(f"[AI] arrived at station cell {(self.drone.col, self.drone.row)} - attempting drop (battery={int(self.drone.power.percent())}%)")
                        parcel_ref = self.drone.carrying
                        self.drone.perform_drop(parcel_ref)
                        # record last chosen drop so chooser avoids repeats
                        self._last_chosen_drop = (self.drone.col, self.drone.row)
                        # advance plan if this was expected drop
                        step = self._current_step()
                        if step and tuple(step["dropoff"]) == (self.drone.col, self.drone.row):
                            self._advance_plan()
                        return
                # else: not in station or occupied -> don't force drop; let normal logic decide

        # ---- If we reach here we are either moving OR not moving but there's nothing to immediately pick/drop.
        # Now handle planning & navigation logic (request plan, follow plan, energy checks, etc.)

        # request plan if none or time to replan
        time_to_replan = (now - self._last_plan_time) > self._replanning_interval
        if (not self.plan) or time_to_replan:
            self._request_plan(force_refresh=False)

        # if drone currently moving wait for arrival (note: arrival frame above handles pick/drop)
        if self.drone.moving:
            return

        # ---------- carrying branch: aim to drop according to plan ----------
        if self.drone.carrying:
            step = self._current_step()
            # deduce planned dropoff: prefer plan, else nearest station center or chooser
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

            # If planned_drop is occupied or equals the last chosen drop, try alternatives
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

            # energy check: can we reach planned_drop while carrying?
            # When ALLOW_RISKY_TRIPS=True we only require energy to reach planned_drop (not to return)
            enough = self._ensure_energy_for_route(
                (self.drone.col, self.drone.row),
                planned_drop,
                carry_weight=getattr(self.drone.carrying, "weight", 0.0),
                require_return=(not ALLOW_RISKY_TRIPS)
            )

            if not enough:
                if not ALLOW_RISKY_TRIPS:
                    station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                    if station:
                        home = (station.col + station.w // 2, station.row + station.h // 2)
                        self.drone.set_target_cell(*home)
                        self.state = "returning"
                        return
                    else:
                        self.drone.lost = True
                        return
                else:
                    # ALLOW_RISKY_TRIPS -> attempt the trip anyway (drone may die en-route)
                    if not self.drone.moving:
                        print(f"[PLANNER] attempting risky drop to {planned_drop} battery={int(self.drone.power.percent())}%")
                        self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                    return

            # if at planned drop cell, drop (if still free)
            if (self.drone.col, self.drone.row) == planned_drop:
                if not self.terrain.occupied_cell(planned_drop[0], planned_drop[1]):
                    parcel_ref = self.drone.carrying
                    self.drone.perform_drop(parcel_ref)
                    # record chosen drop to bias away in future
                    self._last_chosen_drop = planned_drop
                    # station accounting (station.register_delivery) handled by game loop when receiving last_action
                    if step and tuple(step["dropoff"]) == planned_drop:
                        self._advance_plan()
                    return
                else:
                    # occupied unexpectedly, replan
                    self._request_plan(force_refresh=True)
                    return

            # otherwise, set target to planned_drop
            if not self.drone.moving:
                self.drone.set_target_cell(planned_drop[0], planned_drop[1])
            return

        # ---------- not carrying branch: attempt pickup steps ----------
        step = self._current_step()
        if not step:
            self.state = "idle"
            return

        pickup = tuple(step["pickup"])
        # verify pickup still available (not picked, not delivered)
        pobj = self.terrain.parcel_at_cell(pickup[0], pickup[1])
        if pobj is None or getattr(pobj, "delivered", False):
            # advance plan and consider replanning
            self._advance_plan()
            if (now - self._last_plan_time) > self._replanning_interval:
                self._request_plan(force_refresh=True)
            return

        dropoff = tuple(step["dropoff"])
        # energy checks: current->pickup then pickup->dropoff (with weight)
        # NOTE: when ALLOW_RISKY_TRIPS=True we do not require enough energy to also return home.
        can_reach_pick = self._ensure_energy_for_route((self.drone.col, self.drone.row), pickup,
                                                       carry_weight=0.0,
                                                       require_return=False)
        can_reach_after_pick = self._ensure_energy_for_route(pickup, dropoff,
                                                             carry_weight=step.get("weight", 1.0),
                                                             require_return=(not ALLOW_RISKY_TRIPS))
        if not (can_reach_pick and can_reach_after_pick):
            # if conservative mode, return to station to recharge if possible
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)
            if station and not ALLOW_RISKY_TRIPS:
                home = (station.col + station.w // 2, station.row + station.h // 2)
                self.drone.set_target_cell(*home)
                self.state = "returning"
                return
            else:
                # either replan or attempt risky pickup
                if ALLOW_RISKY_TRIPS:
                    if not self.drone.moving:
                        print(f"[PLANNER] attempting risky pickup at {pickup} battery={int(self.drone.power.percent())}%")
                        self.drone.set_target_cell(pickup[0], pickup[1])
                    return
                else:
                    self._request_plan(force_refresh=True)
                    return

        # if at pickup cell -> pick using drone.perform_pick (should be handled above already but keep as fallback)
        if (self.drone.col, self.drone.row) == pickup:
            success = self.drone.perform_pick(pobj)
            if success:
                # picked, now the carrying branch will run next frame
                self.state = "carrying"
                self._last_action_time = now
                return
            else:
                # pick failed (low battery) -> return home if possible
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    home = (station.col + station.w // 2, station.row + station.h // 2)
                    self.drone.set_target_cell(*home)
                    return
                else:
                    self.drone.lost = True
                    return

        # otherwise set target to pickup
        if not self.drone.moving:
            # validate target parcel hasn't been taken
            if pobj.picked or getattr(pobj, "delivered", False):
                # someone else took it; advance next loop
                self._advance_plan()
                return
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
