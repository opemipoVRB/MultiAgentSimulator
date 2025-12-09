# src/controllers.py
import pygame
import random
import time
from typing import Optional, Tuple, List, Dict

from llm_planner import PlannerClient

# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------

# Set to True if you want to use the LLM-based planner (when LangChain + OpenAI/Ollama
# is configured in llm_planner). If False, only local heuristic planner is used.
USE_LLM_PLANNER = False  # flip to True when you want the LLM

# Local planner behavior (these are used even if USE_LLM_PLANNER is False)
#   search_mode: "greedy" (single pass) or "multi" (multiple simulations)
LOCAL_SEARCH_MODE = "multi"  # "greedy" or "multi"
LOCAL_SEARCH_TRIALS = 8  # number of simulations for "multi"
LOCAL_MAX_ITEMS = 20  # max steps in a plan
LLM_BACKEND = "ollama",
# Create planner client with explicit configuration
PLANNER = PlannerClient(
    use_llm=USE_LLM_PLANNER,
    search_mode=LOCAL_SEARCH_MODE,
    num_trials=LOCAL_SEARCH_TRIALS,
    max_items=LOCAL_MAX_ITEMS,
    llm_backend=LLM_BACKEND,
)

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

        # Movement (only set a new target if drone is not already moving)
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

                # The human controller still uses a conservative check
                if hasattr(self.drone, "can_reach_and_return") and not self.drone.can_reach_and_return(
                        new_col, new_row, home_col, home_row
                ):
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
        self.plan: List[Dict] = []  # each item: {"pickup": (c,r), "dropoff": (c,r), "weight": float}
        self.plan_idx: int = 0
        self.no_feasible_plan = False
        self._no_plan_battery_threshold = 15.0
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
            "agent": {
                "col": int(self.drone.col),
                "row": int(self.drone.row),
                "battery_pct": int(self.drone.power.percent()) if hasattr(self.drone, "power") else 0,
            },
            "carrying": {
                "has": bool(self.drone.carrying),
                "weight": getattr(self.drone.carrying, "weight", 0.0) if self.drone.carrying else 0.0,
            },
            "nearest_station": None,
            "all_parcels": [],
            "timestamp": time.time(),
        }
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if station:
            snap["nearest_station"] = {"col": station.col, "row": station.row, "w": station.w, "h": station.h}

        for p in self.terrain.parcels:
            snap["all_parcels"].append(
                {
                    "col": p.col,
                    "row": p.row,
                    "weight": getattr(p, "weight", 1.0),
                    "picked": getattr(p, "picked", False),
                    "delivered": getattr(p, "delivered", False),
                }
            )
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
            if not self.plan:
                self.no_feasible_plan = True
            else:
                self.no_feasible_plan = False
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

        strategy = plan_obj.get("strategy", "unknown") if isinstance(plan_obj, dict) else "unknown"

        # debug log so you can inspect planner decisions
        try:
            print("[PLANNER] strategy:", strategy)
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

    def _ensure_energy_for_route(
            self,
            from_cell: Tuple[int, int],
            to_cell: Tuple[int, int],
            carry_weight: float = 0.0,
            require_return: bool = False,
    ) -> bool:
        """
        Conservative energy check for the leg from_cell -> to_cell.
        If require_return=True it also checks ability to return home (not used when ALLOW_RISKY_TRIPS=True).
        When ALLOW_RISKY_TRIPS is True, require_return should be False for immediate legs.
        """
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

        # If we allow risky trips, do not force a margin for return; just require energy >= needed.
        return self.drone.power.level >= needed

    # ---------------------------
    # Delivery cell chooser
    # ---------------------------
    def _choose_delivery_cell(self) -> Tuple[int, int]:
        """
        Choose a delivery cell. Preference order:

          1. Nearest free cell inside the nearest station to the drone.
          2. If no free cells, nearest station cell (even if occupied),
             avoiding repeating the last chosen cell when possible.
          3. If no station, pick a free non station cell on the map,
             avoiding last chosen when possible.
          4. Fallback to current drone cell.

        The chosen cell is stored in self._last_chosen_drop.
        """
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)

        def _set_and_return(cell: Tuple[int, int]) -> Tuple[int, int]:
            self._last_chosen_drop = cell
            return cell

        # ------------------------------
        # Case 1 and 2  station present
        # ------------------------------
        if station:
            # 1. Prefer nearest free cell inside this station
            try:
                nearest_free = station.least_used_free_cell(
                    self.terrain,
                    ref_col=self.drone.col,
                    ref_row=self.drone.row,
                )
            except Exception:
                nearest_free = None

            if nearest_free is not None:
                return _set_and_return(nearest_free)

            # 2. No free cells  choose any station cell, nearest to the drone
            cells = [
                (c, r)
                for r in range(station.row, station.row + station.h)
                for c in range(station.col, station.col + station.w)
            ]

            # Sort all station cells by distance to the drone, row, then col
            cells_sorted = sorted(
                cells,
                key=lambda c: (
                    abs(c[0] - self.drone.col) + abs(c[1] - self.drone.row),
                    c[1],
                    c[0],
                ),
            )

            # Try to avoid repeating the last chosen drop if we have options
            if (
                    self._last_chosen_drop is not None
                    and self._last_chosen_drop in cells_sorted
                    and len(cells_sorted) > 1
            ):
                cells_sorted = (
                        [c for c in cells_sorted if c != self._last_chosen_drop]
                        + [self._last_chosen_drop]
                )

            return _set_and_return(cells_sorted[0])

        # --------------------------------
        # Case 3  no station on the map
        # --------------------------------
        cols = self.terrain.screen_size[0] // self.terrain.grid_size
        rows = self.terrain.screen_size[1] // self.terrain.grid_size
        attempts = 0
        chosen: Optional[Tuple[int, int]] = None

        while attempts < 400:
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)

            # Avoid current cell
            if (c, r) == (self.drone.col, self.drone.row):
                attempts += 1
                continue

            # Avoid station cells entirely in this branch
            if self.terrain.is_station_cell(c, r):
                attempts += 1
                continue

            # Prefer free cells that are not the last chosen
            if (
                    not self.terrain.occupied_cell(c, r)
                    and (c, r) != self._last_chosen_drop
            ):
                chosen = (c, r)
                break

            # Relax after many attempts  accept any free cell
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

        # --------------------------------------------------
        # Local immediate actions when not moving
        # --------------------------------------------------
        if not self.drone.moving:
            # 1) Not carrying -> try to pick if parcel is here
            if self.drone.carrying is None:
                p_here = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p_here:
                    print(
                        f"[AI] arrived at parcel cell {(self.drone.col, self.drone.row)} "
                        f"- attempting pick (battery={int(self.drone.power.percent())}%)"
                    )
                    success = self.drone.perform_pick(p_here)
                    if not success:
                        # Pick failed, go to nearest station
                        self.drone._last_action = (
                            "pick_failed",
                            (self.drone.col, self.drone.row),
                            None,
                        )
                        station = self.terrain.nearest_station(
                            self.drone.col, self.drone.row
                        )
                        if station:
                            home = (
                                station.col + station.w // 2,
                                station.row + station.h // 2,
                            )
                            self.drone.set_target_cell(*home)
                            self.state = "returning"
                        return
                    else:
                        self.state = "carrying"
                        self._last_action_time = now
                        return

            else:
                # 2) Carrying -> if inside station, try to drop immediately or move
                #    only to the nearest free station cell.
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and station.contains_cell(self.drone.col, self.drone.row):
                    # If this cell is free, drop here
                    if not self.terrain.occupied_cell(self.drone.col, self.drone.row):
                        print(
                            f"[AI] arrived at station cell {(self.drone.col, self.drone.row)} "
                            f"- attempting drop (battery={int(self.drone.power.percent())}%)"
                        )
                        parcel_ref = self.drone.carrying
                        self.drone.perform_drop(parcel_ref)
                        self._last_chosen_drop = (self.drone.col, self.drone.row)

                        # Treat the plan dropoff as satisfied if it is in the same station
                        step = self._current_step()
                        if step:
                            planned_drop = tuple(step["dropoff"])
                            station_for_plan = self.terrain.nearest_station(
                                planned_drop[0], planned_drop[1]
                            )
                            if station_for_plan and station_for_plan is station:
                                self._advance_plan()
                        return
                    else:
                        # Current station cell is occupied, so move to the nearest free cell
                        alt = None
                        try:
                            alt = station.least_used_free_cell(
                                self.terrain,
                                ref_col=self.drone.col,
                                ref_row=self.drone.row,
                            )
                        except Exception:
                            alt = None

                        if alt is None:
                            # Fallback to global chooser if station helper failed
                            try:
                                alt = self._choose_delivery_cell()
                            except Exception:
                                alt = None

                        if alt and not self.drone.moving:
                            self.drone.set_target_cell(alt[0], alt[1])
                            self.state = "returning"
                        else:
                            # No alternative free cell found, just mark idle inside station
                            self.state = "idle"
                        return

        # --------------------------------------------------
        # Handle "no feasible plan" mode to avoid looping
        # --------------------------------------------------
        if self.no_feasible_plan:
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)
            if station and not station.contains_cell(self.drone.col, self.drone.row):
                if not self.drone.moving:
                    home = (
                        station.col + station.w // 2,
                        station.row + station.h // 2,
                    )
                    self.drone.set_target_cell(*home)
                    self.state = "returning"
            else:
                # Already at station or no station exists
                self.state = "idle"
            return

        # --------------------------------------------------
        # Request plan if none or time to replan
        # --------------------------------------------------
        time_to_replan = (now - self._last_plan_time) > self._replanning_interval

        if (not self.plan or time_to_replan) and not self.drone.moving and not self.no_feasible_plan:
            self._request_plan(force_refresh=False)

            # If planner returns no plan, enter no_feasible_plan mode once,
            # regardless of battery, to avoid the tight loop.
            if not self.plan:
                battery_pct = (
                    int(self.drone.power.percent())
                    if hasattr(self.drone, "power")
                    else 0
                )

                # Log reason once
                print(
                    f"[AI] planner returned empty plan at battery={battery_pct}%. "
                    f"Entering no_feasible_plan mode to avoid replan loop."
                )

                self.no_feasible_plan = True
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station and not station.contains_cell(
                        self.drone.col, self.drone.row
                ):
                    home = (
                        station.col + station.w // 2,
                        station.row + station.h // 2,
                    )
                    self.drone.set_target_cell(*home)
                    self.state = "returning"
                else:
                    self.state = "idle"
                return

        # If drone is moving, wait till arrival
        if self.drone.moving:
            return

        # --------------------------------------------------
        # Carrying branch: follow or adapt planned dropoff
        # --------------------------------------------------
        if self.drone.carrying:
            step = self._current_step()

            # Decide which station we are targeting from the plan, if any.
            target_station = None
            planned_drop = (self.drone.col, self.drone.row)

            if step:
                plan_drop = tuple(step["dropoff"])
                target_station = self.terrain.nearest_station(
                    plan_drop[0], plan_drop[1]
                )

            if target_station is None:
                target_station = self.terrain.nearest_station(
                    self.drone.col, self.drone.row
                )

            if target_station:
                # Always choose the nearest free cell in that station,
                # relative to current drone position.
                try:
                    nearest_free = target_station.least_used_free_cell(
                        self.terrain,
                        ref_col=self.drone.col,
                        ref_row=self.drone.row,
                    )
                except Exception:
                    nearest_free = None

                if nearest_free is not None:
                    planned_drop = nearest_free
                else:
                    # No free cell in this station  fall back to nearest station cell
                    cells = [
                        (c, r)
                        for r in range(target_station.row, target_station.row + target_station.h)
                        for c in range(target_station.col, target_station.col + target_station.w)
                    ]
                    cells.sort(
                        key=lambda c: (
                            abs(c[0] - self.drone.col)
                            + abs(c[1] - self.drone.row),
                            c[1],
                            c[0],
                        )
                    )
                    if cells:
                        planned_drop = cells[0]
            else:
                # No station at all  keep current position as drop
                planned_drop = (self.drone.col, self.drone.row)

            # If planned_drop is occupied or equals the last chosen drop, try alternatives
            if self.terrain.occupied_cell(planned_drop[0], planned_drop[1]) or (
                    self._last_chosen_drop is not None
                    and planned_drop == self._last_chosen_drop
            ):
                station_for_drop = self.terrain.nearest_station(
                    planned_drop[0], planned_drop[1]
                )
                if station_for_drop:
                    try:
                        alt = station_for_drop.least_used_free_cell(
                            self.terrain,
                            ref_col=self.drone.col,
                            ref_row=self.drone.row,
                        )
                    except Exception:
                        alt = None
                    if alt:
                        planned_drop = alt
                    else:
                        planned_drop = self._choose_delivery_cell()
                else:
                    planned_drop = self._choose_delivery_cell()

            # Check energy for route to planned_drop
            enough = self._ensure_energy_for_route(
                (self.drone.col, self.drone.row),
                planned_drop,
                carry_weight=getattr(self.drone.carrying, "weight", 0.0),
                require_return=(not ALLOW_RISKY_TRIPS),
            )

            if not enough:
                if not ALLOW_RISKY_TRIPS:
                    station = self.terrain.nearest_station(
                        self.drone.col, self.drone.row
                    )
                    if station:
                        home = (
                            station.col + station.w // 2,
                            station.row + station.h // 2,
                        )
                        self.drone.set_target_cell(*home)
                        self.state = "returning"
                        return
                    else:
                        self.drone.lost = True
                        return
                else:
                    if not self.drone.moving:
                        print(
                            f"[PLANNER] attempting risky drop to {planned_drop} "
                            f"battery={int(self.drone.power.percent())}%"
                        )
                        self.drone.set_target_cell(planned_drop[0], planned_drop[1])
                    return

            # If already at planned drop cell, try to drop if free
            if (self.drone.col, self.drone.row) == planned_drop:
                if not self.terrain.occupied_cell(planned_drop[0], planned_drop[1]):
                    parcel_ref = self.drone.carrying
                    self.drone.perform_drop(parcel_ref)
                    self._last_chosen_drop = planned_drop

                    step = self._current_step()
                    if step:
                        plan_drop = tuple(step["dropoff"])
                        st_plan = self.terrain.nearest_station(
                            plan_drop[0], plan_drop[1]
                        )
                        st_here = self.terrain.nearest_station(
                            planned_drop[0], planned_drop[1]
                        )
                        # If this drop is in the same station as the plan, the step is satisfied
                        if st_plan and st_here and st_plan is st_here:
                            self._advance_plan()
                    return
                else:
                    # The cell is occupied, replan drop target
                    self._request_plan(force_refresh=True)
                    return

            # Move towards planned drop cell
            if not self.drone.moving:
                self.drone.set_target_cell(planned_drop[0], planned_drop[1])
            return

        # --------------------------------------------------
        # Not carrying: follow pickup steps in the plan
        # --------------------------------------------------
        step = self._current_step()
        if not step:
            self.state = "idle"
            return

        pickup = tuple(step["pickup"])
        pobj = self.terrain.parcel_at_cell(pickup[0], pickup[1])

        # Parcel no longer available, advance plan
        if pobj is None or getattr(pobj, "delivered", False):
            self._advance_plan()
            if (now - self._last_plan_time) > self._replanning_interval:
                self._request_plan(force_refresh=True)
            return

        dropoff = tuple(step["dropoff"])

        # Check reachability
        can_reach_pick = self._ensure_energy_for_route(
            (self.drone.col, self.drone.row),
            pickup,
            carry_weight=0.0,
            require_return=False,
        )
        can_reach_after_pick = self._ensure_energy_for_route(
            pickup,
            dropoff,
            carry_weight=step.get("weight", 1.0),
            require_return=(not ALLOW_RISKY_TRIPS),
        )

        if not (can_reach_pick and can_reach_after_pick):
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)
            if station and not ALLOW_RISKY_TRIPS:
                home = (station.col + station.w // 2, station.row + station.h // 2)
                self.drone.set_target_cell(*home)
                self.state = "returning"
                return
            else:
                if ALLOW_RISKY_TRIPS:
                    if not self.drone.moving:
                        print(
                            f"[PLANNER] attempting risky pickup at {pickup} "
                            f"battery={int(self.drone.power.percent())}%"
                        )
                        self.drone.set_target_cell(pickup[0], pickup[1])
                    return
                else:
                    self._request_plan(force_refresh=True)
                    return

        # If at pickup cell, perform pick
        if (self.drone.col, self.drone.row) == pickup:
            success = self.drone.perform_pick(pobj)
            if success:
                self.state = "carrying"
                self._last_action_time = now
                return
            else:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    home = (station.col + station.w // 2, station.row + station.h // 2)
                    self.drone.set_target_cell(*home)
                    return
                else:
                    self.drone.lost = True
                    return

        # Otherwise set target to pickup
        if not self.drone.moving:
            if pobj.picked or getattr(pobj, "delivered", False):
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
