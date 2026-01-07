# src/controllers.py
import os
import pygame
import random
import time
from typing import Optional, Tuple, List, Dict

from strategies.naive import NaiveStrategy

# ---------------------------------------------------------------------------
# Strategy configuration (Naive decentralised baseline)
# ---------------------------------------------------------------------------

NAIVE_NUM_TRIALS = 10  # Monte Carlo greedy rollouts per agent

STRATEGY = NaiveStrategy(
    num_trials=NAIVE_NUM_TRIALS,
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

    def update(self, dt: float, step: int, sim_time: float):
        pass


class HumanAgentController(BaseAgentController):
    """Manual control for the drone (keyboard)."""

    def __init__(self, drone, terrain):
        super().__init__(drone, terrain)
        self._last_keys = pygame.key.get_pressed()

    def handle_event(self, event):
        pass

    def update(self, dt: float, step: int, sim_time: float):
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
                if callable(getattr(self.drone, "can_reach_and_return", None)):
                    if not self.drone.can_reach_and_return(
                            new_col, new_row, home_col, home_row
                    ):
                        self.drone.set_target_cell(home_col, home_row)
                    else:
                        self.drone.set_target_cell(new_col, new_row)
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

    def __init__(self, drone, terrain, strategy, flight_recorder=None):
        super().__init__(drone, terrain)

        self.state = "idle"
        self.plan: List[Dict] = []  # each item: {"pickup": (c,r), "dropoff": (c,r), "weight": float}
        self.plan_idx: int = 0
        self.no_feasible_plan = False
        self._no_plan_battery_threshold = 15.0
        self.cooldown = 0.0
        self._last_plan_time = 0.0
        self._last_action_time = 0.0
        self.last_narration: Optional[str] = None
        # used to bias away from repeating the very last chosen drop cell
        self._last_chosen_drop: Optional[Tuple[int, int]] = None
        self.parked = False
        self.parking_in_progress = False
        self.parking_target = None  # (col, row)
        self.plan_confidence: float = 0.0
        self.flight_recorder = flight_recorder
        self.strategy = strategy

        print(f"[INIT] AI controller for {drone.agent_id}, parked={self.parked}")
        print(
            f"[INIT] AI controller for {drone.agent_id} "
            f"strategy={type(strategy).__name__}"
        )

    # ---------------------------
    # Planner interaction
    # ---------------------------
    def _make_snapshot(self) -> Dict:
        """Create a compact snapshot of the world state for the planner input."""
        snap = {
            "agent": {
                "id": self.drone.agent_id,
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

    def _trim_narration(self, text: str) -> Optional[str]:
        if not text:
            return None

        sentences = [
            s.strip()
            for s in text.replace("\n", " ").split(".")
            if s.strip()
        ]

        truncated = ". ".join(sentences[:3])
        if len(truncated) > 300:
            truncated = truncated[:297].rstrip() + "..."

        return truncated

    def _load_plan(self, plan_obj: Dict):
        """
        Install a strategy-provided plan into the controller.

        This method is the SINGLE commit point where:
          - execution state is initialized
          - confidence is captured
          - narration is prepared
          - flight recording (if any) is triggered

        The strategy has already decided WHAT to do.
        This method only handles HOW the agent will execute it.
        """

        # -------------------------------------------------
        # Normalize input (defensive against legacy planners)
        # -------------------------------------------------
        if isinstance(plan_obj, list):
            # Legacy shape: raw list of steps
            plan_obj = {
                "plan": plan_obj,
                "confidence": self.plan_confidence,
                "narration": None,
                "strategy": "unknown",
            }

            print(
                f"[PLAN-WARN] agent={self.drone.agent_id} "
                f"received legacy plan list. Coercing into plan envelope."
            )

        if not isinstance(plan_obj, dict):
            print(
                f"[PLAN-ERROR] agent={self.drone.agent_id} "
                f"invalid plan object type={type(plan_obj)}. Ignoring."
            )
            self.no_feasible_plan = True
            return

        # -------------------------------------------------
        # Parse plan steps
        # -------------------------------------------------
        plan_list = plan_obj.get("plan", [])
        parsed: List[Dict] = []

        for step in plan_list:
            try:
                pickup = (int(step["pickup"][0]), int(step["pickup"][1]))
                dropoff = (int(step["dropoff"][0]), int(step["dropoff"][1]))
                weight = float(step.get("weight", 1.0))

                parsed.append(
                    {
                        "pickup": pickup,
                        "dropoff": dropoff,
                        "weight": weight,
                    }
                )
            except Exception:
                continue

        # -------------------------------------------------
        # Install execution state
        # -------------------------------------------------
        self.plan = parsed
        self.plan_idx = 0
        self.no_feasible_plan = not bool(self.plan)
        self._last_plan_time = time.time()

        # -------------------------------------------------
        # Capture metadata
        # -------------------------------------------------
        self.plan_confidence = float(
            plan_obj.get("confidence", self.plan_confidence)
        )

        raw_narration = plan_obj.get("narration", "")
        self.last_narration = self._trim_narration(raw_narration)

        # -------------------------------------------------
        # Flight recorder (PRIOR belief)
        # -------------------------------------------------
        if self.flight_recorder and self.plan:
            self.flight_recorder.record_initial_plan(
                agent_id=self.drone.agent_id,
                plan=self.plan,
                confidence=self.plan_confidence,
                projected_battery=plan_obj.get("projected_battery_remaining"),
                step=0,
                sim_time=0.0,
            )

        # -------------------------------------------------
        # Debug logging
        # -------------------------------------------------
        try:
            print(
                f"[PLAN-COMMIT] agent={self.drone.agent_id} "
                f"steps={len(self.plan)} "
                f"confidence={self.plan_confidence:.3f} "
                f"strategy={plan_obj.get('strategy', 'unknown')}"
            )
            if self.last_narration:
                print(f"[PLAN-NARRATION] {self.last_narration}")
        except Exception:
            pass

    # ---------------------------
    # Plan helpers
    # ---------------------------
    def _current_step(self) -> Optional[Dict]:
        if 0 <= self.plan_idx < len(self.plan):
            return self.plan[self.plan_idx]
        return None

    def _advance_plan(self, *, step: int, sim_time: float):
        """
        Advance the current plan step.

        If reactive fallback is enabled, attempt to locally replace the
        current step with a feasible alternative before advancing.
        """

        if getattr(self.strategy, "enable_reactive_fallback", False):
            progress = {
                "current_cell": (int(self.drone.col), int(self.drone.row)),
                "battery_pct": float(self.drone.power.percent()),
                "attempted_parcels": set(),
                "failed_parcels": set(),
            }

            # Snapshot the station (planner must never see live objects)
            station_obj = self.terrain.nearest_station(self.drone.col, self.drone.row)
            station = None
            if station_obj:
                station = {
                    "col": int(station_obj.col),
                    "row": int(station_obj.row),
                    "w": int(station_obj.w),
                    "h": int(station_obj.h),
                }

            next_choice = STRATEGY.choose_best_feasible_next_parcel(
                progress=progress,
                parcels=[
                    {
                        "id": getattr(p, "id", None),
                        "col": p.col,
                        "row": p.row,
                        "weight": getattr(p, "weight", 1.0),
                        "picked": getattr(p, "picked", False),
                        "delivered": getattr(p, "delivered", False),
                    }
                    for p in self.terrain.parcels
                ],
                station=station,
                known_weight=True,
            )

            if next_choice:
                old_step = self.plan[self.plan_idx]

                self.plan[self.plan_idx] = {
                    "pickup": tuple(next_choice["pickup"]),
                    "dropoff": tuple(next_choice["dropoff"]),
                    "weight": next_choice["weight"],
                }

                # ---------------------------------------------
                # Posterior confidence update (belief revision)
                # ---------------------------------------------
                evidence = next_choice.get("posterior_evidence")
                if evidence is not None:
                    prior = self.plan_confidence

                    modifier = STRATEGY.compute_posterior_confidence_modifier(
                        prior_confidence=prior,
                        evidence=evidence,
                    )

                    posterior = prior * modifier
                    self.plan_confidence = posterior

                    if self.flight_recorder:
                        # 1) Log structural change FIRST
                        self.flight_recorder.record_plan_change(
                            agent_id=self.drone.agent_id,
                            old_step=old_step,
                            new_step=self.plan[self.plan_idx],
                            estimated_cost=next_choice.get("estimated_cost"),
                            battery_before=progress["battery_pct"],
                            battery_after=self.drone.power.percent(),
                            prior_confidence=prior,
                            posterior_confidence=posterior,
                            reason="reactive_fallback",
                            step=step,
                            sim_time=sim_time,
                        )

                        # 2) Then log belief evolution
                        self.flight_recorder.record_confidence_update(
                            agent_id=self.drone.agent_id,
                            confidence=posterior,
                            modifier=modifier,
                            note="reactive_fallback_confidence_revision",
                            step=step,
                            sim_time=sim_time,
                        )

                    print(
                        f"[CONFIDENCE-UPDATE] agent={self.drone.agent_id} "
                        f"{prior:.3f} → {self.plan_confidence:.3f}"
                    )

                print(
                    f"[REACTIVE-FALLBACK] agent={self.drone.agent_id} "
                    f"at={progress['current_cell']} "
                    f"battery={progress['battery_pct']:.1f}%\n"
                    f"  replaced pickup {old_step['pickup']} → {next_choice['pickup']} "
                    f"drop {old_step['dropoff']} → {next_choice['dropoff']}"
                )

                return  # do not advance index

        # ---------------------------------------------
        # Normal plan advancement
        # ---------------------------------------------
        self.plan_idx += 1

        if self.plan_idx >= len(self.plan):
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

    def _request_parking(self) -> Optional[Tuple[int, int]]:
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if not station:
            return None
        return station.park_drone(self.drone)

    def _begin_parking(self):
        """
        Request a parking allocation from the nearest station and, if granted,
        initiate movement toward the allocated cell.

        This method does NOT search for free cells or validate occupancy.
        Parking allocation is authoritative and owned by DeliveryStation.
        """
        if self.parking_in_progress or self.parked:
            return

        cell = self._request_parking()
        if cell is None:
            # No parking available yet
            return

        self.parking_target = cell
        self.parking_in_progress = True

        self.drone.set_target_cell(cell[0], cell[1])

        print(
            f"[PARK-START] agent={self.drone.agent_id} "
            f"from=({self.drone.col},{self.drone.row}) "
            f"to={cell}"
        )

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
            if station:
                print(
                    f"[STATION STATE @ PARK DECISION] agent={self.drone.agent_id} "
                    f"{station.debug_state()}"
                )
                print(
                    f"[PARKING] agent={self.drone.agent_id} "
                    f"target_cell={cell} "
                    f"station=({station.col},{station.row}) "
                    f"occupied={self.terrain.occupied_cell(cell[0], cell[1])}"
                )
            else:
                print(
                    f"[PARKING] agent={self.drone.agent_id} "
                    f"target_cell={cell} "
                    f"station=None "
                    f"occupied={self.terrain.occupied_cell(cell[0], cell[1])}"
                )

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
    def update(self, dt: float, step: int, sim_time: float):
        # -----------------------------------------------
        # Cooldown + hard stop conditions
        # -----------------------------------------------
        if self.cooldown > 0:
            self.cooldown -= dt

        if self.drone.lost:
            return

        now = time.time()

        # -----------------------------------------------
        # Parking completion check (arrival-based)
        # -----------------------------------------------
        if self.parking_in_progress:
            if not self.drone.moving:
                self.parking_in_progress = False
                self.parked = True
                print(
                    f"[PARKED] agent={self.drone.agent_id} "
                    f"cell=({self.drone.col},{self.drone.row}) "
                    f"battery={int(self.drone.power.percent())}%"
                )
            return

        # -----------------------------------------------
        # Immediate local actions when NOT moving
        # -----------------------------------------------
        if not self.drone.moving:

            # --------------------------------------------------
            # 1) Not carrying → attempt pick if parcel is here
            # --------------------------------------------------
            if self.drone.carrying is None:
                p_here = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p_here:
                    success = self.drone.perform_pick(p_here)
                    if not success:
                        self.drone._last_action = (
                            "pick_failed",
                            (self.drone.col, self.drone.row),
                            None,
                        )
                    else:
                        self.state = "carrying"
                        self._last_action_time = now
                    return

            # --------------------------------------------------
            # 2) Carrying → attempt drop if inside station
            # --------------------------------------------------
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)
            if (
                    self.drone.carrying
                    and station
                    and station.contains_cell(self.drone.col, self.drone.row)
            ):
                parcel_ref = self.drone.carrying
                success = self.drone.perform_drop(parcel_ref)

                if success:
                    self._last_chosen_drop = (self.drone.col, self.drone.row)

                    step = self._current_step()
                    if step:
                        self._advance_plan(step=step, sim_time=sim_time)
                else:
                    alt = station.least_used_free_cell(
                        self.terrain,
                        ref_col=self.drone.col,
                        ref_row=self.drone.row,
                    )
                    if alt:
                        self.drone.set_target_cell(*alt)
                return
        # -----------------------------------------------
        # Request plan exactly once (open-loop)
        # -----------------------------------------------
        # -----------------------------------------------
        # Strategy decision (single entry point)
        # -----------------------------------------------
        if not self.plan and not self.no_feasible_plan:
            if self.strategy.requires_plan_completion_before_requery:
                if self.state != "idle":
                    return

            if self.terrain.command_center:
                decision = self.terrain.command_center.request_directive(
                    self.drone.agent_id
                )
            else:
                decision = self.strategy.decide(self._make_snapshot())
            mode = decision.get("mode")

            if mode == "plan":
                self._load_plan(decision["plan"])

            elif mode == "task":
                self.current_task = decision["parcel_id"]
            elif mode == "wait":
                # Stay alive, do NOT mark no_feasible_plan
                print(f"AI agent {self.drone.agent_id} Controller Waiting")
                return
            elif mode == "idle":
                self.no_feasible_plan = True
                return
            else:
                raise RuntimeError(f"Unknown strategy mode: {mode}")

        # -----------------------------------------------
        # Handle NO FEASIBLE PLAN → REQUEST PARKING
        # -----------------------------------------------
        if self.no_feasible_plan:
            if not self.parked and not self.parking_in_progress:
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    cell = station.park_drone(self.drone)
                    if cell:
                        self.parking_in_progress = True
                        self.drone.set_target_cell(*cell)
                        print(
                            f"[PARK-REQUEST] agent={self.drone.agent_id} "
                            f"assigned_cell={cell}"
                        )
            return

        if self.drone.moving:
            return

        # -----------------------------------------------
        # Carrying → follow drop intent
        # -----------------------------------------------
        if self.drone.carrying:
            step = self._current_step()
            if step:
                drop_cell = tuple(step["dropoff"])
                self.drone.set_target_cell(*drop_cell)
            return

        # -----------------------------------------------
        # Not carrying → follow pickup intent
        # -----------------------------------------------
        step = self._current_step()
        if not step:
            self.state = "idle"
            return

        pickup = tuple(step["pickup"])
        pobj = self.terrain.parcel_at_cell(pickup[0], pickup[1])

        if pobj is None or getattr(pobj, "delivered", False):
            self._advance_plan(step=step, sim_time=sim_time)
            return

        self.drone.set_target_cell(*pickup)
        self.state = "seeking"


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

    def update(self, dt: float, step: int, sim_time: float):
        self.current.update(dt, step, sim_time)
