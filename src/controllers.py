# src/controllers.py
import random
import time
from typing import Optional, Tuple, List, Dict

import pygame

# If True, AI will attempt trips even if it cannot guarantee a return-to-base.
# The drone will be allowed to attempt pickup/drop and may become lost if battery drains mid-route.

ALLOW_RISKY_TRIPS = True


class BaseAgentController:
    """Base controller API used by the game loop."""

    def __init__(self, drone, terrain):
        self.drone = drone
        self.terrain = terrain

    def handle_event(self, event):
        """Handle pygame events. Override in subclasses."""
        pass

    def update(self, dt: float, step: int, sim_time: float):
        """Update controller state. Override in subclasses."""
        pass

    # ADD THESE METHODS:
    def needs_decision(self) -> bool:
        """
        Return True if the controller is ready to compute a new decision.

        For AI controllers: returns True when idle and ready for a new plan.
        For human controllers: always returns False (human decisions are event-driven).

        Default implementation returns False.
        """
        return False

    def compute_decision(self):
        """
        Compute and return a decision for the agent.

        For AI controllers: returns a decision dict from the strategy.
        For human controllers: should not be called - raises NotImplementedError.

        Returns:
            Dict: The decision object
        Raises:
            NotImplementedError: If called on a controller that doesn't implement it
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_decision()"
        )

    def apply_decision(self, decision):
        """
        Apply a previously computed decision to the controller.

        For AI controllers: commits the decision into controller state.
        For human controllers: typically a no-op.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement apply_decision()"
        )


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
                        self.drone.last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
            else:
                # drop only if target cell has no non-picked parcel (includes delivered)
                if self.terrain.occupied_cell(self.drone.col, self.drone.row):
                    self.drone.last_action = ("drop_failed", (self.drone.col, self.drone.row), None)
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

    def __init__(self, drone, terrain, strategy, topology, flight_recorder=None):
        super().__init__(drone, terrain)
        self.state = "idle"
        self.plan: List[Dict] = []
        self.plan_idx: int = 0
        self.no_feasible_plan = False
        self._no_plan_battery_threshold = 15.0
        self.cooldown = 0.0
        self._last_plan_time = 0.0
        self._last_action_time = 0.0
        self.last_narration: Optional[str] = None
        self._last_chosen_drop: Optional[Tuple[int, int]] = None
        self.parked = False
        self.parking_in_progress = False
        self.parking_target = None
        self.plan_confidence: float = 0.0
        self.flight_recorder = flight_recorder
        self.strategy = strategy
        self.awaiting_cc_response = False
        self.topology = topology
        self.needs_requery = True
        self.current_step = 0

        print(
            f"[INIT] AI controller for {drone.agent_id} "
            f"parked={self.parked} "
            f"strategy={type(strategy).__name__}"
        )

    def needs_decision(self):
        return (
                not self.plan
                and not self.no_feasible_plan
                and not self.drone.moving
        )

    def compute_decision(self):
        if self.topology == "centralized":
            return self.terrain.command_center.request_directive(self.drone.agent_id)
        snap = self._make_snapshot()
        snap["step"] = getattr(self, "current_step", 0)
        return self.strategy.decide(snap)

    def apply_decision(self, decision):
        mode = decision.get("mode")

        if mode == "plan":
            self._load_plan(decision["plan"])
        elif mode == "wait":
            return
        elif mode == "idle":
            self.no_feasible_plan = True
            self.state = "idle"
            print(
                f"[CTRL][STATE] agent={self.drone.agent_id} -> idle | no_feasible_plan={self.no_feasible_plan} | parked={self.parked}"
            )
        else:
            raise RuntimeError(f"Unknown strategy mode: {mode}")

    # ---------------------------
    # Planner interaction
    # ---------------------------
    def _make_snapshot(self) -> Dict:
        """
        Create a compact, perception-only snapshot of the world state.

        This snapshot represents the agent's *local world view* at the current
        decision step. It contains only observable state and internal proprioception.
        It does NOT contain predictions, intentions, plans, or coordination signals.

        The snapshot is the sole input to all decentralized strategies
        (Naive, SPPDEC Phase 1).
        """
        snap = {
            "agent": {
                "id": self.drone.agent_id,
                "col": int(self.drone.col),
                "row": int(self.drone.row),
            },
            "carrying": {
                "has": bool(self.drone.carrying),
                "weight": (
                    getattr(self.drone.carrying, "weight", 0.0)
                    if self.drone.carrying
                    else 0.0
                ),
            },
            "nearest_station": None,
            "all_parcels": [],
            "other_agents": [],
            "step": getattr(self, "current_step", 0),
            "timestamp": time.time(),
        }

        # Nearest station
        station = self.terrain.nearest_station(self.drone.col, self.drone.row)
        if station:
            snap["nearest_station"] = {
                "col": int(station.col),
                "row": int(station.row),
                "w": int(station.w),
                "h": int(station.h),
            }

        # Parcels
        for p in self.terrain.parcels:
            snap["all_parcels"].append(
                {
                    "id": getattr(p, "id", None),
                    "col": int(p.col),
                    "row": int(p.row),
                    "weight": getattr(p, "weight", 1.0),
                    "picked": bool(getattr(p, "picked", False)),
                    "delivered": bool(getattr(p, "delivered", False)),
                }
            )

        # Other agents
        for agent in self.terrain.agents.values():
            if agent is self.drone:
                continue

            snap["other_agents"].append(
                {
                    "id": agent.agent_id,
                    "col": int(agent.col),
                    "row": int(agent.row),
                    "carrying": bool(getattr(agent, "carrying", None)),
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
        # Normalize input
        if isinstance(plan_obj, list):
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
            print(f"1 I got an Invalid plan so I set self.no_feasible_plan to True from CBBA {self.strategy} ")
            self.no_feasible_plan = True
            return

        # Parse plan steps
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

        # Install execution state
        self.plan = parsed
        self.plan_idx = 0
        self.no_feasible_plan = not bool(self.plan)
        self._last_plan_time = time.time()

        # Capture metadata
        self.plan_confidence = float(
            plan_obj.get("confidence", self.plan_confidence)
        )

        raw_narration = plan_obj.get("narration", "")
        self.last_narration = self._trim_narration(raw_narration)

        # Flight recorder
        if self.flight_recorder and self.plan:
            self.flight_recorder.record_initial_plan(
                agent_id=self.drone.agent_id,
                plan=self.plan,
                confidence=self.plan_confidence,
                projected_battery=plan_obj.get("projected_battery_remaining"),
                step=0,
                sim_time=0.0,
            )

        # Debug logging
        print(
            f"[PLAN-COMMIT] agent={self.drone.agent_id} "
            f"steps={len(self.plan)} "
            f"confidence={self.plan_confidence:.3f} "
            f"strategy={plan_obj.get('strategy', 'unknown')}"
        )
        if self.last_narration:
            print(f"[PLAN-NARRATION] {self.last_narration}")

    # ---------------------------
    # Plan helpers
    # ---------------------------
    def _current_step(self) -> Optional[Dict]:
        if 0 <= self.plan_idx < len(self.plan):
            return self.plan[self.plan_idx]
        return None

    def _advance_plan(self, *, step: Optional[Dict], sim_time: float):
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

            station_obj = self.terrain.nearest_station(self.drone.col, self.drone.row)
            station = None
            if station_obj:
                station = {
                    "col": int(station_obj.col),
                    "row": int(station_obj.row),
                    "w": int(station_obj.w),
                    "h": int(station_obj.h),
                }

            next_choice = self.strategy.choose_best_feasible_next_parcel(
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

                # Posterior confidence update
                evidence = next_choice.get("posterior_evidence")
                if evidence is not None:
                    prior = self.plan_confidence

                    modifier = self.strategy.compute_posterior_confidence_modifier(
                        prior_confidence=prior,
                        evidence=evidence,
                    )

                    posterior = prior * modifier
                    self.plan_confidence = posterior

                    if self.flight_recorder:
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

                return

        # Normal plan advancement
        self.plan_idx += 1

        if self.plan_idx >= len(self.plan):
            self.plan = []
            self.plan_idx = 0
            self.state = "idle"
            print(
                f"[CTRL][STATE] agent={self.drone.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )

    def _ensure_energy_for_route(
            self,
            from_cell: Tuple[int, int],
            to_cell: Tuple[int, int],
            carry_weight: float = 0.0,
            require_return: bool = False,
    ) -> bool:
        """
        Conservative energy check for the leg from_cell -> to_cell.
        """
        dist = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])
        needed = self.drone.energy_needed_for_cells(dist, carry_weight)

        if require_return:
            station = self.terrain.nearest_station(from_cell[0], from_cell[1])
            if station:
                home_col = int(station.col + station.w // 2)
                home_row = int(station.row + station.h // 2)
                dist_back = abs(to_cell[0] - home_col) + abs(to_cell[1] - home_row)
                needed += self.drone.energy_needed_for_cells(dist_back, 0.0)

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
        """
        if self.parking_in_progress or self.parked:
            return

        cell = self._request_parking()
        if cell is None:
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

        # Station present
        if station:
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

            # No free cells - choose any station cell
            cells = [
                (c, r)
                for r in range(station.row, station.row + station.h)
                for c in range(station.col, station.col + station.w)
            ]

            cells_sorted = sorted(
                cells,
                key=lambda c: (
                    abs(c[0] - self.drone.col) + abs(c[1] - self.drone.row),
                    c[1],
                    c[0],
                ),
            )

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

        # No station on the map
        cols = self.terrain.screen_size[0] // self.terrain.grid_size
        rows = self.terrain.screen_size[1] // self.terrain.grid_size
        attempts = 0
        chosen: Optional[Tuple[int, int]] = None

        while attempts < 400:
            c = random.randint(0, cols - 1)
            r = random.randint(0, rows - 1)

            if (c, r) == (self.drone.col, self.drone.row):
                attempts += 1
                continue

            if self.terrain.is_station_cell(c, r):
                attempts += 1
                continue

            if (
                    not self.terrain.occupied_cell(c, r)
                    and (c, r) != self._last_chosen_drop
            ):
                chosen = (c, r)
                break

            if attempts > 50 and not self.terrain.occupied_cell(c, r):
                chosen = (c, r)
                break

            attempts += 1

        if chosen is None:
            chosen = (self.drone.col, self.drone.row)

        return _set_and_return(chosen)

    def _consume_inbox(self) -> bool:
        """
        Consume exactly one execution directive from the drone inbox.

        Architectural contract:
        ------------------------
        The AgentController operates strictly at the execution layer.
        It does NOT understand coordination, negotiation, auctions, bids,
        consensus, or protocol-specific messages.

        Every message delivered here must already be a finalized execution
        directive produced by either:
        - a centralized CommandCenter, or
        - a decentralized Strategy that has completed its internal protocol.

        Accepted message envelope:
        ---------------------------
        {
            "mode": "plan" | "wait" | "idle" | "abort",
            "plan": {...}   # required when mode == "plan"
        }

        Any other message reaching this layer is a violation of the
        architecture and indicates a bug in the strategy layer.

        Returns
        -------
        bool
            True if a message was processed.
            False if the inbox was empty.
        """
        if not self.drone.inbox:
            return False

        sender_id, message = self.drone.inbox.pop(0)

        self.awaiting_cc_response = False

        print(
            f"[CTRL][INBOX] agent={self.drone.agent_id} "
            f"from={sender_id} "
            f"payload={message}"
        )

        # -----------------------------
        # Structural validation
        # -----------------------------
        if not isinstance(message, dict) or "mode" not in message:
            raise RuntimeError(
                f"Invalid execution directive delivered to controller: {message}"
            )

        mode = message.get("mode")

        # -----------------------------
        # Semantic validation
        # -----------------------------
        if mode == "plan":
            if "plan" not in message:
                raise RuntimeError(
                    f"Execution directive missing plan payload: {message}"
                )
            self._load_plan(message["plan"])
            return True

        if mode == "wait":
            self.needs_requery = False
            return True

        if mode == "idle":
            print(f"1 I got a request to set self.no_feasible_plan to True from CBBA {self.strategy} ")
            self.no_feasible_plan = True
            self.needs_requery = False
            print(
                f"[CTRL][STATE] agent={self.drone.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )
            return True

        if mode == "abort":
            print(f"[CTRL][ABORT] agent={self.drone.agent_id} received abort")
            self.no_feasible_plan = True
            self.plan = []
            self.state = "idle"
            print(
                f"[CTRL][STATE][ABORT] agent={self.drone.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )

            return True

        # -----------------------------
        # Anything else is illegal
        # -----------------------------
        raise RuntimeError(
            f"Unknown execution directive mode delivered: {mode}"
        )

    def invalidate_decision(self, reason: str):
        """
        Called by the embodied agent when something meaningful occurred.
        This resets the decision lifecycle without time-based polling.
        """
        self.needs_requery = True
        self.awaiting_cc_response = False

    # ---------------------------
    # Main update loop
    # ---------------------------
    def update(self, dt: float, step: int, sim_time: float):
        """
        Main per-tick control loop for the agent.

        Responsibilities handled here, in priority order:

        1. Physical constraints and lifecycle
        2. Network realism
        3. Immediate embodied actions
        4. Strategy decision phase
        5. Execution phase
        """
        self.current_step = step

        # Cooldown and hard stop conditions
        if self.cooldown > 0:
            self.cooldown -= dt

        if self.drone.lost:
            return

        now = time.time()

        # Apply any delivered network decisions first
        if self.topology == "centralized":
            if self._consume_inbox():
                return

        # Parking completion check
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

        # Immediate local embodied actions when stationary
        if not self.drone.moving:

            # Attempt pick if not carrying and parcel is present
            if self.drone.carrying is None:
                p_here = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p_here:
                    success = self.drone.perform_pick(p_here)
                    if not success:
                        self.drone.last_action = (
                            "pick_failed",
                            (self.drone.col, self.drone.row),
                            None,
                        )
                    else:
                        self.state = "carrying"
                        self._last_action_time = now
                    return

            # Attempt drop if carrying and inside station
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

                    step_obj = self._current_step()
                    if step_obj:
                        self._advance_plan(step=step_obj, sim_time=sim_time)

                    # Inform CommandCenter that task is complete
                    if self.topology == "centralized":
                        cc = self.terrain.command_center
                        strategy = cc.strategy

                        record = strategy.agents.get(self.drone.agent_id)
                        record = strategy.agents.get(self.drone.agent_id)
                        if record and record["state"] == "executing":
                            parcel_id = record["task"]

                            strategy.report_task_completed(self.drone.agent_id, parcel_id)
                            # Clear directive lock immediately
                            cc._directive_in_flight.discard(self.drone.agent_id)

                            # Allow immediate requery
                            self.awaiting_cc_response = False
                            print(f"[CENTRAL][COMPLETE] agent={self.drone.agent_id} task={parcel_id}")

                    return
                else:
                    alt = station.least_used_free_cell(
                        self.terrain,
                        ref_col=self.drone.col,
                        ref_row=self.drone.row,
                    )
                    if alt:
                        self.drone.set_target_cell(*alt)
                return

        # Strategy decision phase
        if not self.plan and not self.no_feasible_plan:

            if self.strategy.requires_plan_completion_before_requery:
                if self.state != "idle":
                    return

            if self.topology == "centralized":
                if not self.drone.inbox:
                    decision = self.terrain.command_center.request_directive(self.drone.agent_id)
                    if decision:
                        self.drone.inbox.append(("command_center", decision))
                return

            decision = self.strategy.decide(self._make_snapshot())

            mode = decision.get("mode")

            if mode == "plan":
                self._load_plan(decision["plan"])

            elif mode == "wait":
                return

            elif mode == "idle":
                print(f"2 I got a request to set self.no_feasible_plan to True from CBBA {self.strategy} ")
                self.no_feasible_plan = True
                self.state = "idle"
                print(
                    f"[CTRL][STATE] agent={self.drone.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
                )

                return

            elif mode == "abort":
                print(f"[CTRL][ABORT] agent={self.drone.agent_id} strategy requested abort")
                self.no_feasible_plan = True
                self.plan = []
                self.state = "idle"
                return

            else:

                raise RuntimeError(f"Unknown strategy mode: {mode}")

        # No feasible plan means request parking
        if self.no_feasible_plan:
            station = self.terrain.nearest_station(self.drone.col, self.drone.row)

            # If already safely inside station, park immediately
            if station and station.contains_cell(self.drone.col, self.drone.row):
                if not self.parked:
                    self.parked = True
                    print(
                        f"[PARKED] agent={self.drone.agent_id} already inside station"
                    )
                return

            # Otherwise, initiate parking movement
            if not self.parked and not self.parking_in_progress:
                if station:
                    cell = station.park_drone(self.drone)
                    if cell:
                        self.parking_in_progress = True
                        self.drone.set_target_cell(*cell)
                        print(
                            f"[PARK-REQUEST] agent={self.drone.agent_id} assigned_cell={cell}"
                        )
            return

        # Do not override movement once drone is in motion
        if self.drone.moving:
            return

        # Execution phase: follow plan
        step_obj = self._current_step()

        if self.drone.carrying:
            if step_obj:
                drop_cell = tuple(step_obj["dropoff"])
                self.drone.set_target_cell(*drop_cell)
            return

        if not step_obj:
            self.state = "idle"
            print(
                f"[CTRL][STATE] agent={self.drone.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )
            return

        pickup = tuple(step_obj["pickup"])
        pobj = self.terrain.parcel_at_cell(pickup[0], pickup[1])

        if pobj is None or getattr(pobj, "delivered", False):
            self._advance_plan(step=step_obj, sim_time=sim_time)
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

    # ADD THESE METHODS:
    def needs_decision(self):
        """Delegate to current controller."""
        return self.current.needs_decision()

    def compute_decision(self):
        """Delegate to current controller."""
        return self.current.compute_decision()

    def apply_decision(self, decision):
        """Delegate to current controller."""
        return self.current.apply_decision(decision)
