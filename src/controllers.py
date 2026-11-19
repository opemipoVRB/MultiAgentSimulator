# src/controllers.py
import os
import random
import time
from dataclasses import replace
from typing import Optional, Tuple, List, Dict

import pygame

from src.interface import Command
from src.onboard import OnboardServices
from src.perception import Perception
from src.world_port import WorldPort

# If True, AI will attempt trips even if it cannot guarantee a return-to-base.
# The drone will be allowed to attempt pickup/drop and may become lost if battery drains mid-route.

ALLOW_RISKY_TRIPS = True

# A drone without work asks the ground station over the radio at most this often (steps; 0.5 s at 60 steps/s).
REQUEST_INTERVAL_STEPS = 30


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


# Pick policy, applied identically to every strategy that uses AIAgentController.
#   off      as submitted: the drone picks up whatever parcel is in its cell when it stops
#   log      as submitted, but every stop on a parcel the current plan does not name is recorded
#   enforce  the drone picks up only the parcel its current plan names and leaves any other
# Set with CONTROLLER_PICK_POLICY; recorded in each experiment's provenance.
PICK_POLICY = os.getenv("CONTROLLER_PICK_POLICY", "log")


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

    def __init__(self, drone, terrain, strategy, topology, flight_recorder=None, sensor=None, start_info=None,
                 run_seed=0, port=None):
        super().__init__(drone, terrain)
        self.state = "idle"
        self.plan: List[Dict] = []
        self.plan_idx: int = 0
        self.no_feasible_plan = False
        self.cooldown = 0.0
        self.last_narration: Optional[str] = None
        self._last_chosen_drop: Optional[Tuple[int, int]] = None
        self.parked = False
        self.parking_in_progress = False
        self._ledger_idle = False       # idle because the station is believed full (reversible: a fresh view that says otherwise restarts the drone)
        self.plan_confidence: float = 0.0
        self.flight_recorder = flight_recorder
        self.strategy = strategy
        self.topology = topology
        self.current_step = 0
        self._leg_started = 0        # the step the drone began its present leg toward a pickup (for what a lost race cost)
        self.agent_id = drone.agent_id
        # In a drone process there is no world to read: the world sends observations and carries out commands.
        self.port = port if port is not None else WorldPort(drone, terrain, sensor, start_info, run_seed)
        # What the drone knows is built from what it senses; every strategy acts on this.
        self.perception = Perception(unseen_weight=getattr(strategy, "unseen_parcel_weight", 1.0),
                                     station_memory=getattr(strategy, "station_memory", "observed"),
                                     ledger_stop=getattr(strategy, "ledger_stop", False),
                                     declared_weights=getattr(strategy, "declared_weights", False))
        self.services = OnboardServices(drone, run_seed, start_info)   # what the drone's strategy may use of the drone
        self._obs = None
        self._commands: List[Command] = []      # what the drone will do this step, in order
        self._signals: Dict = {}                # what it tells the world besides commands (for example, abort)
        self._pending = None                    # the command whose answer arrives with the next observation
        self._last_request_step = -REQUEST_INTERVAL_STEPS   # when this drone last asked the ground station for work
        # The onboard side holds no reference to the world: everything goes through the port.
        self.drone = None
        self.terrain = None

        print(
            f"[INIT] AI controller for {self.agent_id} "
            f"parked={self.parked} "
            f"strategy={type(strategy).__name__}"
        )

    def _log_plan_outcome(self, outcome: str, cell=None, planned_cell=None):
        if self.flight_recorder is not None and hasattr(self.flight_recorder, "record_plan_outcome"):
            self.flight_recorder.record_plan_outcome(
                agent_id=self.agent_id, step=self.current_step, outcome=outcome,
                cell=cell, planned_cell=planned_cell, started=self._leg_started)

    def _job_done(self, snap) -> bool:
        """The drone's belief that every parcel is delivered: the station memory says so, or the strategy's own prediction of the station does."""
        if not getattr(self.strategy, "ledger_stop", False):
            return False
        return bool(snap.get("ledger_done")) or bool(self.strategy.predicts_job_done(snap))

    def _strategy_step(self, step: int):
        """Give the strategy its per-cycle call and execute the directive it returns."""
        snap = self._snapshot_from(self._obs)
        snap["step"] = step
        directive = self.strategy.step(snap)
        if directive is None and self._job_done(snap) and not self._obs.carrying:
            directive = {"mode": "interrupt", "reason": "ledger_complete"}      # the station holds every parcel: the plan is a ghost
        if directive and directive.get("mode") == "interrupt":
            return self._interrupt_plan(directive.get("reason"))
        return False

    def _interrupt_plan(self, reason=None) -> bool:
        """Cancel the plan being executed (not while carrying) so the agent decides again. True if it did."""
        if not self.plan or self._obs.carrying:
            return False
        st = self._current_step()
        self._log_plan_outcome("interrupt_" + str(reason), planned_cell=tuple(st["pickup"]) if st else None)
        self.plan = []
        self.plan_idx = 0
        self._commands.append(Command("stop"))
        self.state = "idle"
        return True




    # ---------------------------
    # Planner interaction
    # ---------------------------
    def _dispatch(self, events) -> None:
        """Hand what the drone's senses just noticed to the strategy, if it listens."""
        if events:
            handler = getattr(self.strategy, "on_perception_events", None)
            if handler is not None:
                handler(events)

    def _snapshot_from(self, obs) -> Dict:
        """The legacy snapshot dict for an observation the controller already holds."""
        return self.perception.snapshot(obs)

    @staticmethod
    def _battery_percent(obs) -> float:
        return 0.0 if obs.capacity <= 0 else max(0.0, min(100.0, (obs.battery / obs.capacity) * 100.0))


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
                f"[PLAN-WARN] agent={self.agent_id} "
                f"received legacy plan list. Coercing into plan envelope."
            )

        if not isinstance(plan_obj, dict):
            print(
                f"[PLAN-ERROR] agent={self.agent_id} "
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
        self._leg_started = self.current_step

        # Capture metadata
        self.plan_confidence = float(
            plan_obj.get("confidence", self.plan_confidence)
        )

        raw_narration = plan_obj.get("narration", "")
        self.last_narration = self._trim_narration(raw_narration)

        # Flight recorder
        if self.flight_recorder and self.plan:
            self.flight_recorder.record_initial_plan(
                agent_id=self.agent_id,
                plan=self.plan,
                confidence=self.plan_confidence,
                projected_battery=plan_obj.get("projected_battery_remaining"),
                step=0,
                sim_time=0.0,
            )

        # Debug logging
        print(
            f"[PLAN-COMMIT] agent={self.agent_id} "
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
            o = self._obs
            progress = {
                "current_cell": (o.col, o.row),
                "battery_pct": self._battery_percent(o),
                "battery_energy": o.battery,
                "attempted_parcels": set(),
                "failed_parcels": set(),
            }
            snap_o = self.perception.snapshot(o)
            station = snap_o["nearest_station"]

            next_choice = self.strategy.choose_best_feasible_next_parcel(
                progress=progress,
                parcels=snap_o["all_parcels"],
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
                            agent_id=self.agent_id,
                            old_step=old_step,
                            new_step=self.plan[self.plan_idx],
                            estimated_cost=next_choice.get("estimated_cost"),
                            battery_before=progress["battery_pct"],
                            battery_after=self._battery_percent(o),
                            prior_confidence=prior,
                            posterior_confidence=posterior,
                            reason="reactive_fallback",
                            step=step,
                            sim_time=sim_time,
                        )

                        self.flight_recorder.record_confidence_update(
                            agent_id=self.agent_id,
                            confidence=posterior,
                            modifier=modifier,
                            note="reactive_fallback_confidence_revision",
                            step=step,
                            sim_time=sim_time,
                        )

                    print(
                        f"[CONFIDENCE-UPDATE] agent={self.agent_id} "
                        f"{prior:.3f} → {self.plan_confidence:.3f}"
                    )

                print(
                    f"[REACTIVE-FALLBACK] agent={self.agent_id} "
                    f"at={progress['current_cell']} "
                    f"battery={progress['battery_pct']:.1f}%\n"
                    f"  replaced pickup {old_step['pickup']} → {next_choice['pickup']} "
                    f"drop {old_step['dropoff']} → {next_choice['dropoff']}"
                )

                return

        # Normal plan advancement
        self.plan_idx += 1
        self._leg_started = self.current_step

        if self.plan_idx >= len(self.plan):
            self.plan = []
            self.plan_idx = 0
            self.state = "idle"
            print(
                f"[CTRL][STATE] agent={self.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )

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
        if not self.services.inbox:
            return False

        sender_id, message = self.services.inbox.pop(0)


        print(
            f"[CTRL][INBOX] agent={self.agent_id} "
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
            return True

        if mode == "idle":
            print(f"1 I got a request to set self.no_feasible_plan to True from CBBA {self.strategy} ")
            self.no_feasible_plan = True
            print(
                f"[CTRL][STATE] agent={self.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )
            return True

        if mode == "abort":
            print(f"[CTRL][ABORT] agent={self.agent_id} received abort")
            self.no_feasible_plan = True
            self.plan = []
            self.state = "idle"
            print(
                f"[CTRL][STATE][ABORT] agent={self.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )

            return True

        # -----------------------------
        # Anything else is illegal
        # -----------------------------
        raise RuntimeError(
            f"Unknown execution directive mode delivered: {mode}"
        )


    # ---------------------------
    # Main update loop
    # ---------------------------
    # ---------------------------
    # One step: observe, think, act
    # ---------------------------
    def observe(self, step: int, sim_time: float):
        """The world's reading of this drone for this step (world side; in a separate process it arrives as a message)."""
        return self.port.observe(step, sim_time)

    def act(self, commands: List[Command], step: int) -> None:
        """The world carries out this drone's commands (world side)."""
        self.port.apply_all(commands, step)

    def replay(self, recorder) -> None:
        """Records written in a drone process are replayed here; in-process they were written directly."""
        return None

    def close(self) -> None:
        return None

    def _emit(self, cmd: Command, pending=None) -> None:
        self._commands.append(cmd)
        if pending is not None:
            self._pending = pending

    def _handle_result(self, result, sim_time: float) -> None:
        """The answer to the command issued on the previous step."""
        pending, self._pending = self._pending, None
        if pending is None or result is None:
            return
        kind = pending["kind"]
        if kind == "pick":
            self._log_plan_outcome("picked" if result.success else "pick_failed", cell=pending["here"],
                                   planned_cell=pending["planned"])
            if result.success:
                self.state = "carrying"
        elif kind == "drop" and result.success:
            self._last_chosen_drop = result.cell
            step_obj = self._current_step()
            if step_obj:
                self._advance_plan(step=step_obj, sim_time=sim_time)
            # Tell the ground station over the radio that the task is complete
            if self.topology == "centralized":
                self.services.send("command_center", {"type": "completed"})
        elif kind == "park" and result.success:
            self.parking_in_progress = True
            print(f"[PARK-REQUEST] agent={self.agent_id} assigned_cell={result.cell}")

    def think(self, obs, dt: float, step: int, sim_time: float):
        """
        One step of the drone's own software, from what it has been told. Returns (commands, signals): what the drone
        will do this step, in order, and anything it tells the world besides (for example a request to abort).

        Responsibilities, in priority order:
        1. The answer to last step's command, and what the senses noticed
        2. Physical constraints and lifecycle
        3. Immediate embodied actions
        4. Strategy decision phase
        5. Execution phase
        """
        self.current_step = step
        self._obs = obs
        self._commands = []
        self._signals = {}
        self.services.take_in(obs)
        self._handle_result(obs.last_result, sim_time)
        self._dispatch(self.perception.update(obs))   # the drone senses on every step, not only when it decides

        # Per-cycle strategy call (strategies that declare steps_every_cycle).
        if getattr(self.strategy, "steps_every_cycle", False) or getattr(self.strategy, "ledger_stop", False):
            if self._strategy_step(step):
                # the plan was cancelled and the drone has been told to stop
                obs = replace(obs, moving=False)
                self._obs = obs

        self._act(obs, dt, step, sim_time)
        self._signals["parked"] = bool(self.parked)        # status the drone reports to the world
        outgoing = self.services.take_outbox()
        if outgoing:
            self._commands.append(Command("none", messages=outgoing))
            if self.flight_recorder is not None and hasattr(self.flight_recorder, "record_message"):
                here = (int(obs.col), int(obs.row))
                for to, payload in outgoing:
                    kind = (payload.get("event") or payload.get("type")) if isinstance(payload, dict) else type(payload).__name__
                    self.flight_recorder.record_message(agent_id=self.agent_id, step=step, to=to, kind=kind, cell=here)
        return self._commands, self._signals

    def _act(self, obs, dt: float, step: int, sim_time: float) -> None:
        # Cooldown and hard stop conditions
        if self.cooldown > 0:
            self.cooldown -= dt

        if obs.lost:
            return

        # Apply any delivered network decisions first
        if self.topology == "centralized":
            if self._consume_inbox():
                return

        # Parking completion check
        if self.parking_in_progress:
            if not obs.moving:
                self.parking_in_progress = False
                self.parked = True
                print(
                    f"[PARKED] agent={self.agent_id} "
                    f"cell=({obs.col},{obs.row}) "
                    f"battery={int(self._battery_percent(obs))}%"
                )
            return

        # Immediate local embodied actions when stationary
        if not obs.moving:

            # Attempt pick if not carrying and parcel is present
            if not obs.carrying:
                p_here = obs.here_parcel_weight is not None
                here = (obs.col, obs.row)
                planned = self._current_step()
                planned_cell = tuple(planned["pickup"]) if planned else None
                if p_here and planned_cell != here and PICK_POLICY != "off":
                    # A parcel is here that the current plan does not name
                    self._log_plan_outcome(
                        "unplanned_parcel_ignored" if PICK_POLICY == "enforce" else "unplanned_parcel_present",
                        cell=here, planned_cell=planned_cell)
                    if PICK_POLICY == "enforce":
                        p_here = False
                if p_here:
                    self._emit(Command("pick"), pending={"kind": "pick", "here": here, "planned": planned_cell})
                    return

            # Attempt drop if carrying and inside station
            if obs.carrying and obs.station and obs.inside_station:
                self._emit(Command("drop"), pending={"kind": "drop"})
                return

        if obs.carrying and not self.plan and obs.station:
            # A parcel in hand and no plan (a plan cancelled a step before the drone picked up a parcel that was still there; every strategy, whether or not it uses the ledger stop): deliver it.
            station = self._snapshot_from(obs).get("nearest_station")
            drop = self.strategy._choose_drop_cell((obs.col, obs.row), station)
            self._ledger_idle = False
            self.no_feasible_plan = False
            self.parked = False
            self.parking_in_progress = False
            self._load_plan({"plan": [{"pickup": [obs.col, obs.row], "dropoff": [drop[0], drop[1]], "weight": float(obs.carrying_weight or 1.0)}],
                             "confidence": 1.0, "strategy": "deliver-carried", "narration": "deliver the parcel in hand"})
        if self._ledger_idle and not self._job_done(self._snapshot_from(obs)):
            self._ledger_idle = False
            self.no_feasible_plan = False
            self.parked = False
            self.parking_in_progress = False
            self.state = "idle"

        # Strategy decision phase
        if not self.plan and not self.no_feasible_plan:

            if self.strategy.requires_plan_completion_before_requery:
                if self.state != "idle":
                    return

            if self.topology == "centralized":
                # Ask the ground station for work over the radio (again after a while: the message can be lost)
                if not self.services.inbox and step - self._last_request_step >= REQUEST_INTERVAL_STEPS:
                    self.services.send("command_center", {"type": "request"})
                    self._last_request_step = step
                return

            snap_now = self._snapshot_from(obs)
            if self._job_done(snap_now) and not obs.carrying:
                station = snap_now.get("nearest_station")
                if (getattr(self.strategy, "ledger_verify", False) and station and not snap_now.get("ledger_done")
                        and not snap_now.get("station_in_full_view")):
                    # The job is done by prediction only: look before parking. From the station centre every cell is in view.
                    centre = (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
                    if (obs.col, obs.row) != centre:
                        if not obs.moving:
                            self._emit(Command("move_to", centre))
                        return
                decision = {"mode": "idle", "reason": "ledger_complete"}
                self._ledger_idle = True
            else:
                decision = self.strategy.decide(snap_now)

            mode = decision.get("mode")

            if mode == "plan":
                self._load_plan(decision["plan"])

            elif mode == "wait":
                return

            elif mode == "look":
                # Any strategy may ask to see the station before it decides (active perception): fly to a cell of the station from where every cell is in view (the centre unless the strategy names one), then decide again.
                # The strategy may name the cell to look from (so that idle drones do not stack on one cell); without one the drone goes to the centre.
                station = snap_now.get("nearest_station")
                if station:
                    target = decision.get("cell") or (station["col"] + station["w"] // 2, station["row"] + station["h"] // 2)
                    target = (int(target[0]), int(target[1]))
                    if (obs.col, obs.row) != target and not obs.moving:
                        self._emit(Command("move_to", target))
                return

            elif mode == "idle":
                print(f"2 I got a request to set self.no_feasible_plan to True from CBBA {self.strategy} ")
                self.no_feasible_plan = True
                self.state = "idle"
                print(
                    f"[CTRL][STATE] agent={self.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
                )

                return

            elif mode == "abort":
                print(f"[CTRL][ABORT] agent={self.agent_id} strategy requested abort")
                self._signals["abort"] = decision.get("reason")
                self.no_feasible_plan = True
                self.plan = []
                self.state = "idle"
                return

            else:

                raise RuntimeError(f"Unknown strategy mode: {mode}")

        # No feasible plan means request parking
        if self.no_feasible_plan:
            # If already safely inside station, park immediately
            if obs.station and obs.inside_station:
                if not self.parked:
                    self.parked = True
                    print(
                        f"[PARKED] agent={self.agent_id} already inside station"
                    )
                return

            # Otherwise, initiate parking movement
            if not self.parked and not self.parking_in_progress:
                if obs.station:
                    self._emit(Command("request_parking"), pending={"kind": "park"})
            return

        # Do not override movement once drone is in motion
        if obs.moving:
            return

        # Execution phase: follow plan
        step_obj = self._current_step()

        if obs.carrying:
            if step_obj:
                self._emit(Command("move_to", tuple(step_obj["dropoff"])))
            return

        if not step_obj:
            self.state = "idle"
            print(
                f"[CTRL][STATE] agent={self.agent_id} -> {self.state} | no_feasible_plan = {self.no_feasible_plan} | parked={self.parked}"
            )
            return

        pickup = tuple(step_obj["pickup"])

        # The drone finds a pickup target missing only by being at the cell. Anything it knows earlier comes
        # from its strategy, which can interrupt the plan.
        if (obs.col, obs.row) == pickup and obs.here_parcel_weight is None:
            self._log_plan_outcome("target_gone", planned_cell=pickup)
            self._advance_plan(step=step_obj, sim_time=sim_time)
            return

        self._emit(Command("move_to", pickup))
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
        return self.current.update(dt, step, sim_time)

    def observe(self, step: int, sim_time: float):
        c = self.current
        return c.observe(step, sim_time) if hasattr(c, "observe") else None

    def think(self, obs, dt: float, step: int, sim_time: float):
        """The AI controller thinks from its observation; the human controller acts on the world directly."""
        c = self.current
        if hasattr(c, "think"):
            return c.think(obs, dt, step, sim_time)
        c.update(dt, step, sim_time)
        return [], {}

    def act(self, commands, step: int):
        c = self.current
        if hasattr(c, "act"):
            c.act(commands, step)

    def replay(self, recorder) -> None:
        return None

    def close(self) -> None:
        for c in self.controllers:
            if hasattr(c, "close"):
                c.close()
