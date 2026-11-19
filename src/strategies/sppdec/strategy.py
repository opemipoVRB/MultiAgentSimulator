# src/strategies/sppdec/strategy.py
import logging
from typing import Dict
from .graph import SPPDECAgentState, SPPDECLangGraph
from ..base import BaseStrategy


class SPPDECStrategy(BaseStrategy):
    """
    Phase 1 SPPDEC:
    - local reasoning only
    - no coordination
    - no arbitration
    """

    enable_reactive_fallback = False
    requires_plan_completion_before_requery = False

    uses_declared_energy_model = False   # SPPDEC applies each drone's declared spec itself

    def __init__(self, physics, variant: str = "combined", arbiter: str = "slm",
                 slm_latency_steps: int = 0, slm_model=None, prompt_version=None,
                 symmetric_costs: bool = False, exclude_carrying_peers: bool = False,
                 no_slot_policy: str = "greedy", sensing=None, peer_rate_assumption: str = "known",
                 conflict_policy: str = "trend", coordination_model: str = "hungarian",
                 slm_latency_mode: str = "blocking", cbba_bundle_size=1,
                 assumption_memory: bool = False, observe_every_step: bool = False,
                 observe_gone: bool = True, observe_contested: bool = True,
                 classifier_model: str = "distilled", use_self_check: bool = False, own_rate_known: bool = True,
                 peer_prior: str = "like_self", hardware=None, prompt_cache: bool = False,
                 world_view: bool = False, slm_suggest: bool = True, plan_ahead: bool = False, slm_think: bool = False,
                 rule_perturb: float = 0.0, freshness: bool = False,
                 territory: bool = False, territory_hint=None, ledger_complete: bool = False):
        super().__init__()
        # hardware: name of a measured hardware profile (src/hardware.py); a consultation of the model then takes the time that
        # machine needs to read this prompt and write this answer. prompt_cache: the fixed part of the prompt stays in
        # the model's cache after the first decision. Without a hardware profile, slm_latency_steps is charged.
        self.hardware = None
        if hardware:
            from src.hardware import load_profile
            self.hardware = load_profile(hardware)
        self.prompt_cache = bool(prompt_cache)
        self._prefix_warm = False
        # plan_ahead (needs the world view, which lists the open tasks the model names): the model plans a bundle of tasks;
        # the drone takes the next one still open without asking, and a new bundle is planned in the background while it
        # delivers, so the model's thinking time overlaps flying (see _decide_ahead and _maybe_plan_in_background).
        if plan_ahead and not world_view:
            raise ValueError("plan_ahead needs world_view (the model names tasks from the open-task list)")
        self.plan_ahead = bool(plan_ahead)
        self._last_bundle = []
        self._last_head = None            # parcel the latest decision went to
        self._bg = None                   # background plan under way: {"ids": [...], "release": step}
        self._last_latency = None         # steps the latest consultation took
        self._task_steps = None           # steps a task takes to fly (running mean over tasks taken from the queue)
        self._issue_step = None           # step the latest plan was issued
        self._from_queue = False          # the latest plan came from the queue (no consultation)
        # Parameter meanings, briefly (details beside each use in graph.py):
        #   symmetric_costs / exclude_carrying_peers  score self and peers with one estimator; skip loaded peers
        #   coordination_model  what the "coordinate" policy means: "hungarian" assignment or "cbba" in the drone
        #   cbba_bundle_size  tasks per drone in CBBA in the drone (a number or "auto" = fair share of the open tasks)
        #   assumption_memory / observe_every_step  remember assumed peer tasks; track peers every step
        #   use_self_check / own_rate_known  ablation of the drone's own energy correction; hide its true rate
        self.physics = physics
        # Simulated SLM inference time (steps at 60 Hz) charged to a decision that consulted
        # the SLM; the strategy holds the decision back this long. 0 = instantaneous.
        self.slm_latency_steps = int(slm_latency_steps)
        # "blocking": the drone waits for the decision; "prefetch": it decides its next task while
        # delivering the current parcel (non-blocking inference).
        self.slm_latency_mode = slm_latency_mode
        self.graph = SPPDECLangGraph(
            peer_prior=peer_prior,
            physics=self.physics,
            variant=variant,
            arbiter=arbiter,
            prompt_version=prompt_version,
            symmetric_costs=symmetric_costs,
            exclude_carrying_peers=exclude_carrying_peers,
            no_slot_policy=no_slot_policy,
            sensing=sensing,
            peer_rate_assumption=peer_rate_assumption,
            conflict_policy=conflict_policy,
            coordination_model=coordination_model,
            cbba_bundle_size=cbba_bundle_size,
            assumption_memory=assumption_memory,
            observe_every_step=observe_every_step,
            observe_gone=observe_gone,
            observe_contested=observe_contested,
            classifier_model=classifier_model,
            use_self_check=use_self_check,
            own_rate_known=own_rate_known,
            world_view=world_view,
            slm_suggest=slm_suggest,
            plan_ahead=plan_ahead,
            slm_think=slm_think,
            rule_perturb=rule_perturb,
            freshness=freshness,
            territory=territory,
            territory_hint=territory_hint,
            ledger_complete=ledger_complete,
        )
        self._was_carrying = False
        # The controller calls step() every simulation step for strategies that set this.
        self.steps_every_cycle = bool(observe_every_step or use_self_check or plan_ahead
                                      or (slm_latency_mode == "prefetch" and self.slm_latency_steps))
        # A decision made but not yet released (still "in inference"): (directive, release step)
        self._held = None
        # The plan this drone is executing, and a counter that identifies it
        self._current_steps = None
        self._plan_serial = 0
        self._prefetched_serial = -1
        # Arm-level SLM model; None keeps the drone's default (OLLAMA_MODEL from .env)
        self.slm_model = slm_model
        self._drone = None
        self.flight_recorder = None
        print("[SPPDEC] Strategy initialized with LangGraph controller")
        print(f"[SPPDEC] [{variant}] New strategy instance created at {id(self)}")
        # import traceback
        #
        # print("[SPPDEC] Strategy initialized")
        # traceback.print_stack(limit=5)

    @property
    def flight_recorder(self):
        return self.graph.flight_recorder

    @flight_recorder.setter
    def flight_recorder(self, value):
        self.graph.flight_recorder = value

    @property
    def unseen_parcel_weight(self) -> float:
        """Weight the drone assumes for a parcel it has never seen (its baseline payload)."""
        return self.graph.baseline_payload

    def set_drone(self, drone):
        self.drone = drone
        if self.slm_model:
            drone.slm_model = self.slm_model
        self.graph.set_drone(drone)
        print(f"[SPPDEC] set_drone called for {drone.agent_id} on strategy {id(self)}")

    # ------------------------------------------------------------------
    # Entry points used by the controller
    # ------------------------------------------------------------------
    def decide(self, snapshot: Dict) -> Dict:
        """
        Called when the drone needs a decision. A decision that consulted the SLM is held back for
        the inference time: until then the drone is told to wait (blocking inference).
        """
        if self.plan_ahead:
            return self._decide_ahead(snapshot)
        step = snapshot.get("step", 0)
        if self._held is not None:
            directive, release = self._held
            if step < release:
                return {"mode": "wait", "reason": "inference"}
            self._held = None
            return self._issue(directive)
        directive, attempted = self._decide_now(snapshot)
        latency = self._latency_steps(attempted)
        if latency > 0:
            agent_id = snapshot.get("agent", {}).get("id", "unknown")
            if self.flight_recorder is not None:
                self.flight_recorder.record_latency(agent_id=agent_id, step=step, latency_steps=latency)
            self._held = (directive, step + latency)
            return {"mode": "wait", "reason": "inference"}
        return self._issue(directive)

    def _latency_steps(self, attempted: bool) -> int:
        """Steps a decision is held back for the model's inference: from the hardware profile and the tokens this
        consultation read and wrote, or the fixed `slm_latency_steps` when there is no profile."""
        if not attempted:
            return 0
        if self.hardware is None:
            return self.slm_latency_steps
        usage = self.graph.last_slm_usage or {"prompt_tokens": 0, "gen_tokens": 0}
        model = getattr(self.drone, "slm_model", None) or ""
        steps = self.hardware.decision_steps(model, usage["prompt_tokens"], usage["gen_tokens"],
                                             cached=self.prompt_cache and self._prefix_warm)
        self._prefix_warm = True      # the fixed part of the prompt is in the model's cache from now on
        return steps

    def predicts_job_done(self, snapshot) -> bool:
        return self.graph.predicts_job_done(snapshot)

    def step(self, snapshot: Dict):
        """
        Per-cycle call (every simulation step). Observation, memory and deliberation run here,
        alongside whatever the drone is executing; the controller only executes what comes back.
        """
        out = None
        if self.graph.drone is not None:
            self.graph.watch_own_energy(snapshot)  # self-check signal, always computed
        if self.graph.observe_every_step:
            if self._was_carrying and not snapshot.get("carrying", {}).get("has"):
                # the drone has just delivered: the step it finished is done, and the next pickup of the plan (if any) is its target from this step on
                self.graph.note_drop(snapshot)
                if self._current_steps:
                    self._current_steps = self._current_steps[1:] or None
                self._was_carrying = False
            own = None
            if not snapshot.get("carrying", {}).get("has") and self._current_steps:
                own = tuple(self._current_steps[0]["pickup"])
            signal = self.graph.observe(snapshot, own_target=own)
            if signal.get("replan") and own is not None:
                self._current_steps = None
                self.graph.note_plan_result(False)
                out = {"mode": "interrupt", "reason": signal["replan"]}
            carrying_now = bool(snapshot.get("carrying", {}).get("has"))
            if carrying_now and not self._was_carrying:
                self.graph.note_plan_result(True)
                self.graph.note_pickup(snapshot)
            self._was_carrying = carrying_now
        if (out is None and self.graph.recheck and self._current_steps and not snapshot.get("carrying", {}).get("has")
                and self.graph._own_cells and self.graph._own_cells[-1][0] == snapshot.get("step", 0)):
            out = self._recheck_target(snapshot)
        # The station holds every parcel: the job is done. A drone flying to a parcel it still believes open stops (the plan is a ghost) and parks.
        if (out is None and self.graph.ledger_complete and self._current_steps
                and not snapshot.get("carrying", {}).get("has") and self.graph._ledger_says_done(snapshot)):
            self._current_steps = None
            out = {"mode": "interrupt", "reason": "ledger_complete"}
        if self.plan_ahead:
            self._maybe_plan_in_background(snapshot)
        elif self.slm_latency_mode == "prefetch" and (self.slm_latency_steps or self.hardware):
            self._maybe_prefetch(snapshot)
        return out

    def _recheck_target(self, snapshot: Dict):
        """
        The drone asks the same reasoning again on what it believes now (a decision that is not issued and not recorded). If the plan it would make no longer holds
        the pickup it is flying to, the target belongs to a peer on the shared reasoning: it gives it up and decides again.
        """
        import contextlib, io
        g = self.graph
        target = tuple(self._current_steps[0]["pickup"])
        saved = (g._leg_start, g._compete_cell, g._looking, g.flight_recorder)
        g.flight_recorder = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                directive, _ = self._decide_now(snapshot)
        finally:
            g._leg_start, g._compete_cell, g._looking, g.flight_recorder = saved
        if directive.get("mode") == "plan":
            plan = directive["plan"]["plan"] if isinstance(directive["plan"], dict) else directive["plan"]
            if any(tuple(s["pickup"]) == target for s in plan):
                return None
        self._current_steps = None
        g.note_plan_result(False)
        return {"mode": "interrupt", "reason": "reassigned"}

    def on_perception_events(self, events):
        """A planned task a peer took is dropped from the drone's planned tasks (and from a plan being made)."""
        if not self.plan_ahead:
            return None
        gone = {e["id"] for e in events if e.get("type") == "parcel_taken"}
        planned = set(self.graph.planned_queue) | set(self._bg["ids"] if self._bg else [])
        if gone & planned:
            print(f"[SPPDEC][{self.graph.agent_id}] Plan ahead: planned tasks taken by peers: {sorted(gone & planned)}")
            self.graph.planned_queue = [p for p in self.graph.planned_queue if p not in gone]
            if self._bg:
                self._bg["ids"] = [p for p in self._bg["ids"] if p not in gone]
        return None

    # ------------------------------------------------------------------
    # Plan ahead: tasks are planned in bundles; the drone follows its queue; the next bundle is planned in the background
    # ------------------------------------------------------------------
    def _decide_ahead(self, snapshot: Dict) -> Dict:
        """
        The decision when the drone is free. A background plan that has finished joins the queue. The drone takes the next
        planned task that is still open; with none left it consults the model (blocking, as without plan ahead) unless a
        background plan is still being made, then it waits for that.
        """
        step = snapshot.get("step", 0)
        if self._bg is not None and step >= self._bg["release"]:
            self.graph.planned_queue += [p for p in self._bg["ids"] if p not in self.graph.planned_queue]
            self._bg = None
        if self._held is not None:
            directive, release = self._held
            if step < release:
                return {"mode": "wait", "reason": "inference"}
            self._held = None
            if self._head_open(directive, snapshot):
                return self._issue_ahead(directive, step, False)
        if not self.graph.planned_queue and self._bg is not None:
            return {"mode": "wait", "reason": "inference"}
        directive, attempted = self._decide_now(snapshot)
        if attempted:
            self.graph.planned_queue = list(self._last_bundle)
        latency = self._latency_steps(attempted)
        if latency > 0:
            self._last_latency = latency
            agent_id = snapshot.get("agent", {}).get("id", "unknown")
            if self.flight_recorder is not None:
                self.flight_recorder.record_latency(agent_id=agent_id, step=step, latency_steps=latency)
            self._held = (directive, step + latency)
            return {"mode": "wait", "reason": "inference"}
        return self._issue_ahead(directive, step, not attempted)

    def _issue_ahead(self, directive: Dict, step: int, from_queue: bool) -> Dict:
        """Issue a plan; note how long the previous task took to fly when this one came straight from the queue."""
        if from_queue and self._issue_step is not None and directive.get("mode") == "plan":
            took = step - self._issue_step
            self._task_steps = took if self._task_steps is None else 0.5 * (self._task_steps + took)
        if directive.get("mode") == "plan":
            self._issue_step = step
        return self._issue(directive)

    def _maybe_plan_in_background(self, snapshot: Dict) -> None:
        """
        Plan the next bundle while the drone flies, started early enough that the model has finished when the queue runs
        out: when the tasks still queued take no longer to fly than the model took last time (with nothing measured yet:
        when the queue is empty). The plan starts where the drone will be after its queued tasks and leaves those
        tasks out. It joins the queue when the model's time has passed (see _decide_ahead).
        """
        if (self._bg is not None or self._held is not None or not self._current_steps
                or not snapshot.get("carrying", {}).get("has")):
            return
        queue = self.graph.planned_queue
        if queue and not (self._last_latency and self._task_steps and len(queue) * self._task_steps <= self._last_latency):
            return
        if not queue and self._prefetched_serial == self._plan_serial:
            return
        by_id = {p["id"]: p for p in snapshot.get("all_parcels", [])}
        station = snapshot.get("nearest_station")
        if queue and queue[-1] in by_id:
            last = by_id[queue[-1]]
            after = self._choose_drop_cell((last["col"], last["row"]), station)
        else:
            after = tuple(self._current_steps[0]["dropoff"])
        self._prefetched_serial = self._plan_serial
        snap = dict(snapshot)
        snap["agent"] = dict(snapshot["agent"], col=int(after[0]), row=int(after[1]))
        snap["carrying"] = {"has": False, "weight": 0.0}
        self.graph.excluded_parcels = set(queue)
        kept, self.graph.planned_queue = self.graph.planned_queue, []     # the plan is made by the model, not from the queue
        try:
            directive, attempted = self._decide_now(snap)
        finally:
            self.graph.planned_queue = kept
            self.graph.excluded_parcels = set()
        if directive.get("mode") != "plan" or not attempted:
            return
        step = snapshot.get("step", 0)
        latency = self._latency_steps(attempted)
        self._last_latency = latency or self._last_latency
        if latency > 0 and self.flight_recorder is not None:
            self.flight_recorder.record_latency(agent_id=snapshot["agent"]["id"], step=step, latency_steps=latency)
        self._bg = {"ids": [self._last_head] + list(self._last_bundle), "release": step + latency}

    @staticmethod
    def _head_open(directive: Dict, snapshot: Dict) -> bool:
        """Whether the parcel a held plan goes to is still in its place, as the drone perceives it."""
        if directive.get("mode") != "plan":
            return True
        plan = directive["plan"]["plan"] if isinstance(directive["plan"], dict) else directive["plan"]
        cell = tuple(plan[0]["pickup"])
        return any((p["col"], p["row"]) == cell and not p.get("picked") and not p.get("delivered")
                   for p in snapshot.get("all_parcels", []))

    # ------------------------------------------------------------------
    def _issue(self, directive: Dict) -> Dict:
        """A decision leaves the strategy: remember the plan it carries."""
        if directive.get("mode") == "plan":
            plan = directive["plan"]
            self._current_steps = plan.get("plan") if isinstance(plan, dict) else plan
            self._plan_serial += 1
        return directive

    def _maybe_prefetch(self, snapshot: Dict) -> None:
        """
        Non-blocking inference: while the drone carries a parcel it already decides its next task,
        as if it were at the drop cell with the parcel delivered, so the inference time overlaps
        the delivery.
        """
        if (not snapshot.get("carrying", {}).get("has") or self._held is not None
                or self._prefetched_serial == self._plan_serial or not self._current_steps):
            return
        self._prefetched_serial = self._plan_serial
        drop = tuple(self._current_steps[0]["dropoff"])
        snap = dict(snapshot)
        snap["agent"] = dict(snapshot["agent"], col=int(drop[0]), row=int(drop[1]))
        snap["carrying"] = {"has": False, "weight": 0.0}
        directive, attempted = self._decide_now(snap)
        if directive.get("mode") == "plan":
            step = snapshot.get("step", 0)
            latency = self._latency_steps(attempted)
            if latency > 0 and self.flight_recorder is not None:
                self.flight_recorder.record_latency(agent_id=snapshot["agent"]["id"], step=step,
                                                    latency_steps=latency)
            self._held = (directive, step + latency)

    def _decide_now(self, snapshot: Dict):
        """Run the decision pipeline once. Returns (directive, whether the SLM was consulted)."""
        from typing import cast

        agent_id = snapshot.get("agent", {}).get("id", "unknown")
        print(f"[SPPDEC][{agent_id}] Starting decision cycle")
        if self.drone is None:
            raise RuntimeError("[SPPDEC] Drone not injected into strategy")

        state = cast(
            SPPDECAgentState,
            {
                "snapshot": snapshot,
                "visible_parcels": [],
                "visible_agents": [],
                "feasible_tasks": [],
                "predicted_conflicts": [],
                "selected_task": None,
                "decision_mode": "wait",
                "arbitration_required": False,
                "arbitration_result": None,
                "directive": None,
            },
        )

        final_state = self.graph.app.invoke(state)
        self._last_bundle = list(final_state.get("slm_bundle") or [])
        self._last_head = (final_state.get("selected_task") or {}).get("parcel_id")
        directive = final_state.get("directive")
        if directive is None:
            print(f"[SPPDEC][{agent_id}] No directive produced → WAIT")
            return {"mode": "wait"}, False

        mode = directive.get("mode", "unknown")
        if mode == "plan":
            print(
                f"[SPPDEC][{agent_id}] Decision: {mode} with confidence={directive.get('plan', {}).get('confidence', 0.0):.2f}")
        else:
            print(f"[SPPDEC][{agent_id}] Decision: {mode}")
        return directive, bool(final_state.get("slm_attempted"))
