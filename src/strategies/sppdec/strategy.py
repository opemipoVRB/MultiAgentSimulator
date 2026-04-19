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

    def __init__(self, physics, variant: str = "combined"):
        super().__init__()
        self.physics = physics
        self.graph = SPPDECLangGraph(
            physics=self.physics,
            variant=variant
        )
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

    def set_drone(self, drone):
        self.drone = drone
        self.graph.set_drone(drone)
        print(f"[SPPDEC] set_drone called for {drone.agent_id} on strategy {id(self)}")

    def decide(self, snapshot: Dict) -> Dict:
        """
        Entry point used by AIAgentController.

        This is a thin wrapper around the LangGraph execution pipeline.
        The LangGraph maintains full control over the decision process.
        """
        from typing import cast

        agent_id = snapshot.get("agent", {}).get("id", "unknown")
        print(f"[SPPDEC][{agent_id}] Starting decision cycle")
        # Inject drone dynamically from controller
        if self.drone is None:
            raise RuntimeError("[SPPDEC] Drone not injected into strategy")

        # Initialize state with snapshot
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

        # Invoke the LangGraph pipeline
        final_state = self.graph.app.invoke(state)
        directive = final_state.get("directive")


        if directive is None:
            print(f"[SPPDEC][{agent_id}] No directive produced → WAIT")
            return {"mode": "wait"}

        # Log the final decision
        mode = directive.get("mode", "unknown")
        if mode == "plan":
            print(
                f"[SPPDEC][{agent_id}] Decision: {mode} with confidence={directive.get('plan', {}).get('confidence', 0.0):.2f}")
        else:
            print(f"[SPPDEC][{agent_id}] Decision: {mode}")

        return directive



