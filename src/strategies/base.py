# src/strategies/base.py
from typing import Dict, Any


class BaseStrategy:
    """
    Base interface for all delivery strategies.

    A strategy is a decision-making policy that produces high-level
    directives (plans, tasks, wait, idle) given a snapshot of the world.

    A strategy:
    - DOES decide what an agent should attempt next
    - DOES express planning authority and autonomy constraints
    - MAY be stateless or stateful across decisions

    A strategy explicitly DOES NOT:
    - execute movement or actions
    - mutate the world or terrain
    - own agents, parcels, or stations
    - perform physics, energy consumption, or timing

    Execution responsibility:
    -------------------------
    The AIAgentController is the sole executor of all plans.
    The strategy never issues commands directly to the world.

    Reactive autonomy contract:
    ---------------------------
    `enable_reactive_fallback` declares whether the controller is allowed
    to locally substitute or repair plan steps when execution constraints
    invalidate the original plan.

    - If True:
        The controller MAY perform local, opportunistic replanning
        (for example, choosing an alternative feasible parcel).

    - If False:
        The controller MUST NOT alter the plan.
        All replanning authority remains external to the agent
        (for example, via a CommandCenter).

    This flag is authoritative and must be respected by the controller.

    requires_plan_completion_before_requery:
        If True, the controller MUST NOT request a new directive
        until the current plan has fully completed or been invalidated.
    """
    enable_reactive_fallback: bool = False
    requires_plan_completion_before_requery: bool = False

    def decide(self, snapshot) -> Dict:
        """
        Returns an ExecutionDirective
        """
        raise NotImplementedError

    def request_plan(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a snapshot of the world, return a plan suggestion.

        Must return a dict containing at least:
          - plan: list
          - confidence: float
          - narration: str
          - strategy: str
        """
        raise NotImplementedError
