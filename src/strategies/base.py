# src/strategies/base.py
from typing import Dict, Any


class BaseStrategy:
    """
    Base interface for all delivery strategies.

    A strategy produces task suggestions (plans) given a snapshot.
    It may be stateless or stateful.

    IMPORTANT:
    - This does NOT execute plans.
    - This does NOT own the world.
    - This does NOT move agents.

    The Controller is the sole executor.
    """

    def request_plan(
        self,
        snapshot: Dict[str, Any],
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Given a snapshot of the world, return a plan suggestion.

        Must return a dict containing at least:
          - plan: list
          - confidence: float
          - narration: str
          - strategy: str
        """
        raise NotImplementedError
