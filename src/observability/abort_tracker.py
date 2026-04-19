from typing import Dict, List, Optional
import time
from enum import Enum


class AbortEvent:
    def __init__(
        self,
        *,
        run_id: str,
        strategy: str,
        agent_id: Optional[str],
        reason: str,
        step: int,
        sim_time: float,
        metadata: Optional[Dict] = None,
    ):
        self.event = {
            "run_id": run_id,
            "strategy": strategy,
            "agent_id": agent_id,
            "reason": str(reason),
            "time": {
                "step": step,
                "sim_time": sim_time,
                "world_time": time.time(),
            },
            "metadata": metadata or {},
        }

    def as_dict(self):
        return dict(self.event)


class AbortTracker:
    """
    Passive recorder for abort events during simulation.

    Purpose:
        Capture structured evidence about why strategies fail,
        when they fail, and under what conditions.

    This enables later analysis such as:
        - Failure mode frequency by strategy
        - Abort reason distributions
        - Abort rate vs agent count
        - Energy infeasibility under parcel density
        - Deadlock emergence conditions
    """

    def __init__(self, run_id: str, strategy: str):
        self.run_id = run_id
        self.strategy = strategy
        self.events: List[Dict] = []

    def record_abort(
        self,
        *,
        reason,
        step: int,
        sim_time: float,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        event = AbortEvent(
            run_id=self.run_id,
            strategy=self.strategy,
            agent_id=agent_id,
            reason=reason,
            step=step,
            sim_time=sim_time,
            metadata=metadata,
        )

        self.events.append(event.as_dict())

        print(
            f"[ABORT-TRACKER] "
            f"strategy={self.strategy} "
            f"agent={agent_id} "
            f"reason={reason} "
            f"step={step}"
        )

    def export(self) -> List[Dict]:
        return list(self.events)

    def count_by_reason(self) -> Dict[str, int]:
        counts = {}
        for e in self.events:
            r = e["reason"]
            counts[r] = counts.get(r, 0) + 1
        return counts