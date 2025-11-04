# src/proposals.py
from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple

class Proposal:
    """
    Simple proposal container:
      agent: controller object
      picks: list of parcel objects (in requested order)
      ts: timestamp
    """
    def __init__(self, agent: object, picks: List[object], ts: float = None):
        self.agent = agent
        self.picks = list(picks)
        self.ts = float(ts) if ts is not None else time.time()

    def as_dict(self) -> Dict:
        return {"agent": self.agent, "picks": self.picks, "ts": self.ts}

def validate_proposal_obj(obj: Dict) -> bool:
    """
    Basic validation for a proposal dict (from LLM or planner).
    Ensures 'picks' is a list and coordinates look plausible.
    """
    if not isinstance(obj, dict):
        return False
    picks = obj.get("picks") or obj.get("plan") or obj.get("picks", [])
    if not isinstance(picks, list):
        return False
    # basic check that each pick has col,row if it's a dict-like; parcel objects are allowed too
    for p in picks:
        if hasattr(p, "col") and hasattr(p, "row"):
            continue
        if isinstance(p, dict):
            if "col" not in p or "row" not in p:
                return False
    return True
