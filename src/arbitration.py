# src/arbitration.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import time

def score_candidate(agent, parcel, agent_power_pct: float, agent_pos: Tuple[int,int]) -> float:
    """
    Simple scoring: distance + battery bias + parcel weight factor.
    Lower is better.
    """
    try:
        d = math.hypot(agent_pos[0] - parcel.col, agent_pos[1] - parcel.row)
    except Exception:
        d = 9999.0
    weight = getattr(parcel, "weight", 1.0)
    battery_bias = 1.0 - min(0.4, agent_power_pct / 250.0)
    return d * (1.0 + weight * 0.1) * battery_bias

def arbitrate_proposals(proposals: List[Dict[str, Any]], controllers: List[Any], max_assign_per_agent: int = 2) -> Dict[Any, List[Any]]:
    """
    proposals: list of {"agent": agent_obj, "picks": [parcel_objs], "ts": float}
    controllers: list of controller objects (for battery, pos)
    Returns mapping controller -> assigned parcel list (no conflicts).
    Deterministic tie-breaking: lower score wins, battery% then timestamp.
    """
    # Build candidate list (score, ts, agent, parcel)
    candidates = []
    for prop in proposals:
        agent = prop.get("agent")
        ts = prop.get("ts", 0.0)
        picks = prop.get("picks", [])
        agent_pos = (int(getattr(agent.drone, "col", 0)), int(getattr(agent.drone, "row", 0)))
        batt_pct = getattr(getattr(agent, "drone", None), "power", None)
        batt_val = batt_pct.percent() if batt_pct is not None else 100.0
        for p in picks:
            s = score_candidate(agent, p, batt_val, agent_pos)
            candidates.append((s, -batt_val, ts, agent, p))

    # deterministic sort: score asc, battery desc (so -battery asc), timestamp asc
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    assignments = {c: [] for c in controllers}
    claimed = set()

    for score, neg_batt, ts, agent, parcel in candidates:
        if parcel in claimed:
            continue
        if len(assignments.get(agent, [])) >= max_assign_per_agent:
            continue
        # final validation could be inserted here
        assignments.setdefault(agent, []).append(parcel)
        claimed.add(parcel)

    return assignments
