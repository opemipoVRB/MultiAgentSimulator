from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal


# ============================================================
# Agent-local predictive belief
# ============================================================

@dataclass(frozen=True)
class LocalPrediction:
    """
    Agent-local predictive belief about the near future.

    This is NOT ground truth.
    It is a belief formed from:
    - last observed peer positions
    - inferred trajectories
    - local timing and energy models
    """

    # agent_id -> predicted grid cell (or None if unknown)
    peer_positions: Dict[str, Optional[Tuple[int, int]]]

    # resources this agent predicts may be contested
    # (parcel IDs or station cell IDs)
    predicted_conflicts: List[str]

    # normalized energy margin after intended action [0, 1]
    energy_margin: float


# ============================================================
# Agent intent (broadcastable, soft commitment)
# ============================================================

@dataclass(frozen=True)
class AgentIntent:
    """
    Soft commitment broadcast by an agent.

    Intents are NOT reservations.
    They are advisory signals used for prediction.
    """

    agent_id: str

    # primary target parcel (if any)
    target_parcel_id: Optional[str]

    # intended action category
    intent_type: Literal[
        "pickup",
        "deliver",
        "return",
        "idle",
    ]

    # optional station cell preference (not guaranteed)
    target_station_cell: Optional[Tuple[int, int]]

    # confidence in this intent [0, 1]
    confidence: float

    # agent-local predictive belief used to form this intent
    prediction: LocalPrediction

    # time this intent was formed (monotonic)
    timestamp: float


# ============================================================
# Conflict snapshot (escalation artifact)
# ============================================================

@dataclass(frozen=True)
class ConflictSnapshot:
    """
    Snapshot emitted when local reasoning detects unresolved conflict.

    This is the ONLY object allowed to leave the agent in Phase 1.

    It contains:
    - who is involved
    - what resources are contended
    - how risky the conflict is

    It contains NO decisions.
    """

    # logical timestamp (simulation or wall time)
    timestamp: float

    # agents involved in the predicted conflict
    agents: List[str]

    # contested resources:
    # - parcel IDs
    # - station cell IDs (e.g. "station:(x,y)")
    contested_resources: List[str]

    # normalized overlap / collision risk [0, 1]
    predicted_risk: float
