# src/interface.py
"""
What crosses the boundary between the world and a drone.

A drone sees the world only through an Observation and acts on it only through a Command. Nothing else
crosses: no terrain object, no other drone's object. `Observation.to_snapshot()` renders the same plain dict the
strategies consume today, so strategies keep working while the boundary is introduced.

Both types are plain data (picklable), so they can later travel between processes unchanged.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Cell = Tuple[int, int]


@dataclass
class Result:
    """What the world did with a Command; delivered to the drone with its next observation."""
    kind: str
    step: int                       # the step the command was issued
    success: bool = True
    cell: Optional[Cell] = None


@dataclass
class DeclaredSpec:
    """A drone's declared spec, measured before the mission (see calibration.py)."""
    unloaded_per_cell: float          # energy per cell travelled empty, all peripherals included
    weight_sensitivity: float         # extra consumption per unit of carried weight, as a fraction of unloaded
    capacity: float                   # battery capacity
    start_cell: Cell
    personality: Optional[str] = None  # the drone's declared personality (shared start information; None: the arm's own beta and w-bar)
    speed: float = 1.0                # flight speed relative to the nominal speed (declared before the mission, like the consumption rate)


@dataclass
class StartInfo:
    """What the whole fleet knows before the first step."""
    parcels: List[Dict[str, Any]]     # id, col, row (locations only; state and weight are learned by sensing)
    fleet: Dict[str, DeclaredSpec]    # every drone, the receiving one included
    grid: Optional[Tuple[int, int]] = None   # map size in cells (columns, rows)
    obstacles: Optional[List[Tuple[int, int]]] = None   # no-fly cells of the map (declared to every drone), None on an open map
    total_orders: Optional[int] = None       # orders in the whole shift when some arrive later (dynamic arrival); None: every order is in `parcels`

    @property
    def n_orders(self) -> int:
        return len(self.parcels) if self.total_orders is None else int(self.total_orders)


@dataclass
class Observation:
    step: int
    time: float
    # own state
    id: str
    col: int
    row: int
    battery: float
    capacity: float
    moving: bool
    carrying: bool
    carrying_weight: float
    lost: bool
    # what the drone senses on its own cell and at its station
    here_parcel_weight: Optional[float]
    inside_station: bool
    station: Optional[Dict[str, int]]                     # col, row, w, h of the nearest station
    station_free_cells: List[Cell] = field(default_factory=list)
    # what its sensors report
    parcels: List[Dict[str, Any]] = field(default_factory=list)
    new_orders: List[Dict[str, Any]] = field(default_factory=list)   # orders released since the last observation (shared order feed; dynamic arrival only)
    peers: List[Dict[str, Any]] = field(default_factory=list)
    sensor_radius: float = float("inf")                   # the drone knows its own sensor range
    pos_px: Tuple[float, float] = (0.0, 0.0)              # own odometry: continuous position in map units
    cell_px: float = 1.0                                  # map units per cell
    reserve_floor: float = 0.0                            # hardware reserve of this drone (energy units)
    last_result: Optional[Result] = None                  # answer to the command issued on the previous step
    # radio and services
    inbox: List[Any] = field(default_factory=list)
    parking: Optional[Cell] = None                        # answer to the last parking request
    start_info: Optional[StartInfo] = None                # first step only

    def to_snapshot(self) -> Dict[str, Any]:
        """The plain dict strategies consume (same keys as the controller's snapshot)."""
        return {
            "agent": {"id": self.id, "col": self.col, "row": self.row},
            "carrying": {"has": self.carrying, "weight": self.carrying_weight},
            "nearest_station": dict(self.station) if self.station else None,
            "all_parcels": [dict(p) for p in self.parcels],
            "other_agents": [dict(p) for p in self.peers],
            "step": self.step,
        }


@dataclass
class Command:
    """One action per step, plus any outgoing radio messages."""
    kind: str = "none"            # none | move_to | stop | pick | drop | request_parking
    cell: Optional[Cell] = None   # for move_to
    messages: List[Tuple[str, Any]] = field(default_factory=list)   # (recipient id, payload)

    KINDS = ("none", "move_to", "stop", "pick", "drop", "request_parking")

    def __post_init__(self):
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown command kind {self.kind!r}")
        if self.kind == "move_to" and self.cell is None:
            raise ValueError("move_to needs a cell")
