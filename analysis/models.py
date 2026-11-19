# analysis/models.py
"""
Object-oriented data models for SPPDEC experiment artifacts.

Each class represents a structured view of your JSON data with
convenient access methods and computed properties.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ============================================================================
# Core Experiment Metadata
# ============================================================================

@dataclass
class Parcel:
    """Represents a single parcel in the environment."""
    id: str
    col: int
    row: int
    picked: bool
    delivered: bool
    weight: Optional[float] = None  # stored from the revision onwards

    @property
    def position(self) -> Tuple[int, int]:
        """Return (col, row) tuple."""
        return (self.col, self.row)

    @property
    def is_complete(self) -> bool:
        """Check if parcel has been delivered."""
        return self.delivered


@dataclass
class Agent:
    """Represents a single agent/drone in the experiment."""
    id: str
    type: str
    planner: str
    start_col: int
    start_row: int
    personality: Optional[str] = None  # Add this line

    @property
    def start_position(self) -> Tuple[int, int]:
        """Return starting (col, row) tuple."""
        return (self.start_col, self.start_row)


@dataclass
class Station:
    """Represents the delivery station."""
    col: int
    row: int
    w: int  # width
    h: int  # height

    @property
    def cells(self) -> List[Tuple[int, int]]:
        """Return all cells occupied by the station."""
        return [
            (c, r)
            for r in range(self.row, self.row + self.h)
            for c in range(self.col, self.col + self.w)
        ]

    @property
    def center(self) -> Tuple[float, float]:
        """Return center point of station."""
        return (
            self.col + self.w / 2,
            self.row + self.h / 2
        )


# ============================================================================
# Performance Metrics
# ============================================================================

@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy run."""
    run_id: str
    # strategy_name: str
    seed: int
    planner: str
    n_drones: int
    n_parcels: int

    # Performance metrics
    total_energy_consumed: float
    total_distance_traveled: float
    tasks_completed: int
    mission_makespan_steps: int
    unsafe_failures: int
    messages_sent: int
    runtime_seconds: float

    # Configuration
    comm_reliability: float
    battery_capacity: float
    episodic_interval: float
    timestamp: str

    @property
    def energy_per_task(self) -> float:
        """Average energy consumed per completed task."""
        return self.total_energy_consumed / max(1, self.tasks_completed)

    @property
    def distance_per_task(self) -> float:
        """Average distance traveled per completed task."""
        return self.total_distance_traveled / max(1, self.tasks_completed)

    @property
    def efficiency_ratio(self) -> float:
        """Energy efficiency: distance / energy."""
        return self.total_distance_traveled / max(0.1, self.total_energy_consumed)

    @property
    def failure_rate(self) -> float:
        """Unsafe failures per task."""
        return self.unsafe_failures / max(1, self.tasks_completed)


# ============================================================================
# Heatmap Data
# ============================================================================

@dataclass
class HeatmapCell:
    """Single cell in the exploration heatmap."""
    col: int
    row: int
    visits: int

    @property
    def position(self) -> Tuple[int, int]:
        """Return (col, row) tuple."""
        return (self.col, self.row)


@dataclass
class Heatmap:
    """Exploration heatmap showing agent activity."""
    cells: List[HeatmapCell]

    @property
    def total_visits(self) -> int:
        """Total number of cell visits."""
        return sum(cell.visits for cell in self.cells)

    @property
    def unique_cells(self) -> int:
        """Number of unique cells visited."""
        return len(self.cells)

    def get_visits(self, col: int, row: int) -> int:
        """Get visit count for a specific cell."""
        for cell in self.cells:
            if cell.col == col and cell.row == row:
                return cell.visits
        return 0

    def to_grid(self, width: int, height: int) -> List[List[int]]:
        """Convert to 2D grid representation."""
        grid = [[0 for _ in range(width)] for _ in range(height)]
        for cell in self.cells:
            if 0 <= cell.row < height and 0 <= cell.col < width:
                grid[cell.row][cell.col] = cell.visits
        return grid


# ============================================================================
# Fleet and Agent Trajectory Data
# ============================================================================

@dataclass
class TimeStamp:
    """Unified timestamp representation."""
    step: int
    sim_time: float
    world_time: float

    def __repr__(self) -> str:
        return f"t={self.step}"


@dataclass
class PickAttempt:
    """Record of a parcel pick attempt."""
    type: str  # "pick"
    cell: Tuple[int, int]
    time: TimeStamp
    parcel_id: str


@dataclass
class ParcelMetadata:
    """Optional metadata about parcel registration."""
    registered_at: float
    from_snapshot: bool
    snapshot_step: Optional[int]
    initial_state: Dict[str, bool]  # {picked: bool, delivered: bool}


@dataclass
class ParcelLifecycle:
    """Complete lifecycle of a parcel from spawn to delivery."""
    parcel_id: str
    spawn_cell: Tuple[int, int]

    # Pickup info
    picked_by: Optional[str]
    picked_at: Optional[float]

    # Delivery info
    delivered_by: Optional[str]
    delivered_at: Optional[float]

    # Attempt history
    pick_attempts: List[PickAttempt]

    # Optional metadata
    metadata: Optional[ParcelMetadata] = None

    @property
    def was_picked(self) -> bool:
        """Check if parcel was successfully picked."""
        return self.picked_by is not None

    @property
    def was_delivered(self) -> bool:
        """Check if parcel was successfully delivered."""
        return self.delivered_by is not None

    @property
    def time_to_pickup(self) -> Optional[float]:
        """Time from spawn to pickup."""
        if self.picked_at and self.metadata:
            return self.picked_at - self.metadata.registered_at
        return None

    @property
    def time_to_delivery(self) -> Optional[float]:
        """Time from spawn to delivery."""
        if self.delivered_at and self.metadata:
            return self.delivered_at - self.metadata.registered_at
        return None

    @property
    def handling_time(self) -> Optional[float]:
        """Time from pickup to delivery."""
        if self.picked_at and self.delivered_at:
            return self.delivered_at - self.picked_at
        return None

    @property
    def n_pick_attempts(self) -> int:
        """Number of pick attempts made."""
        return len(self.pick_attempts)


@dataclass
class FleetEvent:
    """Fleet event with flexible structure to handle different event types."""
    type: str
    time: Optional[TimeStamp] = None
    # Optional fields that may or may not be present
    cell: Optional[Tuple[int, int]] = None
    agents: List[str] = field(default_factory=list)
    step: Optional[int] = None
    agent_modes: Optional[Dict[str, str]] = None
    remaining_parcels: Optional[int] = None
    decision: Optional[str] = None
    # Store any additional data as metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_collision(self) -> bool:
        """Check if event is a collision."""
        return self.type == "collision"

    @property
    def n_agents_involved(self) -> int:
        """Number of agents involved in event."""
        return len(self.agents) if self.agents else 0
@dataclass
class TrajectoryPoint:
    """Single point in an agent's trajectory."""
    time: TimeStamp
    cell: Tuple[int, int]
    pos: Tuple[float, float]  # Continuous position

    @property
    def step(self) -> int:
        return self.time.step


@dataclass
class AgentTrajectory:
    """Complete trajectory data for a single agent."""
    agent_id: str
    planner: str
    start_cell: Tuple[int, int]
    trajectory: List[TrajectoryPoint]

    @property
    def n_steps(self) -> int:
        """Total number of trajectory points."""
        return len(self.trajectory)

    @property
    def final_cell(self) -> Tuple[int, int]:
        """Final cell position."""
        return self.trajectory[-1].cell if self.trajectory else self.start_cell

    @property
    def cells_visited(self) -> List[Tuple[int, int]]:
        """List of all cells visited (with duplicates)."""
        return [point.cell for point in self.trajectory]

    @property
    def unique_cells_visited(self) -> int:
        """Count of unique cells visited."""
        return len(set(self.cells_visited))

    def distance_traveled(self) -> float:
        """Calculate total Manhattan distance traveled."""
        distance = 0.0
        prev_cell = self.start_cell

        for point in self.trajectory:
            distance += abs(point.cell[0] - prev_cell[0]) + abs(point.cell[1] - prev_cell[1])
            prev_cell = point.cell

        return distance

    def get_position_at_step(self, step: int) -> Optional[Tuple[int, int]]:
        """Get agent position at specific step."""
        for point in self.trajectory:
            if point.step == step:
                return point.cell
        return None


@dataclass
class FleetData:
    """Complete fleet data including all agents and events."""
    parcels: List[ParcelLifecycle]
    fleet_events: List[FleetEvent]
    agent_trajectories: List[AgentTrajectory]

    @property
    def n_agents(self) -> int:
        """Number of agents in fleet."""
        return len(self.agent_trajectories)

    @property
    def n_parcels(self) -> int:
        """Number of parcels tracked."""
        return len(self.parcels)

    @property
    def n_collisions(self) -> int:
        """Total number of collision events."""
        return sum(1 for event in self.fleet_events if event.is_collision)

    @property
    def parcels_picked(self) -> int:
        """Number of parcels successfully picked."""
        return sum(1 for p in self.parcels if p.was_picked)

    @property
    def parcels_delivered(self) -> int:
        """Number of parcels successfully delivered."""
        return sum(1 for p in self.parcels if p.was_delivered)

    @property
    def total_distance(self) -> float:
        """Total distance traveled by all agents."""
        return sum(traj.distance_traveled() for traj in self.agent_trajectories)

    @property
    def mean_handling_time(self) -> Optional[float]:
        """Average time from pickup to delivery."""
        times = [p.handling_time for p in self.parcels if p.handling_time is not None]
        return sum(times) / len(times) if times else None

    def get_agent_trajectory(self, agent_id: str) -> Optional[AgentTrajectory]:
        """Retrieve trajectory for specific agent."""
        for traj in self.agent_trajectories:
            if traj.agent_id == agent_id:
                return traj
        return None

    def get_parcel(self, parcel_id: str) -> Optional[ParcelLifecycle]:
        """Retrieve parcel by ID."""
        for parcel in self.parcels:
            if parcel.parcel_id == parcel_id:
                return parcel
        return None

    def collisions_involving_agent(self, agent_id: str) -> List[FleetEvent]:
        """Get all collision events involving specific agent."""
        return [
            event for event in self.fleet_events
            if event.is_collision and agent_id in event.agents
        ]


# ============================================================================
# High-Level Experiment Structures
# ============================================================================

@dataclass
class StrategyRun:
    """Complete data for a single strategy execution."""
    strategy_name: str
    metrics: StrategyMetrics
    heatmap: Heatmap
    fleet: FleetData
    meta: Dict
    personality_map: Optional[Dict[str, str]] = None        # NEW: Individual agent personalities
    personality_distribution: Optional[Dict[str, int]] = None  # NEW: Aggregated personality counts

    @property
    def run_id(self) -> str:
        return self.metrics.run_id

    @property
    def seed(self) -> int:
        return self.metrics.seed

    def __repr__(self) -> str:
        return (
            f"StrategyRun(strategy={self.strategy_name}, "
            f"steps={self.metrics.mission_makespan_steps}, "
            f"energy={self.metrics.total_energy_consumed:.1f})"
        )


@dataclass
class Experiment:
    """Complete multi-strategy experiment."""
    run_id: str
    seed: int
    parcels: List[Parcel]
    agents: List[Agent]
    station: Station
    strategies: List[StrategyRun]
    created_at: str

    @property
    def n_strategies(self) -> int:
        """Number of strategies tested."""
        return len(self.strategies)

    @property
    def n_agents(self) -> int:
        """Number of agents in experiment."""
        return len(self.agents)

    @property
    def n_parcels(self) -> int:
        """Number of parcels in experiment."""
        return len(self.parcels)

    def get_strategy(self, name: str) -> Optional[StrategyRun]:
        """Retrieve strategy run by name."""
        for strategy in self.strategies:
            if strategy.strategy_name == name:
                return strategy
        return None

    def compare_strategies(self, metric: str = "mission_makespan_steps") -> Dict[str, float]:
        """Compare all strategies on a given metric."""
        return {
            s.strategy_name: getattr(s.metrics, metric)
            for s in self.strategies
        }

    def __repr__(self) -> str:
        return (
            f"Experiment(run_id={self.run_id}, seed={self.seed}, "
            f"strategies={self.n_strategies}, agents={self.n_agents}, "
            f"parcels={self.n_parcels})"
        )
