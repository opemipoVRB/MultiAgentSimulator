"""
Loaders for converting JSON artifacts into structured Python objects.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

from .models import (
    Experiment, StrategyRun, StrategyMetrics, Heatmap, HeatmapCell,
    FleetData, AgentTrajectory, TrajectoryPoint, TimeStamp,
    ParcelLifecycle, FleetEvent, PickAttempt, ParcelMetadata,
    Parcel, Agent, Station
)


class FleetLoader:
    """Loads fleet.json into structured objects."""

    @staticmethod
    def load_timestamp(data: Dict[str, Any]) -> TimeStamp:
        """Parse timestamp from JSON."""
        return TimeStamp(
            step=data["step"],
            sim_time=data["sim_time"],
            world_time=data["world_time"]
        )

    @staticmethod
    def load_pick_attempt(data: Dict[str, Any]) -> PickAttempt:
        """Parse pick attempt from JSON."""
        return PickAttempt(
            type=data["type"],
            cell=tuple(data["cell"]),
            time=FleetLoader.load_timestamp(data["time"]),
            parcel_id=data["parcel_id"]
        )

    @staticmethod
    def load_parcel_metadata(data: Dict[str, Any]) -> ParcelMetadata:
        """Parse parcel metadata from JSON."""
        return ParcelMetadata(
            registered_at=data["registered_at"],
            from_snapshot=data["from_snapshot"],
            snapshot_step=data.get("snapshot_step"),
            initial_state=data["initial_state"]
        )

    @staticmethod
    def load_parcel(data: Dict[str, Any]) -> ParcelLifecycle:
        """Parse parcel lifecycle from JSON."""
        pick_attempts = [
            FleetLoader.load_pick_attempt(attempt)
            for attempt in data.get("pick_attempts", [])
        ]

        metadata = None
        if "metadata" in data:
            metadata = FleetLoader.load_parcel_metadata(data["metadata"])

        return ParcelLifecycle(
            parcel_id=data["parcel_id"],
            spawn_cell=tuple(data["spawn_cell"]),
            picked_by=data.get("picked_by"),
            picked_at=data.get("picked_at"),
            delivered_by=data.get("delivered_by"),
            delivered_at=data.get("delivered_at"),
            pick_attempts=pick_attempts,
            metadata=metadata
        )

    @staticmethod
    def load_fleet_event(data: Dict[str, Any]) -> FleetEvent:
        """Parse fleet event from JSON."""
        return FleetEvent(
            type=data["type"],
            cell=tuple(data["cell"]),
            agents=data["agents"],
            time=FleetLoader.load_timestamp(data["time"])
        )

    @staticmethod
    def load(data: Dict[str, Any]) -> Tuple[List[ParcelLifecycle], List[FleetEvent]]:
        """Load complete fleet data from JSON dict."""
        parcels = [
            FleetLoader.load_parcel(p)
            for p in data.get("parcels", [])
        ]

        events = [
            FleetLoader.load_fleet_event(e)
            for e in data.get("fleet_events", [])
        ]

        return parcels, events


class AgentLoader:
    """Loads agents.json into structured objects."""

    @staticmethod
    def load_trajectory_point(data: Dict[str, Any]) -> TrajectoryPoint:
        """Parse single trajectory point from JSON."""
        return TrajectoryPoint(
            time=FleetLoader.load_timestamp(data["time"]),
            cell=tuple(data["cell"]),
            pos=tuple(data["pos"])
        )

    @staticmethod
    def load_agent_trajectory(data: Dict[str, Any]) -> AgentTrajectory:
        """Parse agent trajectory from JSON."""
        trajectory = [
            AgentLoader.load_trajectory_point(point)
            for point in data.get("trajectory", [])
        ]

        return AgentTrajectory(
            agent_id=data["agent_id"],
            planner=data["planner"],
            start_cell=tuple(data["start_cell"]),
            trajectory=trajectory
        )

    @staticmethod
    def load(data: List[Dict[str, Any]]) -> List[AgentTrajectory]:
        """Load all agent trajectories from JSON list."""
        return [
            AgentLoader.load_agent_trajectory(agent_data)
            for agent_data in data
        ]


class ExperimentLoader:
    """Loads experiment data from directory structure."""

    def __init__(self, experiment_dir: Path):
        """
        Initialize loader with experiment directory.

        Args:
            experiment_dir: Path to experiment_XXXXXXXX directory
        """
        self.experiment_dir = Path(experiment_dir)
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    def load(self) -> Experiment:
        """Load complete experiment with all strategies."""
        # Load main experiment.json
        exp_json = self._load_json("experiment.json")

        # Parse experiment metadata
        parcels = [Parcel(**p) for p in exp_json["parcels"]]
        agents = [Agent(**a) for a in exp_json["agents"]]
        station = Station(**exp_json["station"])

        # Load each strategy run
        strategies = []
        for result in exp_json["results"]:
            strategy_run = self._load_strategy_run(result)
            strategies.append(strategy_run)

        return Experiment(
            run_id=exp_json["run_id"],
            seed=exp_json["seed"],
            parcels=parcels,
            agents=agents,
            station=station,
            strategies=strategies,
            created_at=exp_json["created_at"]
        )

    def _load_strategy_run(self, result: Dict[str, Any]) -> StrategyRun:
        """Load a single strategy run with all artifacts."""
        strategy_name = result["strategy_name"]
        metrics = StrategyMetrics(**result["metrics"])

        # Load artifacts from strategy subdirectory
        artifacts = result["artifacts"]
        strategy_dir = self.experiment_dir / Path(artifacts["agents"]).parent

        heatmap = self._load_heatmap(strategy_dir / "heatmap.json")
        fleet = self._load_fleet_data(strategy_dir)
        meta = self._load_json(strategy_dir / "meta.json")

        return StrategyRun(
            strategy_name=strategy_name,
            metrics=metrics,
            heatmap=heatmap,
            fleet=fleet,
            meta=meta
        )

    def _load_heatmap(self, path: Path) -> Heatmap:
        """Load heatmap from JSON."""
        data = self._load_json(path)
        cells = [
            HeatmapCell(col=cell["cell"][0], row=cell["cell"][1], visits=cell["count"])
            for cell in data["cell_visits"]
        ]
        return Heatmap(cells=cells)

    def _load_fleet_data(self, strategy_dir: Path) -> FleetData:
        """Load complete fleet data from both fleet.json and agents.json."""
        # Load fleet.json
        fleet_path = strategy_dir / "fleet.json"
        fleet_data = self._load_json(fleet_path)
        parcels, events = FleetLoader.load(fleet_data)

        # Load agents.json
        agents_path = strategy_dir / "agents.json"
        agents_data = self._load_json(agents_path)
        trajectories = AgentLoader.load(agents_data)

        return FleetData(
            parcels=parcels,
            fleet_events=events,
            agent_trajectories=trajectories
        )

    def _load_json(self, filename: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
        """Load and parse JSON file."""
        if isinstance(filename, str):
            path = self.experiment_dir / filename
        else:
            path = filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'r') as f:
            return json.load(f)


class BatchLoader:
    """Loads multiple experiments for batch analysis."""

    def __init__(self, experiments_dir: Path):
        """
        Initialize batch loader.

        Args:
            experiments_dir: Path to directory containing experiment_XXXXXXXX folders
        """
        self.experiments_dir = Path(experiments_dir)

    def load_all(self) -> List[Experiment]:
        """Load all experiments in directory."""
        experiments = []

        for exp_dir in sorted(self.experiments_dir.glob("experiment_*")):
            if exp_dir.is_dir():
                try:
                    loader = ExperimentLoader(exp_dir)
                    exp = loader.load()
                    experiments.append(exp)
                    print(f"✓ Loaded {exp}")
                except Exception as e:
                    print(f"✗ Failed to load {exp_dir.name}: {e}")

        return experiments

    def load_by_seed(self, seed: int) -> Union[Experiment, None]:
        """Load specific experiment by seed."""
        for exp in self.load_all():
            if exp.seed == seed:
                return exp
        return None