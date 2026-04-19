from pathlib import Path
from typing import Dict, Any, List
import json

from .batch_loaders import FleetLoader, AgentLoader
from .models import (
    Experiment, StrategyRun, StrategyMetrics,
    Parcel, Agent, Station
)

class SingleExperimentLoader:
    """
    Loads exactly one experiment directory of the form:

    experiment_xxxx/
        experiment.json
        strategyA/
            agents.json
            fleet.json
            heatmap.json
            meta.json
        strategyB/
            ...
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.experiment_dir}")

    def load(self) -> Experiment:
        exp_json = self._load_json(self.experiment_dir / "experiment.json")

        parcels = [Parcel(**p) for p in exp_json["parcels"]]

        # Handle agents with optional personality field
        agents = []
        for a in exp_json["agents"]:
            # Only include fields that Agent actually accepts
            agent_kwargs = {
                'id': a['id'],
                'type': a['type'],
                'planner': a['planner'],
                'start_col': a['start_col'],
                'start_row': a['start_row']
            }
            # Conditionally add personality if Agent model supports it
            if 'personality' in a:
                # Try to see if Agent accepts personality parameter
                import inspect
                agent_sig = inspect.signature(Agent.__init__)
                if 'personality' in agent_sig.parameters:
                    agent_kwargs['personality'] = a['personality']

            agents.append(Agent(**agent_kwargs))

        station = Station(**exp_json["station"])

        strategies: List[StrategyRun] = []

        # In the load() method, replace this section:
        for result in exp_json["results"]:
            strategy_name = result["strategy_name"]
            metrics = StrategyMetrics(**result["metrics"])

            strategy_dir = self.experiment_dir / strategy_name
            if not strategy_dir.exists():
                raise FileNotFoundError(f"Missing strategy folder: {strategy_dir}")

            heatmap = self._load_heatmap(strategy_dir / "heatmap.json")
            fleet = self._load_fleet(strategy_dir)
            meta = self._load_json(strategy_dir / "meta.json")

            # NEW: Extract personality data from results
            personality_map = result.get("personality_map", {})
            personality_distribution = result.get("personality_distribution", {})

            strategies.append(
                StrategyRun(
                    strategy_name=strategy_name,
                    metrics=metrics,
                    heatmap=heatmap,
                    fleet=fleet,
                    meta=meta,
                    personality_map=personality_map,  # NEW
                    personality_distribution=personality_distribution  # NEW
                )
            )

        return Experiment(
            run_id=exp_json["run_id"],
            seed=exp_json["seed"],
            parcels=parcels,
            agents=agents,
            station=station,
            strategies=strategies,
            created_at=exp_json["created_at"]
        )

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _load_heatmap(self, path: Path):
        data = self._load_json(path)
        from .models import Heatmap, HeatmapCell

        cells = [
            HeatmapCell(
                col=cell["cell"][0],
                row=cell["cell"][1],
                visits=cell["count"]
            )
            for cell in data["cell_visits"]
        ]
        return Heatmap(cells=cells)

    def _load_fleet(self, strategy_dir: Path):
        fleet_json = self._load_json(strategy_dir / "fleet.json")
        agents_json = self._load_json(strategy_dir / "agents.json")

        parcels, events = FleetLoader.load(fleet_json)
        trajectories = AgentLoader.load(agents_json)

        from .models import FleetData

        return FleetData(
            parcels=parcels,
            fleet_events=events,
            agent_trajectories=trajectories
        )
