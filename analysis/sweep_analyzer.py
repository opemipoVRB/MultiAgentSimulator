# analysis/sweep_analyzer.py
"""
Sweep-level analysis for station_scaling experiments.

Loads entire sweep directories (A2_P4, A4_P8, etc.) and provides
comprehensive analysis across different scales.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import re
from collections import defaultdict

from .single_loader import SingleExperimentLoader
from .models import Experiment, StrategyRun
from .aggregators import ExperimentAggregator


class SweepAnalyzer:
    """
    Analyzes complete station_scaling sweeps across multiple agent/parcel configurations.

    Directory structure expected:
    station_scaling/
    ├── A2_P4/
    │   └── experiment_xxxxxx/
    ├── A4_P8/
    │   └── experiment_yyyyyy/
    └── A6_P12/
        └── experiment_zzzzzz/
    """

    def __init__(self, sweep_dir: Union[str, Path]):
        """
        Initialize sweep analyzer.

        Args:
            sweep_dir: Path to station_scaling directory
        """
        self.sweep_dir = Path(sweep_dir)
        if not self.sweep_dir.exists():
            raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

        self.sweep_name = self.sweep_dir.name
        self._experiments: Dict[str, Experiment] = {}
        self._dataframe: Optional[pd.DataFrame] = None

        print(f"📊 SweepAnalyzer initialized for: {self.sweep_name}")

    @property
    def config_names(self) -> List[str]:
        """Get all available config names (A2_P4, A4_P8, etc.)"""
        configs = []
        for config_dir in self.sweep_dir.iterdir():
            if config_dir.is_dir() and re.match(r'A\d+_P\d+', config_dir.name):
                configs.append(config_dir.name)
        return sorted(configs, key=self._parse_config_key)

    @property
    def loaded_configs(self) -> List[str]:
        """Get names of already loaded configs."""
        return list(self._experiments.keys())

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Get comprehensive DataFrame with all loaded experiments.

        Returns:
            DataFrame with one row per strategy run, enhanced with config info
        """
        if self._dataframe is None:
            self._build_dataframe()
        return self._dataframe

    def load_config(self, config_name: str) -> Experiment:
        """
        Load a single config (e.g., 'A4_P8').

        Args:
            config_name: Config directory name

        Returns:
            Loaded experiment
        """
        if config_name in self._experiments:
            print(f"✓ {config_name} already loaded")
            return self._experiments[config_name]

        config_dir = self.sweep_dir / config_name
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        # Find experiment directory (should be only one)
        exp_dirs = list(config_dir.glob("experiment_*"))
        if not exp_dirs:
            raise FileNotFoundError(f"No experiment directory found in {config_dir}")
        if len(exp_dirs) > 1:
            print(f"⚠️  Multiple experiments found in {config_dir}, using most recent")
            exp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        exp_dir = exp_dirs[0]

        try:
            loader = SingleExperimentLoader(exp_dir)
            experiment = loader.load()
            self._experiments[config_name] = experiment

            agents, parcels = self._parse_config_name(config_name)
            print(f"✓ Loaded {config_name}: {len(experiment.strategies)} strategies, "
                  f"{agents}A×{parcels}P, seed={experiment.seed}")

            # Invalidate cached dataframe
            self._dataframe = None

            return experiment

        except Exception as e:
            print(f"✗ Failed to load {config_name}: {e}")
            raise

    def load_all_configs(self, skip_errors: bool = True) -> pd.DataFrame:
        """
        Load all available configs and return combined DataFrame.

        Args:
            skip_errors: If True, skip configs that fail to load

        Returns:
            Combined DataFrame with all strategy runs
        """
        configs = self.config_names
        print(f"🔄 Loading {len(configs)} configs: {configs}")

        for config_name in configs:
            try:
                self.load_config(config_name)
            except Exception as e:
                if skip_errors:
                    print(f"⚠️  Skipping {config_name}: {e}")
                    continue
                else:
                    raise

        print(f"✅ Loaded {len(self.loaded_configs)} configs successfully")
        return self.dataframe

    def get_experiment(self, config_name: str) -> Optional[Experiment]:
        """Get loaded experiment by config name."""
        return self._experiments.get(config_name)

    def get_strategy_across_scales(self, strategy_name: str) -> pd.DataFrame:
        """
        Get performance of one strategy across all loaded scales.

        Args:
            strategy_name: Strategy name to filter

        Returns:
            DataFrame with that strategy across all scales
        """
        df = self.dataframe
        return df[df['strategy_name'].str.contains(strategy_name, case=False, na=False)]

    def get_config_summary(self) -> pd.DataFrame:
        """
        Get summary of each config (agents, parcels, strategies, completion status).

        Returns:
            DataFrame with config-level summaries
        """
        summaries = []

        for config_name in self.config_names:
            config_dir = self.sweep_dir / config_name
            agents, parcels = self._parse_config_name(config_name)

            # Check if loaded
            is_loaded = config_name in self._experiments

            # Locate experiment directories
            exp_dirs = list(config_dir.glob("experiment_*"))

            # Initialize defaults
            n_experiments = 0
            n_strategies = completed_strategies = pending_strategies = seed = None
            experiment_status = "not_found"

            if exp_dirs:
                # Use most recent experiment directory
                exp_dir = sorted(exp_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                exp_json = exp_dir / "experiment.json"

                if exp_json.exists():
                    try:
                        import json
                        with open(exp_json, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        seed = data.get("seed")
                        experiment_status = data.get("status", "unknown")

                        strategies = data.get("strategies", [])

                        # ✅ Correct meaning: number of strategy instances
                        n_experiments = len(strategies)

                        n_strategies = len(strategies)
                        completed_strategies = sum(
                            1 for s in strategies if s.get("status") == "completed"
                        )
                        pending_strategies = sum(
                            1 for s in strategies if s.get("status") in {"pending", "aborted"}
                        )

                    except Exception as e:
                        print(f"Warning: Could not read {exp_json}: {e}")
                        experiment_status = "read_error"

            # Override with loaded data if available
            if is_loaded:
                exp = self._experiments[config_name]

                # Loaded strategies are already per-instance
                instance_ids = {
                    s.get("instance_id")
                    for s in strategies
                    if s.get("instance_id") is not None
                }
                n_experiments = len(instance_ids)
                n_strategies = len(exp.strategies)
                completed_strategies = sum(
                    1 for s in exp.strategies
                    if hasattr(s.metrics, "mission_makespan_steps")
                )
                seed = exp.seed

            summaries.append({
                "config_name": config_name,
                "n_agents": agents,
                "n_parcels": parcels,
                "n_experiments": n_experiments,
                "is_loaded": is_loaded,
                "n_strategies": n_strategies,
                "completed_strategies": completed_strategies,
                "pending_strategies": pending_strategies,
                "experiment_status": experiment_status,
                "seed": seed,
                "config_dir": str(config_dir),
            })

        return pd.DataFrame(summaries).sort_values(["n_agents", "n_parcels"])

    def compare_strategies(self, strategy_names: List[str],
                           metric: str = 'mission_makespan_steps') -> pd.DataFrame:
        """
        Compare specific strategies across all loaded scales.

        Args:
            strategy_names: List of strategy name patterns to compare
            metric: Metric to compare

        Returns:
            Pivot table with strategies vs configs
        """
        df = self.dataframe

        # Filter to requested strategies
        mask = df['strategy_name'].str.contains('|'.join(strategy_names), case=False, na=False)
        filtered = df[mask].copy()

        if filtered.empty:
            print(f"⚠️  No strategies found matching: {strategy_names}")
            return pd.DataFrame()

        # Create pivot table
        pivot = filtered.pivot_table(
            index='strategy_name',
            columns='config_name',
            values=metric,
            aggfunc='mean'
        )

        return pivot

    def get_sppdec_phases(self) -> pd.DataFrame:
        """
        Get SPPDEC Phase 1 variants (control, baseline_only, combined) for analysis.

        Returns:
            DataFrame filtered to SPPDEC variants only
        """
        df = self.dataframe

        sppdec_variants = [
            'sppdec_control',
            'sppdec_baseline_only',
            'sppdec_fairness_only',
            'sppdec_combined'
        ]

        pattern = '|'.join(sppdec_variants)
        mask = df['strategy_name'].str.contains(pattern, case=False, na=False)
        sppdec_df = df[mask].copy()

        if sppdec_df.empty:
            print("⚠️  No SPPDEC strategies found")
            return pd.DataFrame()

        # Add phase classification
        def classify_phase(name):
            name_lower = name.lower()
            if 'control' in name_lower:
                return 'Control'
            elif 'baseline_only' in name_lower:
                return 'Baseline Only'
            elif 'fairness_only' in name_lower:
                return 'Fairness Only'
            elif 'combined' in name_lower:
                return 'Combined'
            else:
                return 'Other'

        sppdec_df['sppdec_phase'] = sppdec_df['strategy_name'].apply(classify_phase)

        return sppdec_df

    def get_personality_variants(self) -> pd.DataFrame:
        """
        Get SPPDEC personality mixing experiments - UPDATED to include more patterns.

        Returns:
            DataFrame filtered to personality variants
        """
        df = self.dataframe

        # UPDATED: Include broader patterns that match your actual strategy names
        personality_patterns = [
            'personality_mix',
            'balanced_quartet',
            'no_reckless',
            'controlled_chaos',
            'efficiency_overdrive',
            'sppdec_fairness_only',  # NEW: Include your actual strategy name
            'sppdec_.*fairness',     # NEW: Pattern for fairness strategies
            'fairness_only'          # NEW: Direct match
        ]

        pattern = '|'.join(personality_patterns)
        mask = df['strategy_name'].str.contains(pattern, case=False, na=False)

        return df[mask].copy()

    def get_baseline_strategies(self) -> pd.DataFrame:
        """
        Get baseline/comparison strategies (centralized, CBBA, naive).

        Returns:
            DataFrame with baseline strategies
        """
        df = self.dataframe

        baseline_patterns = [
            'centralized_cc',
            'cbba_',
            'naive_decentralized',
            'decentralized_auction'
        ]

        pattern = '|'.join(baseline_patterns)
        mask = df['strategy_name'].str.contains(pattern, case=False, na=False)

        return df[mask].copy()

    def export_summary_csv(self, output_path: Union[str, Path]):
        """
        Export comprehensive summary to CSV.

        Args:
            output_path: Where to save CSV file
        """
        output_path = Path(output_path)

        # Get config summary
        config_summary = self.get_config_summary()
        config_summary.to_csv(output_path.parent / f"{output_path.stem}_configs.csv", index=False)

        # Get strategy summary if data loaded
        if self._experiments:
            strategy_summary = self.dataframe.groupby(['config_name', 'strategy_name']).agg({
                'mission_makespan_steps': ['mean', 'std', 'count'],
                'total_energy_consumed': ['mean', 'std'],
                'total_distance_traveled': ['mean', 'std'],
                'unsafe_failures': 'sum',
                'messages_sent': ['mean', 'std']
            }).round(3)

            strategy_summary.to_csv(output_path.parent / f"{output_path.stem}_strategies.csv")

        print(f"✅ Exported summaries to {output_path.parent}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_config_name(self, config_name: str) -> Tuple[int, int]:
        """Parse 'A4_P8' -> (4, 8)"""
        match = re.match(r'A(\d+)_P(\d+)', config_name)
        if not match:
            raise ValueError(f"Invalid config name format: {config_name}")
        return int(match.group(1)), int(match.group(2))

    def _parse_config_key(self, config_name: str) -> Tuple[int, int]:
        """Sorting key for config names."""
        return self._parse_config_name(config_name)

    def _build_dataframe(self):
        """Build comprehensive DataFrame from all loaded experiments."""
        if not self._experiments:
            self._dataframe = pd.DataFrame()
            return

        rows = []

        for config_name, experiment in self._experiments.items():
            agents, parcels = self._parse_config_name(config_name)

            for strategy in experiment.strategies:
                metrics = strategy.metrics

                # Base row from aggregator pattern
                row = {
                    # Config metadata
                    'config_name': config_name,
                    'n_agents': agents,
                    'n_parcels': parcels,
                    'sweep_name': self.sweep_name,

                    # Experiment metadata
                    'run_id': experiment.run_id,
                    'seed': experiment.seed,
                    'strategy_name': strategy.strategy_name,
                    'planner': metrics.planner,

                    # Performance metrics
                    'mission_makespan_steps': metrics.mission_makespan_steps,
                    'total_energy_consumed': metrics.total_energy_consumed,
                    'total_distance_traveled': metrics.total_distance_traveled,
                    'tasks_completed': metrics.tasks_completed,
                    'unsafe_failures': metrics.unsafe_failures,
                    'messages_sent': metrics.messages_sent,
                    'runtime_seconds': metrics.runtime_seconds,

                    # Derived metrics
                    'energy_per_task': metrics.energy_per_task,
                    'distance_per_task': metrics.distance_per_task,
                    'efficiency_ratio': metrics.efficiency_ratio,
                    'failure_rate': metrics.failure_rate,

                    # Configuration
                    'comm_reliability': metrics.comm_reliability,
                    'battery_capacity': metrics.battery_capacity,

                    # Fleet data
                    'n_collisions': strategy.fleet.n_collisions,
                    'parcels_delivered': strategy.fleet.parcels_delivered,
                    'parcels_picked': strategy.fleet.parcels_picked,

                    # Heatmap stats
                    'unique_cells_explored': strategy.heatmap.unique_cells,
                    'total_cell_visits': strategy.heatmap.total_visits,
                }

                # NEW: Add personality data if available
                if hasattr(strategy, 'personality_distribution') and strategy.personality_distribution:
                    row['personality_distribution'] = strategy.personality_distribution
                else:
                    row['personality_distribution'] = None

                if hasattr(strategy, 'personality_map') and strategy.personality_map:
                    row['personality_map'] = strategy.personality_map
                else:
                    row['personality_map'] = None

                rows.append(row)

        self._dataframe = pd.DataFrame(rows)

        # Add useful derived columns
        if not self._dataframe.empty:
            # Scale factor
            self._dataframe['scale_factor'] = (
                    self._dataframe['n_agents'] * self._dataframe['n_parcels']
            )

            # Strategy family (remove instance numbers)
            self._dataframe['strategy_family'] = (
                self._dataframe['strategy_name']
                    .str.replace(r'_instance_\d+', '', regex=True)
            )

            # Performance normalized by scale
            self._dataframe['makespan_per_parcel'] = (
                    self._dataframe['mission_makespan_steps'] / self._dataframe['n_parcels']
            )

            self._dataframe['energy_per_agent'] = (
                    self._dataframe['total_energy_consumed'] / self._dataframe['n_agents']
            )

            # IMPORTANT: Reset index to avoid duplicates
            self._dataframe = self._dataframe.reset_index(drop=True)

        print(f"📊 Built DataFrame: {len(self._dataframe)} strategy runs across {len(self._experiments)} configs")


# Convenience functions for notebook use
def load_sweep(sweep_dir: Union[str, Path]) -> SweepAnalyzer:
    """Convenience function to create and load a sweep analyzer."""
    analyzer = SweepAnalyzer(sweep_dir)
    analyzer.load_all_configs()
    return analyzer


def quick_summary(sweep_dir: Union[str, Path]) -> pd.DataFrame:
    """Quick config summary without loading full experiments."""
    analyzer = SweepAnalyzer(sweep_dir)
    return analyzer.get_config_summary()