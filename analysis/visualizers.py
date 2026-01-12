"""
Publication-quality visualization classes for SPPDEC experiments.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .models import (
    Experiment, StrategyRun, AgentTrajectory, FleetData,
    Station, Parcel
)


class TrajectoryVisualizer:
    """Visualizes agent trajectories and paths."""

    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        self.figsize = figsize
        self.colors = plt.cm.tab10.colors

    def plot_single_run(
            self,
            fleet: FleetData,
            station: Station,
            parcels: List[Parcel],
            title: str = "Agent Trajectories",
            save_path: Optional[Path] = None
    ):
        """
        Plot all agent trajectories on a single figure.

        Args:
            fleet: Fleet data with trajectories
            station: Station location
            parcels: List of parcels
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw station
        self._draw_station(ax, station)

        # Draw parcels
        self._draw_parcels(ax, parcels)

        # Draw agent trajectories
        for i, traj in enumerate(fleet.agent_trajectories):
            color = self.colors[i % len(self.colors)]
            self._draw_trajectory(ax, traj, color)

        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Match grid coordinates

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_experiment_comparison(
            self,
            experiment: Experiment,
            strategy_names: Optional[List[str]] = None,
            save_dir: Optional[Path] = None
    ):
        """
        Plot trajectories for multiple strategies side-by-side.

        Args:
            experiment: Experiment with multiple strategies
            strategy_names: Optional list of strategies to plot (default: all)
            save_dir: Optional directory to save figures
        """
        if strategy_names is None:
            strategies = experiment.strategies
        else:
            strategies = [experiment.get_strategy(name) for name in strategy_names]
            strategies = [s for s in strategies if s is not None]

        n_strategies = len(strategies)
        fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 6))

        if n_strategies == 1:
            axes = [axes]

        for ax, strategy in zip(axes, strategies):
            # Draw station
            self._draw_station(ax, experiment.station)

            # Draw parcels
            self._draw_parcels(ax, experiment.parcels, add_legend=False)

            # Draw agent trajectories
            for i, traj in enumerate(strategy.fleet.agent_trajectories):
                color = self.colors[i % len(self.colors)]
                self._draw_trajectory(ax, traj, color, add_legend=(i == 0))

            ax.set_xlabel("Column", fontsize=10)
            ax.set_ylabel("Row", fontsize=10)
            ax.set_title(
                f"{strategy.strategy_name}\n{strategy.metrics.mission_makespan_steps} steps",
                fontsize=11,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()

        plt.suptitle(f"Trajectory Comparison (Seed: {experiment.seed})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"trajectories_comparison_seed_{experiment.seed}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def _draw_station(self, ax, station: Station):
        """Draw station rectangle."""
        rect = patches.Rectangle(
            (station.col, station.row),
            station.w,
            station.h,
            linewidth=2,
            edgecolor='green',
            facecolor='lightgreen',
            alpha=0.3,
            label='Station'
        )
        ax.add_patch(rect)

    def _draw_parcels(self, ax, parcels: List[Parcel], add_legend: bool = True):
        """Draw parcel locations."""
        for parcel in parcels:
            ax.plot(
                parcel.col,
                parcel.row,
                'x',
                markersize=12,
                markeredgewidth=2,
                color='red',
                alpha=0.7
            )
        # Add single legend entry
        if add_legend:
            ax.plot([], [], 'rx', markersize=12, markeredgewidth=2, label='Parcels')

    def _draw_trajectory(self, ax, traj: AgentTrajectory, color, add_legend: bool = True):
        """Draw single agent trajectory."""
        cells = [traj.start_cell] + traj.cells_visited
        cols = [c[0] for c in cells]
        rows = [c[1] for c in cells]

        # Draw path
        ax.plot(cols, rows, '-', color=color, linewidth=2, alpha=0.6)

        # Draw start point
        if add_legend:
            ax.plot(cols[0], rows[0], 'o', color=color, markersize=10, label=f'{traj.agent_id}')
        else:
            ax.plot(cols[0], rows[0], 'o', color=color, markersize=10)

        # Draw end point
        ax.plot(cols[-1], rows[-1], 's', color=color, markersize=10)


class HeatmapVisualizer:
    """Visualizes exploration heatmaps."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    def plot_heatmap(
            self,
            strategy_run: StrategyRun,
            grid_size: Tuple[int, int] = (25, 16),
            title: Optional[str] = None,
            save_path: Optional[Path] = None
    ):
        """
        Plot exploration heatmap for a strategy run.

        Args:
            strategy_run: Strategy run with heatmap data
            grid_size: (width, height) of grid
            title: Optional custom title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Convert heatmap to grid
        grid = strategy_run.heatmap.to_grid(grid_size[0], grid_size[1])
        grid_array = np.array(grid)

        # Plot heatmap
        im = ax.imshow(grid_array, cmap='hot', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visits', rotation=270, labelpad=20, fontsize=12)

        # Labels
        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        if title is None:
            title = f"Exploration Heatmap: {strategy_run.strategy_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_experiment_heatmaps(
            self,
            experiment: Experiment,
            strategy_names: Optional[List[str]] = None,
            grid_size: Tuple[int, int] = (25, 16),
            save_dir: Optional[Path] = None
    ):
        """
        Plot heatmaps for multiple strategies side-by-side.

        Args:
            experiment: Experiment with multiple strategies
            strategy_names: Optional list of strategies to plot
            grid_size: Grid dimensions
            save_dir: Optional directory to save figure
        """
        if strategy_names is None:
            strategies = experiment.strategies
        else:
            strategies = [experiment.get_strategy(name) for name in strategy_names]
            strategies = [s for s in strategies if s is not None]

        n_strategies = len(strategies)
        fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 5))

        if n_strategies == 1:
            axes = [axes]

        # Find global max for consistent colorbar
        max_visits = max(
            max(cell.visits for cell in s.heatmap.cells) if s.heatmap.cells else 1
            for s in strategies
        )

        for ax, strategy in zip(axes, strategies):
            grid = strategy.heatmap.to_grid(grid_size[0], grid_size[1])
            grid_array = np.array(grid)

            im = ax.imshow(grid_array, cmap='hot', interpolation='nearest', vmin=0, vmax=max_visits)

            ax.set_xlabel("Column", fontsize=10)
            ax.set_ylabel("Row", fontsize=10)
            ax.set_title(
                f"{strategy.strategy_name}\n{strategy.heatmap.total_visits} visits",
                fontsize=11,
                fontweight='bold'
            )

        # Add shared colorbar
        fig.colorbar(im, ax=axes, label='Visits', orientation='horizontal', pad=0.05)

        plt.suptitle(f"Heatmap Comparison (Seed: {experiment.seed})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"heatmaps_comparison_seed_{experiment.seed}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()


class CollisionVisualizer:
    """Visualizes collision events."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    def plot_collisions(
            self,
            fleet: FleetData,
            grid_size: Tuple[int, int] = (25, 16),
            title: str = "Collision Heatmap",
            save_path: Optional[Path] = None
    ):
        """
        Plot collision locations as heatmap.

        Args:
            fleet: Fleet data with collision events
            grid_size: Grid dimensions (width, height)
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Count collisions per cell
        collision_grid = np.zeros(grid_size[::-1])  # (height, width)

        for event in fleet.fleet_events:
            if event.is_collision:
                col, row = event.cell
                if 0 <= row < grid_size[1] and 0 <= col < grid_size[0]:
                    collision_grid[row, col] += 1

        # Plot
        im = ax.imshow(collision_grid, cmap='Reds', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Collisions', rotation=270, labelpad=20, fontsize=12)

        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        ax.set_title(f"{title} (Total: {fleet.n_collisions})", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_collision_timeline(
            self,
            fleet: FleetData,
            title: str = "Collision Timeline",
            save_path: Optional[Path] = None
    ):
        """
        Plot collisions over time.

        Args:
            fleet: Fleet data with collision events
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        collision_events = [e for e in fleet.fleet_events if e.is_collision]

        if not collision_events:
            print("No collisions to plot")
            return

        steps = [e.time.step for e in collision_events]
        agents_involved = [e.n_agents_involved for e in collision_events]

        ax.scatter(steps, agents_involved, s=100, alpha=0.6, c='red')
        ax.set_xlabel("Simulation Step", fontsize=12)
        ax.set_ylabel("Agents Involved", fontsize=12)
        ax.set_title(f"{title} (Total: {len(collision_events)})", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()


class MetricsVisualizer:
    """Visualizes performance metrics comparisons."""

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
        sns.set_style("whitegrid")

    def plot_metric_comparison(
            self,
            experiment: Experiment,
            metric: str = "mission_makespan_steps",
            title: Optional[str] = None,
            save_path: Optional[Path] = None
    ):
        """
        Bar plot comparing single metric across strategies.

        Args:
            experiment: Experiment with multiple strategies
            metric: Metric name (attribute of StrategyMetrics)
            title: Optional custom title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        strategy_names = [s.strategy_name for s in experiment.strategies]
        values = [getattr(s.metrics, metric) for s in experiment.strategies]

        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
        bars = ax.bar(strategy_names, values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        if title is None:
            title = f"{metric.replace('_', ' ').title()} Comparison"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_multi_metric_comparison(
            self,
            experiment: Experiment,
            metrics: List[str] = ["mission_makespan_steps", "total_energy_consumed", "unsafe_failures"],
            save_dir: Optional[Path] = None
    ):
        """
        Plot multiple metrics side-by-side.

        Args:
            experiment: Experiment with multiple strategies
            metrics: List of metric names
            save_dir: Optional directory to save figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        strategy_names = [s.strategy_name for s in experiment.strategies]
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))

        for ax, metric in zip(axes, metrics):
            values = [getattr(s.metrics, metric) for s in experiment.strategies]
            bars = ax.bar(strategy_names, values, color=colors, edgecolor='black', linewidth=1.5)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

            ax.set_xlabel("Strategy", fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f"Metrics Comparison (Seed: {experiment.seed})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"metrics_comparison_seed_{experiment.seed}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_pareto_front(
            self,
            experiment: Experiment,
            x_metric: str = "mission_makespan_steps",
            y_metric: str = "total_energy_consumed",
            title: str = "Speed vs Energy Tradeoff",
            save_path: Optional[Path] = None
    ):
        """
        Plot Pareto front showing tradeoff between two metrics.

        Args:
            experiment: Experiment with multiple strategies
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        x_values = [getattr(s.metrics, x_metric) for s in experiment.strategies]
        y_values = [getattr(s.metrics, y_metric) for s in experiment.strategies]
        labels = [s.strategy_name for s in experiment.strategies]

        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        for x, y, label, color in zip(x_values, y_values, labels, colors):
            ax.scatter(x, y, s=200, alpha=0.7, color=color, edgecolors='black', linewidth=2)
            ax.annotate(
                label,
                (x, y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3)
            )

        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()