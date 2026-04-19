# analysis/visualizers.py
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
    Experiment,
    StrategyRun,
    AgentTrajectory,
    FleetData,
    Station,
    Parcel,
    StrategyMetrics,
)

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


# ============================================================================
# Trajectory Visualization
# ============================================================================

class TrajectoryVisualizer:
    """Visualizes agent trajectories and paths."""

    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        """
        Initialize trajectory visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.colors = plt.cm.tab10.colors

    def plot_single_run(
            self,
            fleet: FleetData,
            station: Station,
            parcels: List[Parcel],
            title: str = "Agent Trajectories",
            save_path: Optional[Path] = None,
            show_grid: bool = True
    ):
        """
        Plot all agent trajectories on a single figure.

        Args:
            fleet: Fleet data with trajectories
            station: Station location
            parcels: List of parcels
            title: Plot title
            save_path: Optional path to save figure
            show_grid: Whether to show grid lines
        """
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # Draw station
        self._draw_station(ax, station)

        # Draw parcels
        self._draw_parcels(ax, parcels)

        # Draw agent trajectories
        for i, traj in enumerate(fleet.agent_trajectories):
            color = self.colors[i % len(self.colors)]
            self._draw_trajectory(ax, traj, color)

        # Formatting
        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)

        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        ax.invert_yaxis()  # Match grid coordinates
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved trajectory plot: {save_path}")

        plt.show()

    def plot_comparison(
            self,
            experiment: Experiment,
            strategy_names: List[str],
            save_dir: Optional[Path] = None
    ):
        """
        Plot trajectories for multiple strategies side-by-side.

        Args:
            experiment: Complete experiment with multiple strategies
            strategy_names: List of strategy names to compare
            save_dir: Optional directory to save figures
        """
        n_strategies = len(strategy_names)
        fig, axes = plt.subplots(1, n_strategies, figsize=(8 * n_strategies, 8))

        if n_strategies == 1:
            axes = [axes]

        for ax, strategy_name in zip(axes, strategy_names):
            strategy = experiment.get_strategy(strategy_name)
            if strategy is None:
                print(f"Warning: Strategy '{strategy_name}' not found")
                continue

            # Draw station
            self._draw_station(ax, experiment.station)

            # Draw parcels
            self._draw_parcels(ax, experiment.parcels)

            # Draw trajectories
            for i, traj in enumerate(strategy.fleet.agent_trajectories):
                color = self.colors[i % len(self.colors)]
                self._draw_trajectory(ax, traj, color, show_legend=(ax == axes[0]))

            # Formatting
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_title(
                f"{strategy_name}\n"
                f"{strategy.metrics.mission_makespan_steps} steps, "
                f"{strategy.metrics.total_energy_consumed:.1f} energy",
                fontsize=12,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            ax.set_aspect('equal')

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "trajectory_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison: {save_path}")

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

    def _draw_parcels(self, ax, parcels: List[Parcel]):
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
        ax.plot([], [], 'rx', markersize=12, markeredgewidth=2, label='Parcels')

    def _draw_trajectory(self, ax, traj: AgentTrajectory, color, show_legend=True):
        """Draw single agent trajectory."""
        cells = [traj.start_cell] + traj.cells_visited
        cols = [c[0] for c in cells]
        rows = [c[1] for c in cells]

        # Draw path
        ax.plot(cols, rows, '-', color=color, linewidth=2, alpha=0.6)

        # Draw start point
        label = f'{traj.agent_id}' if show_legend else None
        ax.plot(cols[0], rows[0], 'o', color=color, markersize=10, label=label)

        # Draw end point
        ax.plot(cols[-1], rows[-1], 's', color=color, markersize=8, alpha=0.7)


# ============================================================================
# Heatmap Visualization
# ============================================================================

class HeatmapVisualizer:
    """Visualizes exploration heatmaps."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize heatmap visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize

    def plot_heatmap(
            self,
            strategy_run: StrategyRun,
            grid_size: Tuple[int, int] = (25, 16),
            title: Optional[str] = None,
            save_path: Optional[Path] = None,
            cmap: str = 'hot'
    ):
        """
        Plot exploration heatmap for a strategy run.

        Args:
            strategy_run: Strategy run with heatmap data
            grid_size: (width, height) of grid
            title: Optional custom title
            save_path: Optional path to save figure
            cmap: Colormap name
        """
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # Convert heatmap to grid
        grid = strategy_run.heatmap.to_grid(grid_size[0], grid_size[1])
        grid_array = np.array(grid)

        # Plot heatmap
        im = ax.imshow(grid_array, cmap=cmap, interpolation='nearest', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visits', rotation=270, labelpad=20, fontsize=12)

        # Labels
        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        if title is None:
            title = f"Exploration Heatmap: {strategy_run.strategy_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add stats text
        stats_text = (
            f"Total visits: {strategy_run.heatmap.total_visits}\n"
            f"Unique cells: {strategy_run.heatmap.unique_cells}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap: {save_path}")

        plt.show()

    def plot_comparison(
            self,
            experiment: Experiment,
            strategy_names: List[str],
            grid_size: Tuple[int, int] = (25, 16),
            save_dir: Optional[Path] = None
    ):
        """
        Compare heatmaps across multiple strategies.

        Args:
            experiment: Complete experiment
            strategy_names: List of strategy names to compare
            grid_size: Grid dimensions
            save_dir: Optional directory to save figure
        """
        n_strategies = len(strategy_names)

        # Use constrained_layout instead of tight_layout
        fig, axes = plt.subplots(
            1,
            n_strategies,
            figsize=(8 * n_strategies, 6),
            constrained_layout=True
        )

        if n_strategies == 1:
            axes = [axes]

        # Find global max for consistent color scale
        global_max = 0
        grids = []
        for strategy_name in strategy_names:
            strategy = experiment.get_strategy(strategy_name)
            if strategy:
                grid = np.array(strategy.heatmap.to_grid(grid_size[0], grid_size[1]))
                grids.append(grid)
                global_max = max(global_max, grid.max())
            else:
                grids.append(None)

        # Plot each heatmap
        for ax, strategy_name, grid in zip(axes, strategy_names, grids):
            if grid is None:
                ax.text(
                    0.5, 0.5,
                    f"Strategy '{strategy_name}' not found",
                    ha='center',
                    va='center',
                    transform=ax.transAxes
                )
                continue

            im = ax.imshow(
                grid,
                cmap='hot',
                interpolation='nearest',
                vmin=0,
                vmax=global_max,
                aspect='auto'
            )

            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            ax.set_title(strategy_name, fontsize=12, fontweight='bold')

        # Single shared colorbar across all axes
        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        cbar.set_label('Visits', rotation=270, labelpad=20)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "heatmap_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap comparison: {save_path}")

        plt.show()


# ============================================================================
# Collision Visualization
# ============================================================================

class CollisionVisualizer:
    """Visualizes collision events."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize collision visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
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
            grid_size: (width, height) of grid
            title: Plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        # Count collisions per cell
        collision_grid = np.zeros(grid_size[::-1])  # (height, width)

        for event in fleet.fleet_events:
            if event.is_collision:
                col, row = event.cell
                if 0 <= row < grid_size[1] and 0 <= col < grid_size[0]:
                    collision_grid[row, col] += 1

        # Plot
        im = ax.imshow(collision_grid, cmap='Reds', interpolation='nearest', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Collision Count', rotation=270, labelpad=20, fontsize=12)

        ax.set_xlabel("Column", fontsize=12)
        ax.set_ylabel("Row", fontsize=12)
        ax.set_title(f"{title} (Total: {fleet.n_collisions})", fontsize=14, fontweight='bold')

        # Add stats
        if fleet.n_collisions > 0:
            stats_text = (
                f"Total collisions: {fleet.n_collisions}\n"
                f"Unique locations: {int((collision_grid > 0).sum())}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10
            )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved collision heatmap: {save_path}")

        plt.show()

    def plot_timeline(
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
        fig, ax = plt.subplots(figsize=(12, 4))

        # Extract collision times
        collision_steps = [
            event.time.step
            for event in fleet.fleet_events
            if event.is_collision
        ]

        if not collision_steps:
            ax.text(0.5, 0.5, "No collisions detected",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14)
        else:
            # Create histogram
            bins = np.arange(0, max(collision_steps) + 50, 50)
            ax.hist(collision_steps, bins=bins, edgecolor='black', alpha=0.7)

            ax.set_xlabel("Simulation Step", fontsize=12)
            ax.set_ylabel("Collision Count", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved collision timeline: {save_path}")

        plt.show()


# ============================================================================
# Metrics Visualization
# ============================================================================

class MetricsVisualizer:
    """Visualizes performance metrics across strategies."""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize metrics visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize

    def plot_comparison(
            self,
            experiment: Experiment,
            metrics: List[str] = None,
            save_dir: Optional[Path] = None
    ):
        """
        Create bar plots comparing metrics across strategies.

        Args:
            experiment: Complete experiment
            metrics: List of metric names to plot (default: key metrics)
            save_dir: Optional directory to save figures
        """
        if metrics is None:
            metrics = [
                "mission_makespan_steps",
                "total_energy_consumed",
                "total_distance_traveled",
                "unsafe_failures"
            ]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        strategy_names = [s.strategy_name for s in experiment.strategies]

        for ax, metric in zip(axes, metrics):
            values = [getattr(s.metrics, metric) for s in experiment.strategies]

            # Create bar plot
            bars = ax.bar(range(len(strategy_names)), values, alpha=0.7, edgecolor='black')

            # Color bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_names)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Formatting
            ax.set_xticks(range(len(strategy_names)))
            ax.set_xticklabels(strategy_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value:.1f}' if isinstance(value, float) else f'{value}',
                        ha='center', va='bottom', fontsize=9)

        plt.suptitle(f"Strategy Comparison (Seed: {experiment.seed})",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved metrics comparison: {save_path}")

        plt.show()

    def plot_pareto(
            self,
            experiment: Experiment,
            x_metric: str = "mission_makespan_steps",
            y_metric: str = "total_energy_consumed",
            save_path: Optional[Path] = None
    ):
        """
        Create Pareto plot showing tradeoffs between two metrics.

        Args:
            experiment: Complete experiment
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        strategy_names = [s.strategy_name for s in experiment.strategies]
        x_values = [getattr(s.metrics, x_metric) for s in experiment.strategies]
        y_values = [getattr(s.metrics, y_metric) for s in experiment.strategies]

        # Plot points
        scatter = ax.scatter(x_values, y_values, s=100, alpha=0.6, edgecolors='black')

        # Add labels
        for i, name in enumerate(strategy_names):
            ax.annotate(
                name,
                (x_values[i], y_values[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )

        # Formatting
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(
            f"Performance Tradeoff (Seed: {experiment.seed})",
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Pareto plot: {save_path}")

        plt.show()

    def plot_metric_distribution(
            self,
            df,
            metric: str,
            group_col: str = "strategy_full_name",
            save_path: Optional[Path] = None,
    ):
        """
        Boxplot showing distribution of a metric across multiple runs.
        """
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        sns.boxplot(
            data=df,
            x=group_col,
            y=metric,
            ax=ax
        )

        sns.stripplot(
            data=df,
            x=group_col,
            y=metric,
            ax=ax,
            color="black",
            alpha=0.4,
            size=3
        )

        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_pareto_runs(
            self,
            df,
            x_metric: str,
            y_metric: str,
            group_col: str = "strategy_full_name",
            save_path: Optional[Path] = None,
    ):
        """
        Scatter plot showing tradeoff across all runs.
        """
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        sns.scatterplot(
            data=df,
            x=x_metric,
            y=y_metric,
            hue=group_col,
            alpha=0.7,
            s=60,
            ax=ax
        )

        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title("Tradeoff Across Runs")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_rank_distribution(
            self,
            df,
            metric: str,
            group_col: str = "strategy_full_name",
    ):
        """
        Ranks strategies per run and shows rank distribution.
        """
        ranked = (
            df
                .groupby("run_id")
                .apply(lambda g: g.assign(rank=g[metric].rank()))
                .reset_index(drop=True)
        )

        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        sns.boxplot(
            data=ranked,
            x=group_col,
            y="rank",
            ax=ax
        )

        ax.invert_yaxis()
        ax.set_ylabel("Rank (lower is better)")
        ax.set_xlabel("Strategy")
        ax.set_title(f"Rank distribution based on {metric.replace('_', ' ')}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.show()

    def plot_mean_ci(
        self,
        df,
        metric: str,
        group_col: str = "strategy_full_name",
        save_path: Optional[Path] = None,
    ):
        grouped = df.groupby(group_col)[metric]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count()

        ci = 1.96 * std / np.sqrt(n)

        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        ax.bar(mean.index, mean.values, yerr=ci.values, capsize=5)

        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Mean ± 95% CI: {metric.replace('_', ' ').title()}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


# ============================================================================
# Scaling Visualization
# ============================================================================

class ScalingVisualizer:
    """
    Visualizes performance scaling across number of agents and tasks.
    Operates on aggregated dataframe (ExperimentAggregator.dataframe).
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize

    def plot_scaling(
        self,
        df,
        metric: str,
        x_axis: str = "n_drones",   # or "n_parcels"
        group_col: str = "strategy_full_name",
        ci: float = 0.95,
        save_path: Optional[Path] = None,
    ):
        """
        Plot scaling curves with confidence intervals.

        Args:
            df: Aggregated dataframe
            metric: Metric column name
            x_axis: 'n_drones' or 'n_parcels'
            group_col: Strategy grouping column
            ci: Confidence interval level (0.95 default)
            save_path: Optional save location
        """

        assert x_axis in df.columns, f"{x_axis} not in dataframe"
        assert metric in df.columns, f"{metric} not in dataframe"
        assert group_col in df.columns, f"{group_col} not in dataframe"

        # Aggregate
        grouped = (
            df
            .groupby([group_col, x_axis])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Compute confidence interval
        z = 1.96 if ci == 0.95 else 1.0
        grouped["ci"] = z * grouped["std"] / np.sqrt(grouped["count"])

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        for name, g in grouped.groupby(group_col):
            ax.plot(g[x_axis], g["mean"], marker="o", label=name)
            ax.fill_between(
                g[x_axis],
                g["mean"] - g["ci"],
                g["mean"] + g["ci"],
                alpha=0.15,
            )

        ax.set_xlabel(x_axis.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Scaling Analysis: {metric.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved scaling plot: {save_path}")

        plt.show()

    def plot_surface(
        self,
        df,
        metric: str,
        group: str,
        save_path: Optional[Path] = None,
    ):
        """
        Heatmap of performance across agents × parcels for one strategy.
        """
        subset = df[df["strategy_full_name"] == group]

        if subset.empty:
            raise ValueError(f"No data found for strategy {group}")

        pivot = subset.pivot_table(
            index="n_drones",
            columns="n_parcels",
            values=metric,
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)

        im = ax.imshow(pivot.values, aspect="auto", origin="lower")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("Number of Parcels")
        ax.set_ylabel("Number of Agents")
        ax.set_title(f"{group} | {metric.replace('_', ' ').title()}")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace("_", " ").title())

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
