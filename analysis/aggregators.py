# analysis/aggregators.py
"""
Batch analysis and statistical aggregation tools.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from pathlib import Path

from .models import Experiment, StrategyMetrics


# ============================================================================
# Experiment Aggregator
# ============================================================================

class ExperimentAggregator:
    """Aggregates data across multiple experiments."""

    def __init__(self, experiments: List[Experiment]):
        """
        Initialize aggregator with list of experiments.

        Args:
            experiments: List of loaded experiments
        """
        self.experiments = experiments
        self._dataframe = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Convert all experiments to a pandas DataFrame.

        Returns:
            DataFrame with one row per strategy run
        """
        if self._dataframe is None:
            self._dataframe = self._build_dataframe()
        return self._dataframe

    def _build_dataframe(self) -> pd.DataFrame:
        """Build comprehensive DataFrame from all experiments."""
        rows = []

        for exp in self.experiments:
            for strategy in exp.strategies:
                metrics = strategy.metrics

                row = {
                    # Experiment metadata
                    'run_id': exp.run_id,
                    'seed': exp.seed,
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
                    'n_drones': metrics.n_drones,
                    'n_parcels': metrics.n_parcels,
                    'comm_reliability': metrics.comm_reliability,
                    'battery_capacity': metrics.battery_capacity,

                    # Fleet data
                    'n_collisions': strategy.fleet.n_collisions,
                    'parcels_delivered': strategy.fleet.parcels_delivered,
                }

                rows.append(row)

        return pd.DataFrame(rows)

    def summary(self, group_by: str = 'strategy_name') -> pd.DataFrame:
        """
        Compute summary statistics grouped by strategy.

        Args:
            group_by: Column to group by (default: strategy_name)

        Returns:
            DataFrame with mean, std, min, max for each metric
        """
        df = self.dataframe

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in [group_by, 'seed', 'run_id']]

        summary = df.groupby(group_by)[numeric_cols].agg(['mean', 'std', 'min', 'max'])

        for col in numeric_cols:
            cv = (summary[(col, 'std')] / summary[(col, 'mean')] * 100).round(1)
            summary[(col, 'cv%')] = cv

        return summary.round(2)

    def export_summary(self, output_path: Path):
        """
        Export summary statistics to CSV.

        Args:
            output_path: Path to save CSV file
        """
        summary = self.summary()
        summary.to_csv(output_path)
        print(f"✓ Exported summary to: {output_path}")

    def export_raw(self, output_path: Path):
        """
        Export raw data to CSV.

        Args:
            output_path: Path to save CSV file
        """
        self.dataframe.to_csv(output_path, index=False)
        print(f"✓ Exported raw data to: {output_path}")


# ============================================================================
# Statistical Analyzer
# ============================================================================

class StatisticalAnalyzer:
    """Performs statistical analysis on experiment data."""

    def __init__(self, aggregator: ExperimentAggregator):
        """
        Initialize analyzer with aggregator.

        Args:
            aggregator: ExperimentAggregator instance
        """
        self.aggregator = aggregator
        self.df = aggregator.dataframe

    def pairwise_ttest(
        self,
        metric: str,
        baseline_strategy: str,
        comparison_strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Perform pairwise t-tests comparing strategies.

        Args:
            metric: Metric to compare
            baseline_strategy: Reference strategy name
            comparison_strategies: Strategies to compare against baseline

        Returns:
            DataFrame with t-test results
        """
        if comparison_strategies is None:
            comparison_strategies = [
                s for s in self.df['strategy_name'].unique()
                if s != baseline_strategy
            ]

        baseline_data = self.df[self.df['strategy_name'] == baseline_strategy][metric]

        results = []

        for strategy in comparison_strategies:
            comparison_data = self.df[self.df['strategy_name'] == strategy][metric]

            t_stat, p_value = stats.ttest_ind(baseline_data, comparison_data)

            pooled_std = np.sqrt(
                ((len(baseline_data) - 1) * baseline_data.std() ** 2 +
                 (len(comparison_data) - 1) * comparison_data.std() ** 2) /
                (len(baseline_data) + len(comparison_data) - 2)
            )
            cohens_d = (baseline_data.mean() - comparison_data.mean()) / pooled_std

            pct_diff = (
                (comparison_data.mean() - baseline_data.mean()) /
                baseline_data.mean() * 100
            )

            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            results.append({
                'comparison_strategy': strategy,
                'baseline_mean': baseline_data.mean(),
                'comparison_mean': comparison_data.mean(),
                'difference': comparison_data.mean() - baseline_data.mean(),
                'pct_difference': pct_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': sig,
                'cohens_d': cohens_d,
                'n_baseline': len(baseline_data),
                'n_comparison': len(comparison_data)
            })

        return pd.DataFrame(results).round(4)

    def anova(self, metric: str) -> Dict:
        """
        Perform one-way ANOVA across all strategies.

        Args:
            metric: Metric to analyze

        Returns:
            Dictionary with ANOVA results
        """
        groups = [
            self.df[self.df['strategy_name'] == strategy][metric].values
            for strategy in self.df['strategy_name'].unique()
        ]

        f_stat, p_value = stats.f_oneway(*groups)

        return {
            'metric': metric,
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_strategies': len(groups),
            'total_n': sum(len(g) for g in groups)
        }

    def plot_boxplot(
            self,
            metric: str,
            title: Optional[str] = None,
            save_path: Optional[Path] = None
    ):
        """
        Create box plot comparing strategies.

        Args:
            metric: Metric to plot
            title: Optional plot title
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=self.df,
            x='strategy_name',
            y=metric,
            hue='strategy_name',
            ax=ax,
            palette='Set3',
            legend=False
        )

        sns.stripplot(
            data=self.df,
            x='strategy_name',
            y=metric,
            ax=ax,
            color='black',
            alpha=0.3,
            size=4
        )

        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)

        if title is None:
            title = f"Distribution of {metric.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Correct way to rotate labels without triggering matplotlib warning
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved boxplot: {save_path}")

        plt.show()

    def plot_parameter_sweep(
        self,
        parameter: str,
        metric: str,
        strategy_filter: Optional[str] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot metric vs parameter for sensitivity analysis.

        Args:
            parameter: Parameter column name
            metric: Metric to plot
            strategy_filter: Optional strategy name to filter
            save_path: Optional path to save figure
        """
        df = self.df
        if strategy_filter:
            df = df[df['strategy_name'].str.contains(strategy_filter)]

        fig, ax = plt.subplots(figsize=(10, 6))

        grouped = df.groupby(parameter)[metric].agg(['mean', 'std', 'count'])
        grouped = grouped.sort_index()

        ax.errorbar(
            grouped.index,
            grouped['mean'],
            yerr=grouped['std'],
            marker='o',
            linewidth=2,
            markersize=8,
            capsize=5,
            capthick=2
        )

        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(
            f"Parameter Sensitivity: {parameter} vs {metric}",
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved parameter sweep: {save_path}")

        plt.show()

    def generate_report(self, output_dir: Path):
        """
        Generate comprehensive statistical report.

        Args:
            output_dir: Directory to save report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating statistical report...")

        summary = self.aggregator.summary()
        summary.to_csv(output_dir / "summary_statistics.csv")
        print("✓ Summary statistics")

        key_metrics = [
            'mission_makespan_steps',
            'total_energy_consumed',
            'total_distance_traveled',
            'unsafe_failures'
        ]

        anova_results = []
        for metric in key_metrics:
            result = self.anova(metric)
            anova_results.append(result)

        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(output_dir / "anova_results.csv", index=False)
        print("✓ ANOVA results")

        for metric in key_metrics:
            self.plot_boxplot(
                metric,
                save_path=output_dir / f"boxplot_{metric}.png"
            )

        print(f"\n✓ Report generated in: {output_dir}")