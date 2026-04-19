# analysis/notebook_utils_redesigned.py
"""
Redesigned notebook utilities with proper visualization design principles.
Enforces readable, focused plots by construction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Union

# Import your existing classes
from .sweep_analyzer import SweepAnalyzer
from .visualizers import MetricsVisualizer, ScalingVisualizer
from .aggregators import StatisticalAnalyzer, ExperimentAggregator


class NotebookAnalyzer:
    """
    Redesigned analyzer that enforces good visualization design by construction.
    """

    def __init__(self, sweep_dir: Union[str, Path]):
        """Initialize notebook analyzer."""
        self.sweep = SweepAnalyzer(sweep_dir)
        self.metrics_viz = MetricsVisualizer()
        self.scaling_viz = ScalingVisualizer()
        self.stats_analyzer = None

        # Strategy display name mapping - use full descriptive names for personalities
        self.DISPLAY_NAMES = {
            "sppdec_personality_mix_efficiency_overdrive": "Efficiency Overdrive",
            "sppdec_personality_mix_3_brains_1_wildcard": "3 Brains + 1 Wildcard",
            "sppdec_personality_mix_balanced": "Balanced Mix",
            "sppdec_personality_mix_balanced_quartet": "Balanced Quartet",
            "sppdec_personality_mix_controlled_chaos": "Controlled Chaos",
            "sppdec_personality_mix_no_reckless": "No Reckless",
            "sppdec_personality_mix_conservative": "Conservative Mix",
            "sppdec_personality_mix_aggressive": "Aggressive Mix",
            "sppdec_personality_mix_explorer": "Explorer Mix",
            "sppdec_personality_mix_optimizer": "Optimizer Mix",
            "sppdec_phase_1_control": "SPPDEC Control",
            "sppdec_phase_1_combined": "SPPDEC Combined",
            "centralized_optimal": "Centralized",
            "decentralized_auction": "Auction",
            "greedy_baseline": "Greedy",
            "random_baseline": "Random",
            "naive_baseline": "Naive"
        }

    def load_all(self, show_progress: bool = True) -> pd.DataFrame:
        """
        Load all configs with optimized sequential loading and progress tracking.

        Args:
            show_progress: Whether to show loading progress
        """
        import time

        if show_progress:
            print("📊 Loading experiment data...")

        start_time = time.time()

        # Use the existing sweep analyzer but with progress tracking
        config_summary = self.sweep.get_config_summary()
        configs_to_load = config_summary[~config_summary['is_loaded']]['config_name'].tolist()

        if configs_to_load and show_progress:
            print(f"📁 Loading {len(configs_to_load)} configurations...")

        # Load with progress tracking
        df = self._load_with_progress(configs_to_load, show_progress)

        if not df.empty:
            class DummyAggregator:
                def __init__(self, dataframe):
                    self.dataframe = dataframe

            dummy_agg = DummyAggregator(df)
            self.stats_analyzer = StatisticalAnalyzer(dummy_agg)

        if show_progress:
            elapsed = time.time() - start_time
            print(f"✅ Loading completed in {elapsed:.1f} seconds")
            print(f"📊 Loaded {len(df)} total runs")

        return df

    def _load_with_progress(self, configs_to_load: List[str], show_progress: bool) -> pd.DataFrame:
        """Load configs with progress updates."""
        if not configs_to_load:
            # Already loaded
            return self.sweep.dataframe if hasattr(self.sweep, 'dataframe') else self.sweep.load_all_configs()

        # For progress tracking during load_all_configs
        if show_progress:
            total_configs = len(configs_to_load)
            print("⏳ Loading configurations...")

            # Monkey patch to show progress (simple approach)
            original_method = None
            if hasattr(self.sweep, '_load_config_data'):
                original_method = self.sweep._load_config_data
                loaded_count = [0]  # Use list for closure

                def progress_wrapper(config_name):
                    result = original_method(config_name)
                    loaded_count[0] += 1
                    if loaded_count[0] % max(1, total_configs // 5) == 0:
                        progress = (loaded_count[0] / total_configs) * 100
                        print(f"⏳ Progress: {progress:.0f}% ({loaded_count[0]}/{total_configs})")
                    return result

                self.sweep._load_config_data = progress_wrapper

        try:
            df = self.sweep.load_all_configs()
        finally:
            # Restore original method if we modified it
            if original_method is not None:
                self.sweep._load_config_data = original_method

        return df

    # def _get_display_name(self, strategy_name: str, max_length: int = None) -> str:
    #     """
    #     Get display name for strategy.
    #
    #     Args:
    #         strategy_name: Full strategy name
    #         max_length: Optional max length for display (None = use full name)
    #     """
    #     display_name = self.DISPLAY_NAMES.get(strategy_name, strategy_name)
    #
    #     # If max_length specified and name is too long, truncate intelligently
    #     if max_length and len(display_name) > max_length:
    #         # For personality names, try to keep the key descriptive part
    #         if "personality_mix" in strategy_name:
    #             # Keep the personality type (the part after the last underscore)
    #             parts = display_name.split()
    #             if len(parts) > 1:
    #                 return parts[-1][:max_length]  # Last word truncated
    #         return display_name[:max_length - 3] + "..."
    #
    #     return display_name

    def _setup_clean_plot(self, figsize=(10, 6), title=None, xlabel=None, ylabel=None):
        """Set up a clean, readable plot with consistent styling."""
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

        fig, ax = plt.subplots(figsize=figsize)

        if title:
            ax.set_title(title, fontweight='bold', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontweight='medium')
        if ylabel:
            ax.set_ylabel(ylabel, fontweight='medium')

        # Clean, subtle grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return fig, ax

    def _select_top_strategies(self, df: pd.DataFrame, metric: str, max_strategies: int = 5) -> pd.DataFrame:
        """
        Select top performing strategies to avoid visual overload.
        Enforces the "never more than 5 strategies" rule.
        """
        if df.empty:
            return df

        # Get mean performance by strategy
        strategy_means = (
            df.groupby('strategy_family')[metric]
                .mean()
                .sort_values()
                .head(max_strategies)
        )

        # Filter to top strategies only
        selected_df = df[df['strategy_family'].isin(strategy_means.index)]

        print(f"📊 Selected top {len(strategy_means)} strategies (of {df['strategy_family'].nunique()} total)")
        for i, (strategy, mean_val) in enumerate(strategy_means.items(), 1):
            print(f"  {i}. {self._get_display_name(strategy)}: {mean_val:.1f}")

        return selected_df

    def _plot_with_percentiles(self, ax, x_vals, y_data, label, color, **kwargs):
        """
        Plot median line with percentile bands instead of mean ± std.
        Much cleaner visual representation.
        """
        if len(y_data) == 0:
            return

        # Calculate percentiles
        median = np.percentile(y_data, 50)
        q25 = np.percentile(y_data, 25)
        q75 = np.percentile(y_data, 75)

        # Plot median line
        ax.plot(x_vals, [median] * len(x_vals),
                label=label, color=color, linewidth=2.5,
                marker='o', markersize=6, **kwargs)

        # Add percentile band
        ax.fill_between(x_vals, [q25] * len(x_vals), [q75] * len(x_vals),
                        alpha=0.25, color=color)

    def quick_overview(self) -> None:
        """Print quick overview of the sweep."""
        config_summary = self.sweep.get_config_summary()

        print("🔍 SWEEP OVERVIEW")
        print("=" * 50)
        print(f"Sweep name: {self.sweep.sweep_name}")
        print(f"Available configs: {len(config_summary)}")
        print(f"Loaded configs: {len(self.sweep.loaded_configs)}")

        if not config_summary.empty:
            print("\nConfig Details:")
            summary_display = config_summary[['config_name', 'n_agents', 'n_parcels',
                                              'is_loaded', 'n_strategies', 'completed_strategies']]
            print(summary_display.to_string(index=False))

            print(f"\nScale range:")
            print(f"  Agents: {config_summary['n_agents'].min()}-{config_summary['n_agents'].max()}")
            print(f"  Parcels: {config_summary['n_parcels'].min()}-{config_summary['n_parcels'].max()}")

    def strategy_overview(self) -> pd.DataFrame:
        """Get overview of strategies across all configs."""
        df = self.sweep.dataframe
        if df.empty:
            print("⚠️  No data loaded. Run load_all() first.")
            return pd.DataFrame()

        # Two-stage aggregation: first per-config mean, then grand mean
        # This weights each configuration equally regardless of instance count
        config_means = df.groupby(['config_name', 'strategy_family']).agg({
            'mission_makespan_steps': 'mean',
            'total_energy_consumed': 'mean',
            'unsafe_failures': 'sum'
        }).reset_index()

        strategy_summary = (
            config_means.groupby('strategy_family')
                .agg({
                'config_name': 'nunique',
                'mission_makespan_steps': ['count', 'mean', 'std'],
                'total_energy_consumed': ['mean', 'std'],
                'unsafe_failures': 'sum'
            })
                .round(2)
        )

        strategy_summary.columns = [
            'n_configs', 'n_runs', 'mean_makespan', 'std_makespan',
            'mean_energy', 'std_energy', 'total_failures'
        ]

        return strategy_summary.sort_values('mean_makespan')

    def compare_sppdec_phases(self, metric: str = 'mission_makespan_steps') -> None:
        """
        Compare SPPDEC phases only - focused question, clean comparison.
        """
        sppdec_df = self.sweep.get_sppdec_phases()

        if sppdec_df.empty:
            print("⚠️  No SPPDEC data found")
            return

        # Clean the DataFrame
        sppdec_df = sppdec_df.reset_index(drop=True)
        sppdec_df = sppdec_df.loc[:, ~sppdec_df.columns.duplicated()]

        print(f"📈 SPPDEC Phase Comparison: {metric}")
        print("=" * 50)

        # Summary table
        summary = (
            sppdec_df.groupby(['sppdec_phase', 'config_name'])[metric]
                .mean()
                .unstack(fill_value=np.nan)
        )
        print("\nMean performance by phase and config:")
        print(summary.round(1).to_string())

        # Create focused plot - SPPDEC phases only
        fig, ax = self._setup_clean_plot(
            figsize=(10, 6),
            title=f'SPPDEC Phase Scaling: {metric.replace("_", " ").title()}',
            xlabel='Number of Agents',
            ylabel=metric.replace('_', ' ').title()
        )

        phases = sorted(sppdec_df['sppdec_phase'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(phases)))  # Max 5 phases - clean colors

        for i, phase in enumerate(phases):
            phase_data = sppdec_df[sppdec_df['sppdec_phase'] == phase]

            # Group by agents and get median with IQR
            grouped = phase_data.groupby('n_agents')[metric].agg(['median', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 2]  # Only show where we have multiple runs

            if len(grouped) == 0:
                continue

            x_vals = grouped['n_agents'].values
            y_vals = grouped['median'].values

            # Plot median line only - no error bars cluttering the view
            ax.plot(x_vals, y_vals,
                    marker='o', markersize=7, linewidth=2.5,
                    label=phase, color=colors[i],
                    markeredgecolor='white', markeredgewidth=1)

        # Clean legend - no overwhelming number of items
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

        # Reasonable axis limits - no extreme padding
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    def compare_baselines_only(self, metric: str = 'mission_makespan_steps') -> None:
        """
        Compare baseline strategies only - separate from SPPDEC.
        Enforces separation by question.
        """
        baseline_df = self.sweep.get_baseline_strategies()

        if baseline_df.empty:
            print("⚠️  No baseline data found")
            return

        print("📊 Baseline Strategy Comparison")
        print("=" * 40)

        # Select top 5 baselines to avoid overload
        selected_df = self._select_top_strategies(baseline_df, metric, max_strategies=5)

        fig, ax = self._setup_clean_plot(
            figsize=(10, 6),
            title=f'Baseline Strategy Scaling: {metric.replace("_", " ").title()}',
            xlabel='Number of Agents',
            ylabel=metric.replace('_', ' ').title()
        )

        strategies = selected_df['strategy_family'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for i, strategy in enumerate(strategies):
            strategy_data = selected_df[selected_df['strategy_family'] == strategy]

            # Group and plot median
            grouped = strategy_data.groupby('n_agents')[metric].agg(['median', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 1]

            if len(grouped) == 0:
                continue

            x_vals = grouped['n_agents'].values
            y_vals = grouped['median'].values

            display_name = self._get_display_name(strategy)
            ax.plot(x_vals, y_vals,
                    marker='s', markersize=6, linewidth=2,
                    label=display_name, color=colors[i])

        ax.legend(loc='best', frameon=True)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    def sppdec_vs_best_baseline(self, metric: str = 'mission_makespan_steps') -> None:
        """
        Direct comparison: SPPDEC Combined vs best baseline.
        Focused question with minimal visual complexity.
        """
        # Get SPPDEC combined
        sppdec_df = self.sweep.get_sppdec_phases()
        sppdec_combined = sppdec_df[sppdec_df['sppdec_phase'] == 'Combined'] if not sppdec_df.empty else pd.DataFrame()

        # Get best baseline
        baseline_df = self.sweep.get_baseline_strategies()
        if baseline_df.empty:
            print("⚠️  No baseline data found")
            return

        best_baseline_name = baseline_df.groupby('strategy_family')[metric].mean().idxmin()
        best_baseline = baseline_df[baseline_df['strategy_family'] == best_baseline_name]

        if sppdec_combined.empty:
            print("⚠️  No SPPDEC Combined data found")
            return

        print("🎯 SPPDEC vs Best Baseline")
        print("=" * 30)
        print(f"Best baseline: {self._get_display_name(best_baseline_name)}")

        # Combine for comparison
        comparison_df = pd.concat([sppdec_combined, best_baseline], ignore_index=True)

        fig, ax = self._setup_clean_plot(
            figsize=(9, 6),
            title=f'SPPDEC vs Best Baseline: {metric.replace("_", " ").title()}',
            xlabel='Number of Agents',
            ylabel=metric.replace('_', ' ').title()
        )

        strategies = comparison_df['strategy_family'].unique()
        colors = ['#2E86AB', '#A23B72']  # Just two contrasting colors

        for i, strategy in enumerate(strategies):
            strategy_data = comparison_df[comparison_df['strategy_family'] == strategy]

            grouped = strategy_data.groupby('n_agents')[metric].agg(['median', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 1]

            if len(grouped) == 0:
                continue

            x_vals = grouped['n_agents'].values
            y_vals = grouped['median'].values

            display_name = self._get_display_name(strategy)
            ax.plot(x_vals, y_vals,
                    marker='o', markersize=8, linewidth=3,
                    label=display_name, color=colors[i],
                    markeredgecolor='white', markeredgewidth=2)

        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    def compare_baselines_and_sppdec_combined(
            self,
            metric: str = 'mission_makespan_steps'
    ) -> None:
        """
        Single plot: all baselines + SPPDEC Combined overlaid together.
        Baselines use dashed lines, SPPDEC uses a thick solid line to stand out.
        """
        baseline_df = self.sweep.get_baseline_strategies()
        sppdec_df = self.sweep.get_sppdec_phases()

        if baseline_df.empty:
            print("⚠️  No baseline data found")
            return

        fig, ax = self._setup_clean_plot(
            figsize=(11, 6),
            title=f'Baseline Strategies vs SPPDEC: {metric.replace("_", " ").title()}',
            xlabel='Number of Agents',
            ylabel=metric.replace('_', ' ').title()
        )

        # ── Baselines: dashed lines, muted palette ─────────────────────────────
        selected_df = self._select_top_strategies(baseline_df, metric, max_strategies=5)
        strategies = selected_df['strategy_family'].unique()
        colors_baseline = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for i, strategy in enumerate(strategies):
            data = selected_df[selected_df['strategy_family'] == strategy]
            grouped = (
                data.groupby('n_agents')[metric]
                    .agg(['median', 'count'])
                    .reset_index()
            )
            grouped = grouped[grouped['count'] >= 1]
            if grouped.empty:
                continue

            ax.plot(
                grouped['n_agents'], grouped['median'],
                marker='s', markersize=6, linewidth=1.8,
                linestyle='--',
                label=self._get_display_name(strategy),
                color=colors_baseline[i],
                alpha=0.75
            )

        # ── SPPDEC Combined: thick solid line, prominent color ─────────────────
        if not sppdec_df.empty:
            sppdec_combined = sppdec_df[sppdec_df['sppdec_phase'] == 'Combined']
            if sppdec_combined.empty:
                sppdec_combined = sppdec_df  # fallback to whatever phase exists

            grouped_sppdec = (
                sppdec_combined.groupby('n_agents')[metric]
                    .agg(['median', 'count'])
                    .reset_index()
            )
            grouped_sppdec = grouped_sppdec[grouped_sppdec['count'] >= 1]

            if not grouped_sppdec.empty:
                ax.plot(
                    grouped_sppdec['n_agents'], grouped_sppdec['median'],
                    marker='o', markersize=9, linewidth=3.5,
                    linestyle='-',
                    label='SPPDEC: Combined',
                    color='#2E86AB',
                    markeredgecolor='white', markeredgewidth=2,
                    zorder=5  # draw on top of baselines
                )

        # ── Legend: SPPDEC entry first ─────────────────────────────────────────
        handles, labels = ax.get_legend_handles_labels()
        # Move SPPDEC to front of legend
        sppdec_idx = next(
            (i for i, l in enumerate(labels) if 'SPPDEC' in l), None
        )
        if sppdec_idx is not None:
            handles = [handles[sppdec_idx]] + [h for i, h in enumerate(handles) if i != sppdec_idx]
            labels = [labels[sppdec_idx]] + [l for i, l in enumerate(labels) if i != sppdec_idx]

        ax.legend(handles, labels, loc='upper right', frameon=True,
                  fancybox=True, shadow=True)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

        # ── Console summary ────────────────────────────────────────────────────
        print(f"\n📊 Summary ({metric}, median across all scales):")
        print("─" * 50)
        if not sppdec_df.empty and not grouped_sppdec.empty:
            sppdec_overall = sppdec_combined[metric].median()
            print(f"  {'SPPDEC: Combined':<40} {sppdec_overall:.1f}  ◄ SPPDEC")
        for strategy in strategies:
            data = selected_df[selected_df['strategy_family'] == strategy]
            overall = data[metric].median()
            print(f"  {self._get_display_name(strategy):<40} {overall:.1f}")

    def personality_analysis(self, metric: str = 'mission_makespan_steps') -> None:
        """
        Analyze SPPDEC personality variants with clear visualization and personality distribution context.
        """
        personality_df = self.sweep.get_personality_variants()

        if personality_df.empty:
            print("⚠️  No personality mixing data found")
            return

        print("🎭 SPPDEC Personality Analysis")
        print("=" * 50)

        # First, show personality distribution/composition for context
        print("\n🧠 Personality Mix Compositions:")
        print("-" * 40)

        personalities = personality_df['strategy_family'].unique()

        # Show inferred composition from names and any available data
        for personality in sorted(personalities):
            display_name = self._get_display_name(personality)
            composition = self._infer_personality_composition(personality)
            print(f"  • {display_name}:")
            print(f"    └─ {composition}")

        print("\n💡 Understanding: Each mix represents different agent behavior combinations")
        print("   affecting coordination strategies, risk tolerance, and decision speed.")

        # Summary table
        summary = (
            personality_df.groupby('strategy_family')
                .agg({
                'mission_makespan_steps': ['mean', 'std', 'count'],
                'total_energy_consumed': ['mean', 'std'],
                'unsafe_failures': 'sum',
                'config_name': 'nunique'
            })
                .round(2)
        )
        print("\n📊 Performance Summary:")
        print(summary.to_string())

        # Create clean plot similar to sppdec_vs_best_baseline style
        fig, ax = self._setup_clean_plot(
            figsize=(14, 8),
            title='SPPDEC Personality Mix Performance Comparison',
            xlabel='Number of Agents',
            ylabel=metric.replace('_', ' ').title()
        )

        personalities = sorted(personality_df['strategy_family'].unique())

        # Limit to top 5 personalities to avoid clutter
        if len(personalities) > 5:
            print(f"\n📊 Showing top 5 personalities (of {len(personalities)} total) by performance:")
            personality_means = (
                personality_df.groupby('strategy_family')[metric]
                    .mean()
                    .sort_values()
                    .head(5)
            )
            personalities = personality_means.index.tolist()
            personality_df = personality_df[personality_df['strategy_family'].isin(personalities)]

            for i, (strategy, mean_val) in enumerate(personality_means.items(), 1):
                display_name = self._get_display_name(strategy)
                print(f"  {i}. {display_name}: {mean_val:.1f}")

        # Use highly distinguishable colors - similar to sppdec_vs_best_baseline
        # These colors are specifically chosen to be very different from each other
        distinct_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
        ]

        # Different markers for additional distinction
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

        for i, personality in enumerate(personalities):
            personality_data = personality_df[personality_df['strategy_family'] == personality]

            # Group by agents - use median like sppdec_vs_best_baseline
            grouped = personality_data.groupby('n_agents')[metric].agg(['median', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 1]

            if len(grouped) == 0:
                continue

            x_vals = grouped['n_agents'].values
            y_vals = grouped['median'].values

            # Get full descriptive name
            display_name = self._get_display_name(personality)

            # Plot with sppdec_vs_best_baseline style - thick lines, large markers
            ax.plot(x_vals, y_vals,
                    marker=markers[i % len(markers)],
                    markersize=10,
                    linewidth=4,  # Thick lines like sppdec_vs_best_baseline
                    label=display_name,
                    color=distinct_colors[i % len(distinct_colors)],
                    markeredgecolor='white',
                    markeredgewidth=2)

        # Legend styling similar to sppdec_vs_best_baseline
        ax.legend(loc='upper right', fontsize=11, frameon=True,
                  fancybox=True, shadow=True,
                  borderpad=1, columnspacing=1)

        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()

        # Show the personality composition details after the plot
        print(f"\n📋 Personality Display Names Reference:")
        for personality in personalities:
            display_name = self._get_display_name(personality)
            print(f"  • {display_name} = {personality}")

    def personality_distribution_radar(self, config_name: str = None) -> None:
        """
        Visualize true SPPDEC personality distributions as agent-count radar charts.
        Uses personality_distribution from experiment.json.
        Radius corresponds to number of agents of each personality.

        Args:
            config_name: Optional config to filter to (e.g., "_A12_P36_belief_tracking_disabled"). If None, uses all configs.
        """
        personality_df = self.sweep.get_personality_variants()
        if personality_df.empty:
            print("⚠️  No personality mixing data found")
            return

        # Filter to specific config if requested
        if config_name:
            personality_df = personality_df[personality_df['config_name'] == config_name]
            if personality_df.empty:
                print(f"⚠️  No personality data found for config: {config_name}")
                available_configs = self.sweep.dataframe['config_name'].unique()
                print(f"💡  Available configs: {list(available_configs)}")
                return

        print("🕸️ SPPDEC Personality Distribution Radar")
        if config_name:
            agents = personality_df['n_agents'].iloc[0]
            parcels = personality_df['n_parcels'].iloc[0]
            print(f"🎯 Config: {config_name} ({agents} agents, {parcels} parcels)")
        print("=" * 50)

        # Expect personality_distribution column to exist and contain dicts
        if "personality_distribution" not in personality_df.columns:
            print("⚠️  personality_distribution not found in data. Check loader plumbing.")
            print("💡  Available columns:", list(personality_df.columns))
            return

        # Collect one representative distribution per strategy_family
        distributions = {}
        for strategy, group in personality_df.groupby("strategy_family"):
            dist = group["personality_distribution"].dropna()
            if not dist.empty:
                distributions[strategy] = dist.iloc[0]

        if not distributions:
            print("⚠️  No valid personality distributions available")
            return

        print(f"📊 Found personality distributions for {len(distributions)} strategy families")

        # Limit to max 6 for readability (increased from 5 since you have 6 good strategies)
        strategies = list(distributions.keys())
        max_strategies = 6
        if len(distributions) > max_strategies:
            strategies = strategies[:max_strategies]
            print(f"🎯 Showing first {max_strategies} strategies for readability")

        # Determine all personality dimensions present
        personality_dimensions = sorted({
            p for dist in distributions.values() for p in dist.keys()
        })

        print(f"🧠 Personality dimensions found: {personality_dimensions}")

        # Radar setup
        angles = np.linspace(0, 2 * np.pi, len(personality_dimensions), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection="polar"))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Determine max agent count across all mixes
        max_agents = max(
            sum(dist.values()) for dist in distributions.values()
        )

        print(f"📈 Agent count range: 0 to {max_agents}")

        for i, strategy in enumerate(strategies):
            dist = distributions[strategy]
            values = [dist.get(p, 0) for p in personality_dimensions]
            values += values[:1]  # Complete the circle

            display_name = self._get_display_name(strategy)

            ax.plot(
                angles,
                values,
                linewidth=3,
                marker="o",
                markersize=8,
                label=display_name,
                color=colors[i % len(colors)],
                markeredgecolor="white",
                markeredgewidth=1.5
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        # Axis styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [p.title() for p in personality_dimensions],
            fontsize=11
        )
        ax.set_ylim(0, max_agents)
        ax.set_yticks(range(0, max_agents + 1))
        ax.set_yticklabels([str(i) for i in range(0, max_agents + 1)], fontsize=10)
        ax.grid(True, alpha=0.3)

        # Dynamic title based on config
        title = "SPPDEC Personality Mix Distributions\n(Agent Counts per Personality)"
        if config_name:
            title = f"{config_name}: SPPDEC Personality Mix Distributions\n(Agent Counts per Personality)"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=30)

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=11)
        plt.tight_layout()
        plt.show()

        # Show the actual distribution data for reference
        print(f"\n📋 Personality Distribution Details:")
        for strategy in strategies:
            display_name = self._get_display_name(strategy)
            dist = distributions[strategy]
            total_agents = sum(dist.values())

            print(f"\n• {display_name} ({total_agents} agents total):")
            for personality, count in sorted(dist.items()):
                percentage = (count / total_agents) * 100 if total_agents > 0 else 0
                print(f"  └─ {personality.title()}: {count} agents ({percentage:.1f}%)")

    def _infer_personality_composition(self, strategy_name: str) -> str:
        """Infer personality composition from strategy name."""
        compositions = {
            "efficiency_overdrive": "High efficiency agents, minimal exploration",
            "3_brains_1_wildcard": "3 analytical agents + 1 unpredictable agent",
            "balanced": "Equal distribution of all personality types",
            "balanced_quartet": "Balanced mix optimized for 4-agent teams",
            "controlled_chaos": "Structured agents with controlled randomness",
            "no_reckless": "Conservative agents, no high-risk behaviors",
            "conservative": "Risk-averse, safety-first agent personalities",
            "aggressive": "High-risk, high-reward agent personalities",
        }

        for key, description in compositions.items():
            if key in strategy_name:
                return description

        return "Custom personality configuration"

    def polygon_performance_comparison(self,
                                       metrics: List[str] = None,
                                       include_baselines: bool = True,
                                       include_sppdec: bool = True,
                                       include_personalities: bool = True,
                                       normalize_metrics: bool = True) -> None:
        """
        Create comprehensive polygon plots comparing performance across multiple metrics.
        Shows relative strengths of different strategy types.

        Args:
            metrics: List of metrics to compare (default: key performance metrics)
            include_baselines: Include centralized/decentralized baselines
            include_sppdec: Include SPPDEC phases
            include_personalities: Include SPPDEC personality variants
            normalize_metrics: Normalize metrics to 0-100 scale for comparison
        """
        if metrics is None:
            metrics = [
                'mission_makespan_steps',
                'total_energy_consumed',
                'unsafe_failures',
                'messages_sent',
                'task_completion_rate',
                'coordination_efficiency'
            ]

        print("🔸 Multi-Metric Polygon Performance Comparison")
        print("=" * 60)

        # Collect all strategy data
        all_strategies = {}

        if include_baselines:
            baseline_strategies = self._get_baseline_polygon_data(metrics)
            all_strategies.update(baseline_strategies)

        if include_sppdec:
            sppdec_strategies = self._get_sppdec_polygon_data(metrics)
            all_strategies.update(sppdec_strategies)

        if include_personalities:
            personality_strategies = self._get_personality_polygon_data(metrics)
            all_strategies.update(personality_strategies)

        if not all_strategies:
            print("⚠️  No strategy data available for polygon comparison")
            return

        # Create separate plots for different strategy categories
        self._create_baseline_vs_sppdec_polygon(all_strategies, metrics, normalize_metrics)
        self._create_personality_polygon_comparison(all_strategies, metrics, normalize_metrics)
        self._create_comprehensive_polygon_view(all_strategies, metrics, normalize_metrics)

    def _get_baseline_polygon_data(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract baseline strategy performance data for polygon plotting."""
        baseline_df = self.sweep.get_baseline_strategies()
        if baseline_df.empty:
            return {}

        strategies = {}

        # Get top centralized and decentralized baselines
        baseline_types = baseline_df.groupby('strategy_family')

        for strategy_name, strategy_data in baseline_types:
            # Calculate mean performance across metrics
            performance = {}
            for metric in metrics:
                if metric in strategy_data.columns:
                    performance[metric] = strategy_data[metric].mean()
                else:
                    performance[metric] = 0.0

            display_name = self._get_display_name(strategy_name)
            strategies[display_name] = performance

        return strategies

    def _get_sppdec_polygon_data(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract SPPDEC phase performance data for polygon plotting."""
        sppdec_df = self.sweep.get_sppdec_phases()
        if sppdec_df.empty:
            return {}

        strategies = {}
        phases = sppdec_df['sppdec_phase'].unique()

        for phase in phases:
            phase_data = sppdec_df[sppdec_df['sppdec_phase'] == phase]

            performance = {}
            for metric in metrics:
                if metric in phase_data.columns:
                    performance[metric] = phase_data[metric].mean()
                else:
                    performance[metric] = 0.0

            display_name = f"SPPDEC {phase}"
            strategies[display_name] = performance

        return strategies

    def _get_personality_polygon_data(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract personality variant performance data for polygon plotting."""
        personality_df = self.sweep.get_personality_variants()
        if personality_df.empty:
            return {}

        strategies = {}
        personalities = personality_df['strategy_family'].unique()

        # Limit to top 5 personalities for clarity
        if len(personalities) > 5:
            personality_means = (
                personality_df.groupby('strategy_family')[metrics[0]]
                    .mean()
                    .sort_values()
                    .head(5)
            )
            personalities = personality_means.index.tolist()

        for personality in personalities:
            personality_data = personality_df[personality_df['strategy_family'] == personality]

            performance = {}
            for metric in metrics:
                if metric in personality_data.columns:
                    performance[metric] = personality_data[metric].mean()
                else:
                    performance[metric] = 0.0

            display_name = self._get_display_name(personality)
            strategies[display_name] = performance

        return strategies

    def _normalize_polygon_data(self, strategies: Dict[str, Dict[str, float]],
                                metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Normalize all metrics to 0-100 scale for fair comparison."""
        if not strategies:
            return strategies

        normalized = {}

        # Find min/max for each metric across all strategies
        metric_ranges = {}
        for metric in metrics:
            all_values = [strategy_data.get(metric, 0) for strategy_data in strategies.values()]
            if all_values:
                metric_ranges[metric] = {
                    'min': min(all_values),
                    'max': max(all_values)
                }
            else:
                metric_ranges[metric] = {'min': 0, 'max': 1}

        # Normalize each strategy
        for strategy_name, strategy_data in strategies.items():
            normalized[strategy_name] = {}
            for metric in metrics:
                value = strategy_data.get(metric, 0)
                metric_range = metric_ranges[metric]

                # For metrics where lower is better (makespan, failures, energy)
                if metric in ['mission_makespan_steps', 'total_energy_consumed', 'unsafe_failures']:
                    # Invert scale - lower values get higher scores
                    if metric_range['max'] > metric_range['min']:
                        normalized_value = 100 * (1 - (value - metric_range['min']) /
                                                  (metric_range['max'] - metric_range['min']))
                    else:
                        normalized_value = 100
                else:
                    # Higher is better
                    if metric_range['max'] > metric_range['min']:
                        normalized_value = 100 * (value - metric_range['min']) / (
                                metric_range['max'] - metric_range['min'])
                    else:
                        normalized_value = 100

                normalized[strategy_name][metric] = max(0, min(100, normalized_value))

        return normalized

    def _create_baseline_vs_sppdec_polygon(self, all_strategies: Dict, metrics: List[str], normalize: bool):
        """Create polygon comparison between baselines and SPPDEC."""

        # Filter for baselines and SPPDEC only
        baseline_sppdec = {name: data for name, data in all_strategies.items()
                           if 'SPPDEC' in name or any(baseline in name.lower()
                                                      for baseline in ['centralized', 'greedy', 'random', 'auction'])}

        if len(baseline_sppdec) < 2:
            print("⚠️  Insufficient baseline/SPPDEC data for comparison")
            return

        self._plot_polygon_chart(
            baseline_sppdec, metrics, normalize,
            title='Baseline vs SPPDEC Performance Comparison',
            figsize=(12, 10)
        )

    def _create_personality_polygon_comparison(self, all_strategies: Dict, metrics: List[str], normalize: bool):
        """Create polygon comparison for SPPDEC personalities."""

        # Filter for personality variants only
        personalities = {name: data for name, data in all_strategies.items()
                         if any(keyword in name.lower() for keyword in
                                ['efficiency', 'chaos', 'reckless', 'balanced', 'brain', 'conservative', 'aggressive'])}

        if len(personalities) < 2:
            print("⚠️  Insufficient personality data for comparison")
            return

        self._plot_polygon_chart(
            personalities, metrics, normalize,
            title='SPPDEC Personality Performance Comparison',
            figsize=(12, 10)
        )

    def _create_comprehensive_polygon_view(self, all_strategies: Dict, metrics: List[str], normalize: bool):
        """Create comprehensive view with top performers from each category."""

        # Select best from each category
        selected_strategies = {}

        # Best baseline
        baselines = {name: data for name, data in all_strategies.items()
                     if any(baseline in name.lower() for baseline in ['centralized', 'greedy', 'auction'])}
        if baselines:
            best_baseline = min(baselines.items(), key=lambda x: x[1].get(metrics[0], float('inf')))
            selected_strategies[best_baseline[0]] = best_baseline[1]

        # Best SPPDEC
        sppdec_strategies = {name: data for name, data in all_strategies.items() if
                             'SPPDEC' in name and 'personality' not in name.lower()}
        if sppdec_strategies:
            best_sppdec = min(sppdec_strategies.items(), key=lambda x: x[1].get(metrics[0], float('inf')))
            selected_strategies[best_sppdec[0]] = best_sppdec[1]

        # Best personality
        personalities = {name: data for name, data in all_strategies.items()
                         if any(keyword in name.lower() for keyword in ['efficiency', 'chaos', 'reckless', 'balanced'])}
        if personalities:
            best_personality = min(personalities.items(), key=lambda x: x[1].get(metrics[0], float('inf')))
            selected_strategies[best_personality[0]] = best_personality[1]

        if len(selected_strategies) >= 2:
            self._plot_polygon_chart(
                selected_strategies, metrics, normalize,
                title='Best Strategy Comparison Across Categories',
                figsize=(10, 8)
            )

    def _plot_polygon_chart(self, strategies: Dict, metrics: List[str], normalize: bool, title: str, figsize: tuple):
        """Create the actual polygon radar chart."""

        if normalize:
            strategies = self._normalize_polygon_data(strategies, metrics)

        # Setup radar chart
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Clean metric labels
        metric_labels = [metric.replace('_', ' ').title().replace('Steps', '(Lower=Better)')
                         if 'makespan' in metric else metric.replace('_', ' ').title() for metric in metrics]

        # Highly distinguishable colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        # Plot each strategy
        for i, (strategy_name, performance) in enumerate(strategies.items()):
            values = [performance.get(metric, 0) for metric in metrics]
            values += values[:1]  # Complete the circle

            # Plot polygon with thick lines and markers
            ax.plot(angles, values, 'o-', linewidth=4, label=strategy_name,
                    color=colors[i % len(colors)], markersize=8,
                    markeredgecolor='white', markeredgewidth=2)
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12, fontweight='medium')

        if normalize:
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=11)
        else:
            ax.set_ylim(bottom=0)

        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=30)

        # **FIXED: Using the exact pattern from your example**
        # Shrink current axis's height by 15% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.15,
                         box.width, box.height * 0.85])

        # Put a legend below current axis - REMOVED invalid 'outside' parameter
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True, ncol=2, fontsize=10)

        plt.show()

        # Show interpretation guide
        if normalize:
            print(f"\n📊 Performance Interpretation (100 = best possible):")
            print(f"   • Larger polygon area = better overall performance")
            print(f"   • Makespan, Energy, Failures: lower actual values = higher scores")
            print(f"   • Messages, Completion Rate: higher actual values = higher scores")
        else:
            print(f"\n📊 Raw metric values - larger numbers may be better or worse depending on metric")

    def _extract_personality_compositions(self, personality_df) -> Dict[str, List[float]]:
        """
        Extract personality compositions from the actual data.
        You'll need to implement this based on how personality data is stored in your system.
        """
        personalities = personality_df['strategy_family'].unique()
        compositions = {}

        for personality in personalities:
            # Try to find personality composition columns in your data
            sample_data = personality_df[personality_df['strategy_family'] == personality].iloc[0]

            # Look for personality-related columns
            personality_cols = [col for col in sample_data.index
                                if any(keyword in col.lower() for keyword in
                                       ['efficiency', 'exploration', 'risk', 'cooperation', 'adaptability',
                                        'consistency'])]

            if personality_cols:
                # Extract actual values from your data
                values = []
                for col in ['efficiency', 'exploration', 'risk', 'cooperation', 'adaptability', 'consistency']:
                    matching_cols = [c for c in personality_cols if col in c.lower()]
                    if matching_cols:
                        val = sample_data[matching_cols[0]]
                        values.append(float(val) if pd.notna(val) else 0.0)
                    else:
                        values.append(0.0)
                compositions[personality] = values
            else:
                # If no explicit personality data, use inferred values based on name
                compositions[personality] = self._infer_personality_values(personality)

        return compositions if any(compositions.values()) else {}

    def _infer_personality_values(self, strategy_name: str) -> List[float]:
        """Infer personality values from strategy name - replace with actual data extraction."""
        # This is a placeholder - you should replace with actual personality composition extraction
        inference_map = {
            "efficiency_overdrive": [95, 20, 40, 60, 50, 80],
            "3_brains_1_wildcard": [70, 80, 60, 85, 90, 40],
            "balanced": [60, 60, 60, 60, 60, 60],
            "controlled_chaos": [50, 90, 70, 60, 95, 30],
            "no_reckless": [70, 40, 20, 80, 50, 90],
            "conservative": [60, 30, 15, 85, 40, 95],
            "aggressive": [40, 80, 95, 50, 80, 20],
        }

        for key, values in inference_map.items():
            if key in strategy_name:
                return values

        return [50, 50, 50, 50, 50, 50]  # Default balanced

    def _create_example_personality_radar(self):
        """Create an example radar chart to show the expected structure."""
        print("📊 Example Personality Distribution Radar Chart:")

        # Example data structure
        personality_dimensions = ['Efficiency', 'Exploration', 'Risk-Taking', 'Cooperation', 'Adaptability',
                                  'Consistency']

        example_personalities = {
            'Efficiency Overdrive': [95, 20, 40, 60, 50, 80],
            'Controlled Chaos': [50, 90, 70, 60, 95, 30],
            'No Reckless': [70, 40, 20, 80, 50, 90]
        }

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(personality_dimensions), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (name, values) in enumerate(example_personalities.items()):
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=3, label=name,
                    color=colors[i], markersize=8)
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(personality_dimensions, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.set_title('Example: SPPDEC Personality Distribution Radar\n(Replace with actual data)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.tight_layout()
        plt.show()

        print("\n💡 To get actual personality distributions, implement personality data extraction")
        print("   in your SweepAnalyzer to capture the personality composition for each mix.")

    def comprehensive_polygon_comparison(self,
                                         metrics: List[str] = None,
                                         include_baselines: bool = True,
                                         include_sppdec: bool = True,
                                         include_personalities: bool = True,
                                         normalize_metrics: bool = True) -> None:
        """
        Create comprehensive polygon plots comparing performance across multiple metrics.
        Shows relative strengths of different strategy types.

        Args:
            metrics: List of metrics to compare (default: key performance metrics)
            include_baselines: Include centralized/decentralized baselines
            include_sppdec: Include SPPDEC phases
            include_personalities: Include SPPDEC personality variants
            normalize_metrics: Normalize metrics to 0-100 scale for comparison
        """
        if metrics is None:
            metrics = [
                'mission_makespan_steps',
                'total_energy_consumed',
                'unsafe_failures',
                'messages_sent'
            ]

        print("🔸 Multi-Metric Polygon Performance Comparison")
        print("=" * 60)

        # Collect all strategy data
        all_strategies = {}

        if include_baselines:
            baseline_strategies = self._get_baseline_polygon_data(metrics)
            all_strategies.update(baseline_strategies)
            print(f"📊 Loaded {len(baseline_strategies)} baseline strategies")

        if include_sppdec:
            sppdec_strategies = self._get_sppdec_polygon_data(metrics)
            all_strategies.update(sppdec_strategies)
            print(f"📊 Loaded {len(sppdec_strategies)} SPPDEC strategies")

        if include_personalities:
            personality_strategies = self._get_personality_polygon_data(metrics)
            all_strategies.update(personality_strategies)
            print(f"📊 Loaded {len(personality_strategies)} personality strategies")

        if not all_strategies:
            print("⚠️  No strategy data available for polygon comparison")
            return

        print(f"\n🎯 Total strategies for comparison: {len(all_strategies)}")

        # Create separate comparison plots
        self._create_baseline_vs_sppdec_polygon(all_strategies, metrics, normalize_metrics)
        self._create_personality_polygon_comparison(all_strategies, metrics, normalize_metrics)
        self._create_comprehensive_polygon_view(all_strategies, metrics, normalize_metrics)

    def _get_baseline_polygon_data(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract baseline strategy performance data for polygon plotting."""
        baseline_df = self.sweep.get_baseline_strategies()
        if baseline_df.empty:
            return {}

        strategies = {}

        # Get performance for each baseline strategy
        for strategy_name in baseline_df['strategy_family'].unique():
            strategy_data = baseline_df[baseline_df['strategy_family'] == strategy_name]

            performance = {}
            for metric in metrics:
                if metric in strategy_data.columns:
                    performance[metric] = strategy_data[metric].mean()
                else:
                    performance[metric] = 0.0

            display_name = self._get_display_name(strategy_name)
            strategies[display_name] = performance

        return strategies

    def _get_sppdec_polygon_data(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract SPPDEC phase performance data for polygon plotting."""
        sppdec_df = self.sweep.get_sppdec_phases()
        if sppdec_df.empty:
            return {}

        strategies = {}
        phases = sppdec_df['sppdec_phase'].unique()

        for phase in phases:
            phase_data = sppdec_df[sppdec_df['sppdec_phase'] == phase]

            performance = {}
            for metric in metrics:
                if metric in phase_data.columns:
                    performance[metric] = phase_data[metric].mean()
                else:
                    performance[metric] = 0.0

            display_name = f"SPPDEC {phase}"
            strategies[display_name] = performance

        return strategies

    def _get_personality_polygon_data(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract personality variant performance data for polygon plotting."""
        personality_df = self.sweep.get_personality_variants()
        if personality_df.empty:
            return {}

        strategies = {}
        personalities = personality_df['strategy_family'].unique()

        # Limit to top 5 personalities for clarity
        if len(personalities) > 5:
            personality_means = (
                personality_df.groupby('strategy_family')[metrics[0]]
                    .mean()
                    .sort_values()
                    .head(5)
            )
            personalities = personality_means.index.tolist()
            print(f"🎭 Showing top 5 personalities (of {len(personality_df['strategy_family'].unique())} total)")

        for personality in personalities:
            personality_data = personality_df[personality_df['strategy_family'] == personality]

            performance = {}
            for metric in metrics:
                if metric in personality_data.columns:
                    performance[metric] = personality_data[metric].mean()
                else:
                    performance[metric] = 0.0

            display_name = self._get_display_name(personality)
            strategies[display_name] = performance

        return strategies

    def _normalize_polygon_data(self, strategies: Dict[str, Dict[str, float]],
                                metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Normalize all metrics to 0-100 scale for fair comparison."""
        if not strategies:
            return strategies

        normalized = {}

        # Find min/max for each metric across all strategies
        metric_ranges = {}
        for metric in metrics:
            all_values = [strategy_data.get(metric, 0) for strategy_data in strategies.values()]
            all_values = [v for v in all_values if v > 0]  # Remove zeros
            if all_values:
                metric_ranges[metric] = {
                    'min': min(all_values),
                    'max': max(all_values)
                }
            else:
                metric_ranges[metric] = {'min': 0, 'max': 1}

        # Normalize each strategy
        for strategy_name, strategy_data in strategies.items():
            normalized[strategy_name] = {}
            for metric in metrics:
                value = strategy_data.get(metric, 0)
                metric_range = metric_ranges[metric]

                # For metrics where lower is better (makespan, failures, energy)
                if metric in ['mission_makespan_steps', 'total_energy_consumed', 'unsafe_failures']:
                    # Invert scale - lower values get higher scores
                    if metric_range['max'] > metric_range['min']:
                        normalized_value = 100 * (1 - (value - metric_range['min']) /
                                                  (metric_range['max'] - metric_range['min']))
                    else:
                        normalized_value = 100
                else:
                    # Higher is better
                    if metric_range['max'] > metric_range['min']:
                        normalized_value = 100 * (value - metric_range['min']) / (
                                metric_range['max'] - metric_range['min'])
                    else:
                        normalized_value = 100

                normalized[strategy_name][metric] = max(0, min(100, normalized_value))

        return normalized

    def _create_baseline_vs_sppdec_polygon(self, all_strategies: Dict, metrics: List[str], normalize: bool):
        """Create polygon comparison between baselines and SPPDEC."""

        # Filter for baselines and SPPDEC only
        baseline_sppdec = {name: data for name, data in all_strategies.items()
                           if 'SPPDEC' in name or any(baseline in name.lower()
                                                      for baseline in ['centralized', 'greedy', 'random', 'auction'])}

        if len(baseline_sppdec) < 2:
            print("⚠️  Insufficient baseline/SPPDEC data for comparison")
            return

        print(f"\n🎯 Baseline vs SPPDEC Comparison ({len(baseline_sppdec)} strategies)")
        self._plot_polygon_chart(
            baseline_sppdec, metrics, normalize,
            title='Baseline vs SPPDEC Performance Strengths',
            figsize=(12, 10)
        )

    def _create_personality_polygon_comparison(self, all_strategies: Dict, metrics: List[str], normalize: bool):
        """Create polygon comparison for SPPDEC personalities."""

        # Filter for personality variants only
        personalities = {name: data for name, data in all_strategies.items()
                         if any(keyword in name.lower() for keyword in
                                ['efficiency', 'chaos', 'reckless', 'balanced', 'brain', 'conservative', 'aggressive'])}

        if len(personalities) < 2:
            print("⚠️  Insufficient personality data for comparison")
            return

        print(f"\n🎭 Personality Performance Comparison ({len(personalities)} personalities)")
        self._plot_polygon_chart(
            personalities, metrics, normalize,
            title='SPPDEC Personality Performance Strengths',
            figsize=(12, 10)
        )

    def _create_comprehensive_polygon_view(self, all_strategies: Dict, metrics: List[str], normalize: bool):
        """Create comprehensive view with top performers from each category."""

        # Select best from each category based on primary metric (makespan)
        selected_strategies = {}
        primary_metric = metrics[0]

        # Best baseline (lowest makespan)
        baselines = {name: data for name, data in all_strategies.items()
                     if any(baseline in name.lower() for baseline in ['centralized', 'greedy', 'auction', 'random'])}
        if baselines:
            best_baseline = min(baselines.items(), key=lambda x: x[1].get(primary_metric, float('inf')))
            selected_strategies[f"Best Baseline ({best_baseline[0]})"] = best_baseline[1]

        # Best SPPDEC phase
        sppdec_strategies = {name: data for name, data in all_strategies.items()
                             if 'SPPDEC' in name and not any(keyword in name.lower() for keyword in
                                                             ['efficiency', 'chaos', 'reckless', 'balanced'])}
        if sppdec_strategies:
            best_sppdec = min(sppdec_strategies.items(), key=lambda x: x[1].get(primary_metric, float('inf')))
            selected_strategies[f"Best SPPDEC ({best_sppdec[0]})"] = best_sppdec[1]

        # Best personality
        personalities = {name: data for name, data in all_strategies.items()
                         if any(keyword in name.lower() for keyword in ['efficiency', 'chaos', 'reckless', 'balanced'])}
        if personalities:
            best_personality = min(personalities.items(), key=lambda x: x[1].get(primary_metric, float('inf')))
            selected_strategies[f"Best Personality ({best_personality[0]})"] = best_personality[1]

        if len(selected_strategies) >= 2:
            print(f"\n🏆 Champion Comparison ({len(selected_strategies)} top performers)")
            self._plot_polygon_chart(
                selected_strategies, metrics, normalize,
                title='Strategy Champions: Multi-Metric Performance',
                figsize=(12, 8)
            )

    def personality_detailed_comparison(self, metrics: List[str] = None) -> None:
        """
        Detailed multi-metric personality comparison.
        """
        if metrics is None:
            metrics = ['mission_makespan_steps', 'total_energy_consumed', 'unsafe_failures']

        personality_df = self.sweep.get_personality_variants()
        if personality_df.empty:
            print("⚠️  No personality mixing data found")
            return

        print("🎭 SPPDEC Personality Detailed Analysis")
        print("=" * 60)

        # Create subplot for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle('SPPDEC Personality Multi-Metric Comparison',
                     fontsize=16, fontweight='bold', y=1.02)

        personalities = sorted(personality_df['strategy_family'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(personalities)))

        for j, metric in enumerate(metrics):
            ax = axes[j]

            for i, personality in enumerate(personalities):
                personality_data = personality_df[personality_df['strategy_family'] == personality]

                grouped = personality_data.groupby('n_agents')[metric].median().reset_index()

                if len(grouped) > 0:
                    display_name = self._get_display_name(personality)
                    ax.plot(grouped['n_agents'], grouped[metric],
                            marker='o', markersize=6, linewidth=2,
                            label=display_name if j == 0 else "",
                            color=colors[i])

            ax.set_title(metric.replace('_', ' ').title(), fontweight='medium')
            ax.set_xlabel('Number of Agents')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        # Single legend for all subplots
        if personalities:
            axes[0].legend(loc='upper left', fontsize=10, frameon=True)

        plt.tight_layout()
        plt.show()

    def ranking_heatmap(self, metric: str = 'mission_makespan_steps') -> None:
        """
        Strategy ranking visualization - much cleaner than overlapping box plots.
        """
        df = self.sweep.dataframe
        if df.empty:
            print("⚠️  No data loaded")
            return

        # Calculate ranks within each config
        ranking_data = []
        for config_name, group in df.groupby('config_name'):
            n_agents = group['n_agents'].iloc[0]  # Assuming consistent within config

            strategy_means = group.groupby('strategy_family')[metric].mean().sort_values()

            for rank, (strategy, value) in enumerate(strategy_means.items(), 1):
                ranking_data.append({
                    'config': config_name,
                    'n_agents': n_agents,
                    'strategy': self._get_display_name(strategy),
                    'rank': rank,
                    'value': value
                })

        ranking_df = pd.DataFrame(ranking_data)

        # Create heatmap
        pivot_table = ranking_df.pivot_table(
            index='strategy',
            columns='n_agents',
            values='rank',
            aggfunc='mean'
        )

        fig, ax = self._setup_clean_plot(
            figsize=(10, 6),
            title=f'Strategy Rankings Across Scales\n(Lower rank = better {metric.replace("_", " ")})'
        )

        # Create heatmap with clean styling
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlBu_r',
                    cbar_kws={'label': 'Average Rank'}, ax=ax)

        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Strategy')

        plt.tight_layout()
        plt.show()

    # Debug Version of Personality Distribution Radar

    def personality_distribution_radar_debug(self) -> None:
        """
        DEBUG VERSION: Trace where personality data is being lost.
        """
        print("🔍 DEBUG: Personality Distribution Analysis")
        print("=" * 60)

        # Step 1: Check raw DataFrame
        df = self.sweep.dataframe
        print(f"📊 Total DataFrame rows: {len(df)}")
        print(f"📊 Available columns: {[col for col in df.columns if 'personality' in col]}")

        # Step 2: Check personality_distribution column content
        if 'personality_distribution' in df.columns:
            print(f"\n🔍 Checking personality_distribution column:")
            non_null_personality = df['personality_distribution'].dropna()
            print(f"📊 Non-null personality entries: {len(non_null_personality)}")

            if len(non_null_personality) > 0:
                print(f"📊 First few personality distributions:")
                for i, dist in enumerate(non_null_personality.head(3)):
                    print(f"   {i + 1}. {dist} (type: {type(dist)})")

        # Step 3: Check get_personality_variants filtering
        personality_df = self.sweep.get_personality_variants()
        print(f"\n🔍 After get_personality_variants() filtering:")
        print(f"📊 Personality DataFrame rows: {len(personality_df)}")

        if personality_df.empty:
            print("⚠️  No personality mixing data found - checking strategy names...")
            all_strategies = df['strategy_name'].unique()
            print("🔍 All available strategy names:")
            for strategy in all_strategies[:10]:  # Show first 10
                print(f"   • {strategy}")
            return

        # Step 4: Check personality_distribution column in filtered data
        if "personality_distribution" not in personality_df.columns:
            print("⚠️  personality_distribution column missing after filtering")
            print("💡  Available columns in filtered data:", list(personality_df.columns))
            return

        # Step 5: Check distributions content
        print(f"\n🔍 Checking distributions in filtered data:")
        distributions = {}
        for strategy, group in personality_df.groupby("strategy_family"):
            dist_series = group["personality_distribution"].dropna()
            print(f"📊 Strategy '{strategy}': {len(dist_series)} non-null distributions")

            if not dist_series.empty:
                first_dist = dist_series.iloc[0]
                print(f"   └─ First distribution: {first_dist} (type: {type(first_dist)})")
                distributions[strategy] = first_dist

        if not distributions:
            print("⚠️  No valid personality distributions found after grouping")
            return

        print(f"\n📊 Final distributions extracted: {len(distributions)}")
        for strategy, dist in distributions.items():
            if isinstance(dist, dict):
                total_agents = sum(dist.values())
                print(f"   • {strategy}: {dist} (total: {total_agents})")
            else:
                print(f"   • {strategy}: {dist} (type: {type(dist)}) - NOT A DICT!")

        # Step 6: Check personality dimensions
        personality_dimensions = sorted({
            p for dist in distributions.values()
            if isinstance(dist, dict)  # Only process dicts
            for p in dist.keys()
        })

        print(f"\n🧠 Personality dimensions found: {personality_dimensions}")

        # Step 7: Check max agents calculation
        valid_dists = [dist for dist in distributions.values() if isinstance(dist, dict)]
        if valid_dists:
            max_agents = max(sum(dist.values()) for dist in valid_dists)
            print(f"📈 Max agents calculated: {max_agents}")
        else:
            print("⚠️  No valid dictionaries found for max agent calculation")

    def inspect_personality_data_directly(self, config_name: str) -> None:
        """
        DEBUG: Directly inspect experiment.json files for personality data.
        """
        import json

        config_dir = self.sweep.sweep_dir / config_name
        exp_dirs = list(config_dir.glob("experiment_*"))

        if not exp_dirs:
            print(f"⚠️  No experiment directories found in {config_dir}")
            return

        exp_dir = exp_dirs[0]
        exp_json = exp_dir / "experiment.json"

        if not exp_json.exists():
            print(f"⚠️  No experiment.json found in {exp_dir}")
            return

        print(f"🔍 Inspecting {exp_json}")

        with open(exp_json, 'r') as f:
            data = json.load(f)

        results = data.get('results', [])
        print(f"📊 Found {len(results)} results in JSON")

        for i, result in enumerate(results[:3]):  # Check first 3
            strategy_name = result.get('strategy_name', 'unknown')
            personality_dist = result.get('personality_distribution', {})

            print(f"\n📊 Result {i + 1}:")
            print(f"   Strategy: {strategy_name}")
            print(f"   Personality distribution: {personality_dist}")

            if personality_dist:
                total = sum(personality_dist.values())
                print(f"   Total agents: {total}")

    # Energy Efficiency Analysis for Strategy Scaling

    def plot_energy_efficiency_scaling(self, normalize_by: str = "tasks_per_agent") -> None:
        """
        Plot energy efficiency as strategies scale up.

        Args:
            normalize_by: How to normalize efficiency
                - "tasks_per_agent": Energy per task per agent
                - "baseline_ratio": Efficiency relative to smallest team
                - "marginal_cost": Additional energy cost per added agent
        """
        df = self.sweep.dataframe
        if df.empty:
            print("⚠️  No data loaded")
            return

        print(f"⚡ Energy Efficiency Scaling Analysis")
        print("=" * 50)

        # Calculate efficiency metrics
        if normalize_by == "tasks_per_agent":
            # Energy per task per agent - ideal metric for efficiency
            df['energy_efficiency'] = df['total_energy_consumed'] / (df['tasks_completed'] * df['n_agents'])
            ylabel = "Energy per Task per Agent"
            title_suffix = "Task-Agent Normalized"

        elif normalize_by == "baseline_ratio":
            # Efficiency relative to smallest team size for each strategy
            efficiency_data = []
            for strategy_family in df['strategy_family'].unique():
                strategy_data = df[df['strategy_family'] == strategy_family].copy()
                if len(strategy_data) == 0:
                    continue

                # Find baseline (smallest team) efficiency
                baseline_row = strategy_data.loc[strategy_data['n_agents'].idxmin()]
                baseline_efficiency = baseline_row['total_energy_consumed'] / baseline_row['tasks_completed']

                # Calculate relative efficiency for all team sizes
                strategy_data['energy_efficiency'] = (
                        (strategy_data['total_energy_consumed'] / strategy_data['tasks_completed'])
                        / baseline_efficiency
                )
                efficiency_data.append(strategy_data)

            df = pd.concat(efficiency_data, ignore_index=True)
            ylabel = "Efficiency Ratio vs Smallest Team"
            title_suffix = "Baseline Relative"

        elif normalize_by == "marginal_cost":
            # Energy cost of adding one more agent
            efficiency_data = []
            for strategy_family in df['strategy_family'].unique():
                strategy_data = df[df['strategy_family'] == strategy_family].copy()
                if len(strategy_data) < 2:
                    continue

                strategy_data = strategy_data.sort_values('n_agents')

                # Calculate marginal energy cost per additional agent
                marginal_costs = []
                for i in range(1, len(strategy_data)):
                    prev_row = strategy_data.iloc[i - 1]
                    curr_row = strategy_data.iloc[i]

                    energy_delta = curr_row['total_energy_consumed'] - prev_row['total_energy_consumed']
                    agent_delta = curr_row['n_agents'] - prev_row['n_agents']
                    task_delta = curr_row['tasks_completed'] - prev_row['tasks_completed']

                    if agent_delta > 0 and task_delta > 0:
                        marginal_cost = energy_delta / (agent_delta * task_delta)
                        marginal_costs.append(marginal_cost)
                    else:
                        marginal_costs.append(np.nan)

                # Add marginal costs to dataframe (skip first row)
                strategy_data = strategy_data.iloc[1:].copy()
                strategy_data['energy_efficiency'] = marginal_costs
                efficiency_data.append(strategy_data)

            df = pd.concat(efficiency_data, ignore_index=True)
            ylabel = "Marginal Energy Cost per Added Agent-Task"
            title_suffix = "Marginal Cost"

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Separate SPPDEC and baseline strategies
        sppdec_strategies = [s for s in df['strategy_family'].unique() if 'sppdec' in s.lower()]
        baseline_strategies = [s for s in df['strategy_family'].unique() if 'sppdec' not in s.lower()]

        # Plot baseline strategies
        colors_baseline = plt.cm.Set1(np.linspace(0, 1, len(baseline_strategies)))
        for i, strategy in enumerate(baseline_strategies):
            strategy_data = df[df['strategy_family'] == strategy]
            if len(strategy_data) == 0:
                continue

            grouped = strategy_data.groupby('n_agents')['energy_efficiency'].agg(['median', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 1]

            if len(grouped) > 0:
                display_name = self._get_display_name(strategy)
                ax.plot(grouped['n_agents'], grouped['median'],
                        marker='s', markersize=6, linewidth=2,
                        label=f"{display_name} (Baseline)",
                        color=colors_baseline[i],
                        linestyle='--', alpha=0.8)

        # Plot SPPDEC strategies
        colors_sppdec = plt.cm.Set2(np.linspace(0, 1, len(sppdec_strategies)))
        for i, strategy in enumerate(sppdec_strategies):
            strategy_data = df[df['strategy_family'] == strategy]
            if len(strategy_data) == 0:
                continue

            grouped = strategy_data.groupby('n_agents')['energy_efficiency'].agg(['median', 'count']).reset_index()
            grouped = grouped[grouped['count'] >= 1]

            if len(grouped) > 0:
                display_name = self._get_display_name(strategy)
                ax.plot(grouped['n_agents'], grouped['median'],
                        marker='o', markersize=8, linewidth=3,
                        label=f"{display_name} (SPPDEC)",
                        color=colors_sppdec[i])

        # Styling
        ax.set_xlabel('Number of Agents', fontsize=12, fontweight='medium')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='medium')
        ax.set_title(f'Energy Efficiency Scaling: {title_suffix}\n(Lower = More Efficient)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, frameon=True)

        # Set reasonable y-limits
        if normalize_by == "baseline_ratio":
            ax.set_ylim(bottom=0.5)  # Show improvement below baseline

        plt.tight_layout()
        plt.show()

        # Show efficiency summary
        print(f"\n📊 Efficiency Summary ({title_suffix}):")
        summary = df.groupby('strategy_family')['energy_efficiency'].agg(['mean', 'std']).round(3)
        summary = summary.sort_values('mean')
        print(summary.to_string())

    def plot_energy_efficiency_comparison(self) -> None:
        """
        Compare multiple efficiency metrics side by side.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot each efficiency metric
        efficiency_metrics = [
            ("tasks_per_agent", "Energy per Task per Agent"),
            ("baseline_ratio", "Efficiency vs Baseline"),
            ("marginal_cost", "Marginal Energy Cost")
        ]

        for i, (metric, title) in enumerate(efficiency_metrics):
            ax = axes[i]

            # Generate data for this metric (simplified version of above logic)
            df = self.sweep.dataframe

            if metric == "tasks_per_agent":
                df = df.copy()
                df['efficiency'] = df['total_energy_consumed'] / (df['tasks_completed'] * df['n_agents'])
            elif metric == "baseline_ratio":
                # Simplified baseline ratio calculation
                df = df.copy()
                df['efficiency'] = df['total_energy_consumed'] / df['tasks_completed']
            else:  # marginal_cost - simplified
                df = df.copy()
                df['efficiency'] = df['total_energy_consumed'] / df['n_agents']

            # Plot top 5 strategies
            top_strategies = df.groupby('strategy_family')['efficiency'].mean().nsmallest(5).index

            colors = plt.cm.tab10(np.linspace(0, 1, len(top_strategies)))
            for j, strategy in enumerate(top_strategies):
                strategy_data = df[df['strategy_family'] == strategy]
                grouped = strategy_data.groupby('n_agents')['efficiency'].median()

                if len(grouped) > 0:
                    display_name = self._get_display_name(strategy)
                    ax.plot(grouped.index, grouped.values,
                            marker='o', linewidth=2, label=display_name,
                            color=colors[j])

            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Agents')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    # Updated _get_display_name method for better SPPDEC personality labels

    def _get_display_name(self, strategy_name: str, max_length: int = None) -> str:
        """
        Get display name for strategy with SPPDEC branding and team compositions.

        Personality Legend:
        Ra = Rational agents (analytical, optimal decisions)
        S = Selfish agents (self-interested, efficient)
        A = Altruistic agents (cooperative, team-focused)
        Re = Reckless agents (exploratory, risk-taking)

        Example: "SPPDEC: Efficiency Overdrive (3Ra,6S,3Re)"
        = 3 Rational + 6 Selfish + 3 Reckless agents in a 12-agent team

        Args:
            strategy_name: Full strategy name
            max_length: Optional max length for display (None = use full name)
        """
        # Updated display names with SPPDEC branding and team compositions
        DISPLAY_NAMES = {
            # SPPDEC Personality Variants - Show composition and SPPDEC branding
            "sppdec_personality_mix_efficiency_overdrive": "SPPDEC: Efficiency Overdrive (3Ra,6S,3Re)",
            "sppdec_personality_mix_3_brains_1_wildcard": "SPPDEC: 3 Brains + 1 Wildcard (9Ra,3Re)",
            "sppdec_personality_mix_balanced_quartet": "SPPDEC: Balanced Quartet (3Ra,3S,3A,3Re)",
            "sppdec_personality_mix_controlled_chaos": "SPPDEC: Controlled Chaos (6Ra,3S,3Re)",
            "sppdec_personality_mix_no_reckless": "SPPDEC: No Reckless (6Ra,3S,3A)",
            "sppdec_fairness_only``": "SPPDEC: Fairness Only (3Ra,3S,3A,3Re)",

            # SPPDEC Core Variants
            "sppdec_control": "SPPDEC: Control",
            "sppdec_baseline_only": "SPPDEC: Baseline Only",
            "sppdec_combined": "SPPDEC: Combined",

            # Baseline strategies - keep existing clean names
            "centralized_cc_auction": "Centralized Auction",
            "centralized_cc_greedy_greedy": "Centralized Greedy",
            "centralized_cc_hungarian_hungarian": "Centralized Hungarian",
            "cbba_single_task_cbba_single": "CBBA Single Task",
            "cbba_full_cbba_bundle": "CBBA Bundle",
            "decentralized_auction_auction_baseline": "Decentralized Auction",
            "naive_decentralized_reactive": "Naive Decentralized"
        }

        display_name = DISPLAY_NAMES.get(strategy_name, strategy_name)

        # If max_length specified and name is too long, truncate intelligently
        if max_length and len(display_name) > max_length:
            # For SPPDEC personality names, try to keep the key descriptive part
            if "SPPDEC:" in display_name and "(" in display_name:
                # Keep SPPDEC: + personality type + composition
                parts = display_name.split("(")
                base_part = parts[0].strip()
                comp_part = "(" + parts[1] if len(parts) > 1 else ""

                if len(base_part) <= max_length - len(comp_part):
                    return base_part + comp_part
                else:
                    # Truncate base part but keep composition
                    truncated_base = base_part[:max_length - len(comp_part) - 3] + "..."
                    return truncated_base + comp_part
            else:
                return display_name[:max_length - 3] + "..."

        return display_name

    def plot_efficiency_ranking_by_scale(self) -> None:
        """
        Show how strategy efficiency rankings change with scale.
        """
        df = self.sweep.dataframe

        # Calculate efficiency metric
        df = df.copy()
        df['efficiency'] = df['total_energy_consumed'] / (df['tasks_completed'] * df['n_agents'])

        # Create ranking matrix
        ranking_data = []
        for n_agents, group in df.groupby('n_agents'):
            strategy_efficiencies = group.groupby('strategy_family')['efficiency'].mean()
            strategy_ranks = strategy_efficiencies.rank(method='min')

            for strategy, rank in strategy_ranks.items():
                ranking_data.append({
                    'n_agents': n_agents,
                    'strategy': self._get_display_name(strategy),
                    'rank': rank,
                    'efficiency': strategy_efficiencies[strategy]
                })

        ranking_df = pd.DataFrame(ranking_data)

        # Create heatmap
        pivot = ranking_df.pivot(index='strategy', columns='n_agents', values='rank')

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                    cbar_kws={'label': 'Efficiency Rank (1=Best)'}, ax=ax)

        ax.set_title('Strategy Efficiency Rankings Across Scales\n(1 = Most Efficient)',
                     fontweight='bold', fontsize=14)
        ax.set_xlabel('Number of Agents')
        ax.set_ylabel('Strategy')

        plt.tight_layout()
        plt.show()

    # Improved Communication-Efficiency Analysis for SPPDEC Research

    def plot_communication_vs_efficiency(self) -> None:
        """
        Show the trade-off between communication overhead and energy efficiency.
        Key insight: Does more messaging lead to better coordination efficiency?
        """
        df = self.sweep.dataframe.copy()

        # Calculate efficiency and communication metrics
        df['energy_efficiency'] = df['total_energy_consumed'] / (df['tasks_completed'] * df['n_agents'])
        df['messages_per_agent'] = df['messages_sent'] / df['n_agents']

        # Create larger scatter plot
        fig, ax = plt.subplots(figsize=(16, 10))  # Larger plot

        # Color by strategy type
        sppdec_mask = df['strategy_family'].str.contains('sppdec', case=False, na=False)
        centralized_mask = df['strategy_family'].str.contains('centralized', case=False, na=False)
        cbba_mask = df['strategy_family'].str.contains('cbba', case=False, na=False)

        # Plot each strategy type with larger markers
        if sppdec_mask.any():
            sppdec_data = df[sppdec_mask]
            ax.scatter(sppdec_data['messages_per_agent'], sppdec_data['energy_efficiency'],
                       alpha=0.8, s=100, label='SPPDEC Strategies', color='green', marker='o',
                       edgecolors='darkgreen', linewidth=2)

        if centralized_mask.any():
            cent_data = df[centralized_mask]
            ax.scatter(cent_data['messages_per_agent'], cent_data['energy_efficiency'],
                       alpha=0.7, s=80, label='Centralized', color='blue', marker='s')

        if cbba_mask.any():
            cbba_data = df[cbba_mask]
            ax.scatter(cbba_data['messages_per_agent'], cbba_data['energy_efficiency'],
                       alpha=0.7, s=80, label='CBBA', color='orange', marker='^')

        # Other strategies
        other_mask = ~(sppdec_mask | centralized_mask | cbba_mask)
        if other_mask.any():
            other_data = df[other_mask]
            ax.scatter(other_data['messages_per_agent'], other_data['energy_efficiency'],
                       alpha=0.7, s=80, label='Other Decentralized', color='red', marker='d')

        # Highlight SPPDEC advantage zone
        if sppdec_mask.any():
            sppdec_max_comm = df[sppdec_mask]['messages_per_agent'].max()
            sppdec_min_eff = df[sppdec_mask]['energy_efficiency'].min()

            # Add shaded region for SPPDEC advantage
            ax.axvspan(0, sppdec_max_comm * 1.1, alpha=0.1, color='green',
                       label='SPPDEC Communication Zone')

        ax.set_xlabel('Messages per Agent', fontweight='medium', fontsize=14)
        ax.set_ylabel('Energy per Task per Agent (Lower = Better)', fontweight='medium', fontsize=14)
        ax.set_title(
            'Communication vs Energy Efficiency Trade-off\n(SPPDEC Advantage: Low Communication, High Efficiency)',
            fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

        # Show communication summary
        print("\n📡 Communication vs Efficiency Summary:")
        comm_summary = df.groupby('strategy_family').agg({
            'messages_per_agent': ['mean', 'std'],
            'energy_efficiency': ['mean', 'std'],
            'messages_sent': 'mean'
        }).round(3)

        print(comm_summary.sort_values(('messages_per_agent', 'mean')).to_string())

    def plot_communication_scaling(self) -> None:
        """
        Show how communication overhead scales with team size.
        Key question: Do SPPDEC strategies maintain low communication at scale?
        """
        df = self.sweep.dataframe.copy()
        df['messages_per_agent'] = df['messages_sent'] / df['n_agents']

        # Create larger dual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # Larger plots

        # Left plot: Messages per Agent vs Team Size
        strategy_families = ['centralized', 'cbba', 'sppdec', 'decentralized']
        colors = ['blue', 'orange', 'green', 'red']

        for i, strategy_type in enumerate(strategy_families):
            mask = df['strategy_family'].str.contains(strategy_type, case=False, na=False)
            if mask.any():
                strategy_data = df[mask]
                grouped = strategy_data.groupby('n_agents')['messages_per_agent'].median()

                if len(grouped) > 0:
                    linewidth = 4 if strategy_type == 'sppdec' else 2  # Emphasize SPPDEC
                    ax1.plot(grouped.index, grouped.values,
                             marker='o', linewidth=linewidth, markersize=8,
                             label=f'{strategy_type.title()} Strategies',
                             color=colors[i])

        ax1.set_xlabel('Number of Agents', fontsize=14)
        ax1.set_ylabel('Messages per Agent', fontsize=14)
        ax1.set_title('Communication Scaling by Strategy Type', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Right plot: Communication Efficiency (Messages vs Performance)
        df['performance_score'] = 1 / df['energy_efficiency']  # Higher = better

        for i, strategy_type in enumerate(strategy_families):
            mask = df['strategy_family'].str.contains(strategy_type, case=False, na=False)
            if mask.any():
                strategy_data = df[mask]

                # Calculate communication efficiency: performance per message
                strategy_data_copy = strategy_data.copy()
                strategy_data_copy['comm_efficiency'] = strategy_data_copy['performance_score'] / (
                        strategy_data_copy['messages_per_agent'] + 0.1)

                grouped = strategy_data_copy.groupby('n_agents')['comm_efficiency'].median()

                if len(grouped) > 0:
                    linewidth = 4 if strategy_type == 'sppdec' else 2  # Emphasize SPPDEC
                    ax2.plot(grouped.index, grouped.values,
                             marker='o', linewidth=linewidth, markersize=8,
                             label=f'{strategy_type.title()}',
                             color=colors[i])

        ax2.set_xlabel('Number of Agents', fontsize=14)
        ax2.set_ylabel('Performance per Message (Higher = Better)', fontsize=14)
        ax2.set_title('Communication Efficiency Scaling', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Alternative Visualizations for "Communication Load by Efficiency Level"

    def plot_sppdec_communication_advantage(self) -> None:
        """
        Highlight SPPDEC's communication advantage with multiple visualization approaches.
        Each subplot shows a different perspective on communication load by efficiency level.
        """

        df = self.sweep.dataframe.copy()

        # Calculate metrics
        df['energy_efficiency'] = df['total_energy_consumed'] / (df['tasks_completed'] * df['n_agents'])
        df['messages_per_agent'] = df['messages_sent'] / df['n_agents']
        df['efficiency_category'] = pd.cut(df['energy_efficiency'],
                                           bins=df['energy_efficiency'].quantile([0, 0.33, 0.67, 1.0]),
                                           labels=['High Efficiency', 'Medium Efficiency', 'Low Efficiency'])

        def classify_strategy_type(strategy_name):
            if 'sppdec' in strategy_name.lower():
                return 'SPPDEC'
            elif 'centralized' in strategy_name.lower():
                return 'Centralized'
            elif 'cbba' in strategy_name.lower():
                return 'CBBA'
            else:
                return 'Other'

        df['strategy_type'] = df['strategy_family'].apply(classify_strategy_type)

        # Create 2x3 subplot layout for multiple approaches
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))

        # APPROACH 1: Logarithmic Scale Scatter
        ax1 = axes[0, 0]
        strategy_colors = {'SPPDEC': 'green', 'Centralized': 'blue', 'CBBA': 'orange', 'Other': 'red'}

        for strategy_type in strategy_colors.keys():
            mask = df['strategy_type'] == strategy_type
            if mask.any():
                strategy_data = df[mask]

                # Add small offset to zero values for log scale
                comm_values = strategy_data['messages_per_agent'].replace(0, 0.001)

                ax1.scatter(strategy_data['energy_efficiency'], comm_values,
                            alpha=0.7, s=60, color=strategy_colors[strategy_type],
                            label=strategy_type)

        ax1.set_yscale('log')
        ax1.set_xlabel('Energy Efficiency (lower = better)')
        ax1.set_ylabel('Messages per Agent (log scale)')
        ax1.set_title('Log Scale: Communication vs Efficiency\n(Shows SPPDEC clustering near zero)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # APPROACH 2: Communication Efficiency Ratio - DUAL AXIS VERSION (FIXED)
        ax2 = axes[0, 1]

        # Calculate communication efficiency: performance per message
        df['performance'] = 1 / df['energy_efficiency']  # Higher = better performance
        df['comm_efficiency'] = df['performance'] / (df['messages_per_agent'] + 0.001)  # Avoid division by zero

        summary = df.groupby(['efficiency_category', 'strategy_type'])['comm_efficiency'].agg(
            ['mean', 'count']).reset_index()
        summary = summary[summary['count'] >= 1]  # Lowered threshold to include more data

        efficiency_categories = summary['efficiency_category'].unique()
        x_pos = np.arange(len(efficiency_categories))
        width = 0.15  # Narrower bars to fit more

        # Debug: Print summary to check data
        print("Summary data for debugging:")
        print(summary[['efficiency_category', 'strategy_type', 'mean', 'count']])

        # Create secondary y-axis for non-SPPDEC strategies
        ax2_right = ax2.twinx()

        # Plot SPPDEC on left axis (primary)
        sppdec_data = summary[summary['strategy_type'] == 'SPPDEC']
        if not sppdec_data.empty:
            sppdec_values = []
            for cat in efficiency_categories:
                cat_data = sppdec_data[sppdec_data['efficiency_category'] == cat]
                if len(cat_data) > 0:
                    sppdec_values.append(cat_data['mean'].iloc[0])
                else:
                    sppdec_values.append(0)

            bars_sppdec = ax2.bar(x_pos - width, sppdec_values, width,
                                  label='SPPDEC', color=strategy_colors['SPPDEC'], alpha=0.8)

            # Add value labels for SPPDEC
            if sppdec_values and max(sppdec_values) > 0:
                max_sppdec = max(sppdec_values)
                for i, (bar, val) in enumerate(zip(bars_sppdec, sppdec_values)):
                    if val > 0:
                        ax2.text(bar.get_x() + bar.get_width() / 2., val + max_sppdec * 0.02,
                                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

        # Plot other strategies on right axis (secondary) - FIXED VERSION
        offset = 0
        other_strategies = ['Centralized', 'CBBA', 'Other']
        for strategy_type in other_strategies:
            strategy_data = summary[summary['strategy_type'] == strategy_type]
            if not strategy_data.empty:
                print(f"\nProcessing {strategy_type}:")
                print(strategy_data)

                values = []
                for cat in efficiency_categories:
                    cat_data = strategy_data[strategy_data['efficiency_category'] == cat]
                    if len(cat_data) > 0:
                        val = cat_data['mean'].iloc[0]
                        values.append(val)
                        print(f"  {cat}: {val}")
                    else:
                        values.append(0)
                        print(f"  {cat}: 0 (no data)")

                color = strategy_colors[strategy_type]
                bars = ax2_right.bar(x_pos + offset * width, values, width,
                                     label=strategy_type, color=color, alpha=0.8)

                # Add value labels - FIXED
                if values and max(values) > 0:
                    max_val = max(values)
                    for i, (bar, val) in enumerate(zip(bars, values)):
                        if val > 0:
                            ax2_right.text(bar.get_x() + bar.get_width() / 2., val + max_val * 0.02,
                                           f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
                offset += 1

        # Formatting
        ax2.set_xlabel('Efficiency Category')
        ax2.set_ylabel('SPPDEC Performance per Message', color='green', fontweight='bold')
        ax2_right.set_ylabel('Other Strategies Performance per Message', color='blue', fontweight='bold')
        ax2.set_title('Communication Efficiency Ratio: Dual Scale\n(SPPDEC vs Communication-Dependent Strategies)')

        # Color the y-axis labels to match the data
        ax2.tick_params(axis='y', labelcolor='green')
        ax2_right.tick_params(axis='y', labelcolor='blue')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(efficiency_categories)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax2.grid(True, alpha=0.3)

        # APPROACH 3: Bubble Chart - Size = Efficiency, Color = Communication
        ax3 = axes[0, 2]

        strategy_summary = df.groupby(['strategy_type', 'efficiency_category']).agg({
            'energy_efficiency': 'mean',
            'messages_per_agent': 'mean',
            'strategy_family': 'count'  # Number of strategies
        }).reset_index()

        # Create bubble chart
        for i, row in strategy_summary.iterrows():
            x = list(efficiency_categories).index(row['efficiency_category'])
            y = list(strategy_colors.keys()).index(row['strategy_type'])

            # Size based on efficiency (smaller efficiency = larger bubble)
            size = 500 / (row['energy_efficiency'] + 0.1)

            # Color intensity based on communication load
            color_intensity = min(row['messages_per_agent'] / 50, 1.0)  # Normalize to 0-1

            ax3.scatter(x, y, s=size, alpha=0.6,
                        c=[[color_intensity, 0, 1 - color_intensity]],  # Red=high comm, Blue=low comm
                        edgecolors='black', linewidth=2)

            # Add text annotation
            ax3.text(x, y, f"{row['messages_per_agent']:.1f}",
                     ha='center', va='center', fontsize=10, fontweight='bold')

        ax3.set_xticks(range(len(efficiency_categories)))
        ax3.set_xticklabels(efficiency_categories)
        ax3.set_yticks(range(len(strategy_colors)))
        ax3.set_yticklabels(strategy_colors.keys())
        ax3.set_xlabel('Efficiency Category')
        ax3.set_ylabel('Strategy Type')
        ax3.set_title('Bubble Chart: Size=Efficiency, Color=Communication\n(Red=High Comm, Blue=Low Comm)')
        ax3.grid(True, alpha=0.3)

        # APPROACH 4: Communication Burden Index
        ax4 = axes[1, 0]

        # Create "Communication Burden Index" - normalized communication load
        max_comm = df['messages_per_agent'].max()
        df['comm_burden_index'] = df['messages_per_agent'] / max_comm * 100

        burden_summary = df.groupby(['efficiency_category', 'strategy_type'])['comm_burden_index'].agg(
            ['mean', 'std']).reset_index()
        burden_summary.columns = ['efficiency_category', 'strategy_type', 'mean_burden', 'std_burden']

        # Heatmap approach
        pivot_burden = burden_summary.pivot(index='strategy_type', columns='efficiency_category',
                                            values='mean_burden').fillna(0)

        im = ax4.imshow(pivot_burden.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

        # Add text annotations
        for i in range(len(pivot_burden.index)):
            for j in range(len(pivot_burden.columns)):
                value = pivot_burden.iloc[i, j]
                ax4.text(j, i, f'{value:.1f}%', ha='center', va='center',
                         color='white' if value > 50 else 'black', fontweight='bold')

        ax4.set_xticks(range(len(pivot_burden.columns)))
        ax4.set_xticklabels(pivot_burden.columns)
        ax4.set_yticks(range(len(pivot_burden.index)))
        ax4.set_yticklabels(pivot_burden.index)
        ax4.set_xlabel('Efficiency Category')
        ax4.set_ylabel('Strategy Type')
        ax4.set_title('Communication Burden Index\n(% of Maximum Communication Load)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Communication Burden (%)', rotation=270, labelpad=15)

        # APPROACH 5: Violin Plot Distribution
        ax5 = axes[1, 1]

        # Create violin plots for each efficiency category
        high_eff = df[df['efficiency_category'] == 'High Efficiency']

        violin_data = []
        violin_labels = []
        violin_colors = []

        for strategy_type in strategy_colors.keys():
            strategy_data = high_eff[high_eff['strategy_type'] == strategy_type]
            if len(strategy_data) >= 3:  # Need minimum data for violin plot
                violin_data.append(strategy_data['messages_per_agent'].values)
                violin_labels.append(strategy_type)
                violin_colors.append(strategy_colors[strategy_type])

        if violin_data:
            parts = ax5.violinplot(violin_data, positions=range(len(violin_data)),
                                   showmeans=True, showmedians=True)

            # Color the violin plots
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(violin_colors[i])
                pc.set_alpha(0.7)

        ax5.set_xticks(range(len(violin_labels)))
        ax5.set_xticklabels(violin_labels, rotation=45)
        ax5.set_ylabel('Messages per Agent')
        ax5.set_title('Communication Distribution\n(High Efficiency Strategies Only)')
        ax5.grid(True, alpha=0.3)

        # APPROACH 6: Communication Overhead Multiplier
        ax6 = axes[1, 2]

        # Calculate communication overhead relative to SPPDEC
        sppdec_baseline = df[df['strategy_type'] == 'SPPDEC']['messages_per_agent'].mean()
        df['comm_multiplier'] = df['messages_per_agent'] / max(sppdec_baseline, 0.001)

        multiplier_summary = df.groupby(['efficiency_category', 'strategy_type'])[
            'comm_multiplier'].mean().reset_index()

        # Create grouped bar chart
        categories = multiplier_summary['efficiency_category'].unique()
        x_pos = np.arange(len(categories))
        width = 0.2

        for i, strategy_type in enumerate(strategy_colors.keys()):
            strategy_data = multiplier_summary[multiplier_summary['strategy_type'] == strategy_type]
            if not strategy_data.empty:
                values = []
                for cat in categories:
                    cat_data = strategy_data[strategy_data['efficiency_category'] == cat]
                    if len(cat_data) > 0:
                        values.append(cat_data['comm_multiplier'].iloc[0])
                    else:
                        values.append(1 if strategy_type == 'SPPDEC' else 0)

                bars = ax6.bar(x_pos + i * width, values, width,
                               label=strategy_type, color=strategy_colors[strategy_type], alpha=0.8)

                # Add value labels on bars
                for j, (bar, val) in enumerate(zip(bars, values)):
                    if val > 1.1:  # Only label if significantly above baseline
                        ax6.text(bar.get_x() + bar.get_width() / 2., val + max(values) * 0.02,
                                 f'{val:.1f}×', ha='center', va='bottom', fontweight='bold')

        ax6.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.8, label='SPPDEC Baseline (1×)')
        ax6.set_xlabel('Efficiency Category')
        ax6.set_ylabel('Communication Overhead Multiplier\n(Relative to SPPDEC)')
        ax6.set_title('Communication Overhead vs SPPDEC\n(How many times more messages than SPPDEC)')
        ax6.set_xticks(x_pos + width * 1.5)
        ax6.set_xticklabels(categories)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Summary insights
        print("\n🎯 Alternative Visualization Insights:")
        print("1. Log Scale: Shows SPPDEC clustering near zero communication")
        print("2. Dual-Axis Efficiency Ratio: SPPDEC delivers orders-of-magnitude higher performance per message")
        print("3. Bubble Chart: Visualizes efficiency vs communication trade-offs")
        print("4. Burden Index: Shows SPPDEC's minimal communication burden")
        print("5. Violin Plots: Distribution of communication patterns")
        print("6. Overhead Multiplier: Quantifies how much more other strategies communicate")

    # Usage:
    # notebook_analyzer.plot_sppdec_communication_advantage()
    def export_results(self, output_dir: Union[str, Path]) -> None:
        """Export all results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Raw data
        df = self.sweep.dataframe
        if not df.empty:
            df.to_csv(output_dir / 'sweep_raw_data.csv', index=False)

        # Strategy summary
        strategy_summary = self.strategy_overview()
        if not strategy_summary.empty:
            strategy_summary.to_csv(output_dir / 'strategy_summary.csv')

        # Config summary
        config_summary = self.sweep.get_config_summary()
        config_summary.to_csv(output_dir / 'config_summary.csv', index=False)

        # SPPDEC analysis
        sppdec_df = self.sweep.get_sppdec_phases()
        if not sppdec_df.empty:
            sppdec_summary = (
                sppdec_df.groupby(['sppdec_phase', 'config_name'])
                    .agg({
                    'mission_makespan_steps': ['mean', 'std'],
                    'total_energy_consumed': ['mean', 'std'],
                    'unsafe_failures': 'sum'
                })
                    .round(3)
            )
            sppdec_summary.to_csv(output_dir / 'sppdec_analysis.csv')

        print(f"✅ Results exported to: {output_dir}")


# Convenience functions
def analyze_sweep(sweep_dir: Union[str, Path], show_progress: bool = True) -> NotebookAnalyzer:
    """
    One-line function to load and analyze a sweep with good visualization design.

    Args:
        sweep_dir: Path to station_scaling directory
        show_progress: Whether to show loading progress
    """
    analyzer = NotebookAnalyzer(sweep_dir)
    analyzer.load_all(show_progress=show_progress)
    return analyzer


def quick_sppdec_comparison(sweep_dir: Union[str, Path],
                            metric: str = 'mission_makespan_steps',
                            show_personality_radar: bool = True) -> None:
    """Quick SPPDEC phase comparison with clean visualization and personality radar."""
    analyzer = analyze_sweep(sweep_dir)
    analyzer.compare_sppdec_phases(metric)

    # Also run personality analysis if available
    try:
        analyzer.personality_analysis(metric)

        # Show personality distribution radar chart
        if show_personality_radar:
            analyzer.personality_distribution_radar()
    except:
        print("No personality data available for comparison")
