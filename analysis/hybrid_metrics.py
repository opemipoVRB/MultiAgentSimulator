# analysis/hybrid_metrics.py
"""
Phase 10: Hybrid Observability Metrics

Post-run analysis of predictive belief quality from hybrid_log and
correction_events written by FlightRecorder.tick_hybrid() and
FlightRecorder.record_correction().

Tier 1 metrics (single-run, data already available):
  - Prediction error vs visibility state
  - Position error vs time since last seen

Tier 2 metrics (cross-run, require Phase 11 sweep data):
  - Coordination performance vs sensing radius
  - Conflict rate vs prediction accuracy

Usage (single run):
    from analysis.hybrid_metrics import HybridMetricsAnalyzer
    analyzer = HybridMetricsAnalyzer.from_strategy_dir(
        Path("experiments/A3_P36/experiment_xxxx/sppdec_combined")
    )
    analyzer.plot_error_vs_visibility(save_path=Path("figures/fig_error_vis.pdf"))
    analyzer.plot_error_vs_time_hidden(save_path=Path("figures/fig_error_time.pdf"))

Usage (multi-run Tier 2):
    records = HybridMetricsAnalyzer.load_sweep_records(sweep_dir)
    analyzer = HybridMetricsAnalyzer(records)
    analyzer.plot_coordination_vs_radius(save_path=Path("figures/fig_coord_radius.pdf"))
    analyzer.plot_conflict_vs_accuracy(save_path=Path("figures/fig_conflict_accuracy.pdf"))
"""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# Match publication-quality defaults from visualizers.py
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 10


# =============================================================================
# Raw data structures
# =============================================================================

class HybridRecord:
    """
    All hybrid observability data extracted from a single strategy run.

    Attributes
    ----------
    config_name : str
        Sweep config label, e.g. "A3_P36". Empty string for single-run use.
    strategy_name : str
        Strategy identifier, e.g. "sppdec_combined".
    visibility_radius : float
        The sensing radius used for this run (inf for full observability).
    hybrid_df : pd.DataFrame
        One row per (observer, peer, step) tick from hybrid_log.
        Columns: observer_id, peer_id, step, sim_time, is_visible,
                 time_since_last_seen, position_error, predicted_energy,
                 confidence, energy_error.
    corrections_df : pd.DataFrame
        One row per reappearance correction event.
        Columns: observer_id, peer_id, step, sim_time, error_dist,
                 steps_since_seen.
    n_collisions : int
        Total fleet collisions in this run (from fleet_events).
    n_tasks : int
        Parcels delivered (from parcels list).
    """

    def __init__(
        self,
        config_name: str,
        strategy_name: str,
        visibility_radius: float,
        hybrid_df: pd.DataFrame,
        corrections_df: pd.DataFrame,
        n_collisions: int = 0,
        n_tasks: int = 0,
    ):
        self.config_name = config_name
        self.strategy_name = strategy_name
        self.visibility_radius = visibility_radius
        self.hybrid_df = hybrid_df
        self.corrections_df = corrections_df
        self.n_collisions = n_collisions
        self.n_tasks = n_tasks

    def __repr__(self) -> str:
        n_ticks = len(self.hybrid_df)
        n_corrections = len(self.corrections_df)
        return (
            f"HybridRecord(config={self.config_name!r}, "
            f"strategy={self.strategy_name!r}, "
            f"R={self.visibility_radius}, "
            f"ticks={n_ticks}, corrections={n_corrections})"
        )


# =============================================================================
# Loader
# =============================================================================

class HybridLogLoader:
    """
    Extracts hybrid_log and correction_events from the file artifacts written
    by FlightRecorder.export().

    The recorder writes a single export dict. Your runner splits this across
    per-strategy subdirectories. This loader is defensive: it checks multiple
    locations because the exact split depends on your runner implementation.

    Search order for hybrid_log:
      1. fleet.json top-level key "hybrid_log"
      2. Per-agent "hybrid_log" list inside agents.json agent dicts

    Search order for corrections:
      1. fleet.json top-level key "corrections"
      2. Per-agent "correction_events" inside agents.json agent dicts
    """

    @classmethod
    def load(
        cls,
        strategy_dir: Path,
        config_name: str = "",
        strategy_name: str = "",
        visibility_radius: float = float("inf"),
    ) -> HybridRecord:
        """
        Load all hybrid observability data from one strategy run directory.

        Parameters
        ----------
        strategy_dir : Path
            Directory containing fleet.json and agents.json.
        config_name : str
            Human-readable config label (e.g. "A3_P36"). Optional.
        strategy_name : str
            Strategy label. Falls back to directory name if empty.
        visibility_radius : float
            Sensing radius for this run. Pass float("inf") for full
            observability. Used only for Tier 2 cross-run plots.

        Returns
        -------
        HybridRecord
        """
        strategy_dir = Path(strategy_dir)
        if not strategy_dir.exists():
            raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")

        if not strategy_name:
            strategy_name = strategy_dir.name

        fleet_json = cls._load_json(strategy_dir / "fleet.json")
        agents_json = cls._load_json(strategy_dir / "agents.json")

        hybrid_raw = cls._extract_hybrid_log(fleet_json, agents_json)
        corrections_raw = cls._extract_corrections(fleet_json, agents_json)
        n_collisions = cls._count_collisions(fleet_json)
        n_tasks = cls._count_tasks(fleet_json)

        hybrid_df = cls._hybrid_to_dataframe(hybrid_raw)
        corrections_df = cls._corrections_to_dataframe(corrections_raw)

        return HybridRecord(
            config_name=config_name,
            strategy_name=strategy_name,
            visibility_radius=visibility_radius,
            hybrid_df=hybrid_df,
            corrections_df=corrections_df,
            n_collisions=n_collisions,
            n_tasks=n_tasks,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _extract_hybrid_log(fleet_json: Dict, agents_json) -> List[Dict]:
        """
        Try fleet.json top-level "hybrid_log" first, then per-agent in agents.json.
        Returns a flat list of tick dicts annotated with observer_id.
        """
        entries: List[Dict] = []

        # Source 1: fleet.json top-level {"hybrid_log": {agent_id: [entries]}}
        hybrid_map = fleet_json.get("hybrid_log", {})
        if isinstance(hybrid_map, dict) and hybrid_map:
            for observer_id, tick_list in hybrid_map.items():
                for entry in tick_list:
                    entries.append({"observer_id": observer_id, **entry})
            return entries

        # Source 2: per-agent dicts in agents.json
        agent_list = agents_json if isinstance(agents_json, list) else []
        for agent_data in agent_list:
            observer_id = agent_data.get("agent_id", "")
            for entry in agent_data.get("hybrid_log", []):
                entries.append({"observer_id": observer_id, **entry})

        return entries

    @staticmethod
    def _extract_corrections(fleet_json: Dict, agents_json) -> List[Dict]:
        """
        Try fleet.json top-level "corrections" first, then per-agent in agents.json.
        """
        # Source 1: fleet.json top-level corrections list
        corrections = fleet_json.get("corrections", [])
        if corrections:
            return corrections

        # Source 2: per-agent correction_events inside agents.json
        agent_list = agents_json if isinstance(agents_json, list) else []
        entries: List[Dict] = []
        for agent_data in agent_list:
            entries.extend(agent_data.get("correction_events", []))
        return entries

    @staticmethod
    def _count_collisions(fleet_json: Dict) -> int:
        return sum(
            1 for e in fleet_json.get("fleet_events", [])
            if e.get("type") == "collision"
        )

    @staticmethod
    def _count_tasks(fleet_json: Dict) -> int:
        return sum(
            1 for p in fleet_json.get("parcels", [])
            if p.get("delivered_by") is not None
        )

    # ------------------------------------------------------------------
    # DataFrame constructors
    # ------------------------------------------------------------------

    @staticmethod
    def _hybrid_to_dataframe(raw: List[Dict]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=[
                "observer_id", "peer_id", "step", "sim_time",
                "is_visible", "time_since_last_seen", "position_error",
                "predicted_energy", "confidence", "energy_error",
            ])

        rows = []
        for e in raw:
            rows.append({
                "observer_id":         e.get("observer_id", ""),
                "peer_id":             e.get("peer_id", ""),
                "step":                int(e.get("step", 0)),
                "sim_time":            float(e.get("sim_time", 0.0)),
                "is_visible":          bool(e.get("is_visible", False)),
                "time_since_last_seen": int(e.get("time_since_last_seen", 0)),
                "position_error":      float(e.get("position_error", 0.0)),
                "predicted_energy":    float(e.get("predicted_energy", 0.0)),
                "confidence":          float(e.get("confidence", 1.0)),
                "energy_error":        (
                    float(e["energy_error"])
                    if e.get("energy_error") is not None
                    else np.nan
                ),
            })

        df = pd.DataFrame(rows)
        df["visibility_label"] = df["is_visible"].map(
            {True: "Visible", False: "Hidden"}
        )
        return df

    @staticmethod
    def _corrections_to_dataframe(raw: List[Dict]) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame(columns=[
                "observer_id", "peer_id", "step", "sim_time",
                "error_dist", "steps_since_seen",
            ])

        rows = []
        for e in raw:
            t = e.get("time", {})
            rows.append({
                "observer_id":    e.get("observer_id", ""),
                "peer_id":        e.get("peer_id", ""),
                "step":           int(t.get("step", e.get("step", 0))),
                "sim_time":       float(t.get("sim_time", 0.0)),
                "error_dist":     float(e.get("error_dist", 0.0)),
                "steps_since_seen": int(e.get("steps_since_seen", 0)),
            })

        return pd.DataFrame(rows)


# =============================================================================
# Main analyzer
# =============================================================================

class HybridMetricsAnalyzer:
    """
    Phase 10 analysis: predictive belief quality for SPPDEC hybrid observability.

    Accepts one or more HybridRecord objects and exposes one method per metric.
    Works on a single record for Tier 1; uses all loaded records for Tier 2.

    Parameters
    ----------
    records : HybridRecord or list of HybridRecord
        Data to analyse. For single-run Tier 1 plots, pass one record.
        For Tier 2 cross-run plots, pass the full sweep list.

    Class methods
    -------------
    from_strategy_dir(path, ...)
        Convenience constructor for the single-run case.
    load_sweep_records(sweep_dir, strategy_filter)
        Load all matching strategy runs from a sweep directory tree.
    """

    def __init__(self, records: Union["HybridRecord", List["HybridRecord"]]):
        if isinstance(records, HybridRecord):
            records = [records]
        self.records = records

        # Combined DataFrames used by Tier 2 methods
        self._combined_hybrid: Optional[pd.DataFrame] = None
        self._combined_corrections: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_strategy_dir(
        cls,
        strategy_dir: Union[str, Path],
        config_name: str = "",
        strategy_name: str = "",
        visibility_radius: float = float("inf"),
    ) -> "HybridMetricsAnalyzer":
        """
        Construct an analyzer from a single strategy run directory.

        Parameters
        ----------
        strategy_dir : path to the directory containing fleet.json + agents.json
        config_name  : optional sweep label, e.g. "A3_P36"
        strategy_name: optional override; defaults to directory name
        visibility_radius: sensing radius used for this run
        """
        record = HybridLogLoader.load(
            Path(strategy_dir),
            config_name=config_name,
            strategy_name=strategy_name,
            visibility_radius=visibility_radius,
        )
        print(f"  Loaded: {record}")
        return cls(record)

    @classmethod
    def load_sweep_records(
        cls,
        sweep_dir: Union[str, Path],
        strategy_filter: str = "sppdec",
        radius_map: Optional[Dict[str, float]] = None,
    ) -> "HybridMetricsAnalyzer":
        """
        Load all matching strategy runs from a sweep directory tree.

        Expected structure:
            sweep_dir/
            ├── A3_P36/
            │   └── experiment_xxxx/
            │       └── sppdec_combined/
            │           ├── fleet.json
            │           └── agents.json
            └── ...

        Parameters
        ----------
        sweep_dir : root sweep directory
        strategy_filter : only load strategies whose names contain this string
        radius_map : optional {config_name: visibility_radius} override.
                     Used when the same sweep was run at different R values
                     and you want to encode that in the record for Tier 2.
        """
        sweep_dir = Path(sweep_dir)
        records: List[HybridRecord] = []
        radius_map = radius_map or {}

        config_pattern = re.compile(r"A\d+_P\d+")

        for config_dir in sorted(sweep_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name
            if not config_pattern.match(config_name):
                continue

            for exp_dir in sorted(config_dir.glob("experiment_*")):
                if not exp_dir.is_dir():
                    continue

                for strategy_dir in sorted(exp_dir.iterdir()):
                    if not strategy_dir.is_dir():
                        continue
                    if strategy_filter and strategy_filter not in strategy_dir.name:
                        continue

                    radius = radius_map.get(config_name, float("inf"))
                    try:
                        record = HybridLogLoader.load(
                            strategy_dir,
                            config_name=config_name,
                            strategy_name=strategy_dir.name,
                            visibility_radius=radius,
                        )
                        records.append(record)
                        print(f"  Loaded: {record}")
                    except Exception as exc:
                        warnings.warn(f"Failed to load {strategy_dir}: {exc}")

        if not records:
            raise ValueError(
                f"No records found under {sweep_dir} "
                f"matching strategy_filter={strategy_filter!r}"
            )

        print(f"\nLoaded {len(records)} records total.")
        return cls(records)

    # ------------------------------------------------------------------
    # Internal data access
    # ------------------------------------------------------------------

    @property
    def primary(self) -> HybridRecord:
        """Return the first (or only) record. Used by Tier 1 methods."""
        return self.records[0]

    @property
    def combined_hybrid(self) -> pd.DataFrame:
        """Concatenate all hybrid_df records with provenance columns."""
        if self._combined_hybrid is None:
            frames = []
            for r in self.records:
                df = r.hybrid_df.copy()
                df["config_name"] = r.config_name
                df["strategy_name"] = r.strategy_name
                df["visibility_radius"] = r.visibility_radius
                frames.append(df)
            self._combined_hybrid = (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )
        return self._combined_hybrid

    @property
    def combined_corrections(self) -> pd.DataFrame:
        """Concatenate all corrections_df records with provenance columns."""
        if self._combined_corrections is None:
            frames = []
            for r in self.records:
                df = r.corrections_df.copy()
                df["config_name"] = r.config_name
                df["strategy_name"] = r.strategy_name
                df["visibility_radius"] = r.visibility_radius
                frames.append(df)
            self._combined_corrections = (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )
        return self._combined_corrections

    def summary(self) -> pd.DataFrame:
        """
        High-level summary statistics across all loaded records.

        Returns one row per record with key prediction quality metrics.
        """
        rows = []
        for r in self.records:
            hdf = r.hybrid_df
            cdf = r.corrections_df

            hidden = hdf[~hdf["is_visible"]] if not hdf.empty else pd.DataFrame()

            rows.append({
                "config":              r.config_name,
                "strategy":            r.strategy_name,
                "visibility_radius":   r.visibility_radius,
                "n_ticks":             len(hdf),
                "n_hidden_ticks":      len(hidden),
                "n_corrections":       len(cdf),
                "mean_pos_error":      hidden["position_error"].mean() if not hidden.empty else np.nan,
                "median_pos_error":    hidden["position_error"].median() if not hidden.empty else np.nan,
                "p95_pos_error":       hidden["position_error"].quantile(0.95) if not hidden.empty else np.nan,
                "mean_steps_hidden":   hidden["time_since_last_seen"].mean() if not hidden.empty else np.nan,
                "mean_correction_err": cdf["error_dist"].mean() if not cdf.empty else np.nan,
                "n_collisions":        r.n_collisions,
                "n_tasks":             r.n_tasks,
            })

        return pd.DataFrame(rows).round(4)

    # ------------------------------------------------------------------
    # Tier 1: Prediction error vs visibility
    # ------------------------------------------------------------------

    def plot_error_vs_visibility(
        self,
        record: Optional["HybridRecord"] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Tier 1 metric: Position prediction error distribution by visibility state.

        Shows a violin plot of position_error split by is_visible.
        When an agent is visible the error is always 0 (position was snapped),
        so the visible distribution is a degenerate spike at 0. The invisible
        distribution reveals how fast prediction degrades over time.

        The figure is suitable for inclusion in the paper section
        "Hybrid Observability Model".

        Parameters
        ----------
        record : HybridRecord, optional
            Defaults to self.primary (first loaded record).
        save_path : Path, optional
            If provided, saves as PDF/PNG at 300 dpi.
        show : bool
            Call plt.show() after building. Set False in notebook loops.

        Returns
        -------
        matplotlib Figure
        """
        r = record or self.primary
        hdf = r.hybrid_df

        if hdf.empty:
            raise ValueError("hybrid_df is empty — no hybrid_log data found in source files.")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

        # --- Left panel: violin ---
        ax = axes[0]
        vis_data = hdf[hdf["is_visible"]]["position_error"]
        hid_data = hdf[~hdf["is_visible"]]["position_error"]

        plot_df = pd.DataFrame({
            "Position Error (cells)": pd.concat([vis_data, hid_data], ignore_index=True),
            "Visibility": (
                ["Visible"] * len(vis_data) + ["Hidden"] * len(hid_data)
            ),
        })

        sns.violinplot(
            data=plot_df,
            x="Visibility",
            y="Position Error (cells)",
            hue="Visibility",
            palette={"Visible": "#2196F3", "Hidden": "#FF5722"},
            legend=False,
            inner="box",
            cut=0,
            ax=ax,
        )

        ax.set_title("Position Error by Visibility State", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Position Error (grid cells)")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate counts
        for xi, (label, data) in enumerate([("Visible", vis_data), ("Hidden", hid_data)]):
            ax.annotate(
                f"n={len(data):,}\nmedian={data.median():.2f}",
                xy=(xi, data.quantile(0.75)),
                ha="center",
                fontsize=9,
                color="black",
            )

        # --- Right panel: CDF of hidden-only errors ---
        ax2 = axes[1]
        if not hid_data.empty:
            sorted_err = np.sort(hid_data.values)
            cdf_y = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            ax2.plot(sorted_err, cdf_y, color="#FF5722", linewidth=2)
            ax2.axvline(
                hid_data.median(), color="#FF5722",
                linestyle="--", linewidth=1.5,
                label=f"Median = {hid_data.median():.2f}"
            )
            ax2.axvline(
                hid_data.quantile(0.95), color="black",
                linestyle=":", linewidth=1.2,
                label=f"95th pct = {hid_data.quantile(0.95):.2f}"
            )
            ax2.legend(fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No hidden observations", ha="center", va="center",
                     transform=ax2.transAxes, color="grey")

        ax2.set_title("CDF of Position Error (Hidden Peers)", fontweight="bold")
        ax2.set_xlabel("Position Error (grid cells)")
        ax2.set_ylabel("Cumulative Fraction")
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Suptitle with run context
        label = (
            f"{r.config_name} / {r.strategy_name}"
            if r.config_name
            else r.strategy_name
        )
        fig.suptitle(
            f"Prediction Error vs Visibility — {label}",
            fontsize=13,
            fontweight="bold",
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Tier 1: Error vs time since last seen
    # ------------------------------------------------------------------

    def plot_error_vs_time_hidden(
        self,
        record: Optional["HybridRecord"] = None,
        max_steps: int = 60,
        bin_width: int = 5,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Tier 1 metric: How position prediction error grows with occlusion duration.

        Shows mean position_error ± std binned by time_since_last_seen for
        all hidden-peer observations. Overlays a Pearson correlation annotation.
        A linear regression line is fitted and displayed.

        This is the core evidence that the straight-line trajectory extrapolation
        degrades at a characterisable rate, which motivates the hybrid approach
        over pure inference.

        Parameters
        ----------
        record : HybridRecord, optional
        max_steps : int
            Clip the x-axis at this many steps. Bins with fewer than 5
            observations are shown with lighter markers.
        bin_width : int
            Step width for binned aggregation. Default 5.
        save_path : Path, optional
        show : bool

        Returns
        -------
        matplotlib Figure
        """
        r = record or self.primary
        hdf = r.hybrid_df

        hidden = hdf[~hdf["is_visible"]].copy()
        if hidden.empty:
            raise ValueError(
                "No hidden-peer observations found. "
                "Ensure visibility_radius < inf so some peers are ever invisible."
            )

        hidden = hidden[hidden["time_since_last_seen"] <= max_steps]

        # Compute correlation on raw (unbinned) data
        corr, pval = stats.pearsonr(
            hidden["time_since_last_seen"],
            hidden["position_error"],
        )

        # Bin for the error-bar trace
        hidden["t_bin"] = (
            (hidden["time_since_last_seen"] // bin_width) * bin_width
        )
        binned = (
            hidden
            .groupby("t_bin")["position_error"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        binned.columns = ["t_bin", "mean_err", "std_err", "n"]

        # Linear regression for trend line
        slope, intercept, r_val, p_lin, _ = stats.linregress(
            hidden["time_since_last_seen"],
            hidden["position_error"],
        )
        x_line = np.linspace(0, max_steps, 200)
        y_line = slope * x_line + intercept

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # --- Left panel: binned error-bar ---
        ax = axes[0]
        solid = binned[binned["n"] >= 5]
        sparse = binned[binned["n"] < 5]

        ax.errorbar(
            solid["t_bin"],
            solid["mean_err"],
            yerr=solid["std_err"].fillna(0),
            fmt="o-",
            color="#FF5722",
            linewidth=2,
            capsize=4,
            capthick=1.5,
            markersize=6,
            label="Mean ± Std (n ≥ 5)",
        )
        if not sparse.empty:
            ax.scatter(
                sparse["t_bin"],
                sparse["mean_err"],
                marker="x",
                color="#FF5722",
                alpha=0.4,
                zorder=3,
                label="Mean (n < 5)",
            )

        ax.plot(
            x_line, y_line,
            "--",
            color="black",
            linewidth=1.5,
            alpha=0.6,
            label=f"Linear fit (R²={r_val**2:.3f})",
        )

        ax.set_xlabel("Steps Since Last Seen")
        ax.set_ylabel("Mean Position Error (grid cells)")
        ax.set_title("Prediction Degradation Over Occlusion Duration", fontweight="bold")
        ax.set_xlim(0, max_steps)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Correlation annotation box
        sig_str = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax.text(
            0.97, 0.05,
            f"r = {corr:.3f} {sig_str}\np = {pval:.2e}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8),
        )

        # --- Right panel: scatter density (hex) ---
        ax2 = axes[1]
        if len(hidden) > 2:
            hb = ax2.hexbin(
                hidden["time_since_last_seen"],
                hidden["position_error"],
                gridsize=30,
                cmap="YlOrRd",
                mincnt=1,
            )
            fig.colorbar(hb, ax=ax2, label="Observation Count")
        else:
            ax2.scatter(
                hidden["time_since_last_seen"],
                hidden["position_error"],
                alpha=0.4, s=20, color="#FF5722",
            )

        ax2.plot(x_line, y_line, "--", color="black", linewidth=1.5, alpha=0.7)
        ax2.set_xlabel("Steps Since Last Seen")
        ax2.set_ylabel("Position Error (grid cells)")
        ax2.set_title("Observation Density", fontweight="bold")
        ax2.set_xlim(0, max_steps)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.2)

        label = (
            f"{r.config_name} / {r.strategy_name}"
            if r.config_name
            else r.strategy_name
        )
        fig.suptitle(
            f"Error vs Time Hidden — {label}",
            fontsize=13,
            fontweight="bold",
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Tier 1 bonus: confidence decay validation
    # ------------------------------------------------------------------

    def plot_confidence_vs_time(
        self,
        record: Optional["HybridRecord"] = None,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Validate that confidence decays as expected with occlusion time.

        Plots mean confidence vs time_since_last_seen for hidden observations
        and overlays the theoretical 0.5^(t/10) half-life decay curve
        (matching the formula in perceive()).

        Not a primary paper figure but useful as a sanity check during
        Phase 10 validation before the Phase 11 sweep.
        """
        r = record or self.primary
        hdf = r.hybrid_df
        hidden = hdf[~hdf["is_visible"]].copy()

        if hidden.empty:
            raise ValueError("No hidden-peer observations found.")

        t_vals = hidden.groupby("time_since_last_seen")["confidence"].mean()

        t_theory = np.arange(0, hidden["time_since_last_seen"].max() + 1)
        conf_theory = np.maximum(0.05, 0.5 ** (t_theory / 10.0))

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(t_vals.index, t_vals.values, "o-", color="#2196F3",
                linewidth=2, markersize=5, label="Observed mean confidence")
        ax.plot(t_theory, conf_theory, "--", color="black", linewidth=1.5,
                label="Theoretical: 0.5^(t/10)")
        ax.axhline(0.05, color="grey", linestyle=":", linewidth=1, label="Floor (0.05)")

        ax.set_xlabel("Steps Since Last Seen")
        ax.set_ylabel("Confidence")
        ax.set_title("Confidence Decay Validation", fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Tier 2: Coordination performance vs sensing radius
    # ------------------------------------------------------------------

    def plot_coordination_vs_radius(
        self,
        metric: str = "n_tasks",
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Tier 2 metric: How coordination performance changes with sensing radius.

        Requires multiple HybridRecord objects loaded with different
        visibility_radius values (e.g. from load_sweep_records() with
        radius_map provided, or from three separate runs at R=inf, R=2, R=0).

        Plots mean metric per radius value with 95% CI if multiple seeds
        are available. Falls back to a single point per radius otherwise.

        Parameters
        ----------
        metric : str
            "n_tasks" (tasks delivered) or "n_collisions". Can be extended
            to any numeric attribute of HybridRecord once your runner logs it.
        save_path : Path, optional
        show : bool

        Returns
        -------
        matplotlib Figure

        Raises
        ------
        ValueError
            If all records have the same visibility_radius (no sweep to plot).
        """
        radii = [r.visibility_radius for r in self.records]
        if len(set(r for r in radii if r != float("inf"))) == 0 and len(set(radii)) <= 1:
            raise ValueError(
                "All records have the same visibility_radius. "
                "Tier 2 plots require runs at different radius values. "
                "Pass radius_map to load_sweep_records() or load records manually."
            )

        rows = []
        for r in self.records:
            rows.append({
                "radius_label": "inf" if r.visibility_radius == float("inf") else str(r.visibility_radius),
                "radius_numeric": (
                    999 if r.visibility_radius == float("inf")
                    else r.visibility_radius
                ),
                "value": getattr(r, metric, np.nan),
                "config": r.config_name,
                "strategy": r.strategy_name,
            })

        df = pd.DataFrame(rows).sort_values("radius_numeric")

        grouped = (
            df.groupby(["radius_numeric", "radius_label"])["value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped["ci"] = 1.96 * grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

        ax.errorbar(
            grouped["radius_numeric"],
            grouped["mean"],
            yerr=grouped["ci"].fillna(0),
            fmt="o-",
            color="#4CAF50",
            linewidth=2.5,
            capsize=5,
            capthick=1.5,
            markersize=8,
        )

        # Label x-ticks with readable radius names
        ax.set_xticks(grouped["radius_numeric"])
        ax.set_xticklabels(grouped["radius_label"])

        metric_label = metric.replace("_", " ").title()
        ax.set_xlabel("Sensing Radius (grid cells)")
        ax.set_ylabel(metric_label)
        ax.set_title(
            f"Coordination Performance vs Sensing Radius\n{metric_label}",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Annotate individual points
        for _, row in grouped.iterrows():
            ax.annotate(
                f"{row['mean']:.1f}",
                xy=(row["radius_numeric"], row["mean"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Tier 2: Conflict rate vs prediction accuracy
    # ------------------------------------------------------------------

    def plot_conflict_vs_accuracy(
        self,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Tier 2 metric: Relationship between prediction accuracy and fleet conflicts.

        Each point is one run (HybridRecord). X-axis is mean correction error
        (position_error at reappearance events). Y-axis is collision count
        normalised by tasks delivered.

        A significant positive correlation here supports the claim that
        better belief accuracy reduces coordination conflicts — a key
        contribution of the hybrid observability model.

        Parameters
        ----------
        save_path : Path, optional
        show : bool

        Returns
        -------
        matplotlib Figure

        Raises
        ------
        ValueError
            If fewer than 3 records are loaded (cannot fit a meaningful
            regression).
        """
        if len(self.records) < 3:
            raise ValueError(
                f"Tier 2 conflict analysis requires at least 3 runs; "
                f"got {len(self.records)}. Load a full sweep first."
            )

        rows = []
        for r in self.records:
            mean_err = (
                r.corrections_df["error_dist"].mean()
                if not r.corrections_df.empty
                else 0.0
            )
            conflict_rate = (
                r.n_collisions / max(r.n_tasks, 1)
            )
            rows.append({
                "mean_correction_error": mean_err,
                "conflict_rate":         conflict_rate,
                "config":                r.config_name,
                "strategy":              r.strategy_name,
                "visibility_radius":     r.visibility_radius,
            })

        df = pd.DataFrame(rows)

        corr, pval = stats.pearsonr(
            df["mean_correction_error"],
            df["conflict_rate"],
        )
        slope, intercept, r_val, _, _ = stats.linregress(
            df["mean_correction_error"],
            df["conflict_rate"],
        )

        x_line = np.linspace(df["mean_correction_error"].min(),
                             df["mean_correction_error"].max(), 200)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        sc = ax.scatter(
            df["mean_correction_error"],
            df["conflict_rate"],
            c=df["visibility_radius"].replace({float("inf"): 999}),
            cmap="viridis_r",
            s=80,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

        ax.plot(x_line, y_line, "--", color="black", linewidth=1.5, alpha=0.7,
                label=f"Linear fit (R²={r_val**2:.3f})")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Sensing Radius (grid cells)")

        for _, row in df.iterrows():
            ax.annotate(
                row["config"],
                xy=(row["mean_correction_error"], row["conflict_rate"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="grey",
            )

        sig_str = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax.text(
            0.97, 0.95,
            f"r = {corr:.3f} {sig_str}\np = {pval:.2e}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.8),
        )

        ax.legend(fontsize=9)
        ax.set_xlabel("Mean Correction Error (grid cells)")
        ax.set_ylabel("Conflict Rate (collisions / tasks delivered)")
        ax.set_title(
            "Fleet Conflict Rate vs Prediction Accuracy",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Convenience: generate all Tier 1 figures in one call
    # ------------------------------------------------------------------

    def generate_tier1_report(
        self,
        output_dir: Union[str, Path],
        record: Optional["HybridRecord"] = None,
        fmt: str = "pdf",
    ) -> None:
        """
        Generate and save all Tier 1 figures for a single run.

        Writes three files into output_dir:
          - fig_p10_error_vs_visibility.<fmt>
          - fig_p10_error_vs_time_hidden.<fmt>
          - fig_p10_confidence_decay.<fmt>

        Parameters
        ----------
        output_dir : Path
            Directory to write figures into (created if absent).
        record : HybridRecord, optional
            Defaults to self.primary.
        fmt : str
            File format passed to savefig ("pdf", "png", "svg").
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        r = record or self.primary
        print(f"Generating Tier 1 report for: {r.strategy_name}")

        self.plot_error_vs_visibility(
            record=r,
            save_path=output_dir / f"fig_p10_error_vs_visibility.{fmt}",
            show=False,
        )
        plt.close("all")

        try:
            self.plot_error_vs_time_hidden(
                record=r,
                save_path=output_dir / f"fig_p10_error_vs_time_hidden.{fmt}",
                show=False,
            )
        except ValueError as e:
            print(f"  Skipped error_vs_time: {e}")
        plt.close("all")

        try:
            self.plot_confidence_vs_time(
                record=r,
                save_path=output_dir / f"fig_p10_confidence_decay.{fmt}",
                show=False,
            )
        except ValueError as e:
            print(f"  Skipped confidence_decay: {e}")
        plt.close("all")

        print(f"Tier 1 report written to: {output_dir}")