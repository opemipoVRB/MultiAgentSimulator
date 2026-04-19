# analysis/slm_analyzer.py
"""
Decision-level analysis for SLM policy selection events logged in fleet.json.

SweepAnalyzer operates at strategy-run granularity (one row per strategy per
experiment). SLMDecisionAnalyzer operates at event granularity (one row per
SLM call per agent per step). The two can be joined on
(config_name, run_id, strategy_name) when cross-referencing outcome metrics.

Expected directory structure (same as SweepAnalyzer):
    sweep_dir/
    ├── A4_P8/
    │   └── experiment_xxxxxx/
    │       └── <strategy_name>/
    │           └── fleet.json
    └── _A12_P36_belief_tracking_disabled/
        └── experiment_yyyyyy/
            └── <strategy_name>/
                └── fleet.json
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SLMDecisionAnalyzer:
    """
    Loads and analyses SLM policy-selection events from fleet.json files
    across one or more sweep configurations.

    Each slm_decision event captures the full decision context at the moment
    the SLM arbitrated between Greedy and Hungarian assignment:
      - which agent fired it
      - at which simulation step
      - which policy was chosen
      - confidence score
      - three continuous context signals + their trend labels
      - both cost estimates and the chosen cost

    Mirrors the SweepAnalyzer interface: initialise with a sweep directory,
    call load_all() or load_config(), then access .dataframe.
    """

    def __init__(self, sweep_dir: Union[str, Path]):
        self.sweep_dir = Path(sweep_dir)
        if not self.sweep_dir.exists():
            raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

        self.sweep_name = self.sweep_dir.name
        self._raw_events: List[dict] = []
        self._dataframe: Optional[pd.DataFrame] = None

        print(f"📊 SLMDecisionAnalyzer initialized for: {self.sweep_name}")

    # ------------------------------------------------------------------
    # Config discovery (mirrors SweepAnalyzer)
    # ------------------------------------------------------------------

    @property
    def config_names(self) -> List[str]:
        """All A{n}_P{m} directories found under sweep_dir."""
        configs = [
            d.name for d in self.sweep_dir.iterdir()
            if d.is_dir() and re.match(r"A\d+_P\d+", d.name)
        ]
        return sorted(configs, key=self._parse_config_key)

    # ------------------------------------------------------------------
    # Loading interface
    # ------------------------------------------------------------------

    def load_all(self, skip_errors: bool = True) -> "SLMDecisionAnalyzer":
        """
        Walk the full sweep directory, find every fleet.json, and extract
        slm_decision events. Returns self for method chaining.
        """
        fleet_files = list(self.sweep_dir.rglob("fleet.json"))
        if not fleet_files:
            print("⚠️  No fleet.json files found.")
            return self

        print(f"🔄 Found {len(fleet_files)} fleet.json file(s)")

        loaded = 0
        for fleet_path in sorted(fleet_files):
            try:
                self._load_fleet_file(fleet_path)
                loaded += 1
            except Exception as exc:
                if skip_errors:
                    print(f"  ⚠️  Skipping {fleet_path.relative_to(self.sweep_dir)}: {exc}")
                else:
                    raise

        self._dataframe = None  # invalidate cache
        print(f"✅ Loaded {loaded} fleet file(s) — {len(self._raw_events)} total SLM decisions")
        return self

    def load_config(self, config_name: str, skip_errors: bool = True) -> "SLMDecisionAnalyzer":
        """
        Load slm_decision events from a single config directory (e.g. 'A4_P8').
        Safe to call multiple times — does not re-add already-loaded events.
        """
        config_dir = self.sweep_dir / config_name
        if not config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

        fleet_files = list(config_dir.rglob("fleet.json"))
        if not fleet_files:
            print(f"⚠️  No fleet.json files found in {config_name}")
            return self

        loaded = 0
        for fleet_path in sorted(fleet_files):
            try:
                self._load_fleet_file(fleet_path)
                loaded += 1
            except Exception as exc:
                if skip_errors:
                    print(f"  ⚠️  Skipping {fleet_path.relative_to(self.sweep_dir)}: {exc}")
                else:
                    raise

        self._dataframe = None
        print(f"✓ {config_name}: {loaded} fleet file(s) loaded")
        return self

    def load_fleet_file(self, fleet_path: Union[str, Path]) -> "SLMDecisionAnalyzer":
        """Load a single fleet.json file explicitly (for targeted inspection)."""
        self._load_fleet_file(Path(fleet_path))
        self._dataframe = None
        return self

    # ------------------------------------------------------------------
    # Core DataFrame
    # ------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        One row per SLM decision event.

        Columns
        -------
        Provenance
            config_name         e.g. 'A4_P8'
            n_agents, n_parcels parsed from config_name
            run_id              experiment directory name
            strategy_name       subdirectory inside experiment dir
        Decision identity
            agent_id            e.g. 'agent_0'
            step                simulation step (int)
            sim_time            simulation time in minutes (float)
            world_time          unix timestamp of the decision
        Decision outcome
            policy              'greedy' | 'hungarian' | 'greedy_fallback'
            confidence          float [0, 1]
        Context signals
            belief_infeasibility, energy_error, competition_ratio   float
            belief_trend, error_trend, competition_trend            str
        Cost estimates
            cost_greedy, cost_hungarian, cost_chosen                float
            cost_delta          cost_hungarian − cost_greedy
                                negative → hungarian was cheaper at decision time
        Derived flags
            switched            True when policy is not 'greedy' or 'greedy_fallback'
            is_fallback         True when policy == 'greedy_fallback'
        """
        if self._dataframe is None:
            self._dataframe = self._build_dataframe()
        return self._dataframe

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def policy_distribution(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Policy selection counts and percentages, grouped by config and strategy
        (or a custom groupby list).
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        groupby = groupby or ["config_name", "strategy_name"]
        counts = (
            df.groupby(groupby + ["policy"])
            .size()
            .rename("count")
            .reset_index()
        )
        totals = counts.groupby(groupby)["count"].transform("sum")
        counts["pct"] = (counts["count"] / totals * 100).round(1)
        return counts.sort_values(groupby + ["policy"]).reset_index(drop=True)

    def per_agent_summary(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Per-agent breakdown of decision counts, policy mix, and mean context
        signals. Useful for spotting fleet asymmetries — e.g. one agent
        consistently deciding under higher infeasibility than others.
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        groupby = groupby or ["config_name", "strategy_name", "agent_id", "policy"]
        return (
            df.groupby(groupby)
            .agg(
                n_decisions=("policy", "count"),
                mean_confidence=("confidence", "mean"),
                mean_belief=("belief_infeasibility", "mean"),
                mean_energy_error=("energy_error", "mean"),
                mean_competition=("competition_ratio", "mean"),
                mean_cost_delta=("cost_delta", "mean"),
            )
            .round(4)
            .reset_index()
            .sort_values(groupby)
        )

    def switching_rate(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fraction of decisions where the SLM chose something other than greedy.
        A non-zero rate at A4_P8 but zero at _A12_P36_belief_tracking_disabled would indicate scale
        suppresses policy differentiation.
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        groupby = groupby or ["config_name", "strategy_name"]
        return (
            df.groupby(groupby)["switched"]
            .agg(
                n_decisions="count",
                n_switched="sum",
                switch_rate=lambda x: round(float(x.mean()), 4),
            )
            .reset_index()
        )

    def cost_delta_summary(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Statistics for cost_delta (hungarian − greedy) by group.

        A mean near zero confirms what this run showed: the two estimators
        rarely diverge. A large negative mean would mean Hungarian was
        systematically cheaper but never chosen.
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        groupby = groupby or ["config_name", "strategy_name", "policy"]
        return (
            df.groupby(groupby)["cost_delta"]
            .agg(mean="mean", std="std", min="min", max="max", count="count")
            .round(6)
            .reset_index()
        )

    def context_profile(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Mean context signals broken down by chosen policy. The primary tool
        for answering: 'under what conditions does the SLM favour Hungarian?'
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        groupby = groupby or ["config_name", "strategy_name", "policy"]
        signals = [
            "belief_infeasibility",
            "energy_error",
            "competition_ratio",
            "confidence",
        ]
        return (
            df.groupby(groupby)[signals]
            .mean()
            .round(4)
            .reset_index()
        )

    def trend_crosstab(self, groupby: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Frequency of each (belief_trend × error_trend × competition_trend)
        combination, split by chosen policy. Shows which trend states drive
        which decisions.
        """
        df = self.dataframe.copy()
        if df.empty:
            return pd.DataFrame()

        groupby = groupby or ["config_name", "strategy_name"]
        df["trend_combo"] = (
            df["belief_trend"] + " | " +
            df["error_trend"] + " | " +
            df["competition_trend"]
        )
        return (
            df.groupby(groupby + ["trend_combo", "policy"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

    def decisions_over_time(
        self,
        config_name: Optional[str] = None,
        strategy_name: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Decisions in chronological order, optionally filtered. Suitable for
        plotting policy transitions across sim_time per agent.
        """
        df = self.dataframe.copy()
        if config_name:
            df = df[df["config_name"] == config_name]
        if strategy_name:
            df = df[df["strategy_name"] == strategy_name]
        if agent_id:
            df = df[df["agent_id"] == agent_id]

        return df.sort_values(["agent_id", "step"]).reset_index(drop=True)

    def divergence_events(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Decisions where |cost_delta| exceeded a threshold — i.e. where
        Hungarian and Greedy genuinely disagreed and the SLM had a real
        choice to make. The first run showed all deltas at zero; this method
        becomes meaningful once fleet scale or task density forces divergence.

        Parameters
        ----------
        threshold : absolute cost difference required to count as divergence.
                    Default 0.0 returns any row where the two costs differ at all.
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        return (
            df[df["cost_delta"].abs() > threshold]
            .sort_values("cost_delta")
            .reset_index(drop=True)
        )

    def get_config_summary(self) -> pd.DataFrame:
        """
        Config-level summary: fleet file count, total decisions, unique agents,
        switch rate, fallback rate, and mean cost delta per strategy. Mirrors
        SweepAnalyzer.get_config_summary() in spirit.
        """
        df = self.dataframe
        if df.empty:
            return pd.DataFrame()

        summary = (
            df.groupby(["config_name", "strategy_name"])
            .agg(
                n_agents=("agent_id", "nunique"),
                n_decisions=("policy", "count"),
                n_switched=("switched", "sum"),
                switch_rate=("switched", "mean"),
                n_fallbacks=("is_fallback", "sum"),
                mean_confidence=("confidence", "mean"),
                mean_cost_delta=("cost_delta", "mean"),
                std_cost_delta=("cost_delta", "std"),
            )
            .round(4)
            .reset_index()
        )

        # Add parsed agent/parcel counts for sorting
        parsed = summary["config_name"].map(self._parse_config_name)
        summary.insert(1, "n_agents_config", [p[0] for p in parsed])
        summary.insert(2, "n_parcels_config", [p[1] for p in parsed])

        return summary.sort_values(
            ["n_agents_config", "n_parcels_config"]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fleet_file(self, fleet_path: Path):
        """
        Extract slm_decision events from one fleet.json and append to
        self._raw_events with provenance inferred from the path.

        Assumed path structure:
            sweep_dir / config_name / run_id / strategy_name / fleet.json
        """
        parts = fleet_path.relative_to(self.sweep_dir).parts
        config_name   = parts[0] if len(parts) >= 4 else "unknown"
        run_id        = parts[1] if len(parts) >= 4 else "unknown"
        strategy_name = parts[2] if len(parts) >= 4 else "unknown"

        with open(fleet_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        slm_events = [
            e for e in data.get("fleet_events", [])
            if e.get("type") == "slm_decision"
        ]

        for event in slm_events:
            self._raw_events.append({
                "config_name":   config_name,
                "run_id":        run_id,
                "strategy_name": strategy_name,
                "_event":        event,
            })

    def _build_dataframe(self) -> pd.DataFrame:
        if not self._raw_events:
            return pd.DataFrame()

        rows = []
        for record in self._raw_events:
            e              = record["_event"]
            config_name    = record["config_name"]

            n_agents, n_parcels = self._parse_config_name(config_name)
            context        = e.get("context", {})
            costs          = e.get("costs", {})
            time_info      = e.get("time", {})

            cost_greedy    = costs.get("greedy",    np.nan)
            cost_hungarian = costs.get("hungarian", np.nan)
            policy         = e.get("policy", "unknown")

            rows.append({
                # Provenance
                "config_name":          config_name,
                "n_agents":             n_agents,
                "n_parcels":            n_parcels,
                "run_id":               record["run_id"],
                "strategy_name":        record["strategy_name"],
                # Decision identity
                "agent_id":             e.get("agent_id", "unknown"),
                "step":                 time_info.get("step",       np.nan),
                "sim_time":             time_info.get("sim_time",   np.nan),
                "world_time":           time_info.get("world_time", np.nan),
                # Outcome
                "policy":               policy,
                "confidence":           e.get("confidence", np.nan),
                # Context signals
                "belief_infeasibility": context.get("belief_infeasibility", np.nan),
                "energy_error":         context.get("energy_error",         np.nan),
                "competition_ratio":    context.get("competition_ratio",    np.nan),
                "belief_trend":         context.get("belief_trend",         "UNKNOWN"),
                "error_trend":          context.get("error_trend",          "UNKNOWN"),
                "competition_trend":    context.get("competition_trend",    "UNKNOWN"),
                # Costs
                "cost_greedy":          cost_greedy,
                "cost_hungarian":       cost_hungarian,
                "cost_chosen":          costs.get("chosen", np.nan),
                "cost_delta":           cost_hungarian - cost_greedy,
                # Flags
                "switched":             policy not in ("greedy", "greedy_fallback"),
                "is_fallback":          policy == "greedy_fallback",
            })

        df = pd.DataFrame(rows)

        numeric_cols = [
            "step", "sim_time", "world_time", "confidence",
            "belief_infeasibility", "energy_error", "competition_ratio",
            "cost_greedy", "cost_hungarian", "cost_chosen", "cost_delta",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values(
            ["config_name", "strategy_name", "agent_id", "step"]
        ).reset_index(drop=True)

        print(
            f"📊 Built SLM DataFrame: {len(df)} decisions | "
            f"{df['config_name'].nunique()} config(s) | "
            f"{df['strategy_name'].nunique()} strategy(ies) | "
            f"{df['agent_id'].nunique()} unique agent id(s)"
        )
        return df

    def _parse_config_name(self, config_name: str) -> Tuple:
        match = re.match(r"A(\d+)_P(\d+)", config_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return np.nan, np.nan

    def _parse_config_key(self, config_name: str) -> Tuple:
        return self._parse_config_name(config_name)


# ------------------------------------------------------------------
# Convenience functions for notebook use (mirrors sweep_analyzer.py)
# ------------------------------------------------------------------

def load_slm_decisions(sweep_dir: Union[str, Path]) -> SLMDecisionAnalyzer:
    """Create an SLMDecisionAnalyzer and load all fleet files in one call."""
    return SLMDecisionAnalyzer(sweep_dir).load_all()


def quick_slm_summary(sweep_dir: Union[str, Path]) -> pd.DataFrame:
    """One-liner config summary of SLM decisions across a sweep."""
    return load_slm_decisions(sweep_dir).get_config_summary()