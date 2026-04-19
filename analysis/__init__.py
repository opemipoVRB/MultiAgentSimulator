# analysis/__init__.py
"""
SPPDEC Experiment Analysis Framework

Object-oriented tools for loading, analyzing, and visualizing
multi-agent coordination experiments.
"""

from .models import (
    Experiment,
    StrategyRun,
    StrategyMetrics,
    FleetData,
    AgentTrajectory,
    ParcelLifecycle,
    FleetEvent,
    Heatmap,
    Station,
    Parcel,
    Agent,
)

from .batch_loaders import (
    ExperimentLoader,
    BatchLoader,
)

from .single_loader import SingleExperimentLoader

from .visualizers import (
    TrajectoryVisualizer,
    HeatmapVisualizer,
    CollisionVisualizer,
    MetricsVisualizer,
    ScalingVisualizer,
)

from .aggregators import (
    ExperimentAggregator,
    StatisticalAnalyzer,
)

from .sweep_analyzer import (
    SweepAnalyzer,
    load_sweep,
    quick_summary,
)

from .notebook_utils import (
    NotebookAnalyzer,
    analyze_sweep,
    quick_sppdec_comparison,
)

from .hybrid_metrics import (
    HybridRecord,
    HybridLogLoader,
    HybridMetricsAnalyzer,
)


__version__ = "1.2.0"
__all__ = [
    # Models
    "Experiment",
    "StrategyRun",
    "StrategyMetrics",
    "FleetData",
    "AgentTrajectory",
    "ParcelLifecycle",
    "FleetEvent",
    "Heatmap",
    "Station",
    "Parcel",
    "Agent",
    # Loaders
    "ExperimentLoader",
    "BatchLoader",
    "SingleExperimentLoader",
    # Visualizers
    "TrajectoryVisualizer",
    "HeatmapVisualizer",
    "CollisionVisualizer",
    "MetricsVisualizer",
    "ScalingVisualizer",
    # Aggregators
    "ExperimentAggregator",
    "StatisticalAnalyzer",
    # Sweep Analysis
    "SweepAnalyzer",
    "load_sweep",
    "quick_summary",
    # Notebook Utils
    "NotebookAnalyzer",
    "analyze_sweep",
    "quick_sppdec_comparison",
    # Phase 10: Hybrid Metrics (NEW)
    "HybridRecord",
    "HybridLogLoader",
    "HybridMetricsAnalyzer",
]