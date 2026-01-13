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

from .loaders import (
    ExperimentLoader,
    BatchLoader,
)

from .visualizers import (
    TrajectoryVisualizer,
    HeatmapVisualizer,
    CollisionVisualizer,
    MetricsVisualizer,
)

from .aggregators import (
    ExperimentAggregator,
    StatisticalAnalyzer,
)

__version__ = "1.0.0"
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
    # Visualizers
    "TrajectoryVisualizer",
    "HeatmapVisualizer",
    "CollisionVisualizer",
    "MetricsVisualizer",
    # Aggregators
    "ExperimentAggregator",
    "StatisticalAnalyzer",
]