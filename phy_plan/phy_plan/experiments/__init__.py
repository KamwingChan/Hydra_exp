"""
IROS Experiments Module

This module provides tools for running IROS experiments:
- Scene configurations for OmniGibson
- Baseline implementations (Geo-Only, Open-Loop, etc.)
- Metrics collection and evaluation
- Experiment runner scripts

Usage:
    from phy_plan.experiments import (
        MetricsCollector,
        BaselineType,
        create_baseline_planner,
        load_scene_config,
    )
    
    # Create metrics collector
    collector = MetricsCollector("experiment_name", "full")
    
    # Create baseline planner
    planner = create_baseline_planner(BaselineType.GEO_ONLY)
    
    # Load scene configuration
    config = load_scene_config("kitchen")
"""

from .baselines import (
    BaselineType,
    BaselineConfig,
    BASELINE_CONFIGS,
    GeoOnlyPlanner,
    OpenLoopPlanner,
    NoObservePlanner,
    LLMDirectPlanner,
    OpenLoopPipeline,
    create_baseline_planner,
    get_baseline_config,
    list_baselines,
    strip_physics_from_scene_graph,
)

from .metrics import (
    TaskMetrics,
    ExperimentMetrics,
    MetricsCollector,
    compare_baselines,
    export_comparison_csv,
)

__all__ = [
    # Baselines
    "BaselineType",
    "BaselineConfig",
    "BASELINE_CONFIGS",
    "GeoOnlyPlanner",
    "OpenLoopPlanner",
    "NoObservePlanner",
    "LLMDirectPlanner",
    "OpenLoopPipeline",
    "create_baseline_planner",
    "get_baseline_config",
    "list_baselines",
    "strip_physics_from_scene_graph",
    # Metrics
    "TaskMetrics",
    "ExperimentMetrics",
    "MetricsCollector",
    "compare_baselines",
    "export_comparison_csv",
]
