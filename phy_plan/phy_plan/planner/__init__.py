"""
planner: Task Planning Module

Architecture:
    LLMPlanner: Core planning logic (LLM calls, response parsing, physics validation)
        - Public API: plan(), parse_response(), convert_to_task_sequence(), enrich_candidates()
        - Conversation: init_conversation(), chat(), reset_conversation()
        
    LLMPlannerPipeline: Interactive planning pipeline
        - Scene graph management
        - User interaction (clarification, confirmation)
        - Multi-turn dialogue support
        
    DynamicPlannerPipeline: Execution with dynamic replanning
        - Execution monitoring
        - Scene change detection (via ChangeDetector)
        - Hybrid replanning triggers (failure + scene change)
        
    SpatialResolver: Automatic spatial reference resolution
        - Reduces unnecessary user clarification
        - Supports "nearest", "closest to", etc.

Key Features:
    1. Physics-aware planning (via PhysicsAgent)
    2. Dynamic scene monitoring (via ChangeDetector)
    3. Spatial reasoning (via SpatialResolver)
"""

from .llm_planner import LLMPlanner, ClarificationRequest, InfeasiblePlan
from .llm_planner_pipeline import LLMPlannerPipeline
from .spatial_resolver import SpatialResolver, SpatialReference, RankedCandidate
from .dynamic_planner import (
    DynamicPlannerPipeline,
    ReplanTrigger,
    ReplanEvent,
    ExecutionResult,
    PipelineResult
)

__all__ = [
    "LLMPlanner", 
    "LLMPlannerPipeline",
    "ClarificationRequest",
    "InfeasiblePlan",
    "SpatialResolver",
    "SpatialReference",
    "RankedCandidate",
    "DynamicPlannerPipeline",
    "ReplanTrigger",
    "ReplanEvent",
    "ExecutionResult",
    "PipelineResult"
]
