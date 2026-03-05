"""
core: core data structure module

- scene_graph: unified scene graph interface
- task: domain language definition (Action, TaskSequence)
- agent: LLM Agent wrapper
- physics_agent: physics-aware validation agent
- change_detector: scene change detection
"""

from .task import Action, ActionType, TaskSequence, TaskStatus
from .scene_graph import SceneGraph, ObjectNode, RoomNode, PhysicalProperties, BoundingBox
from .pipeline import PhyPlanPipeline
from .agent import LLMAgent
from .physics_agent import (
    PhysicsAwareAgent,
    RobotCapability,
    ValidationResult,
    ConstraintViolation,
    ConstraintType
)
from .change_detector import (
    ChangeDetector,
    ChangeReport,
    ObjectChange,
    ChangeType
)
from .category_filter import EXCLUDED_CATEGORIES, should_include_object

__all__ = [
    "Action", "ActionType", "TaskSequence", "TaskStatus",
    "SceneGraph", "ObjectNode", "RoomNode", "PhysicalProperties", "BoundingBox",
    "PhyPlanPipeline",
    "LLMAgent",
    "PhysicsAwareAgent", "RobotCapability", "ValidationResult",
    "ConstraintViolation", "ConstraintType",
    "ChangeDetector", "ChangeReport", "ObjectChange", "ChangeType",
    "EXCLUDED_CATEGORIES", "should_include_object",
]
