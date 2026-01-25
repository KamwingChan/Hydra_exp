"""
core: core data structure module

- scene_graph: unified scene graph interface
- task: domain language definition (Action, TaskSequence)
- agent: LLM Agent wrapper
- physics_agent: physics-aware validation agent
- change_detector: scene change detection
"""

from .task import Action, ActionType, TaskSequence
from .scene_graph import SceneGraph, ObjectNode, RoomNode, PhysicalProperties, BoundingBox
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

__all__ = [
    "Action", "ActionType", "TaskSequence",
    "SceneGraph", "ObjectNode", "RoomNode", "PhysicalProperties", "BoundingBox",
    "LLMAgent",
    "PhysicsAwareAgent", "RobotCapability", "ValidationResult",
    "ConstraintViolation", "ConstraintType",
    "ChangeDetector", "ChangeReport", "ObjectChange", "ChangeType"
]
