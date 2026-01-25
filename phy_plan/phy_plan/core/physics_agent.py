"""
physics_agent.py: Physics-aware validation for task planning

Validates task plans against physical constraints of objects and robot capabilities.
Key constraints:
- Weight: Can robot lift the object?
- Pushability: Can robot push the object?
- Inference confidence: Is the physical property reliable?
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .task import Action, ActionType, TaskSequence
from .scene_graph import SceneGraph, ObjectNode, PhysicalProperties


class ConstraintType(Enum):
    """Type of physical constraint violation"""
    WEIGHT_EXCEEDED = "weight_exceeded"
    NOT_PUSHABLE = "not_pushable"
    LOW_CONFIDENCE = "low_confidence"
    OBJECT_NOT_FOUND = "object_not_found"
    MISSING_PHYSICS = "missing_physics"


@dataclass
class RobotCapability:
    """Robot physical capabilities"""
    max_weight_level: int = 1          # Maximum weight level robot can handle (0-2)
    can_push_heavy: bool = False       # Can push heavy objects (weight_level=2)
    gripper_max_width: float = 0.15    # Maximum gripper width in meters
    min_confidence_threshold: int = 50  # Minimum inference confidence to trust


@dataclass
class ConstraintViolation:
    """Details of a single constraint violation"""
    action_index: int
    action: Action
    constraint_type: ConstraintType
    object_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"[Action {self.action_index}] {self.message}"


@dataclass
class ValidationResult:
    """Result of plan validation"""
    is_valid: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def reason(self) -> str:
        """Get primary failure reason"""
        if self.is_valid:
            return ""
        if self.violations:
            return str(self.violations[0])
        return "Unknown validation failure"
    
    @property
    def all_reasons(self) -> List[str]:
        """Get all violation reasons"""
        return [str(v) for v in self.violations]
    
    def to_feedback_prompt(self) -> str:
        """Generate feedback message for LLM replanning"""
        if self.is_valid:
            return ""
        
        lines = ["物理约束验证失败:"]
        for v in self.violations:
            lines.append(f"- {v.message}")
        
        if self.warnings:
            lines.append("\n警告:")
            for w in self.warnings:
                lines.append(f"- {w}")
        
        lines.append("\n请考虑以上约束重新规划。")
        return "\n".join(lines)


class PhysicsAwareAgent:
    """
    Physics-aware validation agent
    
    Validates task plans against physical constraints based on:
    1. Robot capabilities (weight limit, pushability)
    2. Object physical properties (from phy_graph VLM inference)
    3. Inference confidence thresholds
    """
    
    def __init__(self, robot_capability: Optional[RobotCapability] = None):
        """
        Initialize physics-aware agent
        
        Args:
            robot_capability: Robot physical capability specification.
                            Uses default if not provided.
        """
        self.capability = robot_capability or RobotCapability()
    
    def validate_plan(
        self,
        task_seq: TaskSequence,
        scene_graph: SceneGraph
    ) -> ValidationResult:
        """
        Validate entire task plan against physical constraints
        
        Args:
            task_seq: Task sequence to validate
            scene_graph: Scene graph with object physical properties
            
        Returns:
            ValidationResult with is_valid flag and any violations
        """
        violations: List[ConstraintViolation] = []
        warnings: List[str] = []
        
        for i, action in enumerate(task_seq.actions):
            # Skip actions that don't involve physical manipulation
            if action.action_type in [ActionType.NAVIGATE, ActionType.OBSERVE, ActionType.LOCATE]:
                continue
            
            # Get target object
            obj_id = action.target_object
            if not obj_id:
                # Some actions like ARRANGE may not have direct target
                if action.action_type == ActionType.ARRANGE:
                    # Validate arrange action separately
                    arrange_result = self._validate_arrange_action(i, action, scene_graph)
                    violations.extend(arrange_result.violations)
                    warnings.extend(arrange_result.warnings)
                continue
            
            # Check action against object
            is_valid, msg, violation = self.check_action(i, action, obj_id, scene_graph)
            
            if not is_valid and violation:
                violations.append(violation)
            elif msg:  # Warning case
                warnings.append(msg)
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )
    
    def check_action(
        self,
        action_index: int,
        action: Action,
        obj_id: str,
        scene_graph: SceneGraph
    ) -> Tuple[bool, str, Optional[ConstraintViolation]]:
        """
        Check single action against physical constraints
        
        Args:
            action_index: Index of action in sequence
            action: Action to validate
            obj_id: Target object ID
            scene_graph: Scene graph for object lookup
            
        Returns:
            Tuple of (is_valid, message, violation)
            - is_valid: True if action is physically feasible
            - message: Warning message if any
            - violation: ConstraintViolation if invalid, None otherwise
        """
        # Get object from scene graph
        obj = scene_graph.get_object(obj_id)
        if not obj:
            return False, "", ConstraintViolation(
                action_index=action_index,
                action=action,
                constraint_type=ConstraintType.OBJECT_NOT_FOUND,
                object_id=obj_id,
                message=f"物体 {obj_id} 在场景图中未找到"
            )
        
        # Get physical properties
        phys = obj.physical_properties
        if not phys:
            # No physical properties - generate warning but allow
            return True, f"物体 {obj_id} ({obj.category}) 缺少物理属性，无法验证可行性", None
        
        # Check based on action type
        if action.action_type == ActionType.PICK:
            return self._check_pick_action(action_index, action, obj, phys)
        
        elif action.action_type == ActionType.PLACE:
            # PLACE typically follows PICK, check target surface if specified
            return self._check_place_action(action_index, action, obj, phys, scene_graph)
        
        elif action.action_type == ActionType.MOVE_OBJECT:
            # MOVE_OBJECT combines PICK and PLACE
            return self._check_pick_action(action_index, action, obj, phys)
        
        elif action.action_type in [ActionType.CLEAN_UP]:
            # High-level actions - check if objects involved are manipulable
            return self._check_pick_action(action_index, action, obj, phys)
        
        return True, "", None
    
    def _check_pick_action(
        self,
        action_index: int,
        action: Action,
        obj: ObjectNode,
        phys: PhysicalProperties
    ) -> Tuple[bool, str, Optional[ConstraintViolation]]:
        """Check PICK action constraints"""
        obj_id = obj.node_id
        
        # Check inference confidence first
        if phys.inference_confidence >= 0 and phys.inference_confidence < self.capability.min_confidence_threshold:
            return True, f"物体 {obj_id} ({obj.category}) 物理属性推断置信度较低 ({phys.inference_confidence}%)，建议先观察确认", None
        
        # Check weight constraint
        if phys.weight_level > self.capability.max_weight_level:
            weight_desc = ["轻", "中等", "重"][phys.weight_level] if phys.weight_level <= 2 else "未知"
            estimated = f" (估计 {phys.estimated_weight_kg} kg)" if phys.estimated_weight_kg else ""
            
            return False, "", ConstraintViolation(
                action_index=action_index,
                action=action,
                constraint_type=ConstraintType.WEIGHT_EXCEEDED,
                object_id=obj_id,
                message=f"物体 {obj_id} ({obj.category}) 太重 (weight_level={phys.weight_level}, {weight_desc}{estimated})，机器人无法搬运 (最大支持 weight_level={self.capability.max_weight_level})",
                details={
                    "object_weight_level": phys.weight_level,
                    "robot_max_weight_level": self.capability.max_weight_level,
                    "estimated_weight_kg": phys.estimated_weight_kg
                }
            )
        
        return True, "", None
    
    def _check_place_action(
        self,
        action_index: int,
        action: Action,
        obj: ObjectNode,
        phys: PhysicalProperties,
        scene_graph: SceneGraph
    ) -> Tuple[bool, str, Optional[ConstraintViolation]]:
        """Check PLACE action constraints"""
        # For now, PLACE validation focuses on the object being placed
        # Future: Could check target surface load capacity
        
        # Check if target surface is specified
        target_surface_id = action.params.get("target_surface")
        if target_surface_id:
            surface = scene_graph.get_object(target_surface_id)
            if surface and surface.physical_properties:
                # Future enhancement: check surface load capacity
                pass
        
        return True, "", None
    
    def _check_push_action(
        self,
        action_index: int,
        action: Action,
        obj: ObjectNode,
        phys: PhysicalProperties
    ) -> Tuple[bool, str, Optional[ConstraintViolation]]:
        """Check PUSH action constraints (for future PUSH action type)"""
        obj_id = obj.node_id
        
        # Check pushability
        if not phys.pushable:
            return False, "", ConstraintViolation(
                action_index=action_index,
                action=action,
                constraint_type=ConstraintType.NOT_PUSHABLE,
                object_id=obj_id,
                message=f"物体 {obj_id} ({obj.category}) 无法推动 (pushable=false，可能固定或太重)",
                details={
                    "pushable": phys.pushable,
                    "weight_level": phys.weight_level
                }
            )
        
        # Check if robot can push heavy objects
        if phys.weight_level >= 2 and not self.capability.can_push_heavy:
            return False, "", ConstraintViolation(
                action_index=action_index,
                action=action,
                constraint_type=ConstraintType.WEIGHT_EXCEEDED,
                object_id=obj_id,
                message=f"物体 {obj_id} ({obj.category}) 太重 (weight_level={phys.weight_level})，机器人无法推动",
                details={
                    "weight_level": phys.weight_level,
                    "can_push_heavy": self.capability.can_push_heavy
                }
            )
        
        return True, "", None
    
    def _validate_arrange_action(
        self,
        action_index: int,
        action: Action,
        scene_graph: SceneGraph
    ) -> ValidationResult:
        """Validate ARRANGE action by checking all objects involved"""
        violations = []
        warnings = []
        
        # Get objects from params
        object_ids = action.params.get("object_ids", [])
        
        for obj_id in object_ids:
            obj = scene_graph.get_object(obj_id)
            if not obj:
                violations.append(ConstraintViolation(
                    action_index=action_index,
                    action=action,
                    constraint_type=ConstraintType.OBJECT_NOT_FOUND,
                    object_id=obj_id,
                    message=f"ARRANGE 动作中的物体 {obj_id} 在场景图中未找到"
                ))
                continue
            
            if not obj.physical_properties:
                warnings.append(f"ARRANGE 动作中的物体 {obj_id} ({obj.category}) 缺少物理属性")
                continue
            
            # Check each object can be picked
            is_valid, msg, violation = self._check_pick_action(
                action_index, action, obj, obj.physical_properties
            )
            
            if not is_valid and violation:
                violations.append(violation)
            elif msg:
                warnings.append(msg)
        
        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )
    
    def suggest_alternative(
        self,
        violation: ConstraintViolation,
        scene_graph: SceneGraph
    ) -> Optional[str]:
        """
        Suggest alternative objects when a constraint is violated
        
        Args:
            violation: The constraint violation
            scene_graph: Scene graph to find alternatives
            
        Returns:
            Suggestion string or None
        """
        if violation.constraint_type == ConstraintType.WEIGHT_EXCEEDED:
            # Find lighter objects of same category
            obj = scene_graph.get_object(violation.object_id)
            if not obj:
                return None
            
            alternatives = []
            for candidate in scene_graph.get_objects_by_category(obj.category):
                if candidate.node_id == obj.node_id:
                    continue
                if candidate.physical_properties:
                    if candidate.physical_properties.weight_level <= self.capability.max_weight_level:
                        alternatives.append(
                            f"{candidate.node_id} (weight_level={candidate.physical_properties.weight_level})"
                        )
            
            if alternatives:
                return f"建议替代物体: {', '.join(alternatives[:3])}"
        
        elif violation.constraint_type == ConstraintType.NOT_PUSHABLE:
            obj = scene_graph.get_object(violation.object_id)
            if not obj:
                return None
            
            alternatives = []
            for candidate in scene_graph.get_objects_by_category(obj.category):
                if candidate.node_id == obj.node_id:
                    continue
                if candidate.physical_properties and candidate.physical_properties.pushable:
                    alternatives.append(candidate.node_id)
            
            if alternatives:
                return f"建议可推动的替代物体: {', '.join(alternatives[:3])}"
        
        return None
    
    def get_capability_description(self) -> str:
        """Get human-readable description of robot capabilities"""
        weight_desc = ["轻量", "中等重量", "重物"][min(self.capability.max_weight_level, 2)]
        push_desc = "可以推动重物" if self.capability.can_push_heavy else "只能推动轻量物体"
        
        return (
            f"机器人能力: 最大可操作{weight_desc}物体 (weight_level<={self.capability.max_weight_level}), "
            f"{push_desc}, 夹爪最大宽度 {self.capability.gripper_max_width*100:.0f}cm"
        )
