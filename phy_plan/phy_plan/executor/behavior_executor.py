"""
BehaviorExecutor: Execute tasks in Omnigibson/BEHAVIOR simulation.

Provides execution feedback for dynamic replanning:
- Detailed success/failure information
- Object state after action
- Error categorization for intelligent replanning
"""
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Callable, TYPE_CHECKING

import numpy as np

from ..core.task import TaskSequence, Action, ActionType, TaskStatus, Position

if TYPE_CHECKING:
    from ..core.scene_graph import SceneGraph

logger = logging.getLogger(__name__)

# Try to import BehaviorActionAPI
try:
    from .behavior_action_api import BehaviorActionAPI, StarterSemanticActionPrimitiveSet
    BEHAVIOR_API_AVAILABLE = True
except ImportError:
    BEHAVIOR_API_AVAILABLE = False
    BehaviorActionAPI = None
    StarterSemanticActionPrimitiveSet = None


class ExecutionErrorType(Enum):
    """Categorization of execution errors for intelligent replanning"""
    SUCCESS = "success"
    OBJECT_NOT_FOUND = "object_not_found"
    OBJECT_UNREACHABLE = "object_unreachable"
    GRASP_FAILED = "grasp_failed"
    PLACE_FAILED = "place_failed"
    NAVIGATION_FAILED = "navigation_failed"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    PHYSICS_ERROR = "physics_error"
    UNKNOWN = "unknown"


@dataclass
class ActionFeedback:
    """
    Detailed feedback from action execution
    
    Used by DynamicPlannerPipeline for intelligent replanning decisions.
    """
    success: bool
    error_type: ExecutionErrorType = ExecutionErrorType.SUCCESS
    message: str = ""
    action: Optional[Action] = None
    
    # Execution details
    execution_time: float = 0.0
    retry_count: int = 0
    
    # Object state after action (if applicable)
    object_id: Optional[str] = None
    object_position: Optional[List[float]] = None
    object_in_hand: bool = False
    
    # Additional context for replanning
    suggested_retry: bool = False
    alternative_targets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "error_type": self.error_type.value,
            "message": self.message,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "object_id": self.object_id,
            "object_position": self.object_position,
            "object_in_hand": self.object_in_hand,
            "suggested_retry": self.suggested_retry,
            "alternative_targets": self.alternative_targets
        }
    
    def to_replan_context(self) -> str:
        """Generate context string for LLM replanning"""
        lines = [f"Action result: {'SUCCESS' if self.success else 'FAILED'}"]
        
        if not self.success:
            lines.append(f"Error type: {self.error_type.value}")
            lines.append(f"Error message: {self.message}")
            
            if self.suggested_retry:
                lines.append("Suggestion: Retry may succeed")
            
            if self.alternative_targets:
                lines.append(f"Alternative targets: {', '.join(self.alternative_targets)}")
        
        if self.object_position:
            lines.append(f"Object position: {self.object_position}")
        
        if self.object_in_hand:
            lines.append("Robot is currently holding an object")
        
        return "\n".join(lines)

class BehaviorExecutor:
    """
    Executor that interfaces with Omnigibson/BEHAVIOR to execute plan actions.
    
    Provides detailed feedback for dynamic replanning:
    - Error categorization (object_not_found, grasp_failed, etc.)
    - Object state tracking
    - Retry suggestions
    
    Args:
        env: omnigibson.Environment instance
        use_real_api: If True, use real BEHAVIOR API. If False, use mock execution.
        max_retries: Maximum retry attempts per action
        on_action_feedback: Callback for action feedback
    """
    def __init__(
        self, 
        env, 
        use_real_api: bool = True,
        max_retries: int = 2,
        on_action_feedback: Optional[Callable[[ActionFeedback], None]] = None,
        scene_graph: Optional["SceneGraph"] = None
    ):
        """
        Initialize executor with Omnigibson environment.
        
        Args:
            env: OmniGibson environment instance
            use_real_api: If True, use real BEHAVIOR API. If False, use mock execution.
            max_retries: Maximum retry attempts per action
            on_action_feedback: Callback for action feedback
            scene_graph: phy_plan SceneGraph for object position/category lookup
        """
        self.env = env
        # Assuming single robot setup common in BEHAVIOR tasks
        self.robot = env.robots[0] if hasattr(env, 'robots') and env.robots else None
        
        # Configuration
        self.max_retries = max_retries
        self.on_action_feedback = on_action_feedback
        
        # Scene graph for object lookup (category + position matching)
        self.scene_graph: Optional["SceneGraph"] = scene_graph
        
        # State tracking
        self._object_in_hand: Optional[str] = None
        self._last_feedback: Optional[ActionFeedback] = None
        self._execution_history: List[ActionFeedback] = []
        
        # Object match cache: node_id -> OmniGibson object
        # Used to avoid repeated position matching for the same object
        self._object_match_cache: Dict[str, Any] = {}
        
        # Initialize BEHAVIOR API if available and requested
        self.api = None
        self.use_real_api = use_real_api and BEHAVIOR_API_AVAILABLE
        
        if self.use_real_api:
            try:
                self.api = BehaviorActionAPI(env, self.robot)
                print("[BehaviorExecutor] Using real BEHAVIOR action primitives")
            except Exception as e:
                print(f"[BehaviorExecutor] Warning: Failed to initialize BEHAVIOR API: {e}")
                print("[BehaviorExecutor] Falling back to mock execution")
                self.use_real_api = False
        else:
            print("[BehaviorExecutor] Using mock execution (real API not available or disabled)")
    
    @classmethod
    def from_task_json(cls, env, json_path: str, use_real_api: bool = True) -> 'BehaviorExecutor':
        """
        Create executor and load task from JSON file.
        
        Args:
            env: omnigibson.Environment instance
            json_path: Path to TaskSequence JSON file
            use_real_api: If True, use real BEHAVIOR API
            
        Returns:
            BehaviorExecutor instance with loaded task
        """
        executor = cls(env, use_real_api=use_real_api)
        
        # Load task
        with open(json_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        executor.current_task = TaskSequence.from_dict(task_data)
        
        print(f"[BehaviorExecutor] Loaded task from: {json_path}")
        print(f"[BehaviorExecutor] Task: {executor.current_task.task_name}")
        print(f"[BehaviorExecutor] Actions: {len(executor.current_task.actions)}")
        
        return executor
        
    def execute_task(self, task: TaskSequence):
        """
        Execute a full task sequence with progress tracking.
        
        Args:
            task: TaskSequence to execute
        """
        task.status = TaskStatus.RUNNING
        print(f"\n{'='*60}")
        print(f"[BehaviorExecutor] Starting task: {task.task_name}")
        print(f"[BehaviorExecutor] Total actions: {len(task.actions)}")
        print(f"{'='*60}\n")
        
        for i, action in enumerate(task.actions):
            # Skip already completed actions (allows resuming)
            if action.status == TaskStatus.COMPLETED:
                print(f"[{i+1}/{len(task.actions)}] Skipping completed action: {action.description}")
                continue
            
            # Update task current index
            task.current_action_index = i
            
            # Execute single action
            print(f"\n[{i+1}/{len(task.actions)}] Executing: {action.description}")
            print(f"  Progress: {task.progress*100:.1f}%")
            
            self.execute_action(action)
            
            duration = action.end_time - action.start_time if action.end_time > 0 else 0
            
            if action.status == TaskStatus.FAILED:
                task.mark_failed(f"Action {i+1} failed: {action.result_message}")
                print(f"  ❌ FAILED after {duration:.2f}s: {action.result_message}")
                print(f"\n[BehaviorExecutor] Task failed: {task.error_message}")
                return
            else:
                task.mark_current_action_complete(f"Completed in {duration:.2f}s")
                print(f"  ✅ SUCCESS ({duration:.2f}s)")
        
        task.status = TaskStatus.COMPLETED
        print(f"\n{'='*60}")
        print(f"[BehaviorExecutor] Task completed successfully!")
        print(f"[BehaviorExecutor] Final progress: {task.progress*100:.0f}%")
        print(f"{'='*60}\n")

    def execute_action(self, action: Action) -> ActionFeedback:
        """
        Execute a single action with detailed feedback.
        
        Args:
            action: Action to execute
            
        Returns:
            ActionFeedback with execution details
        """
        action.status = TaskStatus.RUNNING
        action.start_time = time.time()
        print(f"[BehaviorExecutor] Executing action: {action.action_type.value} {action.description}")
        
        feedback = ActionFeedback(
            success=False,
            action=action,
            object_id=action.target_object
        )
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                success = False
                message = ""
                error_type = ExecutionErrorType.UNKNOWN
                
                if action.action_type == ActionType.NAVIGATE:
                    success, message, error_type = self._navigate_with_feedback(action)
                elif action.action_type == ActionType.PICK:
                    success, message, error_type = self._pick_with_feedback(action)
                elif action.action_type == ActionType.PLACE:
                    success, message, error_type = self._place_with_feedback(action)
                elif action.action_type == ActionType.PLACE_INSIDE:
                    success, message, error_type = self._place_inside_with_feedback(action)
                elif action.action_type == ActionType.MOVE_OBJECT:
                    success, message, error_type = self._move_object_with_feedback(action)
                elif action.action_type == ActionType.OPEN:
                    success, message, error_type = self._open_with_feedback(action)
                elif action.action_type == ActionType.CLOSE:
                    success, message, error_type = self._close_with_feedback(action)
                elif action.action_type == ActionType.OBSERVE:
                    success, message, error_type = self._observe_with_feedback(action)
                elif action.action_type == ActionType.ARRANGE:
                    # ARRANGE should be expanded into sub-actions before reaching executor
                    success = False
                    message = (
                        f"ARRANGE action not expanded: {action.description}. "
                        "ARRANGE must be expanded into NAVIGATE/PICK/PLACE sub-actions "
                        "before execution. This indicates a pipeline bug."
                    )
                    error_type = ExecutionErrorType.UNKNOWN
                    logger.error(message)
                else:
                    # For other actions, just mark as success for now (mock execution)
                    success = True
                    message = f"Simulated execution of {action.action_type}"
                    error_type = ExecutionErrorType.SUCCESS
                
                feedback.success = success
                feedback.message = message
                feedback.error_type = error_type if not success else ExecutionErrorType.SUCCESS
                feedback.retry_count = retry_count
                
                if success:
                    action.status = TaskStatus.COMPLETED
                    action.result_message = message
                    break
                else:
                    # Check if retry might help
                    if error_type in [ExecutionErrorType.GRASP_FAILED, ExecutionErrorType.NAVIGATION_FAILED]:
                        feedback.suggested_retry = True
                        if retry_count < self.max_retries:
                            print(f"[BehaviorExecutor] Retrying ({retry_count + 1}/{self.max_retries})...")
                            retry_count += 1
                            time.sleep(0.5)  # Brief pause before retry
                            continue
                    
                    action.status = TaskStatus.FAILED
                    action.result_message = message
                    break
                    
            except Exception as e:
                action.status = TaskStatus.FAILED
                action.result_message = f"Exception: {str(e)}"
                feedback.success = False
                feedback.message = str(e)
                feedback.error_type = ExecutionErrorType.UNKNOWN
                print(f"[BehaviorExecutor] Error: {e}")
                break
        
        action.end_time = time.time()
        feedback.execution_time = action.end_time - action.start_time
        feedback.object_in_hand = self._object_in_hand is not None
        
        # Store feedback
        self._last_feedback = feedback
        self._execution_history.append(feedback)
        
        # Callback
        if self.on_action_feedback:
            self.on_action_feedback(feedback)
        
        return feedback
    
    def _navigate_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """Navigate with detailed error feedback."""
        target_pos = action.target_position
        target_id = action.params.get("room_id") or action.target_object
        
        if not target_id and not target_pos:
            return False, "No navigation target specified", ExecutionErrorType.NAVIGATION_FAILED
        
        if self.use_real_api and self.api:
            # Case 1: Navigate to position (room navigation or explicit position)
            if target_pos:
                try:
                    success, message, metadata = self.api.navigate_to_position(
                        target_pos.x,
                        target_pos.y,
                        0.0  # Default yaw
                    )
                    if success:
                        return True, message, ExecutionErrorType.SUCCESS
                    else:
                        return False, message, ExecutionErrorType.NAVIGATION_FAILED
                except Exception as e:
                    return False, str(e), ExecutionErrorType.NAVIGATION_FAILED
            
            # Case 2: Navigate to object (fallback)
            else:
                target_obj = self._find_object(target_id)
            if not target_obj:
                return False, f"Target {target_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
            
            try:
                success, message, metadata = self.api.navigate_to(target_obj)
                if success:
                    return True, message, ExecutionErrorType.SUCCESS
                else:
                    return False, message, ExecutionErrorType.NAVIGATION_FAILED
            except Exception as e:
                return False, str(e), ExecutionErrorType.NAVIGATION_FAILED
        
        # Mock execution
        pos = target_pos.to_list() if target_pos else "unknown"
        return True, f"[Mock] Navigated to {target_id} at {pos}", ExecutionErrorType.SUCCESS
    
    def _pick_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """Pick with detailed error feedback."""
        obj_id = action.target_object
        if not obj_id:
            return False, "No object specified for PICK", ExecutionErrorType.GRASP_FAILED
        
        if self.use_real_api and self.api:
            obj = self._find_object(obj_id)
            if not obj:
                return False, f"Object {obj_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
            
            try:
                success, message, metadata = self.api.grasp(obj)
                if success:
                    self._object_in_hand = obj_id
                    return True, message, ExecutionErrorType.SUCCESS
                else:
                    # Analyze failure reason
                    if "unreachable" in message.lower():
                        return False, message, ExecutionErrorType.OBJECT_UNREACHABLE
                    return False, message, ExecutionErrorType.GRASP_FAILED
            except Exception as e:
                return False, str(e), ExecutionErrorType.GRASP_FAILED
        
        # Mock execution
        self._object_in_hand = obj_id
        return True, f"[Mock] Picked {obj_id}", ExecutionErrorType.SUCCESS
    
    def _place_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """Place with detailed error feedback."""
        obj_id = action.target_object
        if not obj_id:
            return False, "No object specified for PLACE", ExecutionErrorType.PLACE_FAILED
        
        surface_id = action.params.get("surface_id")
        room_id = action.params.get("room_id")
        
        if self.use_real_api and self.api:
            if surface_id:
                surface_obj = self._find_object(surface_id)
                if not surface_obj:
                    return False, f"Surface {surface_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
                
                try:
                    success, message, metadata = self.api.place_on_top(surface_obj)
                    if success:
                        self._object_in_hand = None
                        return True, message, ExecutionErrorType.SUCCESS
                    else:
                        return False, message, ExecutionErrorType.PLACE_FAILED
                except Exception as e:
                    return False, str(e), ExecutionErrorType.PLACE_FAILED
            elif room_id:
                return False, "Room-based placement requires explicit surface_id", ExecutionErrorType.PLACE_FAILED
            else:
                return False, "No placement target (surface_id or room_id) specified", ExecutionErrorType.PLACE_FAILED
        
        # Mock execution
        target = surface_id or room_id or "unknown"
        self._object_in_hand = None
        return True, f"[Mock] Placed {obj_id} on/in {target}", ExecutionErrorType.SUCCESS
    
    def _place_inside_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """Place inside a container with detailed error feedback."""
        obj_id = action.target_object
        if not obj_id:
            return False, "No object specified for PLACE_INSIDE", ExecutionErrorType.PLACE_FAILED
        
        container_id = action.params.get("container_id")
        if not container_id:
            return False, "No container_id specified for PLACE_INSIDE", ExecutionErrorType.PLACE_FAILED
        
        if self.use_real_api and self.api:
            container_obj = self._find_object(container_id)
            if not container_obj:
                return False, f"Container {container_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
            
            try:
                success, message, metadata = self.api.place_inside(container_obj)
                if success:
                    self._object_in_hand = None
                    return True, message, ExecutionErrorType.SUCCESS
                else:
                    return False, message, ExecutionErrorType.PLACE_FAILED
            except Exception as e:
                return False, str(e), ExecutionErrorType.PLACE_FAILED
        
        # Mock execution
        self._object_in_hand = None
        return True, f"[Mock] Placed {obj_id} inside {container_id}", ExecutionErrorType.SUCCESS
    
    def _move_object_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """Move object with detailed error feedback."""
        obj_id = action.target_object
        if not obj_id:
            return False, "No object specified for MOVE_OBJECT", ExecutionErrorType.UNKNOWN
        
        if self.use_real_api and self.api:
            return False, "MOVE_OBJECT not yet implemented with real API (use PICK + PLACE)", ExecutionErrorType.UNKNOWN
        
        # Mock execution
        target_pos = action.target_position.to_list() if action.target_position else "unknown"
        return True, f"[Mock] Moved {obj_id} to {target_pos}", ExecutionErrorType.SUCCESS

    def _open_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """
        Open a container or door with detailed error feedback.
        
        After opening, the perception system should detect interior objects.
        """
        obj_id = action.params.get("object_id") or action.target_object
        if not obj_id:
            return False, "No object specified for OPEN action", ExecutionErrorType.UNKNOWN
        
        if self.use_real_api and self.api:
            obj = self._find_object(obj_id)
            if not obj:
                return False, f"Container/door {obj_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
            
            try:
                # BEHAVIOR uses 'open_object' method from API
                success, message, metadata = self.api.open_object(obj)
                if success:
                    return True, f"Opened {obj_id}, perception system will update scene graph", ExecutionErrorType.SUCCESS
                else:
                    return False, message, ExecutionErrorType.UNKNOWN
            except Exception as e:
                return False, f"Error opening {obj_id}: {str(e)}", ExecutionErrorType.UNKNOWN
        
        # Mock execution
        return True, f"[Mock] Opened {obj_id}, waiting for scene update", ExecutionErrorType.SUCCESS
    
    def _close_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """
        Close a container or door with detailed error feedback.
        """
        obj_id = action.params.get("object_id") or action.target_object
        if not obj_id:
            return False, "No object specified for CLOSE action", ExecutionErrorType.UNKNOWN
        
        if self.use_real_api and self.api:
            obj = self._find_object(obj_id)
            if not obj:
                return False, f"Container/door {obj_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
            
            try:
                # BEHAVIOR uses 'close_object' method from API
                success, message, metadata = self.api.close_object(obj)
                if success:
                    return True, f"Closed {obj_id}", ExecutionErrorType.SUCCESS
                else:
                    return False, message, ExecutionErrorType.UNKNOWN
            except Exception as e:
                return False, f"Error closing {obj_id}: {str(e)}", ExecutionErrorType.UNKNOWN
        
        # Mock execution
        return True, f"[Mock] Closed {obj_id}", ExecutionErrorType.SUCCESS

    # Observation wait time in seconds (allows phy_graph to update scene graph)
    OBSERVE_WAIT_TIME = 3.0
    
    def _observe_with_feedback(self, action: Action) -> Tuple[bool, str, ExecutionErrorType]:
        """
        Observe an object to confirm/update its physical properties.
        
        This action:
        1. Navigates the robot closer to the target object
        2. Waits for the perception system (phy_graph) to re-analyze the object
        3. Scene graph will be updated asynchronously via ROS subscriber
        
        Use when:
        - Object has low inference_confidence (< 50)
        - Physical properties are unknown or uncertain
        - Need to verify object state before manipulation
        
        Note: After this action completes, DynamicPlannerPipeline should check
        for scene graph updates to get the new physical properties.
        """
        obj_id = action.params.get("object_id") or action.target_object
        if not obj_id:
            return False, "No object specified for OBSERVE action", ExecutionErrorType.UNKNOWN
        
        if self.use_real_api and self.api:
            obj = self._find_object(obj_id)
            if not obj:
                return False, f"Object {obj_id} not found in scene", ExecutionErrorType.OBJECT_NOT_FOUND
            
            try:
                # 1. Navigate closer to the object for better observation
                logger.info(f"Approaching {obj_id} for observation...")
                nav_success, nav_msg, nav_meta = self.api.navigate_to(obj)
                if not nav_success:
                    return False, f"Failed to approach {obj_id} for observation: {nav_msg}", ExecutionErrorType.NAVIGATION_FAILED
                
                # 2. Wait for phy_graph to process and update scene graph
                # The VLM inference is triggered automatically when the robot
                # observes the object (keyframe capture + inference pipeline)
                logger.info(f"Waiting {self.OBSERVE_WAIT_TIME}s for scene graph update...")
                time.sleep(self.OBSERVE_WAIT_TIME)
                
                # 3. Clear object cache since properties may have changed
                if obj_id in self._object_match_cache:
                    del self._object_match_cache[obj_id]
                
                return True, f"Observed {obj_id}, scene graph should be updated", ExecutionErrorType.SUCCESS
                
            except Exception as e:
                return False, f"Error observing {obj_id}: {str(e)}", ExecutionErrorType.UNKNOWN
        
        # Mock execution - also wait to simulate real behavior
        logger.info(f"[Mock] Observing {obj_id}, waiting {self.OBSERVE_WAIT_TIME}s...")
        time.sleep(self.OBSERVE_WAIT_TIME)
        return True, f"[Mock] Observed {obj_id}, scene graph should be updated", ExecutionErrorType.SUCCESS

    def _find_object(self, obj_id: str):
        """
        Find OmniGibson object by node_id using category + position matching.
        
        Strategy:
        1. Check cache first
        2. Get target object's category and position from scene_graph
        3. Filter OmniGibson objects by category
        4. Select the one with closest position
        
        Args:
            obj_id: Object ID (e.g., "O(13)") or direct OmniGibson name
            
        Returns:
            OmniGibson object instance or None if not found
        """
        if not self.env or not hasattr(self.env, 'scene'):
            return None
        
        # 1. Check cache first
        if obj_id in self._object_match_cache:
            return self._object_match_cache[obj_id]
        
        # 2. Try direct name lookup first (for backward compatibility)
        try:
            obj = self.env.scene.object_registry("name", obj_id)
            if obj is not None:
                self._object_match_cache[obj_id] = obj
                return obj
        except Exception:
            pass
        
        # 3. Use category + position matching if scene_graph is available
        if self.scene_graph is None:
            logger.warning(f"Cannot find object {obj_id}: scene_graph not set")
            return None
        
        # Get target object info from scene_graph
        obj_node = self.scene_graph.get_object(obj_id)
        if obj_node is None:
            logger.warning(f"Object {obj_id} not found in scene_graph")
            return None
        
        target_category = obj_node.category
        target_pos = np.array(obj_node.position)
        
        # 4. Find candidates with same category in OmniGibson scene
        candidates = []
        try:
            for og_obj in self.env.scene.objects:
                # Check if category matches
                og_category = getattr(og_obj, 'category', None)
                if og_category and og_category.lower() == target_category.lower():
                    try:
                        og_pos = np.array(og_obj.get_position())
                        distance = np.linalg.norm(og_pos - target_pos)
                        candidates.append((og_obj, distance))
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Error iterating OmniGibson objects: {e}")
            return None
        
        if not candidates:
            logger.warning(f"No OmniGibson objects found with category '{target_category}'")
            return None
        
        # 5. Select closest match
        candidates.sort(key=lambda x: x[1])
        best_match, best_distance = candidates[0]
        
        # Distance threshold warning
        MAX_MATCH_DISTANCE = 1.0  # meters
        if best_distance > MAX_MATCH_DISTANCE:
            logger.warning(
                f"Best match for {obj_id} ({target_category}) is {best_distance:.2f}m away, "
                f"may be incorrect. OG name: {getattr(best_match, 'name', 'unknown')}"
            )
        else:
            logger.debug(
                f"Matched {obj_id} -> {getattr(best_match, 'name', 'unknown')} "
                f"(distance: {best_distance:.3f}m)"
            )
        
        # Cache the result
        self._object_match_cache[obj_id] = best_match
        return best_match
    
    def set_scene_graph(self, scene_graph: "SceneGraph") -> None:
        """
        Set or update the scene graph for object lookup.
        
        Also clears the object match cache since positions may have changed.
        
        Args:
            scene_graph: phy_plan SceneGraph instance
        """
        self.scene_graph = scene_graph
        self._object_match_cache.clear()
        logger.info(f"Scene graph updated, object cache cleared")
    
    def clear_object_cache(self) -> None:
        """Clear the object match cache."""
        self._object_match_cache.clear()
    
    def _navigate(self, action: Action) -> Tuple[bool, str]:
        """Navigate to a location or object (legacy interface)."""
        success, message, _ = self._navigate_with_feedback(action)
        return success, message

    def _pick(self, action: Action) -> Tuple[bool, str]:
        """Pick up an object (legacy interface)."""
        success, message, _ = self._pick_with_feedback(action)
        return success, message

    def _place(self, action: Action) -> Tuple[bool, str]:
        """Place currently held object (legacy interface)."""
        success, message, _ = self._place_with_feedback(action)
        return success, message
        
    def _move_object(self, action: Action) -> Tuple[bool, str]:
        """Move object (legacy interface)."""
        success, message, _ = self._move_object_with_feedback(action)
        return success, message
    
    # ==================== State Query Methods ====================
    
    def get_object_in_hand(self) -> Optional[str]:
        """Get the ID of object currently held by robot"""
        return self._object_in_hand
    
    def get_last_feedback(self) -> Optional[ActionFeedback]:
        """Get feedback from last executed action"""
        return self._last_feedback
    
    def get_execution_history(self) -> List[ActionFeedback]:
        """Get history of all action feedbacks"""
        return self._execution_history.copy()
    
    def clear_execution_history(self) -> None:
        """Clear execution history"""
        self._execution_history = []
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution history"""
        if not self._execution_history:
            return {"total_actions": 0}
        
        success_count = sum(1 for f in self._execution_history if f.success)
        error_types = {}
        for f in self._execution_history:
            if not f.success:
                et = f.error_type.value
                error_types[et] = error_types.get(et, 0) + 1
        
        total_time = sum(f.execution_time for f in self._execution_history)
        total_retries = sum(f.retry_count for f in self._execution_history)
        
        return {
            "total_actions": len(self._execution_history),
            "success_count": success_count,
            "failure_count": len(self._execution_history) - success_count,
            "success_rate": success_count / len(self._execution_history),
            "error_types": error_types,
            "total_execution_time": total_time,
            "total_retries": total_retries
        }
    
    # ==================== DynamicPipeline Integration ====================
    
    def execute_action_for_pipeline(self, action: Action) -> Dict[str, Any]:
        """
        Execute action and return result in format expected by DynamicPlannerPipeline.
        
        Returns:
            Dictionary with 'success', 'error', and additional details
        """
        feedback = self.execute_action(action)
        
        return {
            "success": feedback.success,
            "error": feedback.message if not feedback.success else "",
            "error_type": feedback.error_type.value,
            "execution_time": feedback.execution_time,
            "retry_count": feedback.retry_count,
            "object_in_hand": feedback.object_in_hand,
            "suggested_retry": feedback.suggested_retry,
            "alternative_targets": feedback.alternative_targets,
            "replan_context": feedback.to_replan_context() if not feedback.success else ""
        }
