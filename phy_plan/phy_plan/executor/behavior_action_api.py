"""
BehaviorActionAPI: Wrapper for BEHAVIOR action primitives with detailed feedback.

This module provides a clean interface to omnigibson's action primitives,
handling error cases and providing structured feedback.

Supports two execution modes:
1. Full Mode (default): Uses StarterSemanticActionPrimitives with CuRobo motion planning
   - Requires high GPU memory (12GB+)
   - Provides realistic motion planning and execution
   
2. Symbolic Mode: Uses SymbolicSemanticActionPrimitives
   - Skips CuRobo initialization (saves GPU memory)
   - Objects teleport directly to target positions
   - Physics engine still validates results
   - Ideal for planning logic validation
"""
from typing import Tuple, Optional, Any
from enum import Enum


class ExecutionMode(Enum):
    """Execution mode for BehaviorActionAPI."""
    FULL = "full"           # Full CuRobo motion planning
    SYMBOLIC = "symbolic"   # Symbolic execution (teleport + physics validation)


# Try to import omnigibson - allow module to load even if omnigibson not available
try:
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
        StarterSemanticActionPrimitiveSet,
    )
    from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
        SymbolicSemanticActionPrimitives,
        SymbolicSemanticActionPrimitiveSet,
    )
    from omnigibson.action_primitives.action_primitive_set_base import (
        ActionPrimitiveError,
        ActionPrimitiveErrorGroup,
    )
    OMNIGIBSON_AVAILABLE = True
except ImportError:
    OMNIGIBSON_AVAILABLE = False
    # Provide stub for type hints
    StarterSemanticActionPrimitiveSet = Any
    SymbolicSemanticActionPrimitiveSet = Any


class BehaviorActionAPI:
    """
    Wrapper for BEHAVIOR action primitives with detailed feedback.
    
    Provides a clean interface to omnigibson's action primitives, converting
    exceptions to structured feedback tuples.
    
    Supports two execution modes:
    - FULL: Uses CuRobo for motion planning (GPU intensive, realistic)
    - SYMBOLIC: Uses teleport + physics validation (GPU efficient, for testing)
    
    Args:
        env: Omnigibson Environment instance
        robot: Robot instance (usually env.robots[0])
        mode: ExecutionMode.FULL or ExecutionMode.SYMBOLIC (default: FULL)
        enable_head_tracking: Whether to enable head tracking (default: True, only for FULL mode)
        curobo_batch_size: Batch size for curobo motion planning (default: 1, only for FULL mode)
        motion_cfg_kwargs: Optional dict of motion config kwargs (only for FULL mode)
    
    Example:
        # For high-GPU systems (full motion planning):
        api = BehaviorActionAPI(env, robot, mode=ExecutionMode.FULL)
        
        # For low-GPU systems (symbolic execution):
        api = BehaviorActionAPI(env, robot, mode=ExecutionMode.SYMBOLIC)
    """
    
    def __init__(
        self, 
        env, 
        robot, 
        mode: ExecutionMode = ExecutionMode.FULL,
        enable_head_tracking: bool = True, 
        curobo_batch_size: int = 1, 
        motion_cfg_kwargs: Optional[dict] = None
    ):
        if not OMNIGIBSON_AVAILABLE:
            raise ImportError(
                "omnigibson is not available. Please install omnigibson to use BehaviorActionAPI."
            )
        
        self.env = env
        self.robot = robot
        self.mode = mode
        
        if mode == ExecutionMode.SYMBOLIC:
            # Symbolic mode: Skip CuRobo, use teleport + physics validation
            print("[BehaviorActionAPI] Using SYMBOLIC mode (no CuRobo, teleport + physics)")
            self.controller = SymbolicSemanticActionPrimitives(
                env=env,
                robot=robot
            )
            self._primitive_set = SymbolicSemanticActionPrimitiveSet
        else:
            # Full mode: Use CuRobo for motion planning
            print("[BehaviorActionAPI] Using FULL mode (CuRobo motion planning)")
            self.controller = StarterSemanticActionPrimitives(
                env=env,
                robot=robot,
                enable_head_tracking=enable_head_tracking,
                curobo_batch_size=curobo_batch_size,
                motion_cfg_kwargs=motion_cfg_kwargs
            )
            self._primitive_set = StarterSemanticActionPrimitiveSet
    
    @property
    def is_symbolic(self) -> bool:
        """Check if running in symbolic mode."""
        return self.mode == ExecutionMode.SYMBOLIC
        
    def execute_primitive(
        self, 
        primitive, 
        *args,
        attempts: int = 5
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        Execute a primitive action and return structured feedback.
        
        Args:
            primitive: The action primitive to execute (e.g., GRASP, PLACE_ON_TOP)
                      Accepts both StarterSemanticActionPrimitiveSet and 
                      SymbolicSemanticActionPrimitiveSet
            *args: Arguments for the primitive (typically target objects)
            attempts: Number of retry attempts (default: 5)
        
        Returns:
            Tuple of (success, message, metadata):
            - success (bool): True if action completed successfully
            - message (str): Human-readable description of result
            - metadata (dict or None): Additional information, especially on failure
                - On single error: {'reason': str, 'metadata': dict}
                - On multiple retries: {'attempts': int, 'errors': list}
        
        Example:
            >>> api = BehaviorActionAPI(env, robot, mode=ExecutionMode.SYMBOLIC)
            >>> success, msg, meta = api.grasp(target_object)
            >>> if not success:
            ...     print(f"Failed: {meta['reason']}")
        """
        try:
            # Execute the primitive by stepping through the generator
            for action in self.controller.apply_ref(primitive, *args, attempts=attempts):
                self.env.step(action)
            
            # Success - generator completed without exception
            mode_str = "SYMBOLIC" if self.is_symbolic else "FULL"
            return True, f"Action completed successfully [{mode_str}]", None
            
        except ActionPrimitiveError as e:
            # Single execution error
            error_msg = f"{e.reason.name}: {str(e)}"
            metadata = {
                "reason": e.reason.name,
                "original_metadata": e.metadata,
                "mode": self.mode.value
            }
            return False, error_msg, metadata
            
        except ActionPrimitiveErrorGroup as eg:
            # All retry attempts failed
            errors = []
            for i, exc in enumerate(eg.exceptions):
                errors.append({
                    "attempt": i + 1,
                    "reason": exc.reason.name,
                    "metadata": exc.metadata
                })
            
            error_msg = f"All {len(eg.exceptions)} attempts failed"
            metadata = {
                "attempts": len(eg.exceptions),
                "errors": errors,
                "mode": self.mode.value
            }
            return False, error_msg, metadata
        
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            metadata = {
                "reason": "UNEXPECTED_ERROR",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "mode": self.mode.value
            }
            return False, error_msg, metadata
    
    def grasp(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Grasp an object."""
        return self.execute_primitive(self._primitive_set.GRASP, target_object)
    
    def place_on_top(self, surface_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Place currently held object on top of a surface."""
        return self.execute_primitive(self._primitive_set.PLACE_ON_TOP, surface_object)
    
    def place_inside(self, container_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Place currently held object inside a container."""
        return self.execute_primitive(self._primitive_set.PLACE_INSIDE, container_object)
    
    def navigate_to(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """
        Convenience method: Navigate to an object.
        
        Note: 
        - FULL mode: Uses CuRobo for collision-aware motion planning
        - SYMBOLIC mode: Uses A* pathfinding on traversable map with waypoint teleportation
        """
        return self.execute_primitive(self._primitive_set.NAVIGATE_TO, target_object)
    
    def navigate_to_position(self, x: float, y: float, yaw: float = 0.0) -> Tuple[bool, str, Optional[dict]]:
        """
        Convenience method: Navigate to a world position.
        
        This is useful for navigating to rooms (using room centroid) or arbitrary positions.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            yaw: Target orientation in radians (default: 0.0)
        
        Returns:
            Tuple of (success, message, metadata)
        
        Note:
        - SYMBOLIC mode: Uses A* pathfinding with waypoint teleportation
        - FULL mode: Not yet implemented (would need CuRobo base planning)
        """
        if self.is_symbolic:
            # Access symbolic primitives' _navigate_to_pose directly
            try:
                import torch as th
                pose_2d = [x, y, yaw]
                
                # Execute navigation (yields actions step by step)
                for action in self.controller._navigate_to_pose(pose_2d):
                    self.env.step(action)
                
                mode_str = "SYMBOLIC"
                return True, f"Navigated to position ({x:.2f}, {y:.2f}) [{mode_str}]", None
                
            except Exception as e:
                return False, f"Position navigation failed: {str(e)}", {"error": str(e)}
        else:
            # FULL mode: not implemented yet (would need CuRobo base motion planning)
            return False, "Position navigation not implemented for FULL mode yet", {"mode": "FULL"}
    
    def open_object(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Open an object."""
        return self.execute_primitive(self._primitive_set.OPEN, target_object)
    
    def close_object(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Close an object."""
        return self.execute_primitive(self._primitive_set.CLOSE, target_object)
    
    def release(self) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Release currently held object."""
        return self.execute_primitive(self._primitive_set.RELEASE)
    
    def toggle_on(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """
        Convenience method: Toggle an object on.
        
        Note: Available in both FULL and SYMBOLIC modes.
        """
        return self.execute_primitive(self._primitive_set.TOGGLE_ON, target_object)
    
    def toggle_off(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """
        Convenience method: Toggle an object off.
        
        Note: Available in both FULL and SYMBOLIC modes.
        """
        return self.execute_primitive(self._primitive_set.TOGGLE_OFF, target_object)


# Export for easy imports
if OMNIGIBSON_AVAILABLE:
    __all__ = [
        "BehaviorActionAPI", 
        "ExecutionMode",
        "StarterSemanticActionPrimitiveSet",
        "SymbolicSemanticActionPrimitiveSet"
    ]
else:
    __all__ = ["BehaviorActionAPI", "ExecutionMode"]
