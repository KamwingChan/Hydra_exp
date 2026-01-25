"""
BehaviorActionAPI: Wrapper for BEHAVIOR action primitives with detailed feedback.

This module provides a clean interface to omnigibson's StarterSemanticActionPrimitives,
handling error cases and providing structured feedback.
"""
from typing import Tuple, Optional, Any

# Try to import omnigibson - allow module to load even if omnigibson not available
try:
    from omnigibson.action_primitives.starter_semantic_action_primitives import (
        StarterSemanticActionPrimitives,
        StarterSemanticActionPrimitiveSet,
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


class BehaviorActionAPI:
    """
    Wrapper for BEHAVIOR action primitives with detailed feedback.
    
    Provides a clean interface to omnigibson's action primitives, converting
    exceptions to structured feedback tuples.
    
    Args:
        env: Omnigibson Environment instance
        robot: Robot instance (usually env.robots[0])
        enable_head_tracking: Whether to enable head tracking (default: True)
    """
    
    def __init__(self, env, robot, enable_head_tracking: bool = True):
        if not OMNIGIBSON_AVAILABLE:
            raise ImportError(
                "omnigibson is not available. Please install omnigibson to use BehaviorActionAPI."
            )
        
        self.env = env
        self.robot = robot
        self.controller = StarterSemanticActionPrimitives(
            env=env,
            robot=robot,
            enable_head_tracking=enable_head_tracking
        )
        
    def execute_primitive(
        self, 
        primitive: StarterSemanticActionPrimitiveSet, 
        *args,
        attempts: int = 5
    ) -> Tuple[bool, str, Optional[dict]]:
        """
        Execute a primitive action and return structured feedback.
        
        Args:
            primitive: The action primitive to execute (e.g., GRASP, PLACE_ON_TOP)
            *args: Arguments for the primitive (typically target objects)
            attempts: Number of retry attempts (default: 5, same as omnigibson default)
        
        Returns:
            Tuple of (success, message, metadata):
            - success (bool): True if action completed successfully
            - message (str): Human-readable description of result
            - metadata (dict or None): Additional information, especially on failure
                - On single error: {'reason': str, 'metadata': dict}
                - On multiple retries: {'attempts': int, 'errors': list}
        
        Example:
            >>> api = BehaviorActionAPI(env, robot)
            >>> success, msg, meta = api.execute_primitive(
            ...     StarterSemanticActionPrimitiveSet.GRASP,
            ...     target_object
            ... )
            >>> if not success:
            ...     print(f"Failed: {meta['reason']}")
        """
        try:
            # Execute the primitive by stepping through the generator
            for action in self.controller.apply_ref(primitive, *args, attempts=attempts):
                self.env.step(action)
            
            # Success - generator completed without exception
            return True, "Action completed successfully", None
            
        except ActionPrimitiveError as e:
            # Single execution error
            error_msg = f"{e.reason.name}: {str(e)}"
            metadata = {
                "reason": e.reason.name,
                "original_metadata": e.metadata
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
                "errors": errors
            }
            return False, error_msg, metadata
        
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
            metadata = {
                "reason": "UNEXPECTED_ERROR",
                "exception_type": type(e).__name__,
                "exception_message": str(e)
            }
            return False, error_msg, metadata
    
    def grasp(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Grasp an object."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.GRASP, target_object)
    
    def place_on_top(self, surface_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Place currently held object on top of a surface."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, surface_object)
    
    def place_inside(self, container_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Place currently held object inside a container."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, container_object)
    
    def navigate_to(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Navigate to an object."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.NAVIGATE_TO, target_object)
    
    def open_object(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Open an object."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.OPEN, target_object)
    
    def close_object(self, target_object) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Close an object."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.CLOSE, target_object)
    
    def release(self) -> Tuple[bool, str, Optional[dict]]:
        """Convenience method: Release currently held object."""
        return self.execute_primitive(StarterSemanticActionPrimitiveSet.RELEASE)


# Export for easy imports
if OMNIGIBSON_AVAILABLE:
    __all__ = ["BehaviorActionAPI", "StarterSemanticActionPrimitiveSet"]
else:
    __all__ = ["BehaviorActionAPI"]
