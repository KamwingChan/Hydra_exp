"""
pipeline.py: Non-blocking execution pipeline

Bridges the DynamicPlannerPipeline output (TaskSequence) to the env PrimitiveController,
enabling non-blocking execution where the main simulation loop keeps running
(ROS publishing, sensor updates, video recording all work during execution).

Architecture:
    PhyPlanPipeline (this file)
    ├── Wraps DynamicPlannerPipeline (planning + replanning logic)
    ├── run(instruction) → plan_initial() [blocking ~3-10s] → submit [non-blocking]
    ├── run_from_json(path) → load TaskSequence → submit [non-blocking]
    ├── run_from_sequence(task_seq) → convert + submit [non-blocking]
    └── Callbacks:
        ├── _on_step_complete → handle_step_result() → may request_replan()
        └── _on_task_complete → mark done

Data flow:
    DynamicPlannerPipeline.plan_initial() → TaskSequence
    PhyPlanPipeline._convert() → List[PrimitiveStep]
    PrimitiveController.start_task_sequence() → per-frame execution
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Any, Union

from .task import TaskSequence, Action, ActionType, TaskStatus
from .scene_graph import SceneGraph

logger = logging.getLogger(__name__)


# Action type -> PrimitiveStep primitive_type mapping
ACTION_TO_PRIMITIVE = {
    ActionType.NAVIGATE: "navigate_to",
    ActionType.PICK: "grasp",
    ActionType.PLACE: "place_on_top",
    ActionType.PLACE_INSIDE: "place_inside",
    ActionType.OPEN: "open",
    ActionType.CLOSE: "close",
    ActionType.OBSERVE: "observe",
}


class PhyPlanPipeline:
    """
    Non-blocking pipeline: DynamicPlannerPipeline → PrimitiveController.
    
    Thin adapter that:
    1. Delegates planning to DynamicPlannerPipeline (full features via LLMPlannerPipeline)
    2. Converts TaskSequence → PrimitiveStep list
    3. Submits to PrimitiveController for per-frame execution
    4. On step complete/failure, delegates to DynamicPlannerPipeline.handle_step_result()
       for potential replanning (physics-validated)
    
    Args:
        primitive_controller: env PrimitiveController instance
        dynamic_pipeline: DynamicPlannerPipeline instance (planning + replanning)
    """
    
    def __init__(self, primitive_controller, dynamic_pipeline=None):
        """
        Initialize PhyPlanPipeline.
        
        Args:
            primitive_controller: PrimitiveController instance (has start_task_sequence)
            dynamic_pipeline: DynamicPlannerPipeline instance (optional, enables run()
                              and replanning. Without it, only run_from_sequence() works.)
        """
        self.controller = primitive_controller
        self._dynamic = dynamic_pipeline
        
        # State tracking
        self._current_task: Optional[TaskSequence] = None
        self._executable_actions: Optional[List[Action]] = None  # Only actions actually converted to steps
        self._instruction: str = ""  # Stored for replanning context
        self._is_running: bool = False
        self._result_success: Optional[bool] = None
        self._result_error: str = ""
    
    # ==================== Execution Entry Points ====================
    
    def run(self, instruction: str = None, debug: bool = True) -> bool:
        """
        Plan from natural language instruction and execute non-blocking.
        
        Planning phase blocks (terminal interaction for instruction input,
        LLM call, user confirmation). Execution phase is fully non-blocking.
        
        Args:
            instruction: Natural language task instruction.
                         If None, run_interactive() will prompt user in terminal.
            debug: Print debug info
            
        Returns:
            bool: True if plan generated and submitted successfully
        """
        if self._dynamic is None:
            print("[Pipeline] No DynamicPlannerPipeline configured. "
                         "Use run_from_sequence() instead.")
            return False
        
        if self._is_running:
            print("[Pipeline] Cannot start - a task is already running")
            return False
        
        if debug:
            print(f"[Pipeline] Planning: {instruction}")
        
        try:
            task_seq = self._dynamic.plan_initial(instruction)
        except ValueError as e:
            print(f"[Pipeline] Planning failed: {e}")
            return False
        except Exception as e:
            import traceback
            import json
            import time
            err_trace = traceback.format_exc()
            try:
                with open('/home/kamwing/catkin_ws/src/phy_graph/.cursor/debug.log', 'a') as f:
                    log_entry = {
                        "id": f"log_{int(time.time()*1000)}",
                        "timestamp": int(time.time()*1000),
                        "location": "pipeline.py:116",
                        "message": "Unexpected planning error",
                        "data": {"error": str(e), "traceback": err_trace},
                        "runId": "run1",
                        "hypothesisId": "H1_openai_pydantic_serialization"
                    }
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as log_e:
                pass
            print(f"[Pipeline] Unexpected planning error: {e}")
            traceback.print_exc()
            return False
        
        self._instruction = instruction
        
        if debug:
            print(f"[Pipeline] Plan generated: {len(task_seq.actions)} actions")
            for i, action in enumerate(task_seq.actions):
                print(f"  [{i+1}] {action.action_type.value}: {action.description}")
        
        return self.run_from_sequence(task_seq, debug=debug)
    
    def run_from_json(self, json_path: str, instruction: str = "",
                      debug: bool = True) -> bool:
        """
        Load TaskSequence from JSON file and execute non-blocking.
        
        Skips LLM planning. Optionally initializes DynamicPlannerPipeline's
        execution state for change detection and replanning.
        
        Args:
            json_path: Path to JSON file containing TaskSequence
            instruction: Optional task instruction (for replanning context)
            debug: Print debug info
            
        Returns:
            bool: True if loaded and submitted successfully
        """
        if self._is_running:
            print("[Pipeline] Cannot start - a task is already running")
            return False
        
        try:
            path = Path(json_path)
            if not path.exists():
                print(f"[Pipeline] JSON file not found: {json_path}")
                return False
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            task_seq = TaskSequence.from_dict(data)
        except Exception as e:
            print(f"[Pipeline] Failed to load JSON: {e}")
            return False
        
        self._instruction = instruction or task_seq.task_name or "loaded_from_json"
        
        # Initialize DynamicPlannerPipeline state for change detection
        if self._dynamic is not None:
            try:
                self._dynamic.init_execution_state(task_seq, self._instruction)
            except Exception as e:
                print(f"[Pipeline] Failed to init execution state "
                               f"(change detection disabled): {e}")
        
        if debug:
            print(f"[Pipeline] Loaded from JSON: {len(task_seq.actions)} actions")
            for i, action in enumerate(task_seq.actions):
                print(f"  [{i+1}] {action.action_type.value}: {action.description}")
        
        return self.run_from_sequence(task_seq, debug=debug)
    
    def run_from_sequence(self, task_seq: TaskSequence, debug: bool = True) -> bool:
        """
        Submit an existing TaskSequence for non-blocking execution.
        
        Converts Action list to PrimitiveStep list and submits to
        PrimitiveController with callbacks.
        
        Args:
            task_seq: TaskSequence to execute
            debug: Print debug info
            
        Returns:
            bool: True if submitted successfully
        """
        if self._is_running:
            print("[Pipeline] Cannot submit - a task is already running")
            return False
        
        # Convert Action list to PrimitiveStep list
        try:
            steps = self._convert(task_seq, debug=debug)
        except Exception as e:
            print(f"[Pipeline] Conversion failed: {e}")
            return False
        
        if not steps:
            print("[Pipeline] No executable steps after conversion")
            return False
        
        # Track state
        self._current_task = task_seq
        self._current_task.status = TaskStatus.RUNNING
        self._is_running = True
        self._result_success = None
        self._result_error = ""
        
        # Submit to PrimitiveController (non-blocking from here on)
        success = self.controller.start_task_sequence(
            steps,
            on_step_complete=self._on_step_complete,
            on_task_complete=self._on_task_complete
        )
        
        if not success:
            self._is_running = False
            self._current_task.status = TaskStatus.FAILED
            print("[Pipeline] Failed to start task sequence on controller")
            return False
        
        if debug:
            print(f"[Pipeline] Submitted {len(steps)} steps to PrimitiveController")
        
        return True
    
    # ==================== Conversion ====================
    
    def _convert(self, task_seq: TaskSequence, debug: bool = True) -> List[Any]:
        """
        Convert TaskSequence.actions to List[PrimitiveStep].
        
        Handles:
        - ActionType -> primitive_type string mapping
        - Object ID (e.g. "O(13)") -> category name resolution via scene_graph
        - PLACE/PLACE_INSIDE target extraction from params
        """
        # Import PrimitiveStep from env controller (avoids circular at module level)
        from env.controllers.primitive_controller import PrimitiveStep
        
        steps = []
        self._executable_actions = []  # Reset and rebuild in sync with steps
        
        for i, action in enumerate(task_seq.actions):
            # Skip non-executable action types
            if action.action_type not in ACTION_TO_PRIMITIVE:
                if debug:
                    print(
                        f"[Pipeline] Skipping unsupported action type: "
                        f"{action.action_type.value} ({action.description})"
                    )
                continue
            
            primitive_type = ACTION_TO_PRIMITIVE[action.action_type]
            
            # Resolve target name based on action type
            raw_target = self._get_raw_target(action)
            target_name = self._resolve_target_name(action)
            
            if target_name is None:
                print(
                    f"[Pipeline] Cannot resolve target for action {i+1}: "
                    f"{action.action_type.value} {action.description}"
                )
                continue
            
            steps.append(PrimitiveStep(
                primitive_type=primitive_type,
                target_name=target_name,
                node_id=raw_target or ""
            ))
            self._executable_actions.append(action)
            
            if debug:
                print(
                    f"[Pipeline] Converted: {action.action_type.value}({action.target_object}) "
                    f"-> {primitive_type}({target_name})"
                )
        
        # Inject scene_graph into controller for position-based disambiguation
        scene_graph = None
        if self._dynamic is not None:
            scene_graph = getattr(self._dynamic, '_current_scene_graph', None)
        if scene_graph is not None:
            self.controller._scene_graph = scene_graph

        return steps
    
    def _get_raw_target(self, action: Action) -> Optional[str]:
        """Return the raw node ID (e.g. 'O(13)') for an action without resolving."""
        if action.action_type == ActionType.PLACE:
            return action.params.get("surface_id") or action.target_object
        elif action.action_type == ActionType.PLACE_INSIDE:
            return action.params.get("container_id") or action.target_object
        else:
            return action.target_object

    def _resolve_target_name(self, action: Action) -> Optional[str]:
        """
        Resolve object ID to a name/category that PrimitiveController can find.
        
        Resolution strategy:
        1. For PLACE: use params["surface_id"] as the target
        2. For PLACE_INSIDE: use params["container_id"] as the target
        3. For other actions: use action.target_object
        4. If target is a node ID like "O(13)", look up category in scene_graph
        5. If no scene_graph, use the raw target string as-is
        """
        raw_target = self._get_raw_target(action)
        
        if not raw_target:
            return None
        
        return self._resolve_node_id(raw_target)
    
    def _resolve_node_id(self, node_id: str) -> str:
        """
        Resolve a node ID like "O(13)" to its category name via scene_graph.
        
        Gets scene_graph from DynamicPlannerPipeline if available.
        """
        # Try to get scene graph from DynamicPlannerPipeline
        scene_graph = None
        if self._dynamic is not None:
            scene_graph = self._dynamic._current_scene_graph
        
        if scene_graph is None:
            return node_id
        
        # Try to look up as object node ID
        obj_node = scene_graph.get_object(node_id)
        if obj_node is not None:
            return obj_node.category
        
        # Not found in scene graph -- return as-is
        # PrimitiveController._find_target_object will try name/category matching
        logger.warning(f"[Pipeline] Node ID '{node_id}' not found in scene graph, using as-is")
        return node_id
    
    # ==================== Callbacks ====================
    
    def _on_step_complete(self, step_index: int, step, success: bool, error: str):
        """
        Called by PrimitiveController when a step completes or fails.
        
        Delegates to DynamicPlannerPipeline.handle_step_result() for potential
        replanning. If replanning produces a new plan, requests queue replacement
        on the controller.
        """
        if self._executable_actions is None or step_index >= len(self._executable_actions):
            return
        
        action = self._executable_actions[step_index]
        
        if success:
            action.status = TaskStatus.COMPLETED
            action.result_message = "Completed"
            print(f"[Pipeline] Step {step_index + 1} completed: {action.description}")
        else:
            action.status = TaskStatus.FAILED
            action.result_message = error
            print(f"[Pipeline] Step {step_index + 1} failed: {error}")
        
        # Delegate to DynamicPlannerPipeline for replanning logic
        if self._dynamic is not None:
            try:
                new_plan = self._dynamic.handle_step_result(
                    action, success, error
                )
                if new_plan is not None:
                    # Replan produced a new TaskSequence
                    print(f"[Pipeline] Replan triggered: {len(new_plan.actions)} new actions")
                    new_steps = self._convert(new_plan, debug=True)
                    if new_steps:
                        self._current_task = new_plan
                        self.controller.request_replan(new_steps)
                    else:
                        print("[Pipeline] Replan conversion produced no steps")
            except Exception as e:
                print(f"[Pipeline] handle_step_result error: {e}")
    
    def _on_task_complete(self, success: bool, error: str):
        """Called by PrimitiveController when the entire task sequence finishes."""
        self._is_running = False
        self._result_success = success
        self._result_error = error
        
        if self._current_task:
            if success:
                self._current_task.status = TaskStatus.COMPLETED
                print(f"[Pipeline] Task completed: {self._current_task.task_name}")
            else:
                self._current_task.status = TaskStatus.FAILED
                self._current_task.error_message = error
                print(f"[Pipeline] Task failed: {error}")
    
    # ==================== Query Interface ====================
    
    @property
    def is_running(self) -> bool:
        """Whether a task is currently being executed."""
        return self._is_running
    
    @property
    def progress(self) -> float:
        """Current task progress (0.0 - 1.0)."""
        if self._current_task:
            return self._current_task.progress
        return 0.0
    
    @property
    def current_task(self) -> Optional[TaskSequence]:
        """The currently executing (or last executed) TaskSequence."""
        return self._current_task
    
    @property
    def result(self) -> Optional[bool]:
        """Result of last task execution: True=success, False=failed, None=not finished."""
        return self._result_success
    
    @property
    def instruction(self) -> str:
        """Current/last instruction."""
        return self._instruction
    
    # ==================== Lifecycle Management ====================
    
    def cancel(self):
        """
        Cancel the currently running task.
        
        Clears the task queue in PrimitiveController and resets internal state.
        The robot will return to idle on the next frame.
        """
        if not self._is_running:
            print("[Pipeline] Nothing to cancel (not running)")
            return
        
        print("[Pipeline] Cancelling current task...")
        
        # Stop execution in PrimitiveController
        self.controller.cancel_task()
        
        # Update task status
        if self._current_task:
            self._current_task.status = TaskStatus.FAILED
            self._current_task.error_message = "Cancelled by user"
        
        self._is_running = False
        self._result_success = False
        self._result_error = "Cancelled by user"
        self._executable_actions = None
        print("[Pipeline] Task cancelled, robot returning to idle")
    
    def destroy(self):
        """
        Destroy the pipeline and release resources.
        
        - Cancels any running task
        - Clears DynamicPlannerPipeline reference
        - Clears controller's pipeline reference
        
        After destroy(), P key will trigger lazy re-initialization (via factory).
        """
        print("[Pipeline] Destroying pipeline...")
        
        # Cancel if running
        if self._is_running:
            self.cancel()
        
        # Clear references
        self._dynamic = None
        self._current_task = None
        self._executable_actions = None
        
        # Clear controller's reference so factory can reinitialize
        self.controller.set_pipeline(None)
        
        print("[Pipeline] Pipeline destroyed. Press P to re-initialize.")