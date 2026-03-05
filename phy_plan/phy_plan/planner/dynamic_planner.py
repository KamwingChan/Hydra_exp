"""
dynamic_planner.py: Dynamic replanning pipeline with conversation context

Implements a complete planning and execution pipeline with:
1. Physics-aware initial planning
2. Task-relevant change detection during execution
3. Hybrid replanning triggers (failure + subtask completion)
4. Graceful handling of environmental changes
5. **Persistent conversation context** across replanning

Architecture:
    DynamicPlannerPipeline: Execution monitoring, scene change detection, replan triggering
    LLMPlannerPipeline: Interactive planning (optional), user interaction
    LLMPlanner: Core planning logic (via public API)

Key innovation: Uses persistent LLMAgent.chat() instead of independent plan() calls,
enabling the LLM to learn from execution failures and scene changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import json

from ..core.scene_graph import SceneGraph
from ..core.task import TaskSequence, Action, ActionType, TaskStatus
from ..core.physics_agent import RobotCapability, ValidationResult
from ..core.change_detector import ChangeDetector, ChangeReport, ChangeType
from ..input.phy_graph_subscriber import SceneGraphSubscriber
from .llm_planner import LLMPlanner, ClarificationRequest, InfeasiblePlan
from .llm_planner_pipeline import LLMPlannerPipeline
from ..prompts.task_planning_prompt import generate_task_planning_prompt


class ReplanTrigger(Enum):
    """Reason for replanning"""
    EXECUTION_FAILURE = "execution_failure"
    SCENE_CHANGE = "scene_change"
    PHYSICS_VIOLATION = "physics_violation"
    USER_REQUEST = "user_request"


@dataclass
class ExecutionResult:
    """Result of action or task execution"""
    success: bool
    action: Optional[Action] = None
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplanEvent:
    """Record of a replanning event"""
    trigger: ReplanTrigger
    timestamp: float
    reason: str
    old_plan_progress: float
    new_plan_actions: int
    change_report: Optional[ChangeReport] = None


@dataclass
class PipelineResult:
    """Final result of pipeline execution"""
    success: bool
    task_sequence: Optional[TaskSequence] = None
    total_actions_executed: int = 0
    replan_events: List[ReplanEvent] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: str = ""
    
    def summary(self) -> str:
        """Generate execution summary"""
        status = "✅ Success" if self.success else "❌ Failed"
        lines = [
            f"Pipeline Result: {status}",
            f"  Actions executed: {self.total_actions_executed}",
            f"  Replan events: {len(self.replan_events)}",
            f"  Execution time: {self.execution_time:.2f}s"
        ]
        
        if self.replan_events:
            lines.append("  Replan triggers:")
            for event in self.replan_events:
                lines.append(f"    - {event.trigger.value}: {event.reason[:50]}...")
        
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        
        return "\n".join(lines)


class DynamicPlannerPipeline:
    """
    Dynamic planning and execution pipeline
    
    Implements hybrid replanning strategy:
    - Trigger 1: Immediate replan on execution failure
    - Trigger 2: Scene change check after subtask completion
    
    Only tracks task-relevant objects for efficiency.
    """
    
    def __init__(
        self,
        pipeline: Optional[LLMPlannerPipeline] = None,
        subscriber: Optional[SceneGraphSubscriber] = None,
        robot_capability: Optional[RobotCapability] = None,
        executor: Optional[Any] = None,  # BehaviorExecutor or mock
        check_scene_after_navigate: bool = True,
        check_scene_after_pick: bool = True,
        position_change_threshold: float = 0.5,
        max_replan_per_task: int = 3,
        use_conversation_mode: bool = True,
        model: str = "gpt-4o-mini",
        debug: bool = True
    ):
        """
        Initialize dynamic planner pipeline
        
        Args:
            pipeline: LLMPlannerPipeline instance for interactive planning (optional)
                      If provided, uses its planner and scene graph source.
            subscriber: Scene graph subscriber for real-time updates (used when no pipeline)
            robot_capability: Robot physical capabilities (used when no pipeline)
            executor: Action executor (BehaviorExecutor or mock)
            check_scene_after_navigate: Check for changes after NAVIGATE
            check_scene_after_pick: Check for changes after PICK
            position_change_threshold: Minimum position change to trigger replan (meters)
            max_replan_per_task: Maximum replanning attempts per task
            use_conversation_mode: Enable persistent conversation (recommended)
            model: LLM model name (used when no pipeline)
            debug: Enable debug output
        """
        if pipeline:
            # Use pipeline's planner and subscriber
            self._pipeline = pipeline
            self.planner = pipeline.planner
            self.subscriber = pipeline._subscriber or subscriber
        else:
            # Create new planner and use subscriber
            self._pipeline = None
            self.planner = LLMPlanner(
                robot_capability=robot_capability,
                enable_physics_validation=True,
                enable_spatial_resolver=True
            )
            self.subscriber = subscriber
        
        self.executor = executor
        
        self._check_after_navigate = check_scene_after_navigate
        self._check_after_pick = check_scene_after_pick
        self._max_replan = max_replan_per_task
        self._debug = debug
        
        # Conversation mode
        self._use_conversation = use_conversation_mode
        self._model = model
        self._conversation_initialized = False
        
        # Robot capability (for prompt generation)
        self._robot_capability = robot_capability or RobotCapability()
        
        # Change detector (initialized per task)
        self._change_detector: Optional[ChangeDetector] = None
        self._position_threshold = position_change_threshold
        
        # State
        self._current_plan: Optional[TaskSequence] = None
        self._current_scene_graph: Optional[SceneGraph] = None
        self._current_instruction: str = ""  # stored for handle_step_result()
        self._execution_index: int = 0
        self._replan_count: int = 0
        self._replan_events: List[ReplanEvent] = []
    
    # ==================== Public Methods (for PhyPlanPipeline) ====================
    
    def plan_initial(
        self,
        instruction: Optional[str] = None,
        scene_graph: Optional[SceneGraph] = None
    ) -> TaskSequence:
        """
        Run initial planning (blocking for LLM call + user interaction).
        
        Uses LLMPlannerPipeline.run_interactive() when available for full features
        (info_request, spatial resolution, physics validation, user confirmation).
        Falls back to conversation mode or direct plan() when no pipeline.
        
        After success, internal state is ready for handle_step_result() calls.
        
        Args:
            instruction: Natural language task instruction.
                         If None and pipeline available, run_interactive() prompts user.
                         If None and no pipeline, raises ValueError.
            scene_graph: Initial scene graph (uses subscriber if None)
            
        Returns:
            TaskSequence ready for execution
            
        Raises:
            ValueError: If planning fails (no scene graph, cancelled, infeasible, etc.)
        """
        self._replan_events = []
        self._replan_count = 0
        
        # 1. Get scene graph
        self._current_scene_graph = scene_graph or self._get_latest_scene_graph()
        if self._current_scene_graph is None:
            raise ValueError("No scene graph available for planning")
        
        if self._debug:
            if self._pipeline:
                print(f"[DynamicPipeline] Planning with LLMPlannerPipeline (interactive)")
            else:
                mode_str = "conversation" if self._use_conversation else "legacy"
                print(f"[DynamicPipeline] Planning for: {instruction} (mode: {mode_str})")
        
        # 2. Planning
        if self._pipeline:
            # run_interactive accepts None → prompts user in terminal
            task_seq = self._pipeline.run_interactive(
                initial_instruction=instruction,
                debug=self._debug
            )
            # Handle special return values from run_interactive
            if task_seq.task_name in ("Empty", "Error", "Cancelled"):
                raise ValueError(f"Interactive planning returned: {task_seq.task_name}")
            self._conversation_initialized = True
            result = task_seq
            # Capture the instruction from the pipeline if we didn't have one
            if instruction is None:
                instruction = task_seq.task_name or "interactive_task"
        elif instruction is not None:
            if self._use_conversation:
                if not self._conversation_initialized:
                    self._init_planning_conversation()
                    self._conversation_initialized = True
                result = self._chat_for_plan(instruction, self._current_scene_graph)
            else:
                result, _ = self.planner.plan(
                    self._current_scene_graph, instruction, debug=self._debug
                )
        else:
            raise ValueError("No instruction provided and no interactive pipeline available")
        
        # Handle non-TaskSequence results
        if isinstance(result, ClarificationRequest):
            raise ValueError(f"Clarification needed: {result.question}")
        if isinstance(result, InfeasiblePlan):
            raise ValueError(f"Task infeasible: {result.reason}")
        
        # 3. Initialize execution state
        self.init_execution_state(result, instruction)
        
        return result
    
    def init_execution_state(
        self,
        task_seq: TaskSequence,
        instruction: str,
        scene_graph: Optional[SceneGraph] = None
    ) -> None:
        """
        Initialize execution state for a TaskSequence.
        
        Called by plan_initial() internally, or directly for external TaskSequence
        (e.g., loaded from JSON file via PhyPlanPipeline.run_from_json()).
        
        Sets up: current plan, instruction, execution index, change detector.
        
        Args:
            task_seq: TaskSequence to prepare for execution
            instruction: Original instruction (stored for replanning context)
            scene_graph: Scene graph override (uses current if None)
        """
        if scene_graph is not None:
            self._current_scene_graph = scene_graph
        
        # If still no scene graph, try subscriber
        if self._current_scene_graph is None:
            self._current_scene_graph = self._get_latest_scene_graph()
        
        self._current_plan = task_seq
        self._current_instruction = instruction
        self._execution_index = 0
        
        # Initialize change detector with task-relevant objects
        relevant_objects = self._extract_relevant_objects(task_seq)
        self._change_detector = ChangeDetector(
            task_relevant_objects=relevant_objects,
            position_threshold=self._position_threshold
        )
        if self._current_scene_graph:
            self._change_detector.update_baseline(self._current_scene_graph)
        
        if self._debug:
            print(f"[DynamicPipeline] Execution state initialized: "
                  f"{len(task_seq.actions)} actions, "
                  f"tracking {len(relevant_objects)} objects")
    
    def handle_step_result(
        self,
        action: Action,
        success: bool,
        error: str = "",
        on_replan: Optional[Callable[[ReplanEvent], None]] = None
    ) -> Optional[TaskSequence]:
        """
        Handle step completion/failure. May trigger replanning.
        
        Called by PhyPlanPipeline._on_step_complete() callback after each
        PrimitiveController step finishes.
        
        Args:
            action: The Action that just completed/failed
            success: Whether the step succeeded
            error: Error message (if failed)
            on_replan: Optional callback when replanning occurs
            
        Returns:
            New TaskSequence if replanned, None if continuing normally
        """
        instruction = self._current_instruction
        
        if not success:
            # Execution failure -> immediate replan
            if self._debug:
                print(f"[DynamicPipeline] Step failed: {action.description} - {error}")
            
            exec_result = ExecutionResult(
                success=False, action=action, error_message=error
            )
            replan_ok = self._handle_failure_replan(
                instruction, action, exec_result, on_replan
            )
            if replan_ok:
                return self._current_plan  # New plan from failure replan
            return None  # Replan failed, caller should stop
        
        # Success path
        action.status = TaskStatus.COMPLETED
        self._execution_index += 1
        
        # Check scene after certain action types
        if self._should_check_scene(action):
            change_detected = self._check_and_handle_scene_change(
                instruction, on_replan
            )
            if change_detected and self._current_plan is not None:
                return self._current_plan  # New plan from scene change
            # If change_detected and _current_plan is None, replan failed
        
        return None  # No replan needed, continue normally
    
    # ==================== Validated Replan ====================
    
    def _validated_replan(
        self,
        context_message: str,
        scene_graph: SceneGraph
    ) -> Union[TaskSequence, InfeasiblePlan]:
        """
        Replan with interactive multi-turn dialogue.
        
        Delegates to LLMPlannerPipeline._interactive_loop(replan_mode=True)
        for full interaction support: info_request, clarification, physics
        validation, user confirmation — all while preserving conversation history.
        
        Args:
            context_message: Context about failure/scene change
            scene_graph: Updated scene graph
            
        Returns:
            TaskSequence if successful, InfeasiblePlan otherwise
        """
        compact_json = scene_graph.to_compact_json()
        replan_prompt = (
            f"{context_message}\n\n"
            f"Updated Scene Graph (supersedes the scene graph in the system prompt):\n"
            f"{compact_json}\n\n"
            f"Please generate a new plan considering the above context and updated scene."
        )
        
        # Update pipeline's scene graph to latest
        self._pipeline._scene_graph = scene_graph
        
        if self._debug:
            print(f"[DynamicPipeline] Interactive replan via _interactive_loop...")
        
        # Send replan context and get initial LLM response
        response_text = self.planner.agent.chat(replan_prompt)
        
        if self._debug:
            print(f"\n{'='*40} DEBUG: REPLAN RESPONSE {'='*40}")
            print(response_text)
            print(f"{'='*97}\n")
        
        # Enter interactive loop for multi-turn processing
        self._pipeline._pending_response = response_text
        result = self._pipeline._interactive_loop(
            self._current_instruction, self._debug, replan_mode=True
        )
        
        # Convert special task names to appropriate return types
        if result.task_name in ("Cancelled", "Empty", "Error"):
            return InfeasiblePlan(
                reason=f"Interactive replan ended: {result.task_name}",
                chain_of_thought="",
                suggestions=[]
            )
        return result
    
    # ==================== Blocking Execution (backward compatible) ====================
    
    def plan_and_execute(
        self,
        instruction: str,
        initial_scene_graph: Optional[SceneGraph] = None,
        on_action_complete: Optional[Callable[[Action, ExecutionResult], None]] = None,
        on_replan: Optional[Callable[[ReplanEvent], None]] = None
    ) -> PipelineResult:
        """
        Plan and execute a task with dynamic replanning (blocking execution).
        
        This is the original blocking API, mainly for offline testing, experiments,
        and BehaviorExecutor-based execution. Internally delegates to plan_initial()
        and handle_step_result().
        
        For non-blocking ROS-compatible execution, use PhyPlanPipeline instead.
        
        Args:
            instruction: Natural language instruction
            initial_scene_graph: Initial scene graph (uses subscriber if None)
            on_action_complete: Callback after each action
            on_replan: Callback when replanning occurs
            
        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()
        total_actions = 0
        
        # 1. Planning (delegates to plan_initial)
        try:
            self.plan_initial(instruction, initial_scene_graph)
        except ValueError as e:
            return PipelineResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
        
        # 2. Execute with monitoring (blocking loop)
        while self._execution_index < len(self._current_plan.actions):
            action = self._current_plan.actions[self._execution_index]
            
            # Lazy expansion for ARRANGE actions (uses latest scene graph)
            if action.action_type == ActionType.ARRANGE and self._pipeline:
                expanded = self._pipeline.expand_arrange_action(action)
                if len(expanded) > 1:
                    if self._debug:
                        print(f"[DynamicPipeline] Expanded ARRANGE to {len(expanded)} sub-actions")
                    self._current_plan.actions[self._execution_index:self._execution_index+1] = expanded
                    action = self._current_plan.actions[self._execution_index]
            
            if self._debug:
                print(f"[DynamicPipeline] Executing action {self._execution_index + 1}/"
                      f"{len(self._current_plan.actions)}: {action.description}")
            
            # Execute action (blocking)
            exec_result = self._execute_action(action)
            total_actions += 1
            
            if on_action_complete:
                on_action_complete(action, exec_result)
            
            # Handle step result (delegates to handle_step_result)
            new_plan = self.handle_step_result(
                action, exec_result.success, exec_result.error_message, on_replan
            )
            
            if not exec_result.success:
                if new_plan is None:
                    # Replan failed
                    return PipelineResult(
                        success=False,
                        task_sequence=self._current_plan,
                        total_actions_executed=total_actions,
                        replan_events=self._replan_events,
                        execution_time=time.time() - start_time,
                        error_message=f"Execution failed and replanning unsuccessful: {exec_result.error_message}"
                    )
                # new_plan is not None -> replan succeeded, _current_plan updated, continue loop
                continue
            
            # Success path: handle_step_result already updated _execution_index
            if new_plan is not None:
                # Scene change triggered replan, _current_plan updated
                pass  # Continue with new plan
            
            if self._current_plan is None:
                # Scene change replan failed
                return PipelineResult(
                    success=False,
                    total_actions_executed=total_actions,
                    replan_events=self._replan_events,
                    execution_time=time.time() - start_time,
                    error_message="Scene change detected but replanning failed"
                )
        
        # Success
        self._current_plan.status = TaskStatus.COMPLETED
        
        return PipelineResult(
            success=True,
            task_sequence=self._current_plan,
            total_actions_executed=total_actions,
            replan_events=self._replan_events,
            execution_time=time.time() - start_time
        )
    
    def _execute_action(self, action: Action) -> ExecutionResult:
        """Execute a single action"""
        if self.executor is None:
            # Mock execution for testing
            if self._debug:
                print(f"[DynamicPipeline] Mock executing: {action.description}")
            time.sleep(0.1)  # Simulate execution time
            return ExecutionResult(success=True, action=action)
        
        # Real execution
        try:
            result = self.executor.execute_action(action)
            return ExecutionResult(
                success=result.get("success", False),
                action=action,
                error_message=result.get("error", ""),
                details=result
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                action=action,
                error_message=str(e)
            )
    
    def _should_check_scene(self, action: Action) -> bool:
        """Determine if scene should be checked after this action"""
        if action.action_type == ActionType.NAVIGATE and self._check_after_navigate:
            return True
        if action.action_type == ActionType.PICK and self._check_after_pick:
            return True
        # Always check after PLACE (object position changed)
        if action.action_type == ActionType.PLACE:
            return True
        if action.action_type == ActionType.PLACE_INSIDE:
            return True
        # Always check after OBSERVE (physical properties may have changed)
        if action.action_type == ActionType.OBSERVE:
            return True
        if action.action_type == ActionType.ARRANGE:
            return True
        if action.action_type == ActionType.CLEAN_UP:
            return True
        if action.action_type == ActionType.LOCATE:
            return True
        if action.action_type == ActionType.OPEN:
            return True
        if action.action_type == ActionType.CLOSE:
            return True
        return False
    
    def _check_and_handle_scene_change(
        self,
        instruction: str,
        on_replan: Optional[Callable[[ReplanEvent], None]]
    ) -> bool:
        """
        Check for scene changes and handle if needed
        
        Returns:
            True if change detected (replan attempted), False otherwise
        """
        # Get latest scene graph
        new_sg = self._get_latest_scene_graph()
        if new_sg is None:
            return False
        
        # Quick check first
        if not self._change_detector.quick_check(new_sg):
            return False
        
        # Full detection
        change_report = self._change_detector.detect(self._current_scene_graph, new_sg)
        
        if not change_report.is_task_affected:
            # Update scene graph but no replan needed
            self._current_scene_graph = new_sg
            return False
        
        if self._debug:
            print(f"[DynamicPipeline] Task-relevant scene change detected!")
            print(change_report.summary())
        
        # Replan
        return self._handle_scene_change_replan(
            instruction, change_report, new_sg, on_replan
        )
    
    def _handle_failure_replan(
        self,
        instruction: str,
        failed_action: Action,
        exec_result: ExecutionResult,
        on_replan: Optional[Callable[[ReplanEvent], None]]
    ) -> bool:
        """
        Handle replanning after execution failure
        
        Returns:
            True if replan successful, False otherwise
        """
        if self._replan_count >= self._max_replan:
            if self._debug:
                print(f"[DynamicPipeline] Max replan attempts ({self._max_replan}) reached")
            return False
        
        self._replan_count += 1
        
        # Get latest scene graph
        new_sg = self._get_latest_scene_graph()
        if new_sg is None:
            new_sg = self._current_scene_graph
        
        # Build failure context (in ENGLISH for LLM)
        failure_context = (
            f"[EXECUTION FAILURE]\n"
            f"Previous action failed: {failed_action.description}\n"
            f"Error: {exec_result.error_message}\n"
            f"Please replan considering this failure."
        )
        
        # Use _validated_replan (interactive multi-turn dialogue)
        result = self._validated_replan(failure_context, new_sg)
        
        if isinstance(result, InfeasiblePlan):
            return False
        
        # Record event
        event = ReplanEvent(
            trigger=ReplanTrigger.EXECUTION_FAILURE,
            timestamp=time.time(),
            reason=exec_result.error_message,
            old_plan_progress=self._current_plan.progress if self._current_plan else 0,
            new_plan_actions=len(result.actions)
        )
        self._replan_events.append(event)
        
        if on_replan:
            on_replan(event)
        
        # Update state
        self._current_plan = result
        self._current_scene_graph = new_sg
        self._execution_index = 0
        
        # Update change detector
        relevant_objects = self._extract_relevant_objects(result)
        self._change_detector.set_task_relevant_objects(relevant_objects)
        self._change_detector.update_baseline(new_sg)
        
        if self._debug:
            print(f"[DynamicPipeline] Replanned with {len(result.actions)} actions")
        
        return True
    
    def _handle_scene_change_replan(
        self,
        instruction: str,
        change_report: ChangeReport,
        new_sg: SceneGraph,
        on_replan: Optional[Callable[[ReplanEvent], None]]
    ) -> bool:
        """
        Handle replanning after scene change
        
        Returns:
            True (change was detected), plan may be None if replan failed
        """
        if self._replan_count >= self._max_replan:
            if self._debug:
                print(f"[DynamicPipeline] Max replan attempts ({self._max_replan}) reached")
            self._current_plan = None
            return True
        
        self._replan_count += 1
        
        # Build change context (in ENGLISH for LLM)
        change_context = change_report.to_replan_context()
        
        # Use _validated_replan (interactive multi-turn dialogue)
        result = self._validated_replan(
            f"[SCENE CHANGE]\nScene changed:\n{change_context}",
            new_sg
        )
        
        # Record event
        event = ReplanEvent(
            trigger=ReplanTrigger.SCENE_CHANGE,
            timestamp=time.time(),
            reason=change_report.summary()[:100],
            old_plan_progress=self._current_plan.progress if self._current_plan else 0,
            new_plan_actions=len(result.actions) if isinstance(result, TaskSequence) else 0,
            change_report=change_report
        )
        self._replan_events.append(event)
        
        if on_replan:
            on_replan(event)
        
        if isinstance(result, InfeasiblePlan):
            self._current_plan = None
            return True
        
        # Update state
        self._current_plan = result
        self._current_scene_graph = new_sg
        self._execution_index = 0
        
        # Update change detector
        relevant_objects = self._extract_relevant_objects(result)
        self._change_detector.set_task_relevant_objects(relevant_objects)
        self._change_detector.update_baseline(new_sg)
        
        if self._debug:
            print(f"[DynamicPipeline] Replanned with {len(result.actions)} actions after scene change")
        
        return True
    
    def _get_latest_scene_graph(self) -> Optional[SceneGraph]:
        """Get latest scene graph from subscriber"""
        if self.subscriber is None:
            return self._current_scene_graph
        
        return self.subscriber.get_latest()
    
    @staticmethod
    def _extract_relevant_objects(task_seq: TaskSequence) -> List[str]:
        """Extract task-relevant object IDs from plan"""
        relevant = set()
        
        for action in task_seq.actions:
            if action.target_object:
                relevant.add(action.target_object)
            
            # Check params
            params = action.params or {}
            if "object_id" in params:
                relevant.add(params["object_id"])
            if "surface_id" in params:
                relevant.add(params["surface_id"])
            if "target_object" in params:
                relevant.add(params["target_object"])
        
        return list(relevant)
    
    # ==================== Convenience Methods ====================
    
    def plan_only(
        self,
        instruction: str,
        scene_graph: Optional[SceneGraph] = None
    ) -> Tuple[Union[TaskSequence, ClarificationRequest, InfeasiblePlan], Dict[str, Any]]:
        """
        Plan without execution (for testing/preview)
        """
        sg = scene_graph or self._get_latest_scene_graph()
        if sg is None:
            raise ValueError("No scene graph available")
        
        return self.planner.plan(sg, instruction, debug=self._debug)
    
    def validate_plan(
        self,
        task_seq: TaskSequence,
        scene_graph: Optional[SceneGraph] = None
    ) -> ValidationResult:
        """
        Validate a plan against physics constraints (delegates to planner)
        """
        sg = scene_graph or self._current_scene_graph
        if sg is None:
            raise ValueError("No scene graph available")
        
        return self.planner.validate_plan(task_seq, sg)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current pipeline state"""
        return {
            "has_plan": self._current_plan is not None,
            "execution_index": self._execution_index,
            "plan_length": len(self._current_plan.actions) if self._current_plan else 0,
            "replan_count": self._replan_count,
            "tracked_objects": list(self._change_detector._task_relevant_objects) if self._change_detector else []
        }
    
    # ==================== NEW: Conversation Mode Helpers ====================
    
    def _init_planning_conversation(self) -> None:
        """Initialize persistent conversation for planning
        
        Note: Uses planner's agent instead of creating a separate one,
        ensuring conversation history is shared between initial planning
        and replanning.
        """
        # Generate system prompt using existing prompt generator
        # Parameters: scene_graph_compact, instruction, include_example
        system_prompt, _ = generate_task_planning_prompt(
            scene_graph_compact="",  # Will be provided in user messages
            instruction="",
            include_example=True
        )
        
        # Use planner's agent to maintain conversation history
        self.planner.init_conversation(system_prompt)
        
        if self._debug:
            print(f"[DynamicPipeline] Initialized conversation with planner's agent")
    
    def _chat_for_plan(
        self,
        instruction: str,
        scene_graph: SceneGraph
    ) -> Union[TaskSequence, ClarificationRequest, InfeasiblePlan]:
        """
        Get initial plan via conversation (delegation to planner's chat interface)
        
        For initial planning, we still use planner.plan() for now.
        Conversation mode shines during replanning when history matters.
        """
        # For initial planning, use standard plan() method
        result, _ = self.planner.plan(scene_graph, instruction, debug=self._debug)
        return result
