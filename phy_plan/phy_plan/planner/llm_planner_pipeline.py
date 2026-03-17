"""
llm_planner_pipeline.py: LLM planning pipeline

Integrate scene graph loading, LLM planning, and task post-processing.

Architecture:
    Pipeline: Scene graph management, user interaction, interactive loop, post-processing
    Planner: LLM calls, response parsing, physics validation, replanning (via public API)
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.agent import LLMAgent
from ..core.scene_graph import SceneGraph
from ..core.physics_agent import RobotCapability
from ..core.task import TaskSequence, Action, ActionType
from ..input.phy_graph_io import load_scene_graph
from .llm_planner import LLMPlanner, ClarificationRequest, InfeasiblePlan


class LLMPlannerPipeline:
    """
    LLM Planning Pipeline
    
    Complete workflow:
    1. Load scene graph (from file or directly)
    2. Call LLM Planner to generate task sequence
    3. Post-processing (e.g., expand arrange actions)
    
    Responsibilities:
    - Pipeline: Scene graph management, user interaction, interactive loop, post-processing
    - Planner: LLM calls, response parsing, physics validation, replanning (via public API)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = True,
        subscriber: Optional[Any] = None,
        scene_file: Optional[Union[str, Path]] = None,
        robot_capability: Optional[RobotCapability] = None,
        enable_physics_validation: bool = True,
        enable_spatial_resolver: bool = True,
        prompt_generator: Optional[Callable[..., Tuple[str, str]]] = None,
        compact_json_kwargs: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize Pipeline
        
        Args:
            model: LLM model name
            api_key: API Key (optional)
            base_url: API base URL (optional, OpenRouter uses "https://openrouter.ai/api/v1")
            verbose: Print detailed information
            subscriber: SceneGraphSubscriber instance (real-time mode)
            scene_file: Scene graph file path (file mode, mutually exclusive with subscriber)
            robot_capability: Robot physical capability specification (for physics validation)
            enable_physics_validation: Enable physics constraint validation
            enable_spatial_resolver: Enable automatic spatial reference resolution
            prompt_generator: Custom prompt generator (passed through to LLMPlanner)
            compact_json_kwargs: Extra kwargs for SceneGraph.to_compact_json() (passed through to LLMPlanner)
        """
        self.agent = LLMAgent(model=model, api_key=api_key, base_url=base_url)
        self.planner = LLMPlanner(
            agent=self.agent,
            model=model,
            robot_capability=robot_capability,
            enable_physics_validation=enable_physics_validation,
            enable_spatial_resolver=enable_spatial_resolver,
            prompt_generator=prompt_generator,
            compact_json_kwargs=compact_json_kwargs
        )
        self.verbose = verbose
        
        # Data source
        self._subscriber = subscriber
        self._use_subscriber = subscriber is not None
        
        # Cache
        self._scene_graph: Optional[SceneGraph] = None
        self._last_response: Optional[Dict[str, Any]] = None
        self._previous_scene_hash: Optional[int] = None
        
        # Conversation state
        self._conversation_started: bool = False
        self._info_provided: bool = False  # Track if info was provided via info_request
        self._provided_objects: List[str] = []  # Track which objects info was provided for
        self._pending_response: Optional[str] = None  # Buffered LLM response from handlers
        self._replan_mode: bool = False  # True during replan to preserve conversation history
        # LLM-only time (excl. user clarification / confirmation) for fair comparison with one-shot planners
        self._llm_planning_time_sec: float = 0.0
        
        # Load static scene graph if file path provided
        if scene_file and not self._use_subscriber:
            self.load_scene_graph(scene_file)
    
    # ==================== Scene Graph Management ====================
    
    def load_scene_graph(self, source: Union[str, Path, SceneGraph]) -> SceneGraph:
        """
        Load scene graph
        
        Args:
            source: Scene graph source (file path or SceneGraph object)
            
        Returns:
            SceneGraph object
        """
        if isinstance(source, SceneGraph):
            self._scene_graph = source
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Scene graph file not found: {path}")
            self._scene_graph = load_scene_graph(str(path))
        
        if self.verbose:
            print(f"[Pipeline] Loaded scene graph: {self._scene_graph}")
            print(self._scene_graph.summary())
        
        return self._scene_graph
    
    def _update_scene_graph(self) -> bool:
        """
        Update scene graph (from subscriber or cache)
        
        Returns:
            True if scene graph is available
        """
        if self._use_subscriber:
            latest_sg = self._subscriber.get_latest()
            if latest_sg is not None:
                self._scene_graph = latest_sg
                return True
        return self._scene_graph is not None
    
    def get_scene_graph(self) -> Optional[SceneGraph]:
        """Get current scene graph"""
        return self._scene_graph
    
    def get_compact_json(self) -> str:
        """Get compact JSON of current scene graph"""
        if self._scene_graph is None:
            return "{}"
        return self._scene_graph.to_compact_json()
    
    def get_verbose_description(self) -> str:
        """Get natural language description of current scene graph"""
        if self._scene_graph is None:
            return "No scene graph loaded."
        return self._scene_graph.to_verbose_description()
    
    # ==================== Simple Planning ====================
    
    def run(
        self, 
        instruction: str,
        scene_graph: Optional[Union[str, Path, SceneGraph]] = None
    ) -> TaskSequence:
        """
        Run planning pipeline (non-interactive)
        
        Args:
            instruction: Natural language instruction
            scene_graph: Scene graph source (optional, uses cache if already loaded)
            
        Returns:
            TaskSequence object
        """
        # 1. Load scene graph
        if scene_graph is not None:
            self.load_scene_graph(scene_graph)
        
        if self._scene_graph is None:
            raise ValueError("No scene graph loaded. Call load_scene_graph() first or pass scene_graph.")
        
        if self.verbose:
            print(f"\n[Pipeline] Instruction: {instruction}")
            print(f"[Pipeline] Scene: {len(self._scene_graph.rooms)} rooms, {len(self._scene_graph.objects)} objects")
        
        # 2. Call LLM Planner
        result, response_dict = self.planner.plan(self._scene_graph, instruction)
        self._last_response = response_dict
        
        # Handle different result types
        if isinstance(result, ClarificationRequest):
            if self.verbose:
                print(f"\n[Pipeline] Clarification needed: {result.question}")
            return TaskSequence(task_name="Clarification needed")
        
        if isinstance(result, InfeasiblePlan):
            if self.verbose:
                print(f"\n[Pipeline] Plan infeasible: {result.reason}")
            return TaskSequence(task_name="Infeasible")
        
        task_seq = result
        
        if self.verbose:
            print(f"\n[Pipeline] LLM Response:")
            print(f"  Chain of thought: {response_dict.get('chain_of_thought', 'N/A')[:200]}...")
            print(f"  Plan steps: {len(task_seq.actions)}")
        
        # 3. Post-process
        task_seq = self._post_process(task_seq)
        
        if self.verbose:
            print(f"\n[Pipeline] Final task sequence:")
            print(task_seq.summary())
        
        return task_seq
    
    # ==================== Interactive Planning ====================
    
    def run_interactive(self, initial_instruction: Optional[str] = None, debug: bool = True) -> TaskSequence:
        """
        Interactive planning loop with multi-turn dialogue
        
        Supports:
        - Multi-turn dialogue with conversation memory
        - Automatic spatial resolution
        - Physics validation and replanning
        - User clarification handling
        
        Args:
            initial_instruction: Initial instruction (optional, prompts user if not provided)
            debug: Enable debug mode
            
        Returns:
            Final TaskSequence
        """
        self._llm_planning_time_sec = 0.0
        self._print_header()
        
        # Get initial instruction
        instruction = self._get_initial_instruction(initial_instruction)
        if not instruction:
            return TaskSequence(task_name="Empty")
        
        # Ensure scene graph is available
        if not self._update_scene_graph():
            print("[error] failed to get scene graph")
            return TaskSequence(task_name="Error")
        
        if self.verbose:
            print(f"\n[scene overview] {len(self._scene_graph.rooms)} rooms, "
                  f"{len(self._scene_graph.objects)} objects")
        
        # Initialize conversation
        self._init_conversation(instruction)
        
        # Main loop
        return self._interactive_loop(instruction, debug)
    
    def _print_header(self) -> None:
        """Print interactive mode header"""
        print("=" * 70)
        print("Interactive Planning Mode (Multi-turn Dialogue)")
        print("=" * 70)
    
    def _get_initial_instruction(self, initial_instruction: Optional[str]) -> str:
        """Get initial instruction from parameter or user input"""
        if initial_instruction is None:
            initial_instruction = input("\nplease enter task instruction: ").strip()
        
        if not initial_instruction:
            print("no instruction provided, exiting.")
        
        return initial_instruction
    
    def _init_conversation(self, instruction: str) -> None:
        """Initialize conversation with system prompt"""
        compact_json = self._scene_graph.to_compact_json(**self.planner._compact_json_kwargs)
        system_content, _ = self.planner._prompt_generator(
            compact_json,
            instruction,
            include_example=True
        )
        
        self.planner.init_conversation(system_content)
        self._conversation_started = False
        self._info_provided = False  # Reset info state
        self._provided_objects = []  # Reset provided objects
        self._pending_response = None  # Reset pending response
        print(f"\n[planning] instruction: {instruction}")
    
    def _restart_conversation(self, instruction: str, debug: bool) -> str:
        """Restart conversation with new instruction"""
        self._init_conversation(instruction)
        return self._get_llm_response(instruction, is_first=True, debug=debug)
    
    def _timed_chat(self, prompt: str) -> str:
        """Call LLM and accumulate time for planning_time (LLM-only, excl. user I/O)."""
        t0 = time.time()
        out = self.planner.chat(prompt)
        self._llm_planning_time_sec += time.time() - t0
        return out

    def get_llm_planning_time_sec(self) -> float:
        """Return accumulated LLM-only planning time (excl. user clarification/confirmation)."""
        return getattr(self, "_llm_planning_time_sec", 0.0)

    def _get_llm_response(self, prompt: str, is_first: bool = False, debug: bool = True) -> str:
        """Get LLM response via chat"""
        if is_first:
            compact_json = self._scene_graph.to_compact_json(**self.planner._compact_json_kwargs)
            _, user_prompt = self.planner._prompt_generator(compact_json, prompt, include_example=True)
            prompt = user_prompt
            self._conversation_started = True
        
        print(f"[LLMPlanner] Calling LLM...")
        response_text = self._timed_chat(prompt)
        
        if debug:
            print("\n" + "="*40 + " DEBUG: LLM RESPONSE " + "="*40)
            print(response_text)
            print("="*97 + "\n")
        
        return response_text
    
    def _interactive_loop(self, initial_instruction: str, debug: bool,
                         replan_mode: bool = False) -> TaskSequence:
        """Main interactive planning loop
        
        Args:
            initial_instruction: Task instruction
            debug: Enable debug output
            replan_mode: If True, preserves conversation history (for replanning).
                         In replan mode, _restart_conversation is skipped to keep
                         execution failure/scene change context in the conversation.
        """
        self._replan_mode = replan_mode
        current_instruction = initial_instruction
        
        while True:
            # Use buffered response from handler if available, otherwise call LLM
            if self._pending_response:
                response_text = self._pending_response
                self._pending_response = None
            else:
                is_first = not self._conversation_started
                response_text = self._get_llm_response(
                    current_instruction,
                    is_first=is_first,
                    debug=debug
                )
            
            # Parse response
            try:
                response_dict = self.planner.parse_response(response_text)
                self._last_response = response_dict
            except ValueError as e:
                new_instruction = self._handle_parse_error(e)
                if not new_instruction:
                    return TaskSequence(task_name="Cancelled")
                current_instruction = new_instruction
                continue
            
            # Handle infeasible plan
            if response_dict.get("infeasible_reason"):
                new_instruction = self._handle_infeasible(response_dict, debug)
                if not new_instruction:
                    return TaskSequence(task_name="Cancelled")
                current_instruction = new_instruction
                continue
            
            # Handle info_request (Front-loaded RAG)
            if response_dict.get("info_request", False):
                result = self._handle_info_request(response_dict, debug)
                if result == "continue":
                    continue
                # If result is None, fall through to normal processing
            
            # Handle clarification request
            if response_dict.get("clarification_needed", False):
                result = self._handle_clarification(response_dict, current_instruction, debug)
                if result is None:
                    return TaskSequence(task_name="Cancelled")
                if isinstance(result, str):
                    # Continue with new response
                    continue
                # Otherwise it's handled, continue loop
                continue
            
            # Backend guard: if LLM skipped info_request but plan uses ambiguous objects
            if not self._info_provided:
                ambiguous_ids = self.planner._check_ambiguous_objects(
                    response_dict, self._scene_graph
                )
                if ambiguous_ids:
                    if debug:
                        print(f"[Pipeline] Backend guard: ambiguous objects {ambiguous_ids}")
                    object_info = self.planner._build_object_info(
                        ambiguous_ids, "position", self._scene_graph
                    )
                    guard_prompt = (
                        f"[SYSTEM GUARD] Your plan uses objects that have multiple candidates "
                        f"of the same category in the same room. You MUST consider all of them:\n"
                        f"{object_info}\n\n"
                        f"Please re-evaluate and generate the correct plan (use info_request if needed, or choose based on the positions above)."
                    )
                    response_text = self._timed_chat(guard_prompt)
                    self._pending_response = response_text
                    self._info_provided = True
                    continue
            
            # Convert to TaskSequence
            task_seq = self.planner.convert_to_task_sequence(
                response_dict, 
                self._scene_graph, 
                current_instruction
            )
            
            # Handle empty plan
            if len(task_seq.actions) == 0:
                new_instruction = self._handle_empty_plan(debug)
                if not new_instruction:
                    return TaskSequence(task_name="Cancelled")
                current_instruction = new_instruction
                continue
            
            # Physics validation
            validation_result = self._handle_physics_validation(task_seq, current_instruction, debug)
            if validation_result is None:
                # Validation failed and user wants new instruction
                new_instruction = self._prompt_new_instruction()
                if not new_instruction:
                    return TaskSequence(task_name="Cancelled")
                current_instruction = new_instruction
                continue
            elif isinstance(validation_result, TaskSequence):
                task_seq = validation_result
            
            # Display plan and get confirmation
            self._display_plan(task_seq)
            
            if self._get_user_confirmation():
                print("[executing] starting to execute task sequence...")
                task_seq = self._expand_arrange_actions(task_seq)
                return task_seq
            else:
                new_instruction = self._prompt_new_instruction()
                if not new_instruction:
                    print("exiting planning.")
                    return TaskSequence(task_name="Cancelled")
                current_instruction = new_instruction
                if not self._replan_mode:
                    self._restart_conversation(new_instruction, debug)
                continue
    
    # ==================== Handler Methods ====================
    
    def _handle_parse_error(self, error: Exception) -> Optional[str]:
        """Handle LLM response parse error"""
        print(f"[error] failed to parse LLM response: {error}")
        return self._prompt_new_instruction()
    
    def _handle_infeasible(self, response_dict: Dict[str, Any], debug: bool) -> Optional[str]:
        """Handle infeasible plan response"""
        print(f"\n❌ Task is infeasible: {response_dict.get('infeasible_reason')}")
        if response_dict.get('chain_of_thought'):
            print(f"Reasoning: {response_dict.get('chain_of_thought')}")
        
        new_instruction = self._prompt_new_instruction()
        if new_instruction and not self._replan_mode:
            self._restart_conversation(new_instruction, debug)
        return new_instruction
    
    def _handle_info_request(
        self,
        response_dict: Dict[str, Any],
        debug: bool
    ) -> Optional[str]:
        """
        Handle info request from LLM (Front-loaded RAG)
        
        Args:
            response_dict: Parsed LLM response with info_request=true
            debug: Enable debug output
            
        Returns:
            "continue" to continue the loop, None otherwise
        """
        requested_objects = response_dict.get("requested_objects", [])
        request_type = response_dict.get("request_type", "position")
        reason = response_dict.get("reason", "")
        
        if debug:
            print(f"\n[Pipeline] LLM requested info for: {requested_objects}")
            print(f"[Pipeline] Reason: {reason}")
        
        # Build object info
        object_info = self.planner._build_object_info(
            requested_objects, request_type, self._scene_graph
        )
        
        if debug:
            print(f"[Pipeline] Providing info:\n{object_info}")
        
        # Continue conversation with the info
        continuation_prompt = (
            f"{object_info}\n\n"
            f"Based on the information above, continue planning. "
            f"If multiple candidates still match, use clarification_needed."
        )
        
        response_text = self._get_llm_response(continuation_prompt, is_first=False, debug=debug)
        self._pending_response = response_text  # Store for next loop iteration
        
        # Mark that info was provided for this conversation
        self._info_provided = True
        self._provided_objects = requested_objects
        
        return "continue"
    
    def _handle_clarification(
        self, 
        response_dict: Dict[str, Any], 
        current_instruction: str,
        debug: bool
    ) -> Optional[str]:
        """
        Handle clarification request
        
        Returns:
            None if user cancels, or continues the loop
        """
        question = response_dict.get("question", "")
        candidates = response_dict.get("candidates", [])
        
        # Display candidates to user (spatial ranking is applied inside _display_candidates)
        self._display_candidates(question, candidates, current_instruction)
        
        # Get user answer
        user_answer = input("\nplease answer: ").strip()
        if not user_answer:
            return None
        
        # Build RAG context and continue conversation
        # Check if info was already provided via info_request
        if self._info_provided:
            # Simplified prompt - LLM already has the detailed info
            response_text = self._timed_chat(f"User selected: {user_answer}")
        else:
            # Provide full RAG context (candidates + room centroids)
            rag_context = self._build_rag_context(candidates, current_instruction)
            response_text = self._timed_chat(f"{rag_context}\nUser Answer: {user_answer}")
        
        self._pending_response = response_text  # Store for next loop iteration
        return "continue"
    
    def _try_spatial_resolution(
        self, 
        candidates: List[Dict[str, Any]], 
        instruction: str,
        debug: bool
    ) -> bool:
        """
        DEPRECATED: No longer called. Spatial disambiguation is now handled
        entirely via info_request + LLM reasoning. Kept for backward compatibility.
        """
        return False
    
    def _display_candidates(
        self, 
        question: str, 
        candidates: List[Dict[str, Any]],
        instruction: str
    ) -> None:
        """Display clarification candidates to user"""
        print(f"\n[Robot] {question}")
        
        if not candidates:
            return
        
        print("candidates:")
        
        # Only enrich if not already done (check for position_desc)
        if candidates and not candidates[0].get("position_desc"):
            self.planner.enrich_candidates(candidates, self._scene_graph)
        
        # Rank by distance if spatial resolver available
        if self.planner._spatial_resolver:
            candidates = self.planner._spatial_resolver.rank_candidates_for_display(
                candidates, instruction, self._scene_graph
            )
        
        for i, cand in enumerate(candidates, 1):
            room_id = cand.get('room_id', '')
            room_name = self._scene_graph.get_room(room_id).category if room_id else ''
            print(f"  {i}. {cand.get('category', '')} ({cand.get('object_id', '')}) "
                  f"located in {cand.get('room_id', ''), room_name}")
            if cand.get('position_desc'):
                print(f"     position: {cand['position_desc']}")
            if cand.get('distance_to_reference') is not None:
                print(f"     distance to reference: {cand['distance_to_reference']}m")
            if cand.get('phys_desc'):
                print(f"     physical properties: {cand['phys_desc']}")
            if cand.get('description'):
                print(f"     description: {cand['description']}")
    
    def _build_rag_context(
        self, 
        candidates: List[Dict[str, Any]],
        instruction: str = ""
    ) -> str:
        """Build RAG context from candidates for LLM
        
        Includes candidate details and all room centroids for spatial reasoning.
        
        Args:
            candidates: Enriched candidate objects
            instruction: Original instruction (for context)
        """
        rag_context = "\n[System Info: Detailed Candidate Information]\n"
        for cand in candidates:
            id_str = cand.get('object_id') or cand.get('room_id') or "unknown"
            line = f"- {cand.get('category', 'object')} ({id_str})"
            if 'position_desc' in cand:
                line += f" Position: {cand['position_desc']}"
            if 'phys_desc' in cand:
                line += f", Properties: {cand['phys_desc']}"
            if 'description' in cand:
                line += f", Description: {cand['description']}"
            rag_context += line + "\n"
        
        # Append room centroids for spatial reasoning
        if self._scene_graph:
            rag_context += "\n[Room Centroids]\n"
            for room in self._scene_graph.all_rooms():
                if room.centroid:
                    rag_context += (
                        f"- {room.category} ({room.room_id}) centroid: "
                        f"[{room.centroid[0]:.2f}, {room.centroid[1]:.2f}, {room.centroid[2]:.2f}]\n"
                    )
        
        return rag_context
    
    def _handle_empty_plan(self, debug: bool) -> Optional[str]:
        """Handle empty plan response"""
        print("\n⚠️  Warning: Generated plan is empty (0 actions).")
        print("This may indicate the task is infeasible or needs clarification.")
        
        new_instruction = self._prompt_new_instruction()
        if new_instruction and not self._replan_mode:
            self._restart_conversation(new_instruction, debug)
        return new_instruction
    
    def _handle_physics_validation(
        self, 
        task_seq: TaskSequence, 
        instruction: str,
        debug: bool
    ) -> Optional[TaskSequence]:
        """
        Handle physics validation and replanning
        
        Returns:
            TaskSequence if valid (possibly after replanning), None if failed
        """
        # Use planner's public API for validation
        validation = self.planner.validate_plan(task_seq, self._scene_graph)
        
        if validation.is_valid:
            return task_seq
        
        print(f"\n❌ Physics validation failed: {validation.reason}")
        for violation in validation.violations:
            print(f"  - {violation}")
        
        # Try replanning
        result = self._replan_with_physics_feedback(task_seq, instruction, validation, debug)
        if result is None:
            suggestions = []
            for v in validation.violations:
                s = self.planner.get_physics_suggestion(v, self._scene_graph)
                if s:
                    suggestions.append(s)
            if suggestions:
                print("\n[Robot] Here are some alternatives you can try:")
                for s in suggestions:
                    print(f"  - {s}")
        return result
    
    def _replan_with_physics_feedback(
        self, 
        task_seq: TaskSequence, 
        instruction: str,
        validation: Any,
        debug: bool
    ) -> Optional[TaskSequence]:
        """Replan with physics constraint feedback"""
        constraint_feedback = validation.to_feedback_prompt()
        augmented_prompt = (
            f"{instruction}\n\n"
            f"[CONSTRAINT FEEDBACK]\n{constraint_feedback}\n\n"
            f"Please generate a new plan that avoids the physics constraint violations mentioned above."
        )
        
        print(f"\n[LLMPlanner] Replanning with physics constraints...")
        replan_response = self._timed_chat(augmented_prompt)
        
        if debug:
            print("\n" + "="*40 + " DEBUG: REPLAN RESPONSE " + "="*40)
            print(replan_response)
            print("="*97 + "\n")
        
        try:
            replan_dict = self.planner.parse_response(replan_response)
        except ValueError as e:
            print(f"[LLMPlanner] Replan parse error: {e}")
            return None
        
        # Check if infeasible
        if replan_dict.get("infeasible_reason"):
            print(f"\n❌ Replanning also failed: {replan_dict.get('infeasible_reason')}")
            return None
        
        plan = replan_dict.get("plan", [])
        if not plan:
            reason = replan_dict.get("chain_of_thought", "No plan generated (empty plan).")
            print(f"\n❌ Replanning failed: LLM returned empty plan. {reason}")
            return None

        # Convert to TaskSequence
        new_task_seq = self.planner.convert_to_task_sequence(
            replan_dict, 
            self._scene_graph, 
            instruction
        )
        
        # Validate again
        new_validation = self.planner.validate_plan(new_task_seq, self._scene_graph)
        if not new_validation.is_valid:
            print(f"\n❌ Replanned plan still invalid: {new_validation.reason}")
            return None
        
        print("[LLMPlanner] ✅ Replanning successful! Physics constraints satisfied.")
        return new_task_seq
    
    def _display_plan(self, task_seq: TaskSequence) -> None:
        """Display generated plan to user"""
        print(f"\n[planning completed] generated {len(task_seq.actions)} actions:")
        for i, action in enumerate(task_seq.actions, 1):
            print(f"  {i}. [{action.action_type.value}] {action.description}")
    
    def _get_user_confirmation(self) -> bool:
        """Get user confirmation to execute plan"""
        confirm = input("\nexecute this plan? (y/n): ").strip().lower()
        return confirm == 'y'
    
    def _prompt_new_instruction(self) -> Optional[str]:
        """Prompt user for new instruction"""
        return input("\nplease enter new instruction (or press Enter to exit): ").strip() or None
    
    # ==================== Post-processing ====================
    
    def _post_process(self, task_seq: TaskSequence) -> TaskSequence:
        """
        Post-process task sequence
        
        Currently marks ARRANGE actions for expansion.
        
        Args:
            task_seq: Original task sequence
            
        Returns:
            Processed task sequence
        """
        for action in task_seq.actions:
            if action.action_type == ActionType.ARRANGE:
                action.params["requires_expansion"] = True
        
        return task_seq
    
    def _expand_arrange_actions(self, task_seq: TaskSequence) -> TaskSequence:
        """
        Expand all ARRANGE actions in the task sequence into concrete sub-actions.
        
        Called after user confirms the plan, before returning from _interactive_loop.
        
        Args:
            task_seq: Task sequence potentially containing ARRANGE actions
            
        Returns:
            Task sequence with ARRANGE actions expanded
        """
        expanded_actions = []
        has_expansion = False
        for action in task_seq.actions:
            if action.action_type == ActionType.ARRANGE:
                sub_actions = self.expand_arrange_action(action)
                if len(sub_actions) > 1:
                    print(f"  [arrange] Expanded to {len(sub_actions)} sub-actions")
                    has_expansion = True
                expanded_actions.extend(sub_actions)
            else:
                expanded_actions.append(action)
        
        if has_expansion:
            task_seq.actions = expanded_actions
            print(f"  [arrange] Total actions after expansion: {len(task_seq.actions)}")
        
        return task_seq
    
    def expand_arrange_action(
        self, 
        action: Action,
        offset: float = 0.6,
        distribution: str = "long_sides"
    ) -> List[Action]:
        """
        Expand ARRANGE action to specific movement tasks
        
        Calls existing chair arrangement algorithm.
        
        Args:
            action: ARRANGE type action
            offset: Distance from object to anchor point
            distribution: Distribution method
            
        Returns:
            List of expanded actions
        """
        if action.action_type != ActionType.ARRANGE:
            return [action]
        
        object_category = action.params.get("object_category", "chair")
        room_id = action.params.get("room_id")
        sg = action.params.get("scene_graph", self._scene_graph)
        
        if sg is None:
            print("[Pipeline] Warning: No scene graph for arrange expansion")
            return [action]
        
        try:
            from ..experiments.chair_arrangement import (
                create_arrangement_task_with_hungarian,
                CHAIR_CATEGORIES,
                TABLE_CATEGORIES
            )
            from ..core.task import Position
        except ImportError as e:
            print(f"[Pipeline] Warning: Could not import chair_arrangement: {e}")
            return [action]
        
        # Get objects
        chairs = []
        tables = []
        
        if room_id:
            objects_in_room = sg.get_objects_in_room(room_id)
            for obj in objects_in_room:
                if obj.category.lower() == object_category.lower() or obj.category.lower() in [c.lower() for c in CHAIR_CATEGORIES]:
                    chairs.append(obj)
                if obj.category.lower() in [t.lower() for t in TABLE_CATEGORIES]:
                    tables.append(obj)
        else:
            for cat in CHAIR_CATEGORIES:
                if cat.lower() == object_category.lower() or object_category.lower() in cat.lower():
                    chairs.extend(sg.get_objects_by_category(cat))
            for cat in TABLE_CATEGORIES:
                tables.extend(sg.get_objects_by_category(cat))
        
        if not chairs or not tables:
            print(f"[Pipeline] Warning: No chairs ({len(chairs)}) or tables ({len(tables)}) found for arrangement")
            return [action]
        
        try:
            arrangement_task_seq, target_positions, _ = create_arrangement_task_with_hungarian(
                sg, chairs, tables, offset=offset, distribution=distribution
            )
            return list(arrangement_task_seq.actions)
        except Exception as e:
            print(f"[Pipeline] Warning: Arrangement failed: {e}")
            return [action]
    
    # ==================== Utility Methods ====================
    
    def get_last_response(self) -> Optional[Dict[str, Any]]:
        """Get last LLM response"""
        return self._last_response
    
    def _check_scene_change(self) -> Tuple[bool, List[str]]:
        """
        Simplified scene change detection
        
        Compares current scene graph with previous state.
        
        Returns:
            (has_change, change_descriptions) tuple
        """
        if self._scene_graph is None:
            return False, []
        
        current_objects = set(self._scene_graph.objects.keys())
        current_rooms = {obj_id: obj.room_id 
                        for obj_id, obj in self._scene_graph.objects.items()
                        if obj.room_id is not None}
        current_hash = hash((frozenset(current_objects), frozenset(current_rooms.items())))
        
        if self._previous_scene_hash is None:
            self._previous_scene_hash = current_hash
            return False, []
        
        if current_hash == self._previous_scene_hash:
            return False, []
        
        changes = []
        if len(current_objects) != len(self._scene_graph.objects):
            changes.append(f"Object count changed: {len(self._scene_graph.objects)} objects")
        
        self._previous_scene_hash = current_hash
        return len(changes) > 0, changes
