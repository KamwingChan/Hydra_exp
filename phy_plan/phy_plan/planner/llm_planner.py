"""
llm_planner.py: LLM task planner

use LLM to generate task sequence based on scene graph and natural language instruction.
Supports physics-aware planning with validation and replanning.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.agent import LLMAgent
from ..core.scene_graph import SceneGraph
from ..core.task import TaskSequence, Action, ActionType, Position
from ..core.physics_agent import PhysicsAwareAgent, RobotCapability, ValidationResult, ConstraintViolation
from ..prompts.task_planning_prompt import generate_task_planning_prompt
from .spatial_resolver import SpatialResolver


@dataclass
class ClarificationRequest:
    """
    clarification request
    
    when LLM needs user clarification, return this object.
    """
    question: str
    candidates: List[Dict[str, Any]]
    chain_of_thought: str


@dataclass
class InfeasiblePlan:
    """
    Represents a physically infeasible plan
    
    Returned when physics validation fails and replanning also fails.
    """
    reason: str
    chain_of_thought: str
    suggestions: List[str] = field(default_factory=list)


class LLMPlanner:
    """
    LLM task planner with physics-aware validation
    
    流程：
    1. convert scene graph to compact JSON (with physics and position)
    2. generate prompt
    3. call LLM
    4. parse response to TaskSequence
    5. (optional) validate physics constraints
    6. (optional) replan if validation fails
    """
    
    def __init__(
        self, 
        agent: Optional[LLMAgent] = None, 
        model: str = "gpt-4o-mini",
        robot_capability: Optional[RobotCapability] = None,
        enable_physics_validation: bool = True,
        enable_spatial_resolver: bool = True,
        max_replan_attempts: int = 2
    ):
        """
        initialize planner
        
        Args:
            agent: LLM Agent (optional, auto-created if not provided)
            model: model name
            robot_capability: Robot physical capability specification
            enable_physics_validation: Enable physics constraint validation
            enable_spatial_resolver: Enable automatic spatial reasoning
            max_replan_attempts: Maximum replanning attempts on validation failure
        """
        self.agent = agent or LLMAgent(model=model)
        
        # Physics validation
        self._enable_physics = enable_physics_validation
        self._physics_agent = PhysicsAwareAgent(robot_capability) if enable_physics_validation else None
        
        # Spatial reasoning
        self._enable_spatial = enable_spatial_resolver
        self._spatial_resolver = SpatialResolver() if enable_spatial_resolver else None
        
        # Replanning
        self._max_replan_attempts = max_replan_attempts
    
    # ==================== Conversation Management ====================
    
    def init_conversation(self, system_content: str) -> None:
        """
        Initialize a new conversation context
        
        This sets up the agent for multi-turn dialogue by clearing
        previous history and setting the system prompt.
        
        Args:
            system_content: System message defining role and constraints
        """
        self.agent.init_conversation(system_content)
    
    def chat(self, user_prompt: str) -> str:
        """
        Continue a multi-turn conversation
        
        Uses the existing conversation context (system prompt and history)
        to generate a response. Call init_conversation() first to set up
        the conversation context.
        
        Args:
            user_prompt: User message to send
            
        Returns:
            LLM response text
        """
        return self.agent.chat(user_prompt)
    
    def reset_conversation(self) -> None:
        """Clear conversation history"""
        self.agent.reset()
    
    @property
    def conversation_history(self) -> list:
        """Get current conversation history (read-only)"""
        return list(self.agent.messages)
    
    # ==================== Planning Methods ====================
    
    def plan(
        self, 
        scene_graph: SceneGraph, 
        instruction: str,
        include_example: bool = True,
        debug: bool = True,
        validate_physics: bool = True
    ) -> Tuple[Union[TaskSequence, ClarificationRequest, InfeasiblePlan], Dict[str, Any]]:
        """
        generate task sequence or clarification request based on scene graph and instruction
        
        Args:
            scene_graph: scene graph object
            instruction: natural language instruction
            include_example: whether to include example in prompt
            debug: whether to print debug information
            validate_physics: whether to validate physics constraints (uses class setting if True)
            
        Returns:
            (TaskSequence 或 ClarificationRequest 或 InfeasiblePlan, raw_response_dict) 元组
        """
        # 1. generate prompt (compact format to save tokens)
        # Physical validation is done in backend by PhysicsAwareAgent
        # Detailed info is retrieved via candidate enrichment when needed
        compact_json = scene_graph.to_compact_json()
        system_content, user_prompt = generate_task_planning_prompt(
            compact_json, instruction, include_example
        )
        
        # DEBUG: print prompt
        if debug:
            print("\n" + "="*40 + " DEBUG: PROMPT " + "="*40)
            print(f"System Content Preview:\n{system_content[:200]}...")
            print(f"\nUser Prompt:\n{user_prompt}")
            print("="*95 + "\n")
        
        # 2. call LLM
        print(f"[LLMPlanner] Calling {self.agent.model}...")
        response_text = self.agent.llm_call(system_content, user_prompt)
        
        # DEBUG: print Response
        if debug:
            print("\n" + "="*40 + " DEBUG: RESPONSE " + "="*40)
            print(response_text)
            print("="*97 + "\n")
        
        # 3. parse response
        response_dict = self.parse_response(response_text)
        
        # 4. check if LLM already detected infeasibility
        if response_dict.get("infeasible_reason"):
            return InfeasiblePlan(
                reason=response_dict.get("infeasible_reason", ""),
                chain_of_thought=response_dict.get("chain_of_thought", ""),
                suggestions=[]
            ), response_dict
        
        # 5. check if clarification needed
        if response_dict.get("clarification_needed", False):
            candidates = response_dict.get("candidates", [])
            
            # try spatial resolution first
            if self._enable_spatial and self._spatial_resolver:
                resolved_id = self._spatial_resolver.resolve(instruction, candidates, scene_graph)
                if resolved_id:
                    print(f"[LLMPlanner] Spatial resolver selected: {resolved_id}")
                    # replan with resolved object
                    return self._replan_with_resolved_object(
                        scene_graph, instruction, resolved_id, include_example, debug
                    )
                
                # enrich candidates with distance info for display
                candidates = self._spatial_resolver.rank_candidates_for_display(
                    candidates, instruction, scene_graph
                )
            
            # fill in detailed information (coordinates, physical properties)
            self.enrich_candidates(candidates, scene_graph)
            
            clarification = ClarificationRequest(
                question=response_dict.get("question", ""),
                candidates=candidates,
                chain_of_thought=response_dict.get("chain_of_thought", "")
            )
            return clarification, response_dict
        
        # 6. convert to TaskSequence (with detailed information retrieval)
        task_seq = self.convert_to_task_sequence(response_dict, scene_graph, instruction)
        
        # 7. Physics validation (if enabled)
        if validate_physics and self._enable_physics and self._physics_agent:
            result = self._validate_and_replan(task_seq, scene_graph, instruction, debug)
            if result is not None:
                return result, response_dict
        
        return task_seq, response_dict
    
    def validate_plan(
        self,
        task_seq: TaskSequence,
        scene_graph: SceneGraph
    ) -> ValidationResult:
        """
        Validate a task sequence against physics constraints (Public API)
        
        Args:
            task_seq: Task sequence to validate
            scene_graph: Scene graph for context
            
        Returns:
            ValidationResult with is_valid, reason, violations, warnings
        """
        if self._enable_physics and self._physics_agent:
            return self._physics_agent.validate_plan(task_seq, scene_graph)
        return ValidationResult(is_valid=True)

    def get_physics_suggestion(self, violation: ConstraintViolation, scene_graph: SceneGraph) -> Optional[str]:
        """
        Get alternative suggestion for a physics violation (Public API)
        
        Args:
            violation: ConstraintViolation object (not string)
            scene_graph: Scene graph for context
            
        Returns:
            Suggestion string if available, else None
        """
        if self._enable_physics and self._physics_agent:
            return self._physics_agent.suggest_alternative(violation, scene_graph)
        return None
    
    def plan_with_physics_validation(
        self,
        scene_graph: SceneGraph,
        instruction: str,
        debug: bool = True
    ) -> Tuple[Union[TaskSequence, ClarificationRequest, InfeasiblePlan], ValidationResult]:
        """
        Plan with explicit physics validation result
        
        Args:
            scene_graph: Scene graph
            instruction: User instruction
            debug: Enable debug output
            
        Returns:
            (result, validation_result) tuple
        """
        result, response_dict = self.plan(scene_graph, instruction, debug=debug, validate_physics=False)
        
        # If not a TaskSequence, skip validation
        if not isinstance(result, TaskSequence):
            return result, ValidationResult(is_valid=True)
        
        # Validate
        if self._physics_agent:
            validation = self._physics_agent.validate_plan(result, scene_graph)
            
            if not validation.is_valid:
                # Try replanning
                replan_result = self._replan_with_constraint(
                    scene_graph, instruction, validation.to_feedback_prompt(), debug
                )
                if replan_result:
                    return replan_result, validation
                
                # Return infeasible
                suggestions = []
                for v in validation.violations:
                    s = self.get_physics_suggestion(v, scene_graph)
                    if s:
                        suggestions.append(s)
                
                return InfeasiblePlan(
                    reason=validation.reason,
                    chain_of_thought=response_dict.get("chain_of_thought", ""),
                    suggestions=suggestions
                ), validation
            
            return result, validation
        
        return result, ValidationResult(is_valid=True)
    
    def _validate_and_replan(
        self,
        task_seq: TaskSequence,
        scene_graph: SceneGraph,
        instruction: str,
        debug: bool
    ) -> Optional[Union[TaskSequence, InfeasiblePlan]]:
        """
        Validate task sequence and replan if needed
        
        Returns:
            New TaskSequence or InfeasiblePlan if validation fails, None if valid
        """
        validation = self._physics_agent.validate_plan(task_seq, scene_graph)
        
        if validation.is_valid:
            if validation.warnings:
                print(f"[LLMPlanner] Physics warnings: {validation.warnings}")
            return None
        
        print(f"[LLMPlanner] Physics validation failed: {validation.reason}")
        
        # Try replanning with constraint feedback
        for attempt in range(self._max_replan_attempts):
            print(f"[LLMPlanner] Replanning attempt {attempt + 1}/{self._max_replan_attempts}...")
            
            replan_result = self._replan_with_constraint(
                scene_graph, instruction, validation.to_feedback_prompt(), debug
            )
            
            if replan_result is None:
                continue
            
            if isinstance(replan_result, InfeasiblePlan):
                return replan_result
            
            # Validate the new plan
            new_validation = self._physics_agent.validate_plan(replan_result, scene_graph)
            if new_validation.is_valid:
                print("[LLMPlanner] Replanning successful!")
                return replan_result
            
            validation = new_validation
        
        # All replan attempts failed
        suggestions = []
        for v in validation.violations:
            suggestion = self.get_physics_suggestion(v, scene_graph)
            if suggestion:
                suggestions.append(suggestion)
        
        return InfeasiblePlan(
            reason=validation.reason,
            chain_of_thought=f"Replanning failed after {self._max_replan_attempts} attempts",
            suggestions=suggestions
        )
    
    def _replan_with_constraint(
        self,
        scene_graph: SceneGraph,
        instruction: str,
        constraint_feedback: str,
        debug: bool
    ) -> Optional[Union[TaskSequence, InfeasiblePlan]]:
        """
        Replan with physics constraint feedback
        
        Note: constraint_feedback already contains specific physics info
        (e.g., "Object O(4) too heavy, weight_level=2"), so we don't need
        to include full physics in compact JSON.
        """
        # Build augmented instruction with constraint feedback
        augmented_instruction = f"{instruction}\n\n[CONSTRAINT FEEDBACK]\n{constraint_feedback}"
        
        compact_json = scene_graph.to_compact_json()  # Use default compact format
        system_content, user_prompt = generate_task_planning_prompt(
            compact_json, augmented_instruction, include_example=True
        )
        
        if debug:
            print(f"[LLMPlanner] Replanning with constraint: {constraint_feedback[:100]}...")
        
        response_text = self.agent.llm_call(system_content, user_prompt)
        
        try:
            response_dict = self.parse_response(response_text)
        except ValueError as e:
            print(f"[LLMPlanner] Replan parse error: {e}")
            return None
        
        # Check if LLM reports infeasibility
        if response_dict.get("infeasible_reason"):
            return InfeasiblePlan(
                reason=response_dict.get("infeasible_reason", ""),
                chain_of_thought=response_dict.get("chain_of_thought", ""),
                suggestions=[]
            )
        
        # Check if still needs clarification
        if response_dict.get("clarification_needed", False):
            return None  # Cannot resolve in replan
        
        return self.convert_to_task_sequence(response_dict, scene_graph, instruction)
    
    def _replan_with_resolved_object(
        self,
        scene_graph: SceneGraph,
        instruction: str,
        resolved_object_id: str,
        include_example: bool,
        debug: bool
    ) -> Tuple[Union[TaskSequence, ClarificationRequest, InfeasiblePlan], Dict[str, Any]]:
        """
        Replan with spatially resolved object
        """
        obj = scene_graph.get_object(resolved_object_id)
        obj_desc = f"{obj.category} ({resolved_object_id})" if obj else resolved_object_id
        
        augmented_instruction = f"{instruction}\n\n[SPATIAL RESOLUTION] Based on spatial analysis, use object {obj_desc}."
        
        return self.plan(scene_graph, augmented_instruction, include_example, debug, validate_physics=True)
    
    def enrich_candidates(self, candidates: List[Dict[str, Any]], scene_graph: SceneGraph) -> None:
        """
        Fill in candidate object detailed information (Public API, Stage 2 Retrieval)
        
        Defensive filling: only add when attribute exists, ensure it works even when there is no physical property data.
        Also supports filling room information (if there is room ambiguity).
        
        Args:
            candidates: List of candidate dictionaries to enrich (modified in-place)
            scene_graph: Scene graph for retrieving detailed information
        """
        for cand in candidates:
            # === handle object candidates ===
            obj_id = cand.get("object_id")
            if obj_id:
                full_obj = scene_graph.get_object(obj_id)
                if full_obj:
                    # 1. basic filling: coordinates
                    if full_obj.position and len(full_obj.position) >= 3:
                        cand["position_desc"] = f"[{full_obj.position[0]:.2f}, {full_obj.position[1]:.2f}, {full_obj.position[2]:.2f}]"
                    
                    # 2. optional filling: bounding box
                    if full_obj.bounding_box:
                        cand["bounding_box"] = f"[min: {full_obj.bounding_box.min_point}, max: {full_obj.bounding_box.max_point}]"
                    
                    # 3. optional filling: physical properties
                    if full_obj.physical_properties:
                        props = full_obj.physical_properties
                        details = []
                        if props.weight_level is not None:
                            details.append(f"weight level:{props.weight_level}")
                        if props.pushable is not None:
                            details.append(f"pushable: {'yes' if props.pushable else 'no'}")
                        if details:
                            cand["phys_desc"] = ", ".join(details)
                    
                        # 4. optional filling: description
                    if hasattr(full_obj, 'physical_properties') and full_obj.physical_properties and full_obj.physical_properties.description:
                        cand["description"] = full_obj.physical_properties.description
                continue # object handled, skip

            # === handle room candidates (if LLM returns room_id without object_id) ===
            room_id = cand.get("room_id")
            if room_id:
                full_room = scene_graph.get_room(room_id)
                if full_room:
                    # fill in room coordinates
                    if full_room.centroid:
                        cand["position_desc"] = f"[{full_room.centroid[0]:.2f}, {full_room.centroid[1]:.2f}, {full_room.centroid[2]:.2f}]"
                    # fill in description
                    if full_room.description:
                        cand["description"] = full_room.description

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response text to dictionary (Public API)
        
        Args:
            response_text: LLM response text
            
        Returns:
            parsed dictionary
            
        Raises:
            ValueError: if cannot parse JSON
        """
        # 尝试提取 JSON（处理 markdown 代码块）
        json_str = response_text
        
        # 移除 markdown 代码块
        if "```json" in json_str:
            json_str = re.sub(r"```json\s*", "", json_str)
            json_str = re.sub(r"```\s*", "", json_str)
        elif "```" in json_str:
            json_str = re.sub(r"```\s*", "", json_str)
        
        # 提取 JSON 对象
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError(f"No JSON object found in response: {response_text[:200]}")
        
        json_str = json_str[start_idx:end_idx]
        
        # 移除 JSON 中的单行注释（// ...）
        # LLM 有时会在 JSON 中添加注释，但标准 JSON 不支持
        json_str = re.sub(r'//[^\n]*', '', json_str)
        
        # 移除 JSON 中的多行注释（/* ... */）
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 增强错误信息，帮助诊断截断问题
            error_msg = f"Failed to parse JSON: {e}"
            
            # 检查常见的截断模式
            if not json_str.rstrip().endswith('}'):
                error_msg += "\n[HINT] JSON appears to be truncated (missing closing brace)"
                error_msg += "\n[HINT] This may be caused by max_tokens limit. Check agent.py logs for warnings."
            
            # 检查是否还有注释残留
            if '//' in json_str or '/*' in json_str:
                error_msg += "\n[HINT] JSON may contain comments that weren't properly removed"
            
            # 显示问题位置的上下文
            error_position = e.pos if hasattr(e, 'pos') else len(json_str)
            context_start = max(0, error_position - 100)
            context_end = min(len(json_str), error_position + 100)
            error_msg += f"\n\nError context around position {error_position}:\n"
            error_msg += f"...{json_str[context_start:context_end]}..."
            
            raise ValueError(error_msg)
    
    def convert_to_task_sequence(
        self, 
        response_dict: Dict[str, Any],
        scene_graph: SceneGraph,
        instruction: str
    ) -> TaskSequence:
        """
        Convert parsed response to TaskSequence (Public API)
        
        Args:
            response_dict: Parsed LLM response dictionary
            scene_graph: Scene graph for retrieving detailed information
            instruction: Original instruction (used as task name)
            
        Returns:
            TaskSequence object
        """
        task_seq = TaskSequence(
            task_name=instruction[:50] + "..." if len(instruction) > 50 else instruction,
            metadata={
                "chain_of_thought": response_dict.get("chain_of_thought", ""),
                "source": "llm_planner"
            }
        )
        
        plan = response_dict.get("plan", [])
        
        for step in plan:
            action = self._convert_action(step, scene_graph)
            if action:
                task_seq.add_action(action)
        
        return task_seq
    
    def _convert_action(
        self, 
        step: Dict[str, Any],
        scene_graph: SceneGraph
    ) -> Optional[Action]:
        """
        将单个动作步骤转换为 Action 对象
        
        Args:
            step: 动作步骤字典
            scene_graph: 场景图
            
        Returns:
            Action 对象或 None
        """
        action_name = step.get("action", "").lower()
        params = step.get("params", {})
        
        if action_name == "navigate":
            room_id = params.get("room_id", "")
            room = scene_graph.get_room(room_id)
            target_pos = None
            if room and room.centroid:
                target_pos = Position.from_list(room.centroid)
            
            return Action(
                action_type=ActionType.NAVIGATE,
                target_position=target_pos,
                params={"room_id": room_id},
                description=f"Navigate to {room_id}"
            )
        
        elif action_name == "pick":
            object_id = params.get("object_id", "")
            obj = scene_graph.get_object(object_id)
            target_pos = None
            if obj:
                target_pos = Position.from_list(obj.position)
            
            return Action(
                action_type=ActionType.PICK,
                target_object=object_id,
                target_position=target_pos,
                description=f"Pick up {object_id}" + (f" ({obj.category})" if obj else "")
            )
        
        elif action_name == "place":
            object_id = params.get("object_id", "")
            surface_id = params.get("surface_id")  # 优先使用 surface_id
            room_id = params.get("room_id", "")    # 回退到 room_id
            target_pos = None
            description = ""
            
            if surface_id:
                # 放在某个表面物体上（如桌子）
                surface_obj = scene_graph.get_object(surface_id)
                if surface_obj:
                    # 计算物体顶部位置：使用 bounding_box 的最大 z 值
                    if surface_obj.bounding_box:
                        surface_top_z = surface_obj.bounding_box.max_point[2]
                    else:
                        # 无 bbox 时用物体 position 的 z 作为近似
                        surface_top_z = surface_obj.position[2]
                    target_pos = Position(
                        x=surface_obj.position[0],
                        y=surface_obj.position[1],
                        z=surface_top_z
                    )
                    description = f"Place {object_id} on {surface_id} ({surface_obj.category})"
                else:
                    description = f"Place {object_id} on {surface_id}"
            elif room_id:
                # 回退：放在房间质心
                room = scene_graph.get_room(room_id)
                if room and room.centroid:
                    target_pos = Position.from_list(room.centroid)
                description = f"Place {object_id} in {room_id}"
            else:
                description = f"Place {object_id}"
            
            return Action(
                action_type=ActionType.PLACE,
                target_object=object_id,
                target_position=target_pos,
                params={"surface_id": surface_id, "room_id": room_id},
                description=description
            )
        
        elif action_name == "place_inside":
            object_id = params.get("object_id", "")
            container_id = params.get("container_id", "")
            target_pos = None
            description = ""
            
            if container_id:
                container_obj = scene_graph.get_object(container_id)
                if container_obj:
                    # 使用容器中心位置作为目标
                    target_pos = Position.from_list(container_obj.position)
                    description = f"Place {object_id} inside {container_id} ({container_obj.category})"
                else:
                    description = f"Place {object_id} inside {container_id}"
            else:
                description = f"Place {object_id} inside container"
            
            return Action(
                action_type=ActionType.PLACE_INSIDE,
                target_object=object_id,
                target_position=target_pos,
                params={"container_id": container_id},
                description=description
            )
        
        elif action_name == "arrange":
            object_category = params.get("object_category", "")
            room_id = params.get("room_id", "")
            
            return Action(
                action_type=ActionType.ARRANGE,
                params={
                    "object_category": object_category,
                    "room_id": room_id
                },
                description=f"Arrange {object_category} in {room_id}"
            )
        
        elif action_name == "open":
            object_id = params.get("object_id", "")
            obj = scene_graph.get_object(object_id)
            target_pos = None
            if obj and obj.position:
                target_pos = Position.from_list(obj.position)
            
            return Action(
                action_type=ActionType.OPEN,
                target_object=object_id,
                target_position=target_pos,
                params={"object_id": object_id},
                description=f"Open {object_id}" + (f" ({obj.category})" if obj else "")
            )
        
        elif action_name == "close":
            object_id = params.get("object_id", "")
            obj = scene_graph.get_object(object_id)
            target_pos = None
            if obj and obj.position:
                target_pos = Position.from_list(obj.position)
            
            return Action(
                action_type=ActionType.CLOSE,
                target_object=object_id,
                target_position=target_pos,
                params={"object_id": object_id},
                description=f"Close {object_id}" + (f" ({obj.category})" if obj else "")
            )
        
        elif action_name == "observe":
            object_id = params.get("object_id", "")
            obj = scene_graph.get_object(object_id)
            target_pos = None
            if obj and obj.position:
                target_pos = Position.from_list(obj.position)
            
            return Action(
                action_type=ActionType.OBSERVE,
                target_object=object_id,
                target_position=target_pos,
                params={"object_id": object_id},
                description=f"Observe {object_id} to confirm properties" + (f" ({obj.category})" if obj else "")
            )
        
        else:
            print(f"[LLMPlanner] Warning: Unknown action '{action_name}'")
            return None
    
    def plan_with_verbose(
        self, 
        scene_graph: SceneGraph, 
        instruction: str
    ) -> Tuple[Union[TaskSequence, ClarificationRequest], Dict[str, Any], str]:
        """
        规划并返回详细信息（包括 prompt）
        
        用于调试和可视化。
        
        Returns:
            (TaskSequence 或 ClarificationRequest, response_dict, prompt_text) 元组
        """
        compact_json = scene_graph.to_compact_json()
        system_content, user_prompt = generate_task_planning_prompt(
            compact_json, instruction, include_example=True
        )
        
        full_prompt = f"=== SYSTEM ===\n{system_content}\n\n=== USER ===\n{user_prompt}"
        
        result, response_dict = self.plan(scene_graph, instruction)
        
        return result, response_dict, full_prompt
    
    def replan_from_context(
        self,
        conversation_agent: LLMAgent,
        context_message: str,
        scene_graph: SceneGraph,
        debug: bool = True
    ) -> Union[TaskSequence, ClarificationRequest, InfeasiblePlan]:
        """
        Replan using conversation context (NEW for Dynamic Replanning)
        
        This method continues an existing conversation instead of creating a new one,
        allowing the LLM to see execution history and previous failures.
        
        Args:
            conversation_agent: Ongoing conversation agent (with history)
            context_message: Context about failure/change (in ENGLISH)
            scene_graph: Updated scene graph
            debug: Enable debug output
            
        Returns:
            TaskSequence, ClarificationRequest, or InfeasiblePlan
        """
        # Build replan prompt (ensure English for LLM)
        compact_json = scene_graph.to_compact_json()
        
        replan_prompt = f"""{context_message}

Updated Scene Graph:
{compact_json}

Please generate a new plan considering the above context and updated scene."""
        
        if debug:
            print(f"[LLMPlanner] Replanning via conversation...")
            print(f"[LLMPlanner] Context (first 100 chars): {context_message[:100]}...")
        
        # Continue conversation (preserves history)
        response_text = conversation_agent.chat(replan_prompt)
        
        if debug:
            print(f"[LLMPlanner] Replan response received (length: {len(response_text)})")
        
        try:
            response_dict = self.parse_response(response_text)
        except ValueError as e:
            print(f"[LLMPlanner] Failed to parse replan response: {e}")
            return InfeasiblePlan(
                reason=f"Failed to parse LLM response: {str(e)}",
                chain_of_thought="",
                suggestions=[]
            )
        
        # Check if LLM reports infeasibility
        if response_dict.get("infeasible_reason"):
            return InfeasiblePlan(
                reason=response_dict.get("infeasible_reason", ""),
                chain_of_thought=response_dict.get("chain_of_thought", ""),
                suggestions=[]
            )
        
        # Check if clarification needed
        if response_dict.get("clarification_needed", False):
            candidates = response_dict.get("candidates", [])
            self.enrich_candidates(candidates, scene_graph)
            
            return ClarificationRequest(
                question=response_dict.get("question", ""),
                candidates=candidates,
                chain_of_thought=response_dict.get("chain_of_thought", "")
            )
        
        # Convert to TaskSequence
        task_seq = self.convert_to_task_sequence(
            response_dict, scene_graph, "Replanned task"
        )
        
        # Physics validation
        if self._enable_physics and self._physics_agent:
            validation = self._physics_agent.validate_plan(task_seq, scene_graph)
            if not validation.is_valid:
                if debug:
                    print(f"[LLMPlanner] Replan physics validation failed: {validation.reason}")
                return InfeasiblePlan(
                    reason=validation.reason,
                    chain_of_thought=response_dict.get("chain_of_thought", ""),
                    suggestions=[]
                )
        
        return task_seq
