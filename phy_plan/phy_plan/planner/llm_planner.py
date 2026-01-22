"""
llm_planner.py: LLM task planner

use LLM to generate task sequence based on scene graph and natural language instruction.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.agent import LLMAgent
from ..core.scene_graph import SceneGraph
from ..core.task import TaskSequence, Action, ActionType, Position
from ..prompts.task_planning_prompt import generate_task_planning_prompt


@dataclass
class ClarificationRequest:
    """
    clarification request
    
    when LLM needs user clarification, return this object.
    """
    question: str
    candidates: List[Dict[str, Any]]
    chain_of_thought: str


class LLMPlanner:
    """
    LLM task planner
    
    流程：
    1. convert scene graph to compact JSON
    2. generate prompt
    3. call LLM
    4. parse response to TaskSequence
    """
    
    def __init__(self, agent: Optional[LLMAgent] = None, model: str = "gpt-4o-mini"):
        """
        initialize planner
        
        Args:
            agent: LLM Agent (optional, auto-created if not provided)
            model: model name
        """
        self.agent = agent or LLMAgent(model=model)
    
    def plan(
        self, 
        scene_graph: SceneGraph, 
        instruction: str,
        include_example: bool = True,
        debug: bool = True
    ) -> Tuple[Union[TaskSequence, ClarificationRequest], Dict[str, Any]]:
        """
        generate task sequence or clarification request based on scene graph and instruction
        
        Args:
            scene_graph: scene graph object
            instruction: natural language instruction
            include_example: whether to include example in prompt
            debug: whether to print debug information
            
        Returns:
            (TaskSequence 或 ClarificationRequest, raw_response_dict) 元组
        """
        # 1. generate prompt
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
        
        # DEBUG: 打印 Response
        if debug:
            print("\n" + "="*40 + " DEBUG: RESPONSE " + "="*40)
            print(response_text)
            print("="*97 + "\n")
        
        # 3. 解析响应
        response_dict = self._parse_response(response_text)
        
        # 4. check if clarification is needed
        if response_dict.get("clarification_needed", False):
            candidates = response_dict.get("candidates", [])
            # fill in detailed information (coordinates, physical properties)
            self._enrich_candidates(candidates, scene_graph)
            
            clarification = ClarificationRequest(
                question=response_dict.get("question", ""),
                candidates=candidates,
                chain_of_thought=response_dict.get("chain_of_thought", "")
            )
            return clarification, response_dict
        
        # 5. convert to TaskSequence (with detailed information retrieval)
        task_seq = self._convert_to_task_sequence(response_dict, scene_graph, instruction)
        
        return task_seq, response_dict
    
    def _enrich_candidates(self, candidates: List[Dict[str, Any]], scene_graph: SceneGraph) -> None:
        """
        fill in candidate object detailed information (Stage 2 Retrieval)
        
        defensive filling: only add when attribute exists, ensure it works even when there is no physical property data.
        also supports filling room information (if there is room ambiguity).
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

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        parse LLM response text to dictionary
        
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
    
    def _convert_to_task_sequence(
        self, 
        response_dict: Dict[str, Any],
        scene_graph: SceneGraph,
        instruction: str
    ) -> TaskSequence:
        """
        将解析后的响应转换为 TaskSequence
        
        Args:
            response_dict: 解析后的 LLM 响应
            scene_graph: 场景图（用于检索详细信息）
            instruction: 原始指令
            
        Returns:
            TaskSequence 对象
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
                    # 计算物体顶部位置：position + bbox高度/2
                    surface_top_z = surface_obj.position[2]
                    if hasattr(surface_obj, 'bbox') and surface_obj.bbox:
                        surface_top_z += surface_obj.bbox[2] / 2  # bbox[2] 是高度
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
