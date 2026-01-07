"""
llm_planner.py: LLM 任务规划器

使用 LLM 根据场景图和自然语言指令生成任务序列。
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..core.agent import LLMAgent
from ..core.scene_graph import SceneGraph
from ..core.task import TaskSequence, Action, ActionType, Position
from ..prompts.task_planning_prompt import generate_task_planning_prompt


class LLMPlanner:
    """
    LLM 任务规划器
    
    流程：
    1. 将场景图转为 compact JSON
    2. 生成 prompt
    3. 调用 LLM
    4. 解析响应为 TaskSequence
    """
    
    def __init__(self, agent: Optional[LLMAgent] = None, model: str = "gpt-4o-mini"):
        """
        初始化规划器
        
        Args:
            agent: LLM Agent（可选，不提供则自动创建）
            model: 模型名称
        """
        self.agent = agent or LLMAgent(model=model)
    
    def plan(
        self, 
        scene_graph: SceneGraph, 
        instruction: str,
        include_example: bool = True
    ) -> Tuple[TaskSequence, Dict[str, Any]]:
        """
        根据场景图和指令生成任务序列
        
        Args:
            scene_graph: 场景图对象
            instruction: 自然语言指令
            include_example: 是否在 prompt 中包含示例
            
        Returns:
            (TaskSequence, raw_response_dict) 元组
        """
        # 1. 生成 prompt
        compact_json = scene_graph.to_compact_json()
        system_content, user_prompt = generate_task_planning_prompt(
            compact_json, instruction, include_example
        )
        
        # 2. 调用 LLM
        print(f"[LLMPlanner] Calling {self.agent.model}...")
        response_text = self.agent.llm_call(system_content, user_prompt)
        
        # 3. 解析响应
        response_dict = self._parse_response(response_text)
        
        # 4. 转换为 TaskSequence（带详细信息检索）
        task_seq = self._convert_to_task_sequence(response_dict, scene_graph, instruction)
        
        return task_seq, response_dict
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析 LLM 响应文本为字典
        
        Args:
            response_text: LLM 响应文本
            
        Returns:
            解析后的字典
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
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {json_str[:500]}")
    
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
    ) -> Tuple[TaskSequence, Dict[str, Any], str]:
        """
        规划并返回详细信息（包括 prompt）
        
        用于调试和可视化。
        
        Returns:
            (TaskSequence, response_dict, prompt_text) 元组
        """
        compact_json = scene_graph.to_compact_json()
        system_content, user_prompt = generate_task_planning_prompt(
            compact_json, instruction, include_example=True
        )
        
        full_prompt = f"=== SYSTEM ===\n{system_content}\n\n=== USER ===\n{user_prompt}"
        
        task_seq, response_dict = self.plan(scene_graph, instruction)
        
        return task_seq, response_dict, full_prompt
