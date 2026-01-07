"""
llm_planner_pipeline.py: LLM 规划 Pipeline

整合场景图加载、LLM 规划、任务后处理的完整流程。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.agent import LLMAgent
from ..core.scene_graph import SceneGraph
from ..core.task import TaskSequence, Action, ActionType
from ..input.phy_graph_io import load_scene_graph
from .llm_planner import LLMPlanner


class LLMPlannerPipeline:
    """
    LLM 规划 Pipeline
    
    完整流程：
    1. 加载场景图（从文件或直接传入）
    2. 调用 LLM Planner 生成任务序列
    3. 后处理（如 arrange 动作展开为具体的移动任务）
    """
    
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化 Pipeline
        
        Args:
            model: LLM 模型名称
            api_key: API Key（可选）
            base_url: API base URL（可选，OpenRouter 使用 "https://openrouter.ai/api/v1"）
            verbose: 是否打印详细信息
        """
        self.agent = LLMAgent(model=model, api_key=api_key, base_url=base_url)
        self.planner = LLMPlanner(agent=self.agent, model=model)
        self.verbose = verbose
        
        # 缓存
        self._scene_graph: Optional[SceneGraph] = None
        self._last_response: Optional[Dict[str, Any]] = None
    
    def load_scene_graph(self, source: Union[str, Path, SceneGraph]) -> SceneGraph:
        """
        加载场景图
        
        Args:
            source: 场景图来源（文件路径或 SceneGraph 对象）
            
        Returns:
            SceneGraph 对象
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
    
    def run(
        self, 
        instruction: str,
        scene_graph: Optional[Union[str, Path, SceneGraph]] = None
    ) -> TaskSequence:
        """
        运行规划 Pipeline
        
        Args:
            instruction: 自然语言指令
            scene_graph: 场景图来源（可选，如果已加载则使用缓存）
            
        Returns:
            TaskSequence 对象
        """
        # 1. 加载场景图
        if scene_graph is not None:
            self.load_scene_graph(scene_graph)
        
        if self._scene_graph is None:
            raise ValueError("No scene graph loaded. Call load_scene_graph() first or pass scene_graph.")
        
        if self.verbose:
            print(f"\n[Pipeline] Instruction: {instruction}")
            print(f"[Pipeline] Scene: {len(self._scene_graph.rooms)} rooms, {len(self._scene_graph.objects)} objects")
        
        # 2. 调用 LLM Planner
        task_seq, response_dict = self.planner.plan(self._scene_graph, instruction)
        self._last_response = response_dict
        
        if self.verbose:
            print(f"\n[Pipeline] LLM Response:")
            print(f"  Chain of thought: {response_dict.get('chain_of_thought', 'N/A')[:200]}...")
            print(f"  Plan steps: {len(task_seq.actions)}")
        
        # 3. 后处理（如有需要）
        task_seq = self._post_process(task_seq)
        
        if self.verbose:
            print(f"\n[Pipeline] Final task sequence:")
            print(task_seq.summary())
        
        return task_seq
    
    def _post_process(self, task_seq: TaskSequence) -> TaskSequence:
        """
        后处理任务序列
        
        目前主要处理 ARRANGE 动作（标记为需要进一步展开）
        
        Args:
            task_seq: 原始任务序列
            
        Returns:
            处理后的任务序列
        """
        # 标记 arrange 动作为需要展开
        for action in task_seq.actions:
            if action.action_type == ActionType.ARRANGE:
                action.params["requires_expansion"] = True
        
        return task_seq
    
    def expand_arrange_action(
        self, 
        action: Action,
        offset: float = 0.6,
        distribution: str = "long_sides"
    ) -> List[Action]:
        """
        展开 ARRANGE 动作为具体的移动任务
        
        调用现有的椅子摆放算法。
        
        Args:
            action: ARRANGE 类型的动作
            offset: 物体到锚点的距离
            distribution: 分布方式
            
        Returns:
            展开后的动作列表
        """
        if action.action_type != ActionType.ARRANGE:
            return [action]
        
        # 获取参数
        object_category = action.params.get("object_category", "chair")
        room_id = action.params.get("room_id")
        sg = action.params.get("scene_graph", self._scene_graph)
        
        if sg is None:
            print("[Pipeline] Warning: No scene graph for arrange expansion")
            return [action]
        
        # 导入椅子摆放模块
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
        
        # 获取物体
        chairs = []
        tables = []
        
        if room_id:
            # 只获取指定房间的物体
            objects_in_room = sg.get_objects_in_room(room_id)
            for obj in objects_in_room:
                if obj.category.lower() == object_category.lower() or obj.category.lower() in [c.lower() for c in CHAIR_CATEGORIES]:
                    chairs.append(obj)
                if obj.category.lower() in [t.lower() for t in TABLE_CATEGORIES]:
                    tables.append(obj)
        else:
            # 获取所有物体
            for cat in CHAIR_CATEGORIES:
                if cat.lower() == object_category.lower() or object_category.lower() in cat.lower():
                    chairs.extend(sg.get_objects_by_category(cat))
            for cat in TABLE_CATEGORIES:
                tables.extend(sg.get_objects_by_category(cat))
        
        if not chairs or not tables:
            print(f"[Pipeline] Warning: No chairs ({len(chairs)}) or tables ({len(tables)}) found for arrangement")
            return [action]
        
        # 调用摆放算法
        try:
            arrangement_task_seq, target_positions, _ = create_arrangement_task_with_hungarian(
                sg, chairs, tables, offset=offset, distribution=distribution
            )
            return list(arrangement_task_seq.actions)
        except Exception as e:
            print(f"[Pipeline] Warning: Arrangement failed: {e}")
            return [action]
    
    def get_last_response(self) -> Optional[Dict[str, Any]]:
        """获取最后一次 LLM 响应"""
        return self._last_response
    
    def get_scene_graph(self) -> Optional[SceneGraph]:
        """获取当前场景图"""
        return self._scene_graph
    
    def get_compact_json(self) -> str:
        """获取当前场景图的 compact JSON"""
        if self._scene_graph is None:
            return "{}"
        return self._scene_graph.to_compact_json()
    
    def get_verbose_description(self) -> str:
        """获取当前场景图的自然语言描述"""
        if self._scene_graph is None:
            return "No scene graph loaded."
        return self._scene_graph.to_verbose_description()
