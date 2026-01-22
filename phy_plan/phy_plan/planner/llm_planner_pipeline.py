"""
llm_planner_pipeline.py: LLM planning pipeline

integrate scene graph loading, LLM planning, and task post-processing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.agent import LLMAgent
from ..core.scene_graph import SceneGraph
from ..core.task import TaskSequence, Action, ActionType
from ..input.phy_graph_io import load_scene_graph
from .llm_planner import LLMPlanner, ClarificationRequest


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
        verbose: bool = True,
        subscriber: Optional[Any] = None,
        scene_file: Optional[Union[str, Path]] = None
    ):
        """
        初始化 Pipeline
        
        Args:
            model: LLM 模型名称
            api_key: API Key（可选）
            base_url: API base URL（可选，OpenRouter 使用 "https://openrouter.ai/api/v1"）
            verbose: 是否打印详细信息
            subscriber: SceneGraphSubscriber 实例（实时模式）
            scene_file: 场景图文件路径（文件模式，与 subscriber 互斥）
        """
        self.agent = LLMAgent(model=model, api_key=api_key, base_url=base_url)
        self.planner = LLMPlanner(agent=self.agent, model=model)
        self.verbose = verbose
        
        # 数据源
        self._subscriber = subscriber
        self._use_subscriber = subscriber is not None
        
        # 缓存
        self._scene_graph: Optional[SceneGraph] = None
        self._last_response: Optional[Dict[str, Any]] = None
        self._previous_scene_hash: Optional[int] = None
        
        # 如果提供了文件路径，加载静态场景图
        if scene_file and not self._use_subscriber:
            self.load_scene_graph(scene_file)
    
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

    def _update_scene_graph(self) -> bool:
        """
        更新场景图（从订阅器或缓存）
        
        Returns:
            如果场景图已更新返回 True
        """
        if self._use_subscriber:
            latest_sg = self._subscriber.get_latest()
            if latest_sg is not None:
                self._scene_graph = latest_sg
                return True
        return self._scene_graph is not None
    
    def _check_scene_change(self) -> tuple[bool, List[str]]:
        """
        简化版变化检测
        
        比较当前场景图与上次的差异。
        
        Returns:
            (has_change, change_descriptions) 元组
        """
        if self._scene_graph is None:
            return False, []
        
        # 计算当前场景图的哈希
        current_objects = set(self._scene_graph.objects.keys())
        current_rooms = {obj_id: obj.room_id 
                        for obj_id, obj in self._scene_graph.objects.items()
                        if obj.room_id is not None}
        current_hash = hash((frozenset(current_objects), frozenset(current_rooms.items())))
        
        # 首次检查
        if self._previous_scene_hash is None:
            self._previous_scene_hash = current_hash
            return False, []
        
        # 比较哈希
        if current_hash == self._previous_scene_hash:
            return False, []
        
        # 检测变化
        changes = []
        
        # 简化实现：只检测物体数量变化
        if len(current_objects) != len(self._scene_graph.objects):
            changes.append(f"物体数量变化: {len(self._scene_graph.objects)} 个物体")
        
        self._previous_scene_hash = current_hash
        return len(changes) > 0, changes
    
    def run_interactive(self, initial_instruction: Optional[str] = None, debug: bool = True) -> TaskSequence:
        """
        交互式规划循环（使用真正的多轮对话）
        
        支持多轮对话和环境变化检测。
        
        Args:
            initial_instruction: 初始指令（可选，如果不提供则提示用户输入）
            debug: 是否开启调试模式
            
        Returns:
            最终的 TaskSequence
        """
        print("=" * 70)
        print("Interactive Planning Mode (Multi-turn Dialogue)")
        print("=" * 70)
        
        # 获取初始指令
        if initial_instruction is None:
            initial_instruction = input("\nplease enter task instruction: ").strip()
        
        if not initial_instruction:
            print("no instruction provided, exiting.")
            return TaskSequence(task_name="Empty")
        
        # 1. 更新场景图
        if not self._update_scene_graph():
            print("[error] failed to get scene graph")
            return TaskSequence(task_name="Error")
        
        # 2. 初始化对话（只做一次，包含 few-shot 示例）
        from ..prompts.task_planning_prompt import generate_task_planning_prompt
        
        compact_json = self._scene_graph.to_compact_json()
        system_content, user_prompt = generate_task_planning_prompt(
            compact_json, 
            initial_instruction,
            include_example=True  # 包含 few-shot 示例，提升 LLM 性能
        )
        
        self.planner.agent.init_conversation(system_content)
        
        if self.verbose:
            print(f"\n[scene overview] {len(self._scene_graph.rooms)} rooms, "
                  f"{len(self._scene_graph.objects)} objects")
        
        print(f"\n[planning] instruction: {initial_instruction}")
        print(f"[LLMPlanner] Calling {self.planner.agent.model}...")
        
        # 4. 对话循环
        response_text = self.planner.agent.chat(user_prompt)
        
        # DEBUG: 打印初始响应
        if debug:
            print("\n" + "="*40 + " DEBUG: LLM RESPONSE " + "="*40)
            print(response_text)
            print("="*97 + "\n")
        
        while True:
            # 解析响应
            try:
                response_dict = self.planner._parse_response(response_text)
                self._last_response = response_dict
            except Exception as e:
                print(f"[error] failed to parse LLM response: {e}")
                return TaskSequence(task_name="Error")
            
            # 检查是否需要澄清
            if response_dict.get("clarification_needed", False):
                # 需要澄清
                question = response_dict.get("question", "")
                candidates = response_dict.get("candidates", [])

                print(f"\n[Robot] {question}")
                if candidates:
                    print("candidates:")
                    self.planner._enrich_candidates(candidates, self._scene_graph)
                    for i, candidate in enumerate(candidates, 1):
                        obj_id = candidate.get("object_id", "")
                        category = candidate.get("category", "")
                        room_id = candidate.get("room_id", "")
                        
                        # 基础信息
                        print(f"  {i}. {category} ({obj_id}) located in {room_id}")
                        
                        # 详细信息 (如果有)
                        details_shown = False
                        
                        # 1. 坐标
                        pos_desc = candidate.get("position_desc")
                        if pos_desc:
                            print(f"     position: {pos_desc}")
                            details_shown = True
                            
                        # 2. 物理属性
                        phys_desc = candidate.get("phys_desc")
                        if phys_desc:
                            print(f"     physical properties: {phys_desc}")
                            details_shown = True
                        
                        # 3. 描述
                        description = candidate.get("description")
                        if description:
                            print(f"     description: {description}")
                            details_shown = True
                            
                        if details_shown:
                            print("") # 空行分隔
                
                # 等待用户回答
                user_answer = input("\nplease answer: ").strip()
                if not user_answer:
                    print("no answer provided, exiting.")
                    return TaskSequence(task_name="Cancelled")
                
                # 构造 RAG 上下文反馈给 LLM
                rag_context = "\n[System Info: Detailed Candidate Information]\n"
                for cand in candidates:
                    # 优先使用 object_id，如果是房间则使用 room_id
                    id_str = cand.get('object_id') or cand.get('room_id') or "unknown"
                    cat_str = cand.get('category', 'object')
                    
                    line = f"- {cat_str} ({id_str})"
                    
                    # 添加位置信息
                    if 'position_desc' in cand:
                        line += f" Position: {cand['position_desc']}"
                    
                    # 添加物理属性
                    if 'phys_desc' in cand:
                        line += f", Properties: {cand['phys_desc']}"
                        
                    # 添加描述
                    if 'description' in cand:
                        line += f", Description: {cand['description']}"
                        
                    rag_context += line + "\n"
                
                # 将 RAG 信息注入到用户回答之前
                full_prompt = f"{rag_context}\nUser Instruction: {user_answer}"

                # 继续对话（LLM 会记住之前的候选项）
                print(f"\n[LLMPlanner] Calling {self.planner.agent.model}...")
                response_text = self.planner.agent.chat(full_prompt)
                
                # DEBUG: 打印多轮响应
                if debug:
                    print("\n" + "="*40 + " DEBUG: LLM RESPONSE " + "="*40)
                    print(response_text)
                    print("="*97 + "\n")
                
            else:
                # 生成了计划
                task_seq = self.planner._convert_to_task_sequence(
                    response_dict, 
                    self._scene_graph, 
                    initial_instruction
                )
                
                print(f"\n[planning completed] generated {len(task_seq.actions)} actions:")
                for i, action in enumerate(task_seq.actions, 1):
                    print(f"  {i}. [{action.action_type.value}] {action.description}")
                
                # 询问是否执行
                confirm = input("\nexecute this plan? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("[executing] starting to execute task sequence...")
                    return task_seq
                else:
                    # 用户拒绝，可以重新输入指令
                    new_instruction = input("\nplease enter new instruction (or press Enter to exit): ").strip()
                    if not new_instruction:
                        print("exiting planning.")
                        return task_seq
                    
                    # 重新开始对话
                    compact_json = self._scene_graph.to_compact_json()
                    system_content, user_prompt = generate_task_planning_prompt(
                        compact_json, 
                        new_instruction,
                        include_example=True
                    )
                    self.planner.agent.init_conversation(system_content)
                    
                    print(f"\n[planning] instruction: {new_instruction}")
                    print(f"[LLMPlanner] Calling {self.planner.agent.model}...")
                    response_text = self.planner.agent.chat(user_prompt)