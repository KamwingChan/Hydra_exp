"""
task.py: 任务规划领域语言定义

定义机器人任务的动作类型和任务序列结构，用于：
1. LLM 输出的结构化表示
2. 任务执行器的输入格式
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ActionType(Enum):
    """动作类型枚举"""
    # 物体操作
    PICK = "pick"                 # 抓取物体
    PLACE = "place"               # 放置物体到指定位置
    MOVE_OBJECT = "move_object"   # 移动物体（pick + place 的组合）
    
    # 机器人移动
    NAVIGATE = "navigate"         # 导航到位置
    
    # 高层动作（可分解为低层动作序列）
    ARRANGE = "arrange"           # 摆放物体（如椅子归位）
    CLEAN_UP = "clean_up"         # 整理区域
    
    # 感知动作
    OBSERVE = "observe"           # 观察/获取物体信息
    LOCATE = "locate"             # 定位物体


@dataclass
class Position:
    """3D 位置"""
    x: float
    y: float
    z: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_list(cls, pos: List[float]) -> "Position":
        return cls(x=pos[0], y=pos[1], z=pos[2] if len(pos) > 2 else 0.0)
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Position":
        return cls(x=d["x"], y=d["y"], z=d.get("z", 0.0))


@dataclass
class Action:
    """单个动作定义"""
    action_type: ActionType
    target_object: Optional[str] = None      # 目标物体 ID，如 "O(13)"
    target_position: Optional[Position] = None  # 目标位置
    params: Dict[str, Any] = field(default_factory=dict)  # 额外参数
    description: str = ""                    # 动作描述（用于调试/可视化）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "action_type": self.action_type.value,
            "description": self.description,
        }
        if self.target_object:
            result["target_object"] = self.target_object
        if self.target_position:
            result["target_position"] = self.target_position.to_dict()
        if self.params:
            result["params"] = self.params
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Action":
        """从字典创建 Action"""
        action_type = ActionType(d["action_type"])
        target_object = d.get("target_object")
        target_position = None
        if "target_position" in d and d["target_position"]:
            tp = d["target_position"]
            if isinstance(tp, list):
                target_position = Position.from_list(tp)
            elif isinstance(tp, dict):
                target_position = Position.from_dict(tp)
        params = d.get("params", {})
        description = d.get("description", "")
        return cls(
            action_type=action_type,
            target_object=target_object,
            target_position=target_position,
            params=params,
            description=description
        )


@dataclass
class TaskSequence:
    """任务序列：一系列有序的动作"""
    actions: List[Action] = field(default_factory=list)
    task_name: str = ""                      # 任务名称
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def add_action(self, action: Action) -> None:
        """添加动作到序列"""
        self.actions.append(action)
    
    def add_move_object(
        self,
        object_id: str,
        target_position: Position,
        description: str = ""
    ) -> None:
        """便捷方法：添加移动物体动作"""
        self.actions.append(Action(
            action_type=ActionType.MOVE_OBJECT,
            target_object=object_id,
            target_position=target_position,
            description=description or f"Move {object_id} to {target_position.to_list()}"
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "task_name": self.task_name,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSequence":
        """从字典创建 TaskSequence"""
        actions = [Action.from_dict(a) for a in d.get("actions", [])]
        return cls(
            actions=actions,
            task_name=d.get("task_name", ""),
            metadata=d.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "TaskSequence":
        """从 JSON 字符串创建 TaskSequence"""
        return cls.from_dict(json.loads(json_str))
    
    def __len__(self) -> int:
        return len(self.actions)
    
    def __iter__(self):
        return iter(self.actions)
    
    def summary(self) -> str:
        """生成任务序列摘要"""
        lines = [f"Task: {self.task_name}", f"Actions ({len(self.actions)}):"]
        for i, action in enumerate(self.actions, 1):
            lines.append(f"  {i}. [{action.action_type.value}] {action.description}")
        return "\n".join(lines)

