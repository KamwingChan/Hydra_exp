"""
task.py: domain language definition for task planning

define the action types and task sequence structure for robot tasks, used for:
1. structured representation of LLM output
2. input format for task executor
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ActionType(Enum):
    """action type enumeration"""
    # 物体操作
    PICK = "pick"                 # pick object
    PLACE = "place"               # place object on a surface
    MOVE_OBJECT = "move_object"   # move object (combination of pick and place)
    # robot movement
    NAVIGATE = "navigate"         # navigate to a position
    # high-level actions (can be decomposed into low-level action sequences)
    ARRANGE = "arrange"           # arrange objects (e.g. place chairs around tables)
    CLEAN_UP = "clean_up"         # clean up the area
    # perception actions
    OBSERVE = "observe"           # observe/get object information
    LOCATE = "locate"             # locate object


@dataclass
class Position:
    """3D position"""
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
    """single action definition"""
    action_type: ActionType
    target_object: Optional[str] = None      # target object ID, e.g. "O(13)"
    target_position: Optional[Position] = None  # target position
    params: Dict[str, Any] = field(default_factory=dict)  # additional parameters
    description: str = ""                    # action description (for debugging/visualization)
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary format"""
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
        """create Action from dictionary"""
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
    """task sequence: a sequence of ordered actions"""
    actions: List[Action] = field(default_factory=list)
    task_name: str = ""                      # task name
    metadata: Dict[str, Any] = field(default_factory=dict)  # metadata
    
    def add_action(self, action: Action) -> None:
        """add action to sequence"""
        self.actions.append(action)
    
    def add_move_object(
        self,
        object_id: str,
        target_position: Position,
        description: str = ""
    ) -> None:
        """convenient method: add move object action"""
        self.actions.append(Action(
            action_type=ActionType.MOVE_OBJECT,
            target_object=object_id,
            target_position=target_position,
            description=description or f"Move {object_id} to {target_position.to_list()}"
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary format"""
        return {
            "task_name": self.task_name,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSequence":
        """create TaskSequence from dictionary"""
        actions = [Action.from_dict(a) for a in d.get("actions", [])]
        return cls(
            actions=actions,
            task_name=d.get("task_name", ""),
            metadata=d.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "TaskSequence":
        """create TaskSequence from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def __len__(self) -> int:
        return len(self.actions)
    
    def __iter__(self):
        return iter(self.actions)
    
    def summary(self) -> str:
        """generate task sequence summary"""
        lines = [f"Task: {self.task_name}", f"Actions ({len(self.actions)}):"]
        for i, action in enumerate(self.actions, 1):
            lines.append(f"  {i}. [{action.action_type.value}] {action.description}")
        return "\n".join(lines)

