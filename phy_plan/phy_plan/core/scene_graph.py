"""
scene_graph.py: 统一场景图接口

抽象不同来源（phy_graph JSON / Hydra DSG）的场景图，
提供统一的查询接口供 planner 和 visualization 使用。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class BoundingBox:
    """物体包围盒"""
    min_point: List[float]  # [x, y, z]
    max_point: List[float]  # [x, y, z]
    
    @property
    def center(self) -> List[float]:
        """包围盒中心点"""
        return [
            (self.min_point[0] + self.max_point[0]) / 2,
            (self.min_point[1] + self.max_point[1]) / 2,
            (self.min_point[2] + self.max_point[2]) / 2,
        ]
    
    @property
    def dimensions(self) -> List[float]:
        """包围盒尺寸 [width, depth, height]"""
        return [
            self.max_point[0] - self.min_point[0],
            self.max_point[1] - self.min_point[1],
            self.max_point[2] - self.min_point[2],
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": {"x": self.min_point[0], "y": self.min_point[1], "z": self.min_point[2]},
            "max": {"x": self.max_point[0], "y": self.max_point[1], "z": self.max_point[2]}
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["BoundingBox"]:
        """从字典创建 BoundingBox"""
        if not d or "min" not in d or "max" not in d:
            return None
        min_pt = d["min"]
        max_pt = d["max"]
        return cls(
            min_point=[min_pt["x"], min_pt["y"], min_pt["z"]],
            max_point=[max_pt["x"], max_pt["y"], max_pt["z"]]
        )


@dataclass
class PhysicalProperties:
    """物体物理属性（由 phy_graph 推断）"""
    friction_level: int = 1        # 摩擦等级 0-2
    pushable: bool = True          # 是否可推动
    weight_level: int = 1          # 重量等级 0-2
    description: str = ""          # 物体描述
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicalProperties":
        if not d:
            return cls()
        return cls(
            friction_level=d.get("friction_level", 1),
            pushable=d.get("pushable", True),
            weight_level=d.get("weight_level", 1),
            description=d.get("description", "")
        )


@dataclass
class ObjectNode:
    """物体节点"""
    node_id: str                   # 如 "O(13)"
    category: str                  # 物体类别，如 "chair", "table"
    position: List[float]          # [x, y, z]
    bounding_box: Optional[BoundingBox] = None
    physical_properties: Optional[PhysicalProperties] = None
    room_id: Optional[str] = None  # 所属房间 ID
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "category": self.category,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]}
        }
        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()
        if self.physical_properties:
            result["physical_properties"] = {
                "friction_level": self.physical_properties.friction_level,
                "pushable": self.physical_properties.pushable,
                "weight_level": self.physical_properties.weight_level,
                "description": self.physical_properties.description
            }
        if self.room_id:
            result["room_id"] = self.room_id
        return result
    
    def to_compact(self) -> Dict[str, Any]:
        """转换为 compact 格式（用于 LLM）"""
        return {
            "node_id": self.node_id,
            "category": self.category,
            "room_id": self.room_id
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], room_id: Optional[str] = None) -> "ObjectNode":
        """从 phy_graph JSON 格式创建"""
        pos = d.get("position", {})
        position = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
        
        bbox = None
        if "bounding_box" in d:
            bbox = BoundingBox.from_dict(d["bounding_box"])
        
        phys_props = None
        if "physical_properties" in d:
            phys_props = PhysicalProperties.from_dict(d["physical_properties"])
        
        return cls(
            node_id=d.get("node_id", ""),
            category=d.get("category", "unknown"),
            position=position,
            bounding_box=bbox,
            physical_properties=phys_props,
            room_id=room_id
        )


@dataclass
class RoomNode:
    """房间节点"""
    room_id: str                   # 如 "R(0)"
    category: str                  # 房间类别，如 "LivingRoom", "DiningRoom"
    centroid: Optional[List[float]] = None  # 房间中心 [x, y, z]
    object_ids: List[str] = field(default_factory=list)  # 包含的物体 ID 列表
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "room_id": self.room_id,
            "category": self.category,
            "object_ids": self.object_ids
        }
        if self.centroid:
            result["centroid"] = {"x": self.centroid[0], "y": self.centroid[1], "z": self.centroid[2]}
        if self.description:
            result["description"] = self.description
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoomNode":
        centroid = None
        if "centroid" in d:
            c = d["centroid"]
            centroid = [c.get("x", 0), c.get("y", 0), c.get("z", 0)]
        
        return cls(
            room_id=d.get("room_id", ""),
            category=d.get("category", "Unknown"),
            centroid=centroid,
            object_ids=d.get("object_ids", []),
            description=d.get("description", "")
        )


class SceneGraph:
    """
    统一场景图接口
    
    封装 phy_graph 或 hydra 输出的场景图，提供统一查询接口。
    """
    
    def __init__(self):
        self.objects: Dict[str, ObjectNode] = {}
        self.rooms: Dict[str, RoomNode] = {}
        self.timestamp: str = ""
        self.source: str = ""  # "phy_graph" or "hydra"
        self.metadata: Dict[str, Any] = {}
    
    # ==================== 查询接口 ====================
    
    def get_object(self, node_id: str) -> Optional[ObjectNode]:
        """根据 ID 获取物体"""
        return self.objects.get(node_id)
    
    def get_objects_by_category(self, category: str) -> List[ObjectNode]:
        """根据类别获取物体列表"""
        return [obj for obj in self.objects.values() if obj.category.lower() == category.lower()]
    
    def get_objects_in_room(self, room_id: str) -> List[ObjectNode]:
        """获取房间内的所有物体"""
        room = self.rooms.get(room_id)
        if not room:
            return []
        return [self.objects[oid] for oid in room.object_ids if oid in self.objects]
    
    def get_room(self, room_id: str) -> Optional[RoomNode]:
        """根据 ID 获取房间"""
        return self.rooms.get(room_id)
    
    def get_rooms_by_category(self, category: str) -> List[RoomNode]:
        """根据类别获取房间列表"""
        return [room for room in self.rooms.values() if room.category.lower() == category.lower()]
    
    def all_objects(self) -> List[ObjectNode]:
        """获取所有物体"""
        return list(self.objects.values())
    
    def all_rooms(self) -> List[RoomNode]:
        """获取所有房间"""
        return list(self.rooms.values())
    
    # ==================== 转换接口 ====================
    
    def to_compact_json(self) -> str:
        """
        生成 compact 格式 JSON（用于 LLM 上下文）
        只包含 node_id, category, room_id
        """
        compact = {
            "rooms": [],
            "objects": []
        }
        for room in self.rooms.values():
            compact["rooms"].append({
                "room_id": room.room_id,
                "category": room.category,
                "object_ids": room.object_ids
            })
        for obj in self.objects.values():
            compact["objects"].append(obj.to_compact())
        return json.dumps(compact, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典格式"""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "rooms": [room.to_dict() for room in self.rooms.values()],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    # ==================== 统计信息 ====================
    
    def summary(self) -> str:
        """生成场景图摘要"""
        lines = [
            f"SceneGraph (source: {self.source})",
            f"  Timestamp: {self.timestamp}",
            f"  Rooms: {len(self.rooms)}",
            f"  Objects: {len(self.objects)}",
            "",
            "Categories:"
        ]
        
        # 统计物体类别
        category_counts: Dict[str, int] = {}
        for obj in self.objects.values():
            cat = obj.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {cat}: {count}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"SceneGraph(rooms={len(self.rooms)}, objects={len(self.objects)}, source={self.source})"

