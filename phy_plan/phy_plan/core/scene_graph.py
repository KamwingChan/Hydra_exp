"""
scene_graph.py: unified scene graph interface

abstract the scene graph from different sources (phy_graph JSON / Hydra DSG),
provide a unified query interface for planner and visualization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


@dataclass
class BoundingBox:
    """bounding box of the object"""
    min_point: List[float]  # [x, y, z]
    max_point: List[float]  # [x, y, z]
    
    @property
    def center(self) -> List[float]:
        """bounding box center point"""
        return [
            (self.min_point[0] + self.max_point[0]) / 2,
            (self.min_point[1] + self.max_point[1]) / 2,
            (self.min_point[2] + self.max_point[2]) / 2,
        ]
    
    @property
    def dimensions(self) -> List[float]:
        """bounding box dimensions [width, depth, height]"""
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
        """create BoundingBox from dictionary"""
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
    """physical properties of the object (inferred from phy_graph)"""
    friction_level: int = 1              # friction level 0-2
    pushable: bool = True                # whether the object is pushable
    weight_level: int = 1                # weight level 0-2
    description: str = ""                # object description
    estimated_weight_kg: str = ""        # estimated weight range (e.g. "5-10")
    inference_confidence: int = -1       # inference confidence (image score 0-100)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhysicalProperties":
        if not d:
            return cls()
        return cls(
            friction_level=d.get("friction_level", 1),
            pushable=d.get("pushable", True),
            weight_level=d.get("weight_level", 1),
            description=d.get("description", ""),
            estimated_weight_kg=d.get("estimated_weight_kg", ""),
            inference_confidence=d.get("inference_confidence", -1)
        )


@dataclass
class ObjectNode:
    """object node"""
    node_id: str                   # e.g. "O(13)"
    category: str                  # object category, e.g. "chair", "table"
    position: List[float]          # [x, y, z]
    orientation: Optional[List[float]] = None  # [roll, pitch, yaw]
    bounding_box: Optional[BoundingBox] = None
    physical_properties: Optional[PhysicalProperties] = None
    room_id: Optional[str] = None  # room ID
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "node_id": self.node_id,
            "category": self.category,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]}
        }
        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()
        if self.physical_properties:
            props = {
                "friction_level": self.physical_properties.friction_level,
                "pushable": self.physical_properties.pushable,
                "weight_level": self.physical_properties.weight_level,
                "description": self.physical_properties.description
            }
            if self.physical_properties.estimated_weight_kg:
                props["estimated_weight_kg"] = self.physical_properties.estimated_weight_kg
            if self.physical_properties.inference_confidence >= 0:
                props["inference_confidence"] = self.physical_properties.inference_confidence
            result["physical_properties"] = props
        if self.room_id:
            result["room_id"] = self.room_id
        if self.orientation:
            result["orientation"] = {"roll": self.orientation[0], "pitch": self.orientation[1], "yaw": self.orientation[2]}
        return result
    
    def to_compact(self) -> Dict[str, Any]:
        """convert to compact format (for LLM)"""
        return {
            "node_id": self.node_id,
            "category": self.category,
            "room_id": self.room_id
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], room_id: Optional[str] = None) -> "ObjectNode":
        """create ObjectNode from phy_graph JSON format"""
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
    """room node"""
    room_id: str                   # e.g. "R(0)"
    category: str                  # room category, e.g. "LivingRoom", "DiningRoom"
    centroid: Optional[List[float]] = None  # room center [x, y, z]
    object_ids: List[str] = field(default_factory=list)  # list of object IDs
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

@dataclass
class PlaceGvdNode:
    """place node in GVD"""
    place_id: int                   # e.g. 1             # location category, e.g. "LivingRoom", "DiningRoom"
    centroid: List[float]          # [x, y, z]
    distance: float

@dataclass
class PlaceGvdEdge:
    """place edge in GVD"""
    source_id: int
    target_id: int
    weight: float
    
@dataclass
class PlaceGvdGraph:
    """place graph in GVD"""
    PlaceGvdNodes: List[PlaceGvdNode] = field(default_factory=list)
    PlaceGvdEdges: List[PlaceGvdEdge] = field(default_factory=list)

class SceneGraph:
    """
    unified scene graph interface
    
    encapsulate scene graph output from phy_graph or hydra, provide unified query interface.
    """
    
    def __init__(self):
        self.objects: Dict[str, ObjectNode] = {}
        self.rooms: Dict[str, RoomNode] = {}
        self.gvd_graph: Optional[PlaceGvdGraph] = None
        self.timestamp: str = ""
        self.source: str = ""  # "phy_graph" or "hydra"
        self.metadata: Dict[str, Any] = {}
    
    # ==================== 查询接口 ====================
    
    def get_object(self, node_id: str) -> Optional[ObjectNode]:
        """get object by ID"""
        return self.objects.get(node_id)
    
    def get_objects_by_category(self, category: str) -> List[ObjectNode]:
        """get objects by category"""
        return [obj for obj in self.objects.values() if obj.category.lower() == category.lower()]
    
    def get_objects_in_room(self, room_id: str) -> List[ObjectNode]:
        """get all objects in the room"""
        room = self.rooms.get(room_id)
        if not room:
            return []
        return [self.objects[oid] for oid in room.object_ids if oid in self.objects]
    
    def get_room(self, room_id: str) -> Optional[RoomNode]:
        """get room by ID"""
        return self.rooms.get(room_id)
    
    def get_rooms_by_category(self, category: str) -> List[RoomNode]:
        """get rooms by category"""
        return [room for room in self.rooms.values() if room.category.lower() == category.lower()]
    
    def all_objects(self) -> List[ObjectNode]:
        """get all objects"""
        return list(self.objects.values())
    
    def all_rooms(self) -> List[RoomNode]:
        """get all rooms"""
        return list(self.rooms.values())
    
    def get_gvd_graph(self) -> Optional[PlaceGvdGraph]:
        """get GVD graph"""
        return self.gvd_graph
    # ==================== 转换接口 ====================
    
    def to_compact_json(self) -> str:
        """
        generate compact format JSON (for LLM context)
        only contains node_id, category, room_id
        """
        compact = {
            "rooms": [],
            "objects": []
        }
        for room in self.rooms.values():
            room_data = {
                "room_id": room.room_id,
                "category": room.category,
                "object_ids": room.object_ids
            }
            if room.centroid:
                room_data["centroid"] = {
                    "x": room.centroid[0],
                    "y": room.centroid[1],
                    "z": room.centroid[2]
                }
            compact["rooms"].append(room_data)
        for obj in self.objects.values():
            compact["objects"].append(obj.to_compact())
        return json.dumps(compact, indent=2, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to complete dictionary format"""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "rooms": [room.to_dict() for room in self.rooms.values()],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_verbose_description(self, include_objects: bool = True) -> str:
        """
        generate natural language description (for LLM prompt)
        
        reference context-matters's get_verbose_scene_graph() design.
        
        Args:
            include_objects: whether to include detailed object list
            
        Returns:
            natural language description of the scene
        """
        lines = []
        
        # room overview
        room_labels = [f"{r.category}({r.room_id})" for r in self.rooms.values()]
        if room_labels:
            lines.append(f"The scene contains {len(self.rooms)} rooms: {', '.join(room_labels)}.")
        else:
            lines.append("The scene has no defined rooms.")
        
        # content of each room
        if include_objects:
            for room in self.rooms.values():
                objects_in_room = self.get_objects_in_room(room.room_id)
                if objects_in_room:
                    obj_list = [f"{obj.category}({obj.node_id})" for obj in objects_in_room]
                    lines.append(f"The {room.category}({room.room_id}) contains: {', '.join(obj_list)}.")
                else:
                    lines.append(f"The {room.category}({room.room_id}) has no objects.")
            
            # objects without room assignment
            orphan_objects = [obj for obj in self.objects.values() if not obj.room_id]
            if orphan_objects:
                obj_list = [f"{obj.category}({obj.node_id})" for obj in orphan_objects]
                lines.append(f"Objects without room assignment: {', '.join(obj_list)}.")
        
        # object statistics
        category_counts: Dict[str, int] = {}
        for obj in self.objects.values():
            cat = obj.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if category_counts:
            stats = [f"{count} {cat}(s)" for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])]
            lines.append(f"Total objects: {len(self.objects)} ({', '.join(stats)}).")
        
        return "\n".join(lines)
    
    # ==================== statistics ====================
    
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
        
        # count object categories
        category_counts: Dict[str, int] = {}
        for obj in self.objects.values():
            cat = obj.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {cat}: {count}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"SceneGraph(rooms={len(self.rooms)}, objects={len(self.objects)}, source={self.source})"

