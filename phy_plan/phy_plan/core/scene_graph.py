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
    
    STALE_CONFIDENCE_THRESHOLD: int = 50
    
    @property
    def is_stale(self) -> bool:
        """True when confidence is known but below the reliability threshold."""
        return 0 <= self.inference_confidence < self.STALE_CONFIDENCE_THRESHOLD
    
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
    
    def to_compact(self, include_physics: bool = False, include_position: bool = False) -> Dict[str, Any]:
        """
        Convert to compact format (for LLM)
        
        Default is minimal format to save tokens. Physical validation is done
        in backend by PhysicsAwareAgent, not by LLM.
        
        Args:
            include_physics: Include physical properties (weight_level, pushable)
            include_position: Include object position coordinates
            
        Returns:
            Compact dictionary representation
        """
        result = {
            "node_id": self.node_id,
            "category": self.category,
            "room_id": self.room_id
        }
        
        has_phys = bool(self.physical_properties)
        if has_phys and self.physical_properties.is_stale:
            has_phys = False
        result["has_physics"] = has_phys
        
        # Add position for spatial reasoning (optional, increases token usage)
        if include_position and self.position:
            result["position"] = {
                "x": round(self.position[0], 2),
                "y": round(self.position[1], 2),
                "z": round(self.position[2], 2)
            }
        
        # Add physical properties (optional, increases token usage)
        # Note: Physical validation is done in backend by PhysicsAwareAgent
        if include_physics and self.physical_properties:
            phys = self.physical_properties
            result["physical_properties"] = {
                "weight_level": phys.weight_level,
                "pushable": phys.pushable
            }
            if phys.estimated_weight_kg:
                result["physical_properties"]["estimated_weight_kg"] = phys.estimated_weight_kg
        
        return result
    
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
    bounding_box: Optional[BoundingBox] = None  # room bounding box (from phy_graph)
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
        if self.bounding_box:
            result["bounding_box"] = self.bounding_box.to_dict()
        if self.description:
            result["description"] = self.description
        return result
    
    def get_corner(self, corner_type: str) -> Optional[List[float]]:
        """
        Get a corner position of the room based on bounding box.
        
        Args:
            corner_type: One of "front_left", "front_right", "back_left", "back_right",
                         "center", "front", "back", "left", "right"
                         
        Returns:
            [x, y, z] position or None if bounding_box unavailable
        """
        if not self.bounding_box:
            return self.centroid  # fallback to centroid
        
        min_pt = self.bounding_box.min_point
        max_pt = self.bounding_box.max_point
        center = self.bounding_box.center
        
        # Use floor level (z = min_pt[2]) for corner positions
        z = min_pt[2]
        
        corner_map = {
            "center": center,
            "front_left": [min_pt[0], min_pt[1], z],
            "front_right": [max_pt[0], min_pt[1], z],
            "back_left": [min_pt[0], max_pt[1], z],
            "back_right": [max_pt[0], max_pt[1], z],
            "front": [(min_pt[0] + max_pt[0]) / 2, min_pt[1], z],
            "back": [(min_pt[0] + max_pt[0]) / 2, max_pt[1], z],
            "left": [min_pt[0], (min_pt[1] + max_pt[1]) / 2, z],
            "right": [max_pt[0], (min_pt[1] + max_pt[1]) / 2, z],
        }
        
        return corner_map.get(corner_type.lower(), center)
    
    def point_in_room(self, point: List[float]) -> bool:
        """
        Check if a point is inside the room's bounding box (2D, ignores z).
        
        Args:
            point: [x, y, z] position
            
        Returns:
            True if point is inside the room's 2D bounding box
        """
        if not self.bounding_box:
            return False
        
        min_pt = self.bounding_box.min_point
        max_pt = self.bounding_box.max_point
        
        return (min_pt[0] <= point[0] <= max_pt[0] and
                min_pt[1] <= point[1] <= max_pt[1])
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoomNode":
        centroid = None
        if "centroid" in d:
            c = d["centroid"]
            centroid = [c.get("x", 0), c.get("y", 0), c.get("z", 0)]
        
        # Parse bounding_box (same format as ObjectNode)
        bbox = None
        if "bounding_box" in d:
            bbox = BoundingBox.from_dict(d["bounding_box"])
        
        return cls(
            room_id=d.get("room_id", ""),
            category=d.get("category", "Unknown"),
            centroid=centroid,
            bounding_box=bbox,
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
    
    # ==================== 空间计算接口 ====================
    
    @staticmethod
    def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
        """
        Calculate Euclidean distance between two positions
        
        Args:
            pos1: First position [x, y, z] or [x, y]
            pos2: Second position [x, y, z] or [x, y]
            
        Returns:
            Euclidean distance in meters
        """
        import math
        if len(pos1) < 2 or len(pos2) < 2:
            return float("inf")
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = 0.0
        if len(pos1) > 2 and len(pos2) > 2:
            dz = pos2[2] - pos1[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_distance_to_room(self, obj_id: str, room_id: str) -> Optional[float]:
        """
        Calculate distance from object to room centroid
        
        Args:
            obj_id: Object node ID
            room_id: Room node ID
            
        Returns:
            Distance in meters, or None if positions unavailable
        """
        obj = self.get_object(obj_id)
        room = self.get_room(room_id)
        
        if not obj or not obj.position:
            return None
        if not room or not room.centroid:
            return None
        
        return self.calculate_distance(obj.position, room.centroid)
    
    def get_distance_between_objects(self, obj_id1: str, obj_id2: str) -> Optional[float]:
        """
        Calculate distance between two objects
        
        Args:
            obj_id1: First object node ID
            obj_id2: Second object node ID
            
        Returns:
            Distance in meters, or None if positions unavailable
        """
        obj1 = self.get_object(obj_id1)
        obj2 = self.get_object(obj_id2)
        
        if not obj1 or not obj1.position:
            return None
        if not obj2 or not obj2.position:
            return None
        
        return self.calculate_distance(obj1.position, obj2.position)
    
    def get_nearest_object(
        self,
        reference_point: List[float],
        category: Optional[str] = None,
        room_id: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None
    ) -> Optional[ObjectNode]:
        """
        Find nearest object to a reference point
        
        Args:
            reference_point: [x, y, z] reference position
            category: Optional category filter
            room_id: Optional room filter
            exclude_ids: Object IDs to exclude
            
        Returns:
            Nearest matching object or None
        """
        candidates = self.all_objects()
        
        # Apply filters
        if category:
            category_lower = category.lower()
            candidates = [obj for obj in candidates if obj.category.lower() == category_lower]
        
        if room_id:
            candidates = [obj for obj in candidates if obj.room_id == room_id]
        
        if exclude_ids:
            exclude_set = set(exclude_ids)
            candidates = [obj for obj in candidates if obj.node_id not in exclude_set]
        
        # Find nearest
        nearest = None
        min_distance = float("inf")
        
        for obj in candidates:
            if not obj.position:
                continue
            
            distance = self.calculate_distance(obj.position, reference_point)
            if distance < min_distance:
                min_distance = distance
                nearest = obj
        
        return nearest
    
    def get_objects_sorted_by_distance(
        self,
        reference_point: List[float],
        category: Optional[str] = None,
        room_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[tuple]:
        """
        Get objects sorted by distance to reference point
        
        Args:
            reference_point: [x, y, z] reference position
            category: Optional category filter
            room_id: Optional room filter
            limit: Maximum number of results
            
        Returns:
            List of (ObjectNode, distance) tuples sorted by distance
        """
        candidates = self.all_objects()
        
        # Apply filters
        if category:
            category_lower = category.lower()
            candidates = [obj for obj in candidates if obj.category.lower() == category_lower]
        
        if room_id:
            candidates = [obj for obj in candidates if obj.room_id == room_id]
        
        # Calculate distances
        with_distances = []
        for obj in candidates:
            if obj.position:
                distance = self.calculate_distance(obj.position, reference_point)
                with_distances.append((obj, distance))
        
        # Sort by distance
        with_distances.sort(key=lambda x: x[1])
        
        if limit:
            with_distances = with_distances[:limit]
        
        return with_distances
    
    def get_nearest_room(self, position: List[float]) -> Optional[RoomNode]:
        """
        Find nearest room to a position (by centroid)
        
        Args:
            position: [x, y, z] position
            
        Returns:
            Nearest room or None
        """
        nearest = None
        min_distance = float("inf")
        
        for room in self.rooms.values():
            if not room.centroid:
                continue
            
            distance = self.calculate_distance(position, room.centroid)
            if distance < min_distance:
                min_distance = distance
                nearest = room
        
        return nearest
    
    def get_room_corner(self, room_id: str, corner_type: str) -> Optional[List[float]]:
        """
        Get a specific corner/region position of a room.
        
        Args:
            room_id: Room node ID (e.g., "R(0)")
            corner_type: One of "front_left", "front_right", "back_left", "back_right",
                         "center", "front", "back", "left", "right"
                         
        Returns:
            [x, y, z] position or None if room not found or no bounding_box
        """
        room = self.get_room(room_id)
        if not room:
            return None
        return room.get_corner(corner_type)
    
    def get_room_containing_point(self, position: List[float]) -> Optional[RoomNode]:
        """
        Find which room contains a given point (using bounding box).
        
        Args:
            position: [x, y, z] position
            
        Returns:
            RoomNode containing the point, or None if not in any room
        """
        for room in self.rooms.values():
            if room.point_in_room(position):
                return room
        return None
    
    def get_objects_near_corner(
        self,
        room_id: str,
        corner_type: str,
        max_distance: float = 2.0,
        category: Optional[str] = None
    ) -> List[tuple]:
        """
        Find objects near a specific corner of a room.
        
        Args:
            room_id: Room node ID
            corner_type: Type of corner ("front_left", "back_right", etc.)
            max_distance: Maximum distance from corner in meters
            category: Optional category filter
            
        Returns:
            List of (ObjectNode, distance) tuples sorted by distance
        """
        corner_pos = self.get_room_corner(room_id, corner_type)
        if not corner_pos:
            return []
        
        results = self.get_objects_sorted_by_distance(
            reference_point=corner_pos,
            category=category,
            room_id=room_id
        )
        
        # Filter by max distance
        return [(obj, dist) for obj, dist in results if dist <= max_distance]
    
    # ==================== 转换接口 ====================
    
    def to_compact_json(
        self,
        include_physics: bool = False,
        include_position: bool = False,
        include_room_bbox: bool = False
    ) -> str:
        """
        Generate compact format JSON (for LLM context)
        
        Default is minimal format to save tokens. Detailed info is retrieved
        via candidate enrichment (RAG-like mechanism) when needed.
        
        Args:
            include_physics: Include object physical properties (weight_level, pushable)
            include_position: Include object and room positions for spatial reasoning
            include_room_bbox: Include room bounding boxes (for corner/region reasoning)
            
        Returns:
            Compact JSON string suitable for LLM prompt
        """
        compact = {
            "rooms": [],
            "objects": []
        }
        
        # Add rooms with optional centroid and bounding_box
        for room in self.rooms.values():
            room_data = {
                "room_id": room.room_id,
                "category": room.category,
                "object_ids": room.object_ids
            }
            if include_position and room.centroid:
                room_data["centroid"] = {
                    "x": round(room.centroid[0], 2),
                    "y": round(room.centroid[1], 2),
                    "z": round(room.centroid[2], 2)
                }
            if include_room_bbox and room.bounding_box:
                room_data["bounding_box"] = {
                    "min": {
                        "x": round(room.bounding_box.min_point[0], 2),
                        "y": round(room.bounding_box.min_point[1], 2),
                        "z": round(room.bounding_box.min_point[2], 2)
                    },
                    "max": {
                        "x": round(room.bounding_box.max_point[0], 2),
                        "y": round(room.bounding_box.max_point[1], 2),
                        "z": round(room.bounding_box.max_point[2], 2)
                    }
                }
            compact["rooms"].append(room_data)
        
        # Add objects with optional physics and position
        for obj in self.objects.values():
            compact["objects"].append(obj.to_compact(
                include_physics=include_physics,
                include_position=include_position
            ))
        
        try:
            return json.dumps(compact, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
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

