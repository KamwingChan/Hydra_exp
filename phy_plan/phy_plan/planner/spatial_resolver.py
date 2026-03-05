"""
spatial_resolver.py: Spatial reasoning for automatic ambiguity resolution

Resolves spatial references in instructions (e.g., "closest to", "nearest", "left of")
by computing distances and spatial relationships using scene graph coordinates.

This allows the planner to automatically select objects without user clarification
when spatial references provide sufficient information.
"""

import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.scene_graph import SceneGraph, ObjectNode, RoomNode


@dataclass
class SpatialReference:
    """Parsed spatial reference from instruction"""
    reference_type: str  # "closest", "nearest", "farthest", "left_of", "right_of", etc.
    reference_target: str  # The target of the reference (room name, object, etc.)
    reference_id: Optional[str] = None  # Resolved room/object ID if found


@dataclass
class RankedCandidate:
    """Candidate object with computed distance/score"""
    object_id: str
    category: str
    room_id: Optional[str]
    distance: float
    position: List[float]
    score: float = 0.0  # Higher is better for selection
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "room_id": self.room_id,
            "distance": round(self.distance, 2),
            "position": [round(p, 2) for p in self.position]
        }


class SpatialResolver:
    """
    Spatial reasoning resolver for automatic object selection
    
    Parses spatial references in instructions and uses scene graph
    coordinates to automatically resolve ambiguous object references.
    """
    
    # Spatial reference patterns (English and Chinese)
    SPATIAL_PATTERNS = {
        "closest": [
            r"closest\s+to\s+(?:the\s+)?(.+)",
            r"nearest\s+to\s+(?:the\s+)?(.+)",
            r"最近的?(?:的|到)?(.+)",
            r"靠近(.+)的",
            r"离(.+)最近",
        ],
        "farthest": [
            r"farthest\s+from\s+(?:the\s+)?(.+)",
            r"furthest\s+from\s+(?:the\s+)?(.+)",
            r"最远的?(?:的|到)?(.+)",
            r"离(.+)最远",
        ],
        "left_of": [
            r"(?:to\s+the\s+)?left\s+of\s+(?:the\s+)?(.+)",
            r"(.+)的?左边",
            r"(.+)左侧",
        ],
        "right_of": [
            r"(?:to\s+the\s+)?right\s+of\s+(?:the\s+)?(.+)",
            r"(.+)的?右边",
            r"(.+)右侧",
        ],
        "in_front_of": [
            r"in\s+front\s+of\s+(?:the\s+)?(.+)",
            r"(.+)的?前面",
            r"(.+)前方",
        ],
        "behind": [
            r"behind\s+(?:the\s+)?(.+)",
            r"(.+)的?后面",
            r"(.+)后方",
        ],
        "inside": [
            r"inside\s+(?:the\s+)?(.+)",
            r"in\s+(?:the\s+)?(.+)",
            r"(.+)里面的?",
            r"(.+)内的?",
        ]
    }
    
    # Room corner/region patterns (for target locations)
    CORNER_PATTERNS = {
        "corner": [
            r"(?:the\s+)?corner(?:\s+of)?",
            r"角落",
            r"墙角",
        ],
        "front_left": [
            r"front[\s-]?left\s+corner",
            r"左前角",
            r"前左角",
        ],
        "front_right": [
            r"front[\s-]?right\s+corner",
            r"右前角",
            r"前右角",
        ],
        "back_left": [
            r"back[\s-]?left\s+corner",
            r"rear[\s-]?left\s+corner",
            r"左后角",
            r"后左角",
        ],
        "back_right": [
            r"back[\s-]?right\s+corner",
            r"rear[\s-]?right\s+corner",
            r"右后角",
            r"后右角",
        ],
        "center": [
            r"(?:the\s+)?center(?:\s+of)?",
            r"(?:the\s+)?middle(?:\s+of)?",
            r"中间",
            r"中心",
            r"中央",
        ],
        "front": [
            r"(?:the\s+)?front(?:\s+of)?",
            r"前面",
            r"前方",
        ],
        "back": [
            r"(?:the\s+)?back(?:\s+of)?",
            r"(?:the\s+)?rear(?:\s+of)?",
            r"后面",
            r"后方",
        ],
        "left": [
            r"(?:the\s+)?left\s+side",
            r"左边",
            r"左侧",
        ],
        "right": [
            r"(?:the\s+)?right\s+side",
            r"右边",
            r"右侧",
        ],
    }
    
    def __init__(self):
        """Initialize spatial resolver"""
        pass
    
    def resolve(
        self,
        instruction: str,
        candidates: List[Dict[str, Any]],
        scene_graph: SceneGraph
    ) -> Optional[str]:
        """
        DEPRECATED: No longer called by the planning pipeline. Spatial disambiguation
        is now handled entirely via info_request + LLM reasoning, which can consider
        multiple dimensions (color, size, etc.) beyond just distance.
        
        Kept for backward compatibility and potential utility usage.
        """
        spatial_ref = self._parse_spatial_reference(instruction)
        if not spatial_ref:
            return None
        
        ref_point = self._get_reference_point(spatial_ref, scene_graph)
        if not ref_point:
            return None
        
        candidate_ids = [c.get("object_id") for c in candidates if c.get("object_id")]
        if not candidate_ids:
            return None
        
        ranked = self.rank_by_distance(candidate_ids, ref_point, scene_graph)
        if not ranked:
            return None
        
        if spatial_ref.reference_type in ["closest", "nearest", "inside"]:
            return ranked[0].object_id
        elif spatial_ref.reference_type in ["farthest", "furthest"]:
            return ranked[-1].object_id
        else:
            return ranked[0].object_id
    
    def has_spatial_reference(self, instruction: str) -> bool:
        """
        Check if instruction contains spatial reference
        
        Args:
            instruction: User instruction
            
        Returns:
            True if spatial reference detected
        """
        return self._parse_spatial_reference(instruction) is not None
    
    def rank_by_distance(
        self,
        object_ids: List[str],
        reference_point: List[float],
        scene_graph: SceneGraph
    ) -> List[RankedCandidate]:
        """
        Rank objects by distance to reference point
        
        Args:
            object_ids: List of object IDs to rank
            reference_point: [x, y, z] reference position
            scene_graph: Scene graph for object positions
            
        Returns:
            List of RankedCandidate sorted by distance (closest first)
        """
        ranked = []
        
        for obj_id in object_ids:
            obj = scene_graph.get_object(obj_id)
            if not obj or not obj.position:
                continue
            
            distance = self._calculate_distance(obj.position, reference_point)
            ranked.append(RankedCandidate(
                object_id=obj_id,
                category=obj.category,
                room_id=obj.room_id,
                distance=distance,
                position=obj.position,
                score=1.0 / (distance + 0.001)  # Inverse distance as score
            ))
        
        # Sort by distance (ascending)
        ranked.sort(key=lambda x: x.distance)
        return ranked
    
    def rank_candidates_for_display(
        self,
        candidates: List[Dict[str, Any]],
        instruction: str,
        scene_graph: SceneGraph
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates for display to user, adding distance info
        
        Args:
            candidates: List of candidate objects
            instruction: User instruction
            scene_graph: Scene graph
            
        Returns:
            Candidates with added distance/ranking info
        """
        spatial_ref = self._parse_spatial_reference(instruction)
        if not spatial_ref:
            return candidates
        
        ref_point = self._get_reference_point(spatial_ref, scene_graph)
        if not ref_point:
            return candidates
        
        # Add distance info to candidates
        enriched = []
        for cand in candidates:
            obj_id = cand.get("object_id")
            if not obj_id:
                enriched.append(cand)
                continue
            
            obj = scene_graph.get_object(obj_id)
            if not obj or not obj.position:
                enriched.append(cand)
                continue
            
            distance = self._calculate_distance(obj.position, ref_point)
            enriched_cand = dict(cand)
            enriched_cand["distance_to_reference"] = round(distance, 2)
            enriched_cand["position"] = {
                "x": round(obj.position[0], 2),
                "y": round(obj.position[1], 2),
                "z": round(obj.position[2], 2) if len(obj.position) > 2 else 0.0
            }
            enriched.append(enriched_cand)
        
        # Sort by distance
        enriched.sort(key=lambda x: x.get("distance_to_reference", float("inf")))
        
        return enriched
    
    def _parse_spatial_reference(self, instruction: str) -> Optional[SpatialReference]:
        """Parse spatial reference from instruction"""
        instruction_lower = instruction.lower()
        
        for ref_type, patterns in self.SPATIAL_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, instruction_lower)
                if match:
                    target = match.group(1).strip()
                    return SpatialReference(
                        reference_type=ref_type,
                        reference_target=target
                    )
        
        return None
    
    def _get_reference_point(
        self,
        spatial_ref: SpatialReference,
        scene_graph: SceneGraph
    ) -> Optional[List[float]]:
        """Get reference point coordinates from scene graph"""
        target = spatial_ref.reference_target.lower()
        
        # Try to match room by ID pattern (e.g., "R(1)", "r(1)")
        room_id_match = re.search(r'r\((\d+)\)', target)
        if room_id_match:
            room_id = f"R({room_id_match.group(1)})"
            room = scene_graph.get_room(room_id)
            if room and room.centroid:
                spatial_ref.reference_id = room_id
                return room.centroid
        
        # Try to match room by category name
        for room in scene_graph.all_rooms():
            room_category_lower = room.category.lower()
            # Match various formats: "conference room", "conferenceroom", "会议室"
            if (room_category_lower in target or 
                target in room_category_lower or
                room_category_lower.replace(" ", "") in target.replace(" ", "")):
                if room.centroid:
                    spatial_ref.reference_id = room.room_id
                    return room.centroid
        
        # Try to match object by ID pattern (e.g., "O(5)")
        obj_id_match = re.search(r'o\((\d+)\)', target)
        if obj_id_match:
            obj_id = f"O({obj_id_match.group(1)})"
            obj = scene_graph.get_object(obj_id)
            if obj and obj.position:
                spatial_ref.reference_id = obj_id
                return obj.position
        
        # Try to match object by category
        for obj in scene_graph.all_objects():
            obj_category_lower = obj.category.lower()
            if obj_category_lower in target or target in obj_category_lower:
                if obj.position:
                    spatial_ref.reference_id = obj.node_id
                    return obj.position
        
        return None
    
    @staticmethod
    def _calculate_distance(pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two positions"""
        if len(pos1) < 2 or len(pos2) < 2:
            return float("inf")
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = 0.0
        if len(pos1) > 2 and len(pos2) > 2:
            dz = pos2[2] - pos1[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_objects_in_room(
        self,
        room_id: str,
        category: Optional[str],
        scene_graph: SceneGraph
    ) -> List[ObjectNode]:
        """
        Get objects in a specific room, optionally filtered by category
        
        Args:
            room_id: Target room ID
            category: Optional category filter
            scene_graph: Scene graph
            
        Returns:
            List of matching objects
        """
        objects = scene_graph.get_objects_in_room(room_id)
        
        if category:
            category_lower = category.lower()
            objects = [obj for obj in objects if obj.category.lower() == category_lower]
        
        return objects
    
    def find_nearest_object(
        self,
        category: str,
        reference_point: List[float],
        scene_graph: SceneGraph,
        room_id: Optional[str] = None
    ) -> Optional[ObjectNode]:
        """
        Find the nearest object of a category to a reference point
        
        Args:
            category: Object category to search
            reference_point: Reference position
            scene_graph: Scene graph
            room_id: Optional room constraint
            
        Returns:
            Nearest matching object or None
        """
        if room_id:
            candidates = self.get_objects_in_room(room_id, category, scene_graph)
        else:
            candidates = scene_graph.get_objects_by_category(category)
        
        if not candidates:
            return None
        
        # Find nearest
        nearest = None
        min_distance = float("inf")
        
        for obj in candidates:
            if not obj.position:
                continue
            
            distance = self._calculate_distance(obj.position, reference_point)
            if distance < min_distance:
                min_distance = distance
                nearest = obj
        
        return nearest
    
    def parse_corner_reference(self, instruction: str) -> Optional[Tuple[str, str]]:
        """
        Parse room corner/region reference from instruction
        
        Args:
            instruction: User instruction (e.g., "move to the corner of the office")
            
        Returns:
            Tuple of (corner_type, room_reference) or None if not found
        """
        instruction_lower = instruction.lower()
        
        # First find corner type
        corner_type = None
        for ctype, patterns in self.CORNER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    corner_type = ctype
                    break
            if corner_type:
                break
        
        if not corner_type:
            return None
        
        # Then find room reference
        # Common patterns: "corner of the kitchen", "客厅的角落"
        room_patterns = [
            r"corner\s+of\s+(?:the\s+)?(.+?)(?:\s|$)",
            r"center\s+of\s+(?:the\s+)?(.+?)(?:\s|$)",
            r"middle\s+of\s+(?:the\s+)?(.+?)(?:\s|$)",
            r"(.+?)的?(?:角落|中间|中心|中央)",
            r"到(.+?)(?:的)?(?:角落|中间|中心|前面|后面|左边|右边)",
        ]
        
        room_ref = None
        for pattern in room_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                room_ref = match.group(1).strip()
                break
        
        if not room_ref:
            # If no explicit room, default corner type still useful
            return (corner_type, "")
        
        return (corner_type, room_ref)
    
    def resolve_corner_position(
        self,
        instruction: str,
        scene_graph: SceneGraph,
        default_room_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve corner/region reference to actual position
        
        Args:
            instruction: User instruction containing corner reference
            scene_graph: Scene graph with room bounding boxes
            default_room_id: Default room ID if not specified in instruction
            
        Returns:
            Dict with {"position": [x,y,z], "room_id": str, "corner_type": str}
            or None if cannot resolve
        """
        parsed = self.parse_corner_reference(instruction)
        if not parsed:
            return None
        
        corner_type, room_ref = parsed
        
        # Find room
        room = None
        if room_ref:
            # Try to match by category name
            for r in scene_graph.all_rooms():
                r_cat_lower = r.category.lower()
                room_ref_lower = room_ref.lower()
                if (r_cat_lower in room_ref_lower or 
                    room_ref_lower in r_cat_lower or
                    r_cat_lower.replace(" ", "") == room_ref_lower.replace(" ", "")):
                    room = r
                    break
            
            # Try to match by room ID
            if not room:
                room_id_match = re.search(r'r\((\d+)\)', room_ref.lower())
                if room_id_match:
                    room = scene_graph.get_room(f"R({room_id_match.group(1)})")
        
        # Use default room if not found
        if not room and default_room_id:
            room = scene_graph.get_room(default_room_id)
        
        if not room:
            return None
        
        # Get corner position
        position = room.get_corner(corner_type)
        if not position:
            return None
        
        return {
            "position": position,
            "room_id": room.room_id,
            "corner_type": corner_type
        }
    
    def find_objects_near_corner(
        self,
        instruction: str,
        category: Optional[str],
        scene_graph: SceneGraph,
        max_distance: float = 2.0,
        default_room_id: Optional[str] = None
    ) -> List[Tuple[ObjectNode, float]]:
        """
        Find objects near a corner/region specified in instruction
        
        Args:
            instruction: User instruction with corner reference
            category: Optional object category filter
            scene_graph: Scene graph
            max_distance: Maximum distance from corner (meters)
            default_room_id: Default room if not specified
            
        Returns:
            List of (ObjectNode, distance) tuples sorted by distance
        """
        corner_info = self.resolve_corner_position(
            instruction, scene_graph, default_room_id
        )
        
        if not corner_info:
            return []
        
        position = corner_info["position"]
        room_id = corner_info["room_id"]
        
        # Get objects in room
        objects = scene_graph.get_objects_in_room(room_id)
        
        # Filter by category if specified
        if category:
            cat_lower = category.lower()
            objects = [obj for obj in objects if obj.category.lower() == cat_lower]
        
        # Calculate distances and filter
        results = []
        for obj in objects:
            if not obj.position:
                continue
            dist = self._calculate_distance(obj.position, position)
            if dist <= max_distance:
                results.append((obj, dist))
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results
    
    def suggest_placement_position(
        self,
        instruction: str,
        scene_graph: SceneGraph,
        default_room_id: Optional[str] = None,
        offset_from_wall: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest a placement position based on spatial reference in instruction.
        
        Useful for tasks like "move the sofa to the corner" - returns the
        target position where the object should be placed.
        
        Args:
            instruction: User instruction
            scene_graph: Scene graph with room bounding boxes
            default_room_id: Default room if not specified
            offset_from_wall: Distance to offset from exact corner (meters)
            
        Returns:
            Dict with {"position": [x,y,z], "room_id": str, "description": str}
            or None if cannot determine placement
        """
        corner_info = self.resolve_corner_position(
            instruction, scene_graph, default_room_id
        )
        
        if not corner_info:
            return None
        
        position = corner_info["position"]
        room_id = corner_info["room_id"]
        corner_type = corner_info["corner_type"]
        
        room = scene_graph.get_room(room_id)
        if not room or not room.bounding_box:
            return {
                "position": position,
                "room_id": room_id,
                "description": f"{corner_type} of {room.category if room else 'room'}"
            }
        
        # Offset from walls for corners
        bbox = room.bounding_box
        adjusted_pos = list(position)
        
        # Apply offset based on corner type
        if "left" in corner_type or corner_type == "left":
            adjusted_pos[0] += offset_from_wall
        if "right" in corner_type or corner_type == "right":
            adjusted_pos[0] -= offset_from_wall
        if "front" in corner_type or corner_type == "front":
            adjusted_pos[1] += offset_from_wall
        if "back" in corner_type or corner_type == "back":
            adjusted_pos[1] -= offset_from_wall
        
        return {
            "position": adjusted_pos,
            "room_id": room_id,
            "description": f"{corner_type} of {room.category}",
            "original_corner": position
        }