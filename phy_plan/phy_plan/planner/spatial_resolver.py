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
        Attempt to resolve ambiguous candidates using spatial reasoning
        
        Args:
            instruction: User instruction containing spatial reference
            candidates: List of candidate objects (from LLM clarification)
            scene_graph: Scene graph with position data
            
        Returns:
            Selected object_id if spatial resolution succeeds, None otherwise
        """
        # Parse spatial reference from instruction
        spatial_ref = self._parse_spatial_reference(instruction)
        if not spatial_ref:
            return None
        
        # Resolve reference target to scene graph entity
        ref_point = self._get_reference_point(spatial_ref, scene_graph)
        if not ref_point:
            return None
        
        # Get candidate object IDs
        candidate_ids = [c.get("object_id") for c in candidates if c.get("object_id")]
        if not candidate_ids:
            return None
        
        # Rank candidates by distance
        ranked = self.rank_by_distance(candidate_ids, ref_point, scene_graph)
        if not ranked:
            return None
        
        # Select based on reference type
        if spatial_ref.reference_type in ["closest", "nearest", "inside"]:
            return ranked[0].object_id  # Closest
        elif spatial_ref.reference_type in ["farthest", "furthest"]:
            return ranked[-1].object_id  # Farthest
        else:
            # For directional references, use the closest as default
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
