"""
change_detector.py: Task-relevant scene change detection

Detects changes in the scene graph that may affect the current task execution.
Focuses on task-relevant objects rather than all scene changes.

Change types:
- Object appearance: New objects appear in scene
- Object disappearance: Objects removed from scene  
- Object movement: Objects moved to different rooms
- Position change: Objects moved within same room (significant displacement)

Note on Hydra node_id instability:
  Hydra may assign different node_ids to the same object across scene graph updates.
  This module uses (category + position) matching to identify objects across updates,
  rather than relying solely on node_id.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import math

from .scene_graph import SceneGraph, ObjectNode


class ChangeType(Enum):
    """Type of scene change"""
    OBJECT_APPEARED = "object_appeared"
    OBJECT_DISAPPEARED = "object_disappeared"
    OBJECT_MOVED_ROOM = "object_moved_room"
    OBJECT_POSITION_CHANGED = "object_position_changed"


@dataclass
class ObjectChange:
    """Details of a single object change"""
    object_id: str
    change_type: ChangeType
    category: str = ""
    old_room_id: Optional[str] = None
    new_room_id: Optional[str] = None
    old_position: Optional[List[float]] = None
    new_position: Optional[List[float]] = None
    displacement: float = 0.0  # Distance moved in meters
    
    def __str__(self) -> str:
        if self.change_type == ChangeType.OBJECT_APPEARED:
            return f"物体 {self.object_id} ({self.category}) 出现在 {self.new_room_id}"
        elif self.change_type == ChangeType.OBJECT_DISAPPEARED:
            return f"物体 {self.object_id} ({self.category}) 从 {self.old_room_id} 消失"
        elif self.change_type == ChangeType.OBJECT_MOVED_ROOM:
            return f"物体 {self.object_id} ({self.category}) 从 {self.old_room_id} 移动到 {self.new_room_id}"
        elif self.change_type == ChangeType.OBJECT_POSITION_CHANGED:
            return f"物体 {self.object_id} ({self.category}) 位置变化 {self.displacement:.2f}m"
        return f"物体 {self.object_id} 发生变化"


@dataclass
class ChangeReport:
    """Report of all detected changes"""
    objects_appeared: List[ObjectChange] = field(default_factory=list)
    objects_disappeared: List[ObjectChange] = field(default_factory=list)
    objects_moved_room: List[ObjectChange] = field(default_factory=list)
    objects_position_changed: List[ObjectChange] = field(default_factory=list)
    
    # Task relevance
    task_relevant_changes: List[ObjectChange] = field(default_factory=list)
    is_task_affected: bool = False
    
    @property
    def has_changes(self) -> bool:
        """Check if any changes detected"""
        return bool(
            self.objects_appeared or 
            self.objects_disappeared or 
            self.objects_moved_room or
            self.objects_position_changed
        )
    
    @property
    def all_changes(self) -> List[ObjectChange]:
        """Get all changes as a flat list"""
        return (
            self.objects_appeared + 
            self.objects_disappeared + 
            self.objects_moved_room +
            self.objects_position_changed
        )
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        if not self.has_changes:
            return "场景无变化"
        
        lines = ["场景变化检测:"]
        
        if self.objects_appeared:
            lines.append(f"  新出现物体: {len(self.objects_appeared)}")
            for c in self.objects_appeared[:3]:  # Show first 3
                lines.append(f"    - {c}")
        
        if self.objects_disappeared:
            lines.append(f"  消失物体: {len(self.objects_disappeared)}")
            for c in self.objects_disappeared[:3]:
                lines.append(f"    - {c}")
        
        if self.objects_moved_room:
            lines.append(f"  换房间物体: {len(self.objects_moved_room)}")
            for c in self.objects_moved_room[:3]:
                lines.append(f"    - {c}")
        
        if self.objects_position_changed:
            lines.append(f"  位置变化物体: {len(self.objects_position_changed)}")
            for c in self.objects_position_changed[:3]:
                lines.append(f"    - {c}")
        
        if self.is_task_affected:
            lines.append(f"\n⚠️ 任务相关变化: {len(self.task_relevant_changes)}")
            for c in self.task_relevant_changes:
                lines.append(f"    - {c}")
        
        return "\n".join(lines)
    
    def to_replan_context(self) -> str:
        """Generate context for LLM replanning"""
        if not self.task_relevant_changes:
            return ""
        
        lines = ["检测到环境变化，影响当前任务:"]
        for c in self.task_relevant_changes:
            lines.append(f"- {c}")
        lines.append("\n请根据最新场景图重新规划。")
        
        return "\n".join(lines)


class ChangeDetector:
    """
    Task-relevant scene change detector
    
    Only tracks changes to objects that are relevant to the current task,
    ignoring background changes that don't affect execution.
    
    Handles Hydra node_id instability by matching objects via (category + position).
    """
    
    def __init__(
        self,
        task_relevant_objects: Optional[List[str]] = None,
        position_threshold: float = 0.5,  # Minimum displacement to consider as change (meters)
        track_all: bool = False,  # If True, track all objects not just task-relevant
        match_threshold: float = 0.3  # Max distance to consider same object (for node_id instability)
    ):
        """
        Initialize change detector
        
        Args:
            task_relevant_objects: List of object IDs relevant to current task.
                                  If None, tracks all objects.
            position_threshold: Minimum position change to consider significant (meters)
            track_all: If True, detect changes in all objects
            match_threshold: Max distance to match objects with different node_ids (meters).
                            Used to handle Hydra's node_id instability.
        """
        self._task_relevant_objects: Set[str] = set(task_relevant_objects or [])
        self._task_relevant_categories: Set[str] = set()  # Categories for semantic tracking
        self._position_threshold = position_threshold
        self._track_all = track_all
        self._match_threshold = match_threshold
        
        # Cache of previous scene state
        self._previous_objects: Dict[str, ObjectNode] = {}
        self._previous_timestamp: str = ""
        
        # node_id mapping: {old_id: new_id} to track objects across Hydra id changes
        self._id_mapping: Dict[str, str] = {}
    
    def set_task_relevant_objects(
        self, 
        object_ids: List[str],
        scene_graph: Optional[SceneGraph] = None
    ) -> None:
        """
        Update the set of task-relevant objects
        
        Also extracts categories for semantic matching (handles node_id instability).
        """
        self._task_relevant_objects = set(object_ids)
        self._task_relevant_categories.clear()
        
        # Extract categories from scene graph for semantic tracking
        if scene_graph:
            for obj_id in object_ids:
                obj = scene_graph.get_object(obj_id)
                if obj:
                    self._task_relevant_categories.add(obj.category)
    
    def add_task_relevant_object(self, object_id: str, category: Optional[str] = None) -> None:
        """Add an object to task-relevant set"""
        self._task_relevant_objects.add(object_id)
        if category:
            self._task_relevant_categories.add(category)
    
    def _match_objects_by_semantic(
        self, 
        old_objects: Dict[str, ObjectNode],
        new_objects: Dict[str, ObjectNode]
    ) -> Dict[str, str]:
        """
        Match objects between two scene graphs using (category + position)
        
        Handles Hydra's node_id instability by finding corresponding objects
        even when their IDs have changed.
        
        Args:
            old_objects: Objects from previous scene graph
            new_objects: Objects from current scene graph
            
        Returns:
            Mapping from old_id to new_id for matched objects
        """
        matches: Dict[str, str] = {}
        used_new_ids: Set[str] = set()
        
        for old_id, old_obj in old_objects.items():
            if old_id in new_objects:
                # Same ID exists, direct match
                matches[old_id] = old_id
                used_new_ids.add(old_id)
                continue
            
            # ID changed, try to find by (category + position)
            best_match: Optional[str] = None
            best_distance = float('inf')
            
            for new_id, new_obj in new_objects.items():
                if new_id in used_new_ids:
                    continue
                
                # Must be same category
                if old_obj.category != new_obj.category:
                    continue
                
                # Calculate position distance
                if old_obj.position and new_obj.position:
                    distance = self._calculate_distance(old_obj.position, new_obj.position)
                    
                    # Check if within match threshold
                    if distance < self._match_threshold and distance < best_distance:
                        best_distance = distance
                        best_match = new_id
            
            if best_match:
                matches[old_id] = best_match
                used_new_ids.add(best_match)
        
        return matches
    
    def update_baseline(self, scene_graph: SceneGraph) -> None:
        """
        Update the baseline scene state for comparison
        
        Call this after successful task completion or when starting new task.
        """
        self._previous_objects = {
            obj_id: obj for obj_id, obj in scene_graph.objects.items()
        }
        self._previous_timestamp = scene_graph.timestamp
    
    def detect(
        self,
        old_sg: Optional[SceneGraph],
        new_sg: SceneGraph
    ) -> ChangeReport:
        """
        Detect changes between two scene graphs
        
        Handles Hydra's node_id instability by matching objects via (category + position)
        before checking for changes.
        
        Args:
            old_sg: Previous scene graph (uses cached baseline if None)
            new_sg: Current scene graph
            
        Returns:
            ChangeReport with all detected changes
        """
        report = ChangeReport()
        
        # Use cached baseline if old_sg not provided
        if old_sg is None:
            old_objects = self._previous_objects
        else:
            old_objects = old_sg.objects
        
        new_objects = new_sg.objects
        
        # ========================================================
        # Step 0: Semantic matching to handle node_id instability
        # ========================================================
        id_matches = self._match_objects_by_semantic(old_objects, new_objects)
        self._id_mapping = id_matches  # Save for external reference
        
        matched_old_ids = set(id_matches.keys())
        matched_new_ids = set(id_matches.values())
        
        old_ids = set(old_objects.keys())
        new_ids = set(new_objects.keys())
        
        # ========================================================
        # Step 1: Detect truly disappeared objects (no semantic match found)
        # ========================================================
        disappeared_ids = old_ids - matched_old_ids
        for obj_id in disappeared_ids:
            old_obj = old_objects[obj_id]
            change = ObjectChange(
                object_id=obj_id,
                change_type=ChangeType.OBJECT_DISAPPEARED,
                category=old_obj.category,
                old_room_id=old_obj.room_id,
                old_position=old_obj.position
            )
            report.objects_disappeared.append(change)
            
            # Check task relevance by ID or category
            if self._is_task_relevant(obj_id, old_obj.category):
                report.task_relevant_changes.append(change)
        
        # ========================================================
        # Step 2: Detect truly new objects (not matched to any old object)
        # ========================================================
        appeared_ids = new_ids - matched_new_ids
        for obj_id in appeared_ids:
            new_obj = new_objects[obj_id]
            change = ObjectChange(
                object_id=obj_id,
                change_type=ChangeType.OBJECT_APPEARED,
                category=new_obj.category,
                new_room_id=new_obj.room_id,
                new_position=new_obj.position
            )
            report.objects_appeared.append(change)
            
            # New objects of task-relevant categories might be important
            if self._track_all or new_obj.category in self._task_relevant_categories:
                report.task_relevant_changes.append(change)
        
        # ========================================================
        # Step 3: Detect changes in matched objects
        # ========================================================
        for old_id, new_id in id_matches.items():
            old_obj = old_objects[old_id]
            new_obj = new_objects[new_id]
            
            # Use the NEW id for reporting (more relevant for subsequent operations)
            report_id = new_id
            
            # Check room change
            if old_obj.room_id != new_obj.room_id:
                change = ObjectChange(
                    object_id=report_id,
                    change_type=ChangeType.OBJECT_MOVED_ROOM,
                    category=new_obj.category,
                    old_room_id=old_obj.room_id,
                    new_room_id=new_obj.room_id,
                    old_position=old_obj.position,
                    new_position=new_obj.position
                )
                report.objects_moved_room.append(change)
                
                if self._is_task_relevant(old_id, old_obj.category):
                    report.task_relevant_changes.append(change)
            
            # Check position change (within same room)
            elif old_obj.position and new_obj.position:
                displacement = self._calculate_distance(old_obj.position, new_obj.position)
                # Only report if exceeds position_threshold (but still within match_threshold)
                if displacement >= self._position_threshold:
                    change = ObjectChange(
                        object_id=report_id,
                        change_type=ChangeType.OBJECT_POSITION_CHANGED,
                        category=new_obj.category,
                        old_room_id=old_obj.room_id,
                        new_room_id=new_obj.room_id,
                        old_position=old_obj.position,
                        new_position=new_obj.position,
                        displacement=displacement
                    )
                    report.objects_position_changed.append(change)
                    
                    if self._is_task_relevant(old_id, old_obj.category):
                        report.task_relevant_changes.append(change)
        
        # Set task affected flag
        report.is_task_affected = len(report.task_relevant_changes) > 0
        
        # ========================================================
        # Step 4: Update task-relevant object IDs if they changed
        # ========================================================
        self._update_task_relevant_ids(id_matches)
        
        # Update baseline for next comparison
        self.update_baseline(new_sg)
        
        return report
    
    def _update_task_relevant_ids(self, id_matches: Dict[str, str]) -> None:
        """Update task-relevant object IDs when Hydra assigns new IDs"""
        updated_relevant = set()
        for old_id in self._task_relevant_objects:
            if old_id in id_matches:
                new_id = id_matches[old_id]
                updated_relevant.add(new_id)
            elif old_id in id_matches.values():
                # ID hasn't changed
                updated_relevant.add(old_id)
        self._task_relevant_objects = updated_relevant
    
    def get_current_id(self, original_id: str) -> str:
        """
        Get the current node_id for an object, accounting for Hydra ID changes.
        
        Args:
            original_id: The original node_id used when setting up task
            
        Returns:
            Current node_id (may differ from original due to Hydra instability)
        """
        return self._id_mapping.get(original_id, original_id)
    
    def quick_check(self, new_sg: SceneGraph) -> bool:
        """
        Quick check if any task-relevant objects have changed
        
        Faster than full detect() - use for polling during execution.
        Handles Hydra node_id instability by matching via (category + position).
        
        Args:
            new_sg: Current scene graph
            
        Returns:
            True if any task-relevant changes detected
        """
        if not self._task_relevant_objects and not self._task_relevant_categories:
            return False
        
        # Build semantic matches
        id_matches = self._match_objects_by_semantic(self._previous_objects, new_sg.objects)
        
        for obj_id in self._task_relevant_objects:
            old_obj = self._previous_objects.get(obj_id)
            if not old_obj:
                continue
            
            # Find the matched new ID (may be different due to Hydra instability)
            new_id = id_matches.get(obj_id)
            
            if new_id is None:
                # Object disappeared (no semantic match found)
                return True
            
            new_obj = new_sg.objects.get(new_id)
            if not new_obj:
                return True
            
            # Check if object moved rooms
            if old_obj.room_id != new_obj.room_id:
                return True
            
            # Check significant position change
            if old_obj.position and new_obj.position:
                displacement = self._calculate_distance(old_obj.position, new_obj.position)
                if displacement >= self._position_threshold:
                    return True
        
        return False
    
    def _is_task_relevant(self, obj_id: str, category: Optional[str] = None) -> bool:
        """
        Check if object is relevant to current task
        
        Checks both by ID and by category to handle Hydra node_id instability.
        """
        if self._track_all:
            return True
        if not self._task_relevant_objects and not self._task_relevant_categories:
            return True  # If no specific objects set, consider all relevant
        
        # Check by ID
        if obj_id in self._task_relevant_objects:
            return True
        
        # Check by category (for semantic matching)
        if category and category in self._task_relevant_categories:
            return True
            
        return False
    
    @staticmethod
    def _calculate_distance(pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two positions"""
        if len(pos1) < 2 or len(pos2) < 2:
            return 0.0
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = (pos2[2] - pos1[2]) if len(pos1) > 2 and len(pos2) > 2 else 0.0
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    @staticmethod
    def extract_relevant_objects_from_plan(plan_actions: List[Dict[str, Any]]) -> List[str]:
        """
        Extract task-relevant object IDs from a plan
        
        Args:
            plan_actions: List of action dictionaries from LLM plan
            
        Returns:
            List of object IDs involved in the plan
        """
        relevant_ids = set()
        
        for action in plan_actions:
            params = action.get("params", {})
            
            # Direct object references
            if "object_id" in params:
                relevant_ids.add(params["object_id"])
            
            # Surface/target references
            if "surface_id" in params:
                relevant_ids.add(params["surface_id"])
            
            # Target object for move actions
            if "target_object" in params:
                relevant_ids.add(params["target_object"])
        
        return list(relevant_ids)
