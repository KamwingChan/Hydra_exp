"""
phy_graph_io.py: Load phy_graph JSON scene graph

supports two formats:
1. full version: scene_graph_latest.json (contains position, bounding_box, physical_properties)
2. compact version: scene_graph_compact.json (only contains node_id, category, bounding_box)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..core.scene_graph import (
    SceneGraph, 
    ObjectNode, 
    RoomNode, 
    BoundingBox, 
    PhysicalProperties
)


def load_scene_graph(file_path: Union[str, Path]) -> SceneGraph:
    """
    load scene graph from phy_graph JSON file
    
    Args:
        file_path: JSON file path
        
    Returns:
        SceneGraph object
        
    Example:
        >>> sg = load_scene_graph("data/scene_graph_office.json")
        >>> chairs = sg.get_objects_by_category("chair")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scene graph file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return load_scene_graph_from_dict(data)


def load_scene_graph_from_dict(data: Dict[str, Any]) -> SceneGraph:
    """
    create SceneGraph from dictionary data
    
    supports two formats:
    1. phy_graph output format: {"scene_graph": {"rooms": [...], "objects": [...]}}
    2. direct format: {"rooms": [...], "objects": [...]}
    
    Args:
        data: dictionary data after JSON parsing
        
    Returns:
        SceneGraph object
    """
    sg = SceneGraph()
    sg.source = "phy_graph"
    
    # process nested structure
    if "scene_graph" in data:
        scene_data = data["scene_graph"]
        sg.timestamp = scene_data.get("timestamp", "")
    else:
        scene_data = data
        sg.timestamp = data.get("timestamp", "")
    
    # save metadata
    if "source" in data:
        sg.metadata["original_source"] = data["source"]
    if "schema_version" in data:
        sg.metadata["schema_version"] = data["schema_version"]
    
    # parse rooms
    rooms_data = scene_data.get("rooms", [])
    for room_dict in rooms_data:
        room = RoomNode.from_dict(room_dict)
        sg.rooms[room.room_id] = room
    
    # build object to room mapping
    object_to_room: Dict[str, str] = {}
    for room in sg.rooms.values():
        for obj_id in room.object_ids:
            object_to_room[obj_id] = room.room_id
    
    # parse objects
    objects_data = scene_data.get("objects", [])
    for obj_dict in objects_data:
        node_id = obj_dict.get("node_id", "")
        room_id = object_to_room.get(node_id)
        obj = ObjectNode.from_dict(obj_dict, room_id=room_id)
        sg.objects[obj.node_id] = obj
    
    return sg


def save_scene_graph(sg: SceneGraph, file_path: Union[str, Path], compact: bool = False) -> None:
    """
    save scene graph to JSON file
    
    Args:
        sg: SceneGraph object
        file_path: output file path
        compact: whether to use compact format
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compact:
        content = sg.to_compact_json()
    else:
        content = sg.to_json()
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

