"""
phy_graph_io.py: 加载 phy_graph 输出的 JSON 场景图

支持两种格式：
1. 完整版：scene_graph_latest.json（含 position, bounding_box, physical_properties）
2. compact 版：scene_graph_compact.json（仅 node_id, category, bounding_box）
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
    从 phy_graph JSON 文件加载场景图
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        SceneGraph 对象
        
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
    从字典数据创建 SceneGraph
    
    支持两种格式：
    1. phy_graph 输出格式: {"scene_graph": {"rooms": [...], "objects": [...]}}
    2. 直接格式: {"rooms": [...], "objects": [...]}
    
    Args:
        data: JSON 解析后的字典
        
    Returns:
        SceneGraph 对象
    """
    sg = SceneGraph()
    sg.source = "phy_graph"
    
    # 处理嵌套结构
    if "scene_graph" in data:
        scene_data = data["scene_graph"]
        sg.timestamp = scene_data.get("timestamp", "")
    else:
        scene_data = data
        sg.timestamp = data.get("timestamp", "")
    
    # 保存元数据
    if "source" in data:
        sg.metadata["original_source"] = data["source"]
    if "schema_version" in data:
        sg.metadata["schema_version"] = data["schema_version"]
    
    # 解析房间
    rooms_data = scene_data.get("rooms", [])
    for room_dict in rooms_data:
        room = RoomNode.from_dict(room_dict)
        sg.rooms[room.room_id] = room
    
    # 构建物体到房间的映射
    object_to_room: Dict[str, str] = {}
    for room in sg.rooms.values():
        for obj_id in room.object_ids:
            object_to_room[obj_id] = room.room_id
    
    # 解析物体
    objects_data = scene_data.get("objects", [])
    for obj_dict in objects_data:
        node_id = obj_dict.get("node_id", "")
        room_id = object_to_room.get(node_id)
        obj = ObjectNode.from_dict(obj_dict, room_id=room_id)
        sg.objects[obj.node_id] = obj
    
    return sg


def save_scene_graph(sg: SceneGraph, file_path: Union[str, Path], compact: bool = False) -> None:
    """
    保存场景图到 JSON 文件
    
    Args:
        sg: SceneGraph 对象
        file_path: 输出文件路径
        compact: 是否使用 compact 格式
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if compact:
        content = sg.to_compact_json()
    else:
        content = sg.to_json()
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

