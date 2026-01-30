#!/usr/bin/env python3
"""
dsg_builder.py: 从 BEHAVIOR JSON 文件构建 spark_dsg.DynamicSceneGraph

用于离线模式（rosbag 回放）下从 JSON 文件生成 DSG。
复用 behavior_json_2_scene_graph.py 的解析逻辑。

NOTE:
- 离线模式需要 OBJECT_REAL_DIMENSIONS 因为 JSON 不包含完整边界框信息
- 实时模式（dsg_utils.py）直接从 OmniGibson 对象获取 aabb_extent
- semantic_label 对 phy_graph 不重要，统一用 0
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import spark_dsg as dsg
    SPARK_DSG_AVAILABLE = True
except ImportError:
    SPARK_DSG_AVAILABLE = False
    dsg = None


# 真实尺寸配置（从 Isaac Sim 测量得到）
# 格式: (category, model): (width, depth, height) 单位：米
# 仅用于离线 JSON 模式，实时模式直接从 OmniGibson 获取
OBJECT_REAL_DIMENSIONS = {
    ("conference_table", "hdomxc"): (2.37, 5.17, 0.82),
    ("conference_table", "jxixdw"): (1.0, 2.0, 0.73),
    ("swivel_chair", None): (0.89, 0.70, 1.09),
    ("eames_chair", "mmqvnh"): (0.58, 0.62, 0.78),
    ("eames_chair", "svlwdg"): (0.85, 0.94, 0.75),
    ("coffee_cup", "ckkwmj"): (0.09, 0.07, 0.070),
}


def _get_real_dimensions(category: str, model: Optional[str] = None) -> Tuple[float, float, float]:
    """获取类别的真实尺寸"""
    if model:
        key = (category, model)
        if key in OBJECT_REAL_DIMENSIONS:
            return OBJECT_REAL_DIMENSIONS[key]
    
    key_default = (category, None)
    if key_default in OBJECT_REAL_DIMENSIONS:
        return OBJECT_REAL_DIMENSIONS[key_default]
    
    # 回退到估算值
    if "chair" in category.lower():
        return (0.5, 0.5, 1.0)
    elif "table" in category.lower():
        return (2.0, 1.0, 0.75)
    else:
        return (1.0, 1.0, 1.0)




def _infer_room_category(room_name: str) -> str:
    """从房间名称推断类别名称（用于 DSG node.name）"""
    room_lower = room_name.lower()
    
    if "office" in room_lower:
        return "Office"
    elif "meeting" in room_lower or "conference" in room_lower:
        return "ConferenceRoom"
    elif "kitchen" in room_lower:
        return "Kitchen"
    elif "bedroom" in room_lower or "bed" in room_lower:
        return "Bedroom"
    elif "bathroom" in room_lower or "bath" in room_lower:
        return "Bathroom"
    elif "living" in room_lower:
        return "LivingRoom"
    elif "dining" in room_lower:
        return "DiningRoom"
    else:
        return room_name.replace("_", " ").title()


class BehaviorDsgBuilder:
    """
    从 BEHAVIOR JSON 文件构建 spark_dsg.DynamicSceneGraph
    
    用于离线模式（rosbag 回放）
    """
    
    def __init__(self, target_categories: Optional[List[str]] = None):
        """
        初始化构建器
        
        Args:
            target_categories: 要提取的类别列表，None 表示提取所有
        """
        if not SPARK_DSG_AVAILABLE:
            raise ImportError("spark_dsg is not available. Please install it first.")
        
        self.target_categories = target_categories
        self.object_counter = 0
        self.room_counter = 0
        self._obj_name_to_node_id: Dict[str, int] = {}
    
    def build_from_json(self, json_path: str) -> "dsg.DynamicSceneGraph":
        """
        从 BEHAVIOR JSON 文件构建 DSG
        
        Args:
            json_path: JSON 文件路径
            
        Returns:
            构建好的 DynamicSceneGraph
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        G = dsg.DynamicSceneGraph()
        self.object_counter = 0
        self.room_counter = 0
        self._obj_name_to_node_id.clear()
        
        # 提取对象和房间信息
        objects, object_to_rooms = self._extract_objects(data)
        rooms = self._build_rooms(objects, object_to_rooms)
        
        # 添加房间节点
        room_name_to_node_id: Dict[str, int] = {}
        for room in rooms:
            room_node_id = self._add_room_node(G, room)
            room_name_to_node_id[room["name"]] = room_node_id
        
        # 添加对象节点
        for obj in objects:
            obj_node_id = self._add_object_node(G, obj)
            
            # 添加 object -> room 边
            obj_rooms = object_to_rooms.get(obj["name"], [])
            for room_name in obj_rooms:
                if room_name in room_name_to_node_id:
                    try:
                        G.insert_edge(obj_node_id, room_name_to_node_id[room_name])
                    except Exception:
                        pass
        
        return G
    
    def _extract_objects(self, data: Dict[str, Any]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """从 JSON 数据提取对象"""
        objects = []
        object_to_rooms: Dict[str, List[str]] = {}
        
        object_registry = data.get("state", {}).get("registry", {}).get("object_registry", {})
        objects_info = data.get("objects_info", {}).get("init_info", {})
        
        for obj_name, obj_info in objects_info.items():
            obj_args = obj_info.get("args", {})
            category = obj_args.get("category", "")
            
            # 过滤类别
            if self.target_categories and category not in self.target_categories:
                continue
            
            # 获取位置
            obj_state = object_registry.get(obj_name)
            if obj_state is None:
                continue
            
            root_link = obj_state.get("root_link", {})
            pos = root_link.get("pos", [0.0, 0.0, 0.0])
            
            if len(pos) < 3:
                continue
            
            # 获取尺寸信息
            scale = obj_args.get("scale")
            model = obj_args.get("model")
            
            # 计算边界框
            width, depth, height = _get_real_dimensions(category, model)
            if scale and len(scale) >= 3:
                width *= scale[0]
                depth *= scale[1]
                height *= scale[2]
            
            half_w = width / 2.0
            half_d = depth / 2.0
            
            bbox_min = [pos[0] - half_w, pos[1] - half_d, pos[2]]
            bbox_max = [pos[0] + half_w, pos[1] + half_d, pos[2] + height]
            bbox_center = [
                (bbox_min[0] + bbox_max[0]) / 2,
                (bbox_min[1] + bbox_max[1]) / 2,
                (bbox_min[2] + bbox_max[2]) / 2
            ]
            
            objects.append({
                "name": obj_name,
                "category": category,
                "position": pos,
                "bbox_center": bbox_center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            })
            
            # 房间信息
            in_rooms = obj_args.get("in_rooms", [])
            if isinstance(in_rooms, str):
                in_rooms = [in_rooms] if in_rooms else []
            if in_rooms:
                object_to_rooms[obj_name] = in_rooms
        
        return objects, object_to_rooms
    
    def _build_rooms(
        self, 
        objects: List[Dict], 
        object_to_rooms: Dict[str, List[str]]
    ) -> List[Dict]:
        """构建房间列表"""
        all_room_names = set()
        for room_list in object_to_rooms.values():
            all_room_names.update(room_list)
        
        if not all_room_names:
            return []
        
        rooms = []
        for room_name in sorted(all_room_names):
            # 找到属于该房间的对象
            room_objects = []
            room_positions = []
            
            for obj in objects:
                obj_rooms = object_to_rooms.get(obj["name"], [])
                if room_name in obj_rooms:
                    room_objects.append(obj["name"])
                    room_positions.append(obj["position"])
            
            # 计算房间中心
            centroid = None
            if room_positions:
                centroid = [
                    sum(p[0] for p in room_positions) / len(room_positions),
                    sum(p[1] for p in room_positions) / len(room_positions),
                    sum(p[2] for p in room_positions) / len(room_positions),
                ]
            
            category = _infer_room_category(room_name)
            
            rooms.append({
                "name": room_name,
                "category": category,
                "centroid": centroid,
                "object_names": room_objects,
            })
        
        return rooms
    
    def _add_object_node(self, G: "dsg.DynamicSceneGraph", obj: Dict) -> int:
        """添加 Object 节点"""
        attrs = dsg.ObjectNodeAttributes()
        
        attrs.position = np.array(obj["position"], dtype=np.float64)
        attrs.name = obj["category"]  # phy_graph 用 name (category) 来过滤
        attrs.semantic_label = 0  # phy_graph 不需要 semantic_label
        attrs.last_update_time_ns = int(time.time() * 1e9)
        attrs.is_active = True
        
        # 边界框
        if "bbox_center" in obj:
            attrs.bounding_box.world_P_center = np.array(obj["bbox_center"], dtype=np.float64)
        if "bbox_min" in obj and "bbox_max" in obj:
            bbox_min = np.array(obj["bbox_min"], dtype=np.float64)
            bbox_max = np.array(obj["bbox_max"], dtype=np.float64)
            attrs.bounding_box.dimensions = bbox_max - bbox_min
        
        # 创建节点 ID
        node_symbol = dsg.NodeSymbol('O', self.object_counter)
        node_id = node_symbol.value
        self.object_counter += 1
        
        self._obj_name_to_node_id[obj["name"]] = node_id
        
        G.add_node(dsg.DsgLayers.OBJECTS, node_id, attrs)
        
        return node_id
    
    def _add_room_node(self, G: "dsg.DynamicSceneGraph", room: Dict) -> int:
        """添加 Room 节点"""
        attrs = dsg.RoomNodeAttributes()
        
        attrs.name = room["category"]  # phy_graph 用 name (category) 来识别房间
        attrs.semantic_label = 0  # phy_graph 不需要 semantic_label
        attrs.last_update_time_ns = int(time.time() * 1e9)
        
        if room.get("centroid"):
            attrs.position = np.array(room["centroid"], dtype=np.float64)
        
        # 创建节点 ID
        node_symbol = dsg.NodeSymbol('R', self.room_counter)
        node_id = node_symbol.value
        self.room_counter += 1
        
        G.add_node(dsg.DsgLayers.ROOMS, node_id, attrs)
        
        return node_id


def build_dsg_from_behavior_json(
    json_path: str,
    target_categories: Optional[List[str]] = None
) -> "dsg.DynamicSceneGraph":
    """
    便捷函数：从 BEHAVIOR JSON 文件构建 DSG
    
    Args:
        json_path: JSON 文件路径
        target_categories: 要提取的类别列表
        
    Returns:
        构建好的 DynamicSceneGraph
    """
    builder = BehaviorDsgBuilder(target_categories)
    return builder.build_from_json(json_path)
