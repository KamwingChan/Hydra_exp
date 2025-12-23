import pathlib
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
import numpy as np
import spark_dsg as dsg
from spark_dsg import NodeSymbol
import json
import logging

from ..core.scene_graph import (
    SceneGraph, 
    ObjectNode, 
    RoomNode, 
    BoundingBox, 
    PhysicalProperties
)

logger = logging.getLogger(__name__)

def load_scene_graph(file_path: Union[str, pathlib.Path]) -> SceneGraph:
    """
    Load a Hydra DSG file into a unified SceneGraph object.
    
    Functionality:
    1. Loads Objects -> ObjectNodes
    2. Loads Rooms -> RoomNodes
    3. Loads Places (GVD) -> stored in metadata["places"] for navigation/planning
    4. Stores raw DSG object -> metadata["dsg"] for advanced graph queries (A*, mesh access)
    
    Args:
        file_path: Path to the .json or .dsg file
        
    Returns:
        SceneGraph: The populated scene graph object
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DSG file not found: {file_path}")

    # Load DSG using spark_dsg bindings
    try:
        # Note: load expects a string path
        G = dsg.DynamicSceneGraph.load(str(file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load DSG from {file_path}: {e}")
    
    sg = SceneGraph()
    sg.source = "hydra"
    sg.timestamp = "" # TODO: extract timestamp if available in DSG metadata
    sg.metadata["dsg"] = G  # Store raw DSG for planner access (e.g. navigation)
    
    # Identify Layer IDs (Handle potential variations in bindings)
    # Standard Hydra Layers: Objects=2, Places=1, Rooms=4
    OBJECTS_LAYER = getattr(dsg.DsgLayers, "OBJECTS", 2)
    PLACES_LAYER = getattr(dsg.DsgLayers, "PLACES", 1)
    ROOMS_LAYER = getattr(dsg.DsgLayers, "ROOMS", 4)
    
    # --- 1. Process Objects (Static) ---
    if G.has_layer(OBJECTS_LAYER):
        layer = G.get_layer(OBJECTS_LAYER)
        for node in layer.nodes:
            obj_node = _convert_object_node(node, G)
            if obj_node:
                sg.objects[obj_node.node_id] = obj_node

    # --- 2. Process Rooms ---
    if G.has_layer(ROOMS_LAYER):
        layer = G.get_layer(ROOMS_LAYER)
        for node in layer.nodes:
            room_node = _convert_room_node(node, G)
            if room_node:
                sg.rooms[room_node.room_id] = room_node
    
    # --- 3. Link Objects to Rooms ---
    # In Hydra, Objects are usually children of Rooms
    for obj in sg.objects.values():
        # Get raw node ID from the symbol string (e.g. "O(1)" -> 1 if needed, but we use the stored map)
        # We search parents in DSG to find the room
        try:
            # We need the integer ID to query the graph
            # Assuming node_id is formatted as "O(123)"
            node_symbol = NodeSymbol(obj.node_id)
            node_id = node_symbol.value
            
            parents = G.get_parents(node_id)
            for parent_id in parents:
                # Check if parent is a room
                if NodeSymbol(parent_id).category_id == 'R':
                    room_id_str = str(NodeSymbol(parent_id))
                    obj.room_id = room_id_str
                    if room_id_str in sg.rooms:
                        sg.rooms[room_id_str].object_ids.append(obj.node_id)
                    break
        except Exception as e:
            logger.warning(f"Failed to link object {obj.node_id} to room: {e}")

    # --- 4. Extract Places (GVD) for Navigation ---
    places = []
    if G.has_layer(PLACES_LAYER):
        layer = G.get_layer(PLACES_LAYER)
        for node in layer.nodes:
            # Basic place info
            pos = node.attributes.position
            place_info = {
                "node_id": str(NodeSymbol(node.id)),
                "position": [float(x) for x in pos],
                "distance": getattr(node.attributes, "distance", 0.0),
                "neighbors": [] # Can be populated via G.get_node(node.id).siblings if needed
            }
            places.append(place_info)
    
    sg.metadata["places"] = places
    
    # --- [NEW] 5. Process Dynamic Agents (轨迹/动态规划关键) ---
    # Hydra 存储动态层的方式是：DSG -> Dynamic Layers -> Layer(Prefix) -> Nodes(Time Series)
    agents_data = {}
    try:
        # 获取所有动态层的 layer_prefix (例如 "a" 代表 robot agent)
        # 注意: get_dynamic_layer_names 可能会返回 ["a", "human_1", ...]
        dynamic_prefixes = G.get_dynamic_layer_names() if hasattr(G, "get_dynamic_layer_names") else []
        
        AGENTS_LAYER_ID = getattr(dsg.DsgLayers, "AGENTS", 2) # 通常也是 2，但属于动态层
        
        for prefix in dynamic_prefixes:
            try:
                # 获取特定前缀的动态层
                dyn_layer = G.get_dynamic_layer(AGENTS_LAYER_ID, prefix)
                
                agent_traj = []
                for node in dyn_layer.nodes:
                    # 动态节点通常包含时间戳
                    timestamp = node.timestamp if hasattr(node, "timestamp") else 0
                    pos = _to_list_3d(node.attributes.position)
                    
                    agent_traj.append({
                        "node_id": str(NodeSymbol(node.id)),
                        "timestamp": timestamp,
                        "position": pos,
                        # 如果有速度或其他属性也可以在这里提取
                    })
                
                # 按时间排序
                agent_traj.sort(key=lambda x: x["timestamp"])
                agents_data[prefix] = agent_traj
                
            except Exception as e:
                logger.warning(f"Failed to process dynamic layer {prefix}: {e}")
                
        # 将动态数据存入 metadata，供规划器使用
        sg.metadata["agents"] = agents_data
        
    except Exception as e:
        logger.warning(f"Error processing dynamic layers: {e}")

    return sg

# 辅助函数：参考 graphReader.py _vector_to_array
def _to_list_3d(vec) -> List[float]:
    """Robustly convert vector-like object to [x, y, z] list"""
    try:
        # Case 1: numpy array or list/tuple
        arr = np.array(vec, dtype=float)
        if arr.size >= 3:
            return arr.flatten()[:3].tolist()
    except:
        pass
    
    # Case 2: Object with x, y, z attributes
    if hasattr(vec, 'x') and hasattr(vec, 'y') and hasattr(vec, 'z'):
        try:
            return [float(vec.x), float(vec.y), float(vec.z)]
        except:
            pass
            
    return [0.0, 0.0, 0.0] # Default fallback

def _convert_object_node(node, G) -> Optional[ObjectNode]:
    """Convert a spark_dsg Node (Object Layer) to ObjectNode"""
    try:
        attrs = node.attributes
        node_id_str = str(NodeSymbol(node.id))
        
        # Robust position extraction
        position = _to_list_3d(attrs.position)
        
        # Robust Bounding Box extraction
        bbox = None
        if hasattr(attrs, "bounding_box"):
            raw_bbox = attrs.bounding_box
            try:
                # Try min/max first (common in newer bindings)
                if hasattr(raw_bbox, 'min') and hasattr(raw_bbox, 'max'):
                    min_pt = _to_list_3d(raw_bbox.min)
                    max_pt = _to_list_3d(raw_bbox.max)
                    bbox = BoundingBox(min_point=min_pt, max_point=max_pt)
                # Fallback logic if needed (e.g. pos/dim) can be added here
            except Exception as e:
                # logger.debug(f"BBox conversion failed for {node_id_str}: {e}")
                pass
        
        # Extract Category/Name
        # Use 'name' if available (e.g. "chair"), else 'semantic_label'
        category = getattr(attrs, "name", "unknown")
        if not category and hasattr(attrs, "semantic_label"):
            category = f"class_{attrs.semantic_label}"
            
        return ObjectNode(
            node_id=node_id_str,
            category=category,
            position=position,
            bounding_box=bbox,
            physical_properties=PhysicalProperties(description=category), # Default props
            room_id=None # Linked later
        )
    except Exception as e:
        logger.warning(f"Error converting object node {node.id}: {e}")
        return None

def _convert_room_node(node, G) -> Optional[RoomNode]:
    """Convert a spark_dsg Node (Room Layer) to RoomNode"""
    try:
        attrs = node.attributes
        room_id_str = str(NodeSymbol(node.id))
        
        pos = attrs.position
        centroid = [float(pos[0]), float(pos[1]), float(pos[2])]
        
        # Use 'name' or 'semantic_label' for category
        category = getattr(attrs, "name", "Room")
        if not category and hasattr(attrs, "semantic_label"):
             category = f"Room_{attrs.semantic_label}"

        return RoomNode(
            room_id=room_id_str,
            category=category,
            centroid=centroid,
            object_ids=[], # Linked later
            description=f"{category} at {centroid}"
        )
    except Exception as e:
        logger.warning(f"Error converting room node {node.id}: {e}")
        return None 