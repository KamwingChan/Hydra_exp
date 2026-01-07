#!/usr/bin/env python3
"""
Convert Behavior JSON to scene graph format.

Reads a Behavior JSON file and extracts swivel_chair and conference_table objects,
outputting a scene graph JSON similar to offline_graph_gen.py format.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加 phy_plan 路径（如果不在同一项目下）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _now_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


# 真实尺寸配置（从 Isaac Sim 测量得到）
# 格式: (category, model): (width, depth, height) 单位：米
# width = x 方向, depth = y 方向, height = z 方向
OBJECT_REAL_DIMENSIONS = {
    ("conference_table", "hdomxc"): (2.37, 5.17, 0.82),  # 从 get_real_scale.py 测量
    ("conference_table", "jxixdw"): (1.0, 2.0, 0.73),  # 默认单个桌子（回退值）
    ("swivel_chair", None): (0.89, 0.70, 1.09),  # 从 get_real_scale.py 测量
    ("eames_chair", "mmqvnh"): (0.58, 0.62, 0.78), 
    ("eames_chair", "svlwdg"): (0.85, 0.94, 0.75),  # 通用椅子（回退值）
    ("coffee_cup", "ckkwmj"): (0.09, 0.07, 0.070),  # 咖啡杯
}


def _get_real_dimensions(category: str, model: Optional[str] = None) -> Tuple[float, float, float]:
    """
    从配置字典获取真实尺寸
    
    Args:
        category: 物体类别
        model: 模型名称（可选）
        
    Returns:
        (width, depth, height) 元组，单位：米
    """
    # 先尝试精确匹配 (category, model)
    if model:
        key = (category, model)
        if key in OBJECT_REAL_DIMENSIONS:
            return OBJECT_REAL_DIMENSIONS[key]
    
    # 回退到 (category, None)
    key_default = (category, None)
    if key_default in OBJECT_REAL_DIMENSIONS:
        return OBJECT_REAL_DIMENSIONS[key_default]
    
    # 最后回退到估算值，并给出提示
    model_str = f" (model: {model})" if model else ""
    print(f"[WARN] No real dimensions found for category '{category}'{model_str} in OBJECT_REAL_DIMENSIONS, using estimated values")
    
    if "chair" in category.lower():
        return (0.5, 0.5, 1.0)
    elif "table" in category.lower() or "conference_table" in category.lower():
        return (2.0, 1.0, 0.75)
    else:
        return (1.0, 1.0, 1.0)


def _estimate_bbox_from_position_and_scale(
    pos: List[float], 
    category: str, 
    scale: Optional[List[float]] = None,
    model: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Estimate bounding box from position, category, and optional scale.
    Uses real measured dimensions from Isaac Sim when available.
    
    Args:
        pos: Position [x, y, z]
        category: Object category (e.g., "swivel_chair", "conference_table")
        scale: Optional scale factors [sx, sy, sz] (typically close to 1.0)
        model: Model name (e.g., "hdomxc") for precise dimension lookup
    
    Returns:
        Bounding box with min and max coordinates
    """
    x, y, z = pos[0], pos[1], pos[2]
    
    # 从配置获取真实尺寸（已包含 scale 的影响）
    base_width, base_depth, base_height = _get_real_dimensions(category, model)
    
    # 注意：真实尺寸已经是从 Isaac Sim 测量得到的实际尺寸
    # scale 在 Behavior JSON 中通常是接近 1.0 的缩放因子
    # 如果测量时使用的 scale 与当前物体的 scale 不同，可能需要调整
    # 但通常差异很小，直接使用测量值即可
    if scale and len(scale) >= 3:
        # 如果 scale 与测量时的 scale 显著不同，可以应用调整
        # 但为了简单，我们直接使用测量值（假设 scale 接近 1.0）
        # 如果需要精确，可以记录测量时的 scale 并做对比
        width = base_width * scale[0]
        depth = base_depth * scale[1]
        height = base_height * scale[2]
    else:
        width, depth, height = base_width, base_depth, base_height
    
    half_w = width / 2.0
    half_d = depth / 2.0
    
    return {
        "min": {
            "x": x - half_w,
            "y": y - half_d,
            "z": z
        },
        "max": {
            "x": x + half_w,
            "y": y + half_d,
            "z": z + height
        }
    }


def extract_objects_from_behavior_json(
    json_path: Path, 
    target_categories: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Extract objects from Behavior JSON file.
    
    Args:
        json_path: Path to Behavior JSON file
        target_categories: List of categories to extract (e.g., ["swivel_chair", "conference_table"])
    
    Returns:
        Tuple of (objects_list, object_to_rooms_dict)
        - objects_list: List of object dictionaries with node_id, category, position, and bounding_box
        - object_to_rooms_dict: Dict mapping object node_id to list of room names
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = []
    object_counter = 0
    object_to_rooms: Dict[str, List[str]] = {}  # node_id -> [room_name1, room_name2, ...]
    
    # Get object registry (state information) - corrected path
    object_registry = data.get("state", {}).get("registry", {}).get("object_registry", {})
    
    # Get objects info (category information) - corrected path
    objects_info = data.get("objects_info", {}).get("init_info", {})
    
    if not object_registry:
        print("[WARN] No object_registry found in state.registry")
    if not objects_info:
        print("[WARN] No objects_info.init_info found")
    
    # Iterate through all objects in objects_info to get category
    for obj_name, obj_info in objects_info.items():
        # Get category from args
        obj_args = obj_info.get("args", {})
        category = obj_args.get("category", "")
        
        # Filter by target categories
        if category not in target_categories:
            continue
        
        # Get scale and model if available
        scale = obj_args.get("scale")
        model = obj_args.get("model")  # 获取模型名称，如 "hdomxc"
        
        # Get in_rooms from args
        in_rooms = obj_args.get("in_rooms", [])
        if isinstance(in_rooms, str):
            # 处理空字符串的情况
            in_rooms = [] if in_rooms == "" else [in_rooms]
        elif not isinstance(in_rooms, list):
            in_rooms = []
        
        # Get position from object_registry (state information)
        obj_state = object_registry.get(obj_name)
        if obj_state is None:
            print(f"[WARN] Object {obj_name} found in objects_info but not in object_registry, skipping")
            continue
        
        root_link = obj_state.get("root_link", {})
        pos = root_link.get("pos", [0.0, 0.0, 0.0])
        
        if len(pos) < 3:
            print(f"[WARN] Object {obj_name} has invalid position, skipping")
            continue
        
        # Generate node_id (similar to offline_graph_gen.py format: O(0), O(1), ...)
        node_id = f"O({object_counter})"
        object_counter += 1
        
        # Create object entry
        obj_entry: Dict[str, Any] = {
            "node_id": node_id,
            "category": category,
        }
        
        # Add bounding box (using real dimensions from Isaac Sim measurements)
        bbox = _estimate_bbox_from_position_and_scale(pos, category, scale, model=model)
        obj_entry["bounding_box"] = bbox
        
        # Optionally add position for reference
        obj_entry["position"] = {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        }
        
        objects.append(obj_entry)
        
        # 记录物体所属的房间
        if in_rooms:
            object_to_rooms[node_id] = in_rooms
    
    return objects, object_to_rooms


def _infer_room_category(room_name: str) -> str:
    """
    从房间名称推断房间类别，保留原始名称作为基础
    
    Args:
        room_name: 房间名称，如 "shared_office_0", "meeting_room_0"
        
    Returns:
        房间类别，保留原始名称或推断类型
    """
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
        # 默认使用首字母大写的房间名称
        return room_name.replace("_", " ").title()


def build_rooms(
    objects: List[Dict[str, Any]], 
    object_to_rooms: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    构建房间层，严格按照 phy_graph 格式，保留原始房间名称
    
    Args:
        objects: 物体列表
        object_to_rooms: 物体ID到房间名称列表的映射
        
    Returns:
        房间列表，格式符合 phy_graph 标准
    """
    # 收集所有唯一的房间名称
    all_room_names = set()
    for room_list in object_to_rooms.values():
        all_room_names.update(room_list)
    
    if not all_room_names:
        print("[WARN] No rooms found in objects, creating default room")
        # 如果没有房间信息，创建一个默认房间包含所有物体
        all_room_names = {"default_room"}
        for obj in objects:
            object_to_rooms[obj["node_id"]] = ["default_room"]
    
    # 为每个房间创建房间节点
    rooms = []
    room_counter = 0
    room_name_to_id: Dict[str, str] = {}  # room_name -> room_id
    
    for room_name in sorted(all_room_names):
        room_id = f"R({room_counter})"
        room_counter += 1
        room_name_to_id[room_name] = room_id
        
        # 找到属于该房间的所有物体
        room_object_ids = []
        room_positions = []
        room_bboxes = []
        
        for obj in objects:
            obj_id = obj["node_id"]
            obj_rooms = object_to_rooms.get(obj_id, [])
            if room_name in obj_rooms:
                room_object_ids.append(obj_id)
                # 收集位置和包围盒用于计算 centroid 和 bounding_box
                if "position" in obj:
                    pos = obj["position"]
                    room_positions.append([pos["x"], pos["y"], pos["z"]])
                if "bounding_box" in obj:
                    bbox = obj["bounding_box"]
                    room_bboxes.append(bbox)
        
        # 计算 centroid（所有物体位置的平均值）
        centroid = None
        if room_positions:
            avg_x = sum(p[0] for p in room_positions) / len(room_positions)
            avg_y = sum(p[1] for p in room_positions) / len(room_positions)
            avg_z = sum(p[2] for p in room_positions) / len(room_positions)
            centroid = {"x": avg_x, "y": avg_y, "z": avg_z}
        
        # 计算 bounding_box（所有物体包围盒的并集）
        bbox = None
        if room_bboxes:
            min_x = min(b["min"]["x"] for b in room_bboxes)
            min_y = min(b["min"]["y"] for b in room_bboxes)
            min_z = min(b["min"]["z"] for b in room_bboxes)
            max_x = max(b["max"]["x"] for b in room_bboxes)
            max_y = max(b["max"]["y"] for b in room_bboxes)
            max_z = max(b["max"]["z"] for b in room_bboxes)
            bbox = {
                "min": {"x": min_x, "y": min_y, "z": min_z},
                "max": {"x": max_x, "y": max_y, "z": max_z}
            }
        
        # 推断房间类别，但保留原始名称在描述中
        category = _infer_room_category(room_name)
        
        # 创建房间节点（严格按照 phy_graph 格式）
        room_node = {
            "room_id": room_id,
            "category": category,
            "description": f"A {category.lower()} named {room_name}",
            "object_ids": room_object_ids
        }
        
        if centroid:
            room_node["centroid"] = centroid
        
        if bbox:
            room_node["bounding_box"] = bbox
        
        rooms.append(room_node)
    
    return rooms


def build_scene_graph(
    objects: List[Dict[str, Any]], 
    rooms: List[Dict[str, Any]],
    source_path: Path
) -> Dict[str, Any]:
    """
    Build scene graph structure similar to phy_graph output format.
    """
    return {
        "schema_version": 1,
        "source": {
            "type": "behavior_json",
            "path": str(source_path),
            "basename": source_path.name
        },
        "scene_graph": {
            "timestamp": _now_timestamp_str(),
            "rooms": rooms,
            "objects": objects
        }
    }


def visualize_scene_graph(
    scene_graph_path: Path,
    output_image_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """
    可视化场景图
    
    Args:
        scene_graph_path: 场景图 JSON 文件路径
        output_image_path: 输出图片路径（可选）
        show_plot: 是否显示图表
    """
    try:
        from phy_plan.input.phy_graph_io import load_scene_graph
        from phy_plan.visualization.arrangement_viz import visualize_scene_2d
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[WARN] Could not import visualization modules: {e}")
        print("[WARN] Skipping visualization. Install matplotlib if needed.")
        return
    
    print(f"\n[INFO] Visualizing scene graph...")
    
    # 加载场景图
    sg = load_scene_graph(str(scene_graph_path))
    
    # 可视化
    fig, ax = visualize_scene_2d(
        sg,
        categories=None,  # 显示所有类别
        show_labels=True,
        show_bbox=True,
        figsize=(14, 10)
    )
    
    # 保存图片
    if output_image_path:
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved visualization to: {output_image_path}")
    
    # 显示图表
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Behavior JSON to scene graph format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json --output scene_graph.json
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json --categories swivel_chair conference_table armchair
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json --visualize
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to Behavior JSON file (e.g., office_vendor_machine_0.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output path (default: scene_graph.json in same directory as input)"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=["swivel_chair", "conference_table", "eames_chair", "coffee_cup"],
        help="Categories to extract (default: swivel_chair conference_table eames_chair coffee_cup)"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Visualize the generated scene graph"
    )
    parser.add_argument(
        "--visualize-output",
        default="",
        help="Path to save visualization image (default: scene_graph_visualization.png in output directory)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save if --visualize-output is set)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / "scene_graph.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract objects and room information
    print(f"[INFO] Reading from: {input_path}")
    print(f"[INFO] Extracting categories: {', '.join(args.categories)}")
    
    objects, object_to_rooms = extract_objects_from_behavior_json(input_path, args.categories)
    print(f"[INFO] Found {len(objects)} objects")
    
    if len(objects) == 0:
        print("[WARN] No objects found! Check if categories match the JSON file.")
        return
    
    # Build rooms
    rooms = build_rooms(objects, object_to_rooms)
    print(f"[INFO] Found {len(rooms)} rooms")
    
    # Build scene graph
    scene_graph = build_scene_graph(objects, rooms, input_path)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scene_graph, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Wrote {len(objects)} objects and {len(rooms)} rooms to: {output_path}")
    
    # Print summary
    category_counts = {}
    for obj in objects:
        cat = obj["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n[SUMMARY]")
    print("Objects:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    print("\nRooms:")
    for room in rooms:
        print(f"  {room['room_id']} ({room['category']}): {len(room['object_ids'])} objects")
    
    # 可视化（如果启用）
    if args.visualize:
        if args.visualize_output:
            viz_output = Path(args.visualize_output).expanduser().resolve()
        else:
            viz_output = output_path.parent / "scene_graph_visualization.png"
        
        visualize_scene_graph(
            scene_graph_path=output_path,
            output_image_path=viz_output,
            show_plot=not args.no_show
        )


if __name__ == "__main__":
    main()
