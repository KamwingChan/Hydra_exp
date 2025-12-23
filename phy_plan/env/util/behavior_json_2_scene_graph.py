#!/usr/bin/env python3
"""
Convert Behavior JSON to scene graph format.

Reads a Behavior JSON file and extracts swivel_chair and conference_table objects,
outputting a scene graph JSON similar to offline_graph_gen.py format.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


# 真实尺寸配置（从 Isaac Sim 测量得到）
# 格式: (category, model): (width, depth, height) 单位：米
# width = x 方向, depth = y 方向, height = z 方向
OBJECT_REAL_DIMENSIONS = {
    ("conference_table", "hdomxc"): (2.37, 5.17, 0.82),  # 从 get_real_scale.py 测量
    ("conference_table", None): (2.0, 1.0, 0.75),  # 默认单个桌子（回退值）
    ("swivel_chair", None): (0.89, 0.70, 1.09),  # 从 get_real_scale.py 测量
    ("chair", None): (0.89, 0.70, 1.09),  # 通用椅子（回退值）
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
    
    # 最后回退到估算值
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


def extract_objects_from_behavior_json(json_path: Path, target_categories: List[str]) -> List[Dict[str, Any]]:
    """
    Extract objects from Behavior JSON file.
    
    Args:
        json_path: Path to Behavior JSON file
        target_categories: List of categories to extract (e.g., ["swivel_chair", "conference_table"])
    
    Returns:
        List of object dictionaries with node_id, category, position, and bounding_box
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = []
    object_counter = 0
    
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
    
    return objects


def build_scene_graph(objects: List[Dict[str, Any]], source_path: Path) -> Dict[str, Any]:
    """
    Build scene graph structure similar to offline_graph_gen.py output.
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
            "rooms": [],  # Empty rooms for now
            "objects": objects
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Behavior JSON to scene graph format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json --output scene_graph.json
  python behavior_json_2_scene_graph.py --input office_vendor_machine_0.json --categories swivel_chair conference_table armchair
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
        default=["swivel_chair", "conference_table"],
        help="Categories to extract (default: swivel_chair conference_table)"
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
    
    # Extract objects
    print(f"[INFO] Reading from: {input_path}")
    print(f"[INFO] Extracting categories: {', '.join(args.categories)}")
    
    objects = extract_objects_from_behavior_json(input_path, args.categories)
    print(f"[INFO] Found {len(objects)} objects")
    
    if len(objects) == 0:
        print("[WARN] No objects found! Check if categories match the JSON file.")
        return
    
    # Build scene graph
    scene_graph = build_scene_graph(objects, input_path)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scene_graph, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Wrote {len(objects)} objects to: {output_path}")
    
    # Print summary
    category_counts = {}
    for obj in objects:
        cat = obj["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n[SUMMARY]")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
