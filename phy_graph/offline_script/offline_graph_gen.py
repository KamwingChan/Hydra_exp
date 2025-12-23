#!/usr/bin/env python3
"""
Offline compact scene graph generator.

Input: a Spark-DSG JSON file (dsg.json)
Output: a compact scene graph JSON similar to phy_graph's ROS output, but without:
- room classification
- physical property inference

Includes object bounding boxes if available in the DSG node attributes.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, DefaultDict

import numpy as np
import spark_dsg as dsg
from spark_dsg import NodeSymbol

from collections import defaultdict


def _now_timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def _node_label(node_id: int) -> str:
    # e.g. "O(10)" / "R(0)" / "p(123)"
    return str(NodeSymbol(node_id))


def _node_category_char(node_id: int) -> str:
    s = _node_label(node_id)
    return s[0] if s else "?"


def _vec3_to_np(v: Any) -> Optional[np.ndarray]:
    # Handles: numpy array, list/tuple, objects with x/y/z
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        if v.shape[0] >= 3:
            return v.astype(float)
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        try:
            return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)
        except (TypeError, ValueError):
            return None
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        try:
            return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
        except (TypeError, ValueError, AttributeError):
            return None
    return None


def _vec3_to_json(v: np.ndarray) -> Dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _extract_bbox_min_max(attrs: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Try to extract an AABB from spark_dsg node attributes.
    Notes:
    - In spark_dsg, BoundingBox is primarily represented by (world_P_center, dimensions, rotation).
    - Python bindings expose:
      - bbox.dimensions, bbox.world_P_center, bbox.world_R_center
      - bbox.min / bbox.max (derived from center/dim, may ignore rotation for AABB extremes)
      - bbox.corners() (best for getting a world-frame AABB by taking min/max over corners)
    """
    if not hasattr(attrs, "bounding_box"):
        return None

    bbox = getattr(attrs, "bounding_box", None)
    if bbox is None:
        return None

    # If bbox is invalid, skip
    try:
        if hasattr(bbox, "is_valid") and not bool(bbox.is_valid()):
            return None
    except Exception:
        # If is_valid exists but errors, continue with best-effort extraction
        pass

    # Best: use corners() -> world-frame AABB
    try:
        if hasattr(bbox, "corners"):
            corners = bbox.corners()
            # corners is array-like of 8 vectors
            pts = []
            for c in corners:
                p = _vec3_to_np(c)
                if p is not None:
                    pts.append(p)
            if pts:
                pts_np = np.stack(pts, axis=0)
                return pts_np.min(axis=0), pts_np.max(axis=0)
    except Exception:
        pass

    # Next: center ± dimensions/2 (works well for AABB/RAABB if rotation is identity)
    try:
        if hasattr(bbox, "world_P_center") and hasattr(bbox, "dimensions"):
            center = _vec3_to_np(getattr(bbox, "world_P_center"))
            dims = _vec3_to_np(getattr(bbox, "dimensions"))
            if center is not None and dims is not None:
                half = dims / 2.0
                return center - half, center + half
    except Exception:
        pass

    # Last resort: use bbox.min / bbox.max properties if present
    try:
        if hasattr(bbox, "min") and hasattr(bbox, "max"):
            vmin = _vec3_to_np(getattr(bbox, "min"))
            vmax = _vec3_to_np(getattr(bbox, "max"))
            if vmin is not None and vmax is not None:
                return vmin, vmax
    except Exception:
        pass

    return None


def _safe_get_name(node: Any) -> str:
    try:
        name = getattr(node.attributes, "name", "")
        return name if name else ""
    except Exception:
        return ""

def _build_interlayer_adjacency(graph: dsg.DynamicSceneGraph) -> Dict[int, Set[int]]:
    adj: DefaultDict[int, Set[int]] = defaultdict(set)

    # static interlayer edges
    try:
        for e in graph.interlayer_edges:
            s = int(e.source)
            t = int(e.target)
            adj[s].add(t)
            adj[t].add(s)
    except Exception:
        pass

    # dynamic interlayer edges (optional)
    try:
        for e in graph.dynamic_interlayer_edges:
            s = int(e.source)
            t = int(e.target)
            adj[s].add(t)
            adj[t].add(s)
    except Exception:
        pass

    return dict(adj)


def build_compact_scene_graph(
    graph: dsg.DynamicSceneGraph, 
    dsg_path: Path,
    filter_categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build compact scene graph from DSG.
    
    Args:
        graph: DynamicSceneGraph instance
        dsg_path: Path to the DSG file
        filter_categories: Optional list of category names to include. If None, include all.
    """
    # Objects
    objects_layer = graph.get_layer(dsg.DsgLayers.OBJECTS) if graph.has_layer(dsg.DsgLayers.OBJECTS) else None
    objects: List[Dict[str, Any]] = []
    object_id_set: Set[int] = set()
    if objects_layer is not None:
        for n in objects_layer.nodes:
            nid = int(n.id.value)
            attrs = n.attributes
            category = _safe_get_name(n) or "Unknown"
            
            # Filter by category if filter_categories is specified
            if filter_categories is not None:
                if category not in filter_categories:
                    continue  # Skip this object
            
            object_id_set.add(nid)
            entry: Dict[str, Any] = {
                "node_id": _node_label(nid),
                "category": category,
            }

            bbox = _extract_bbox_min_max(attrs)
            if bbox is not None:
                bmin, bmax = bbox
                entry["bounding_box"] = {"min": _vec3_to_json(bmin), "max": _vec3_to_json(bmax)}

            objects.append(entry)

    # Place index for room->place->object traversal
    place_nodes: Dict[int, Any] = {}
    if graph.has_layer(dsg.DsgLayers.PLACES):
        places_layer = graph.get_layer(dsg.DsgLayers.PLACES)
        for p in places_layer.nodes:
            place_nodes[int(p.id.value)] = p

    adj = _build_interlayer_adjacency(graph)

    # Rooms
    rooms: List[Dict[str, Any]] = []
    if graph.has_layer(dsg.DsgLayers.ROOMS):
        rooms_layer = graph.get_layer(dsg.DsgLayers.ROOMS)
        for rnode in rooms_layer.nodes:
            rid = int(rnode.id.value)
            rcat = _safe_get_name(rnode) or "Unknown"

            obj_ids: Set[int] = set()

            # --- Preferred: use interlayer adjacency ---
            if adj:
                for nb_id in adj.get(rid, set()):
                    c = _node_category_char(nb_id)
                    if c in ("O", "o"):
                        obj_ids.add(nb_id)
                    elif c == "p":
                        for pnb_id in adj.get(nb_id, set()):
                            if _node_category_char(pnb_id) in ("O", "o"):
                                obj_ids.add(pnb_id)

            # --- Fallback: use siblings() (older graphs / unusual exports) ---
            if not obj_ids:
                try:
                    for nb in list(rnode.siblings()):
                        nb_id = int(nb)
                        c = _node_category_char(nb_id)
                        if c in ("O", "o"):
                            obj_ids.add(nb_id)
                        elif c == "p" and nb_id in place_nodes:
                            pnode = place_nodes[nb_id]
                            for pnb in list(pnode.siblings()):
                                pnb_id = int(pnb)
                                if _node_category_char(pnb_id) in ("O", "o"):
                                    obj_ids.add(pnb_id)
                except Exception:
                    pass

            # Filter object_ids to only include those in object_id_set (which is already filtered)
            if object_id_set:
                obj_ids = {oid for oid in obj_ids if oid in object_id_set}

            rooms.append(
                {
                    "room_id": _node_label(rid),
                    "category": rcat,
                    "object_ids": sorted([_node_label(oid) for oid in obj_ids]),
                }
            )

    return {
        "schema_version": 1,
        "source": {"type": "dsg.json", "path": str(dsg_path), "basename": dsg_path.name},
        "scene_graph": {"timestamp": _now_timestamp_str(), "rooms": rooms, "objects": objects},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsg", required=True, help="Path to dsg.json")
    ap.add_argument(
        "--out",
        default="",
        help="Output path (default: alongside dsg.json as scene_graph_compact.json)",
    )
    ap.add_argument(
        "--categories", "-c",
        nargs="+",
        default=None,
        help="Filter objects by category names (e.g., --categories chair table). If not specified, include all objects."
    )
    args = ap.parse_args()

    dsg_path = Path(args.dsg).expanduser().resolve()
    if not dsg_path.exists():
        raise FileNotFoundError(dsg_path)

    graph = dsg.DynamicSceneGraph.load(str(dsg_path))
    if graph is None:
        raise RuntimeError("Failed to load DSG (spark_dsg returned None)")

    out_path = Path(args.out).expanduser().resolve() if args.out else dsg_path.with_name("scene_graph_compact.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Print filter info if categories are specified
    if args.categories:
        print(f"[INFO] Filtering objects by categories: {', '.join(args.categories)}")
    
    payload = build_compact_scene_graph(graph, dsg_path, filter_categories=args.categories)
    
    # Print summary
    if args.categories:
        category_counts = {}
        for obj in payload["scene_graph"]["objects"]:
            cat = obj["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"[INFO] Found {len(payload['scene_graph']['objects'])} objects:")
        for cat, count in sorted(category_counts.items()):
            print(f"  {cat}: {count}")
    
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()


