#!/usr/bin/env python3
"""
rerun_scene_graph.py: Replay a task recording in Rerun 3-D viewer.

Reads a JSON file produced by TaskRecorder and visualises:
  - Rooms as translucent 3-D boxes
  - Objects as coloured 3-D boxes (red = task-relevant, blue = normal)
  - Room-object edges as connecting lines
  - Robot position as a green sphere + accumulated trajectory
  - Current action, plan progress and LLM chain-of-thought in text panels

Usage:
  python -m phy_plan.visualization.rerun_scene_graph \
      --recording exp_results/recording_T4.json \
      --highlight "O(10)" "O(15)"

Dependencies (replay machine only):
  pip install rerun-sdk numpy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import rerun as rr
except ImportError:
    print("rerun-sdk is required: pip install rerun-sdk")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────── #
#  Colour palette                                                      #
# ──────────────────────────────────────────────────────────────────── #
COLOR_HIGHLIGHT = [255, 60, 60, 255]      # red — task-relevant
COLOR_OBJECT    = [100, 149, 237, 220]    # cornflower blue
COLOR_ROOM      = [255, 255, 0, 225]     # yellow
COLOR_AGENT     = [50, 205, 50, 255]      # green
COLOR_EDGE      = [255, 255, 255, 255]     # white
COLOR_TRAJ      = [50, 205, 50, 120]      # green, semi-transparent
COLOR_BUILDING   = [255, 60, 60, 255]    # red
COLOR_HELD      = [255, 165, 0, 255]     # orange — object held by robot


# ──────────────────────────────────────────────────────────────────── #
#  Geometry helpers                                                     #
# ──────────────────────────────────────────────────────────────────── #

def _bbox_center_and_half(bbox: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return (center, half_size) from a {min:{x,y,z}, max:{x,y,z}} dict."""
    mn = np.array([bbox["min"]["x"], bbox["min"]["y"], bbox["min"]["z"]])
    mx = np.array([bbox["max"]["x"], bbox["max"]["y"], bbox["max"]["z"]])
    return (mn + mx) / 2.0, (mx - mn) / 2.0


def _pos_array(pos: Dict) -> np.ndarray:
    return np.array([pos["x"], pos["y"], pos["z"]])


# ──────────────────────────────────────────────────────────────────── #
#  Per-frame logging                                                    #
# ──────────────────────────────────────────────────────────────────── #

def log_scene_graph(
    sg: Dict,
    highlight_ids: List[str],
    robot_pose: Optional[List[float]],
    trajectory: List[np.ndarray],
    held_object_names: Optional[set] = None,
    held_node_ids: Optional[set] = None,
):
    """Log one scene-graph snapshot to the current Rerun time context.

    Args:
        held_object_names: OG object names currently held (fallback for name matching).
        held_node_ids: SG node_ids (e.g. "O(13)") currently held — exact skip, no ghost.
    """

    room_centroids: Dict[str, np.ndarray] = {}

    # ── Rooms ──────────────────────────────────────────────────────── #
    for room in sg.get("rooms", []):
        rid = room["room_id"]
        cat = room.get("category", rid)

        if "bounding_box" in room:
            center, half = _bbox_center_and_half(room["bounding_box"])
        elif "centroid" in room:
            center = _pos_array(room["centroid"])
            half = np.array([1.0, 1.0, 1.0])
        else:
            continue

        center_vis = center + np.array([0, 0, 3])

        room_centroids[rid] = center_vis

        rr.log(
            f"world/rooms/{rid}",
            rr.Points3D(
                [center_vis],
                radii=[0.35],
                colors=[COLOR_ROOM],
                labels=[cat]
            ),
        )
    # ── Building ────────────────────────────────────────────────────── #
    building_center = None
    if room_centroids:
        room_positions = np.array(list(room_centroids.values()))
        building_center = np.mean(room_positions, axis=0) + np.array([0.0, 0.0, 2])
        rr.log(
            "world/building",
            rr.Points3D(
                [building_center],
                colors=[COLOR_BUILDING],
                radii=[0.4],
                labels=["Office"],
            ),
        )
    # ── Building–room edges ─────────────────────────────────────────── #
    if building_center is not None and room_centroids:
        b_room_starts = np.array([building_center] * len(room_centroids))
        b_room_ends = np.array(list(room_centroids.values()))
        b_room_segments = np.stack([b_room_starts, b_room_ends], axis=1)
        rr.log(
            "world/edges/building_room",
            rr.LineStrips3D(
                b_room_segments,
                colors=[COLOR_EDGE] * len(b_room_segments),
                radii=[0.01],
            ),
        )

    # ── Objects ────────────────────────────────────────────────────── #
    edge_starts: List[np.ndarray] = []
    edge_ends: List[np.ndarray] = []

    _held_names = held_object_names or set()
    _held_nids = held_node_ids or set()

    def _should_skip_held(nid: str, cat: str) -> bool:
        if nid in _held_nids:
            return True
        if not _held_names:
            return False
        cat_norm = cat.lower().replace("_", "").replace(" ", "")
        for h in _held_names:
            h_norm = h.lower().replace("_", "").replace(" ", "")
            if cat_norm in h_norm or h_norm in cat_norm:
                return True
        return False

    for obj in sg.get("objects", []):
        nid = obj["node_id"]
        cat = obj.get("category", nid)

        if _should_skip_held(nid, cat):
            continue

        is_highlight = nid in highlight_ids
        color = COLOR_HIGHLIGHT if is_highlight else COLOR_OBJECT

        if "bounding_box" in obj:
            center, half = _bbox_center_and_half(obj["bounding_box"])
        elif "position" in obj:
            center = _pos_array(obj["position"])
            half = np.array([0.1, 0.1, 0.1])
        else:
            continue

        # Build compact label
        phys = obj.get("physical_properties", {})
        conf = phys.get("inference_confidence", -1)
        conf_str = f" conf={conf}%" if conf >= 0 else ""
        label = f"{nid} {cat}{conf_str}"

        rr.log(
            f"world/objects/{nid}",
            rr.Boxes3D(
                centers=[center],
                half_sizes=[half],
                colors=[color],
                labels=[label],
            ),
        )

        # Room-object edge
        room_id = obj.get("room_id")
        if room_id and room_id in room_centroids:
            edge_starts.append(room_centroids[room_id])
            edge_ends.append(center)

    # ── Room-object edges (batch) ──────────────────────────────────── #
    if edge_starts:
        segments = np.stack(
            [np.array(edge_starts), np.array(edge_ends)], axis=1
        )
        rr.log(
            "world/edges/room_object",
            rr.LineStrips3D(segments, colors=[COLOR_EDGE] * len(segments), radii=[0.01]),
        )

    # ── Agent ──────────────────────────────────────────────────────── #
    if robot_pose and len(robot_pose) >= 3:
        pos = np.array(robot_pose[:3])
        rr.log(
            "world/agent",
            rr.Points3D([pos], colors=[COLOR_AGENT], radii=[0.15]),
        )
        rr.log("world/agent/label", rr.TextDocument(f"Agent {pos.round(2).tolist()}"))

        trajectory.append(pos)
        if len(trajectory) >= 2:
            rr.log(
                "world/agent/trajectory",
                rr.LineStrips3D(
                    [np.array(trajectory)],
                    colors=[COLOR_TRAJ],
                ),
            )


def log_held_objects(held_objects: List[Dict]):
    """Draw objects currently held by the robot as orange boxes (dynamic overlay)."""
    for ho in held_objects:
        name = ho.get("name", "unknown")
        pose = ho.get("pose")
        if not pose or len(pose) < 3:
            continue
        pos = np.array(pose[:3])
        safe_name = name.replace("/", "_")
        rr.log(
            f"world/held/{safe_name}",
            rr.Boxes3D(
                centers=[pos],
                half_sizes=[[0.08, 0.08, 0.08]],
                colors=[COLOR_HELD],
                labels=[f"[held] {name}"],
            ),
        )


def log_text_panels(frame: Dict):
    """Log textual information (action, plan, CoT) to Rerun text panels."""
    event = frame.get("event_type", "")

    # Current action
    action = frame.get("action")
    if action:
        atype = action.get("action_type", "?")
        desc = action.get("description", "")
        target = action.get("target_object", "")
        success = frame.get("action_success")
        status = ""
        if success is True:
            status = " [OK]"
        elif success is False:
            err = frame.get("action_error", "")
            status = f" [FAIL: {err}]"
        rr.log("panels/action", rr.TextDocument(
            f"**{event}** — {atype}({target}) {desc}{status}",
            media_type=rr.MediaType.MARKDOWN,
        ))

    # Plan progress
    plan = frame.get("plan")
    if plan:
        lines = [f"{i+1}. {a.get('action_type','?')}: {a.get('description','')}"
                 for i, a in enumerate(plan)]
        rr.log("panels/plan", rr.TextDocument(
            "## Plan\n" + "\n".join(lines),
            media_type=rr.MediaType.MARKDOWN,
        ))

    # Chain of thought
    cot = frame.get("chain_of_thought")
    if cot:
        rr.log("panels/chain_of_thought", rr.TextDocument(
            f"## Chain of Thought\n{cot}",
            media_type=rr.MediaType.MARKDOWN,
        ))

    # Replan context
    if event == "replan":
        reason = frame.get("replan_reason", "")
        ctx = frame.get("replan_context", "")
        pv = frame.get("physics_validation")
        parts = [f"## Replan\n**Reason:** {reason}"]
        if ctx:
            parts.append(f"\n**Context:**\n```\n{ctx[:1000]}\n```")
        if pv:
            parts.append(f"\n**Physics:** {json.dumps(pv, indent=2)}")
        rr.log("panels/replan", rr.TextDocument(
            "\n".join(parts),
            media_type=rr.MediaType.MARKDOWN,
        ))


# ──────────────────────────────────────────────────────────────────── #
#  Main replay driver                                                   #
# ──────────────────────────────────────────────────────────────────── #

def replay(recording_path: str, extra_highlight: Optional[List[str]] = None):
    """Load a recording JSON and play it back through Rerun."""
    path = Path(recording_path)
    if not path.exists():
        print(f"Recording file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    frames: List[Dict] = data.get("frames", [])
    highlight_ids: List[str] = data.get("highlight_ids", [])
    instruction: str = data.get("instruction", "")

    if extra_highlight:
        highlight_ids = list(set(highlight_ids) | set(extra_highlight))

    rr.init("scene_graph_replay", spawn=True)

    rr.log("panels/task", rr.TextDocument(
        f"## Task\n{instruction}\n\n**Highlighted:** {', '.join(highlight_ids) or 'none'}",
        media_type=rr.MediaType.MARKDOWN,
    ))

    trajectory: List[np.ndarray] = []
    dynamic_objects: set = set()    # OG names of objects currently/previously held
    dynamic_node_ids: set = set()   # SG node_ids (e.g. "O(13)") — exact ghost skip
    prev_held_names: set = set()
    prev_held_node_ids: set = set()
    last_sg: Optional[Dict] = None   # carry forward when scene_update has no SG

    for idx, frame in enumerate(frames):
        rr.set_time("step", sequence=idx)
        rr.set_time("wall_clock", duration=frame.get("timestamp", 0.0))

        sg = frame.get("scene_graph")
        if sg:
            last_sg = sg
        sg_to_use = last_sg  # use carried SG for scene_update frames (no ghost)

        robot_pose = frame.get("robot_pose")
        held_objects: List[Dict] = frame.get("held_objects") or []
        cur_held_names = {ho["name"] for ho in held_objects if "name" in ho}
        cur_held_node_ids = {ho["node_id"] for ho in held_objects if ho.get("node_id")}

        dynamic_objects |= cur_held_names
        dynamic_node_ids |= cur_held_node_ids

        if sg_to_use:
            log_scene_graph(sg_to_use, highlight_ids, robot_pose, trajectory,
                            held_object_names=dynamic_objects,
                            held_node_ids=dynamic_node_ids)
        elif robot_pose and len(robot_pose) >= 3:
            pos = np.array(robot_pose[:3])
            rr.log("world/agent", rr.Points3D([pos], colors=[COLOR_AGENT], radii=[0.15]))
            trajectory.append(pos)
            if len(trajectory) >= 2:
                rr.log(
                    "world/agent/trajectory",
                    rr.LineStrips3D([np.array(trajectory)], colors=[COLOR_TRAJ]),
                )

        if held_objects:
            log_held_objects(held_objects)

        released_names = prev_held_names - cur_held_names
        released_nids = prev_held_node_ids - cur_held_node_ids
        for name in released_names:
            dynamic_objects.discard(name)
        for nid in released_nids:
            dynamic_node_ids.discard(nid)
        for name in released_names:
            safe_name = name.replace("/", "_")
            rr.log(f"world/held/{safe_name}", rr.Clear(recursive=False))

        prev_held_names = cur_held_names
        prev_held_node_ids = cur_held_node_ids

        log_text_panels(frame)

    print(f"[Rerun] Replayed {len(frames)} frames from {path.name}")


# ──────────────────────────────────────────────────────────────────── #
#  Static preview (single JSON scene graph)                             #
# ──────────────────────────────────────────────────────────────────── #

def preview_static(sg_path: str, highlight_ids: Optional[List[str]] = None):
    """Visualise a single scene-graph JSON file (no recording needed)."""
    path = Path(sg_path)
    if not path.exists():
        print(f"Scene graph file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    sg = raw.get("scene_graph", raw)

    rr.init("scene_graph_preview", spawn=True)
    rr.set_time("step", sequence=0)
    log_scene_graph(sg, highlight_ids or [], robot_pose=None, trajectory=[])
    print(f"[Rerun] Static preview of {path.name}")


# ──────────────────────────────────────────────────────────────────── #
#  CLI                                                                  #
# ──────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Replay task recordings or preview scene graphs in Rerun 3-D viewer.",
    )
    sub = parser.add_subparsers(dest="command")

    # replay sub-command
    p_replay = sub.add_parser("replay", help="Replay a task recording JSON")
    p_replay.add_argument("--recording", required=True, help="Path to recording JSON")
    p_replay.add_argument("--highlight", nargs="*", default=[], help="Extra node IDs to highlight")

    # preview sub-command
    p_prev = sub.add_parser("preview", help="Static preview of a scene-graph JSON")
    p_prev.add_argument("--sg_file", required=True, help="Path to scene graph JSON")
    p_prev.add_argument("--highlight", nargs="*", default=[], help="Node IDs to highlight")

    args = parser.parse_args()

    if args.command == "replay":
        replay(args.recording, args.highlight)
    elif args.command == "preview":
        preview_static(args.sg_file, args.highlight)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
