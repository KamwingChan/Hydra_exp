#!/usr/bin/env python3
"""
检查从起点到终点在 traversable map 上是否连通。
用法:
  cd /home/kamwing/catkin_ws/src/phy_plan/env
  python check_trav_path.py
  python check_trav_path.py --target-x -4.38 --target-y -0.92
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
phy_plan_root = os.path.dirname(current_dir)
if phy_plan_root not in sys.path:
    sys.path.insert(0, phy_plan_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch as th
import omnigibson as og
from config.scene_config import choose_scene
from utils.trav_map_utils import inject_custom_trav_map


def main():
    parser = argparse.ArgumentParser(description="Check if path exists on trav map (built-in get_shortest_path)")
    parser.add_argument("--scene", type=str, default="Rs_int")
    parser.add_argument("--scene-file", type=str, default="/home/kamwing/catkin_ws/src/phy_plan/env/config/scene_configs/Rs_int_T4.json")
    parser.add_argument("--trav-map", type=str, default="/home/kamwing/catkin_ws/src/phy_plan/env/config/scene_configs/Rs_int_T4.png")
    parser.add_argument("--start-x", type=float, default=0.0, help="Start world x")
    parser.add_argument("--start-y", type=float, default=0.0, help="Start world y")
    parser.add_argument("--target-x", type=float, default=-4.385, help="Target world x")
    parser.add_argument("--target-y", type=float, default=-0.925, help="Target world y")
    args = parser.parse_args()

    print("Creating env (same as full_pipeline)...")
    env = og.Environment(choose_scene(
        args.scene, args.scene_file, semantic_segmentation=False,
        trav_map_path=args.trav_map, trav_map_resolution=0.01,
    ))
    if args.trav_map and os.path.isfile(args.trav_map):
        inject_custom_trav_map(env, args.trav_map, custom_resolution=0.01)

    robot = env.robots[0]
    scene = env.scene
    if not hasattr(scene, "_trav_map") or scene._trav_map is None:
        print("ERROR: scene has no _trav_map")
        return

    trav = scene._trav_map
    src = th.tensor([args.start_x, args.start_y])
    tgt = th.tensor([args.target_x, args.target_y])

    print(f"Start (world): [{args.start_x}, {args.start_y}]")
    print(f"Target (world): [{args.target_x}, {args.target_y}]")
    print("Calling trav.get_shortest_path(0, src, tgt, entire_path=True, robot=None)...")

    path_world, dist = trav.get_shortest_path(0, src, tgt, entire_path=True, robot=None)

    if path_world is None:
        print("Result: NO PATH (disconnected or start/goal not traversable)")
    else:
        print(f"Result: PATH FOUND, {len(path_world)} waypoints, geodesic distance = {dist:.3f} m")


if __name__ == "__main__":
    main()