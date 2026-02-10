#!/usr/bin/env python3
"""
A* 算法测试脚本
用于验证 OmniGibson 的 astar 实现是否有问题
"""

import torch as th
import cv2
import numpy as np
from omnigibson.utils.motion_planning_utils import astar

print("="*70)
print("A* Algorithm Test")
print("="*70)

# ========== 测试 1: 简单地图 ==========
print("\n[Test 1] Simple 10x10 map with clear path")
test_map = th.ones((10, 10), dtype=th.uint8) * 255  # 全白（可遍历）
start = (1, 1)
goal = (8, 8)

print(f"  Map shape: {test_map.shape}")
print(f"  Start: {start}, Goal: {goal}")
print(f"  Start traversable: {test_map[start] > 0}")
print(f"  Goal traversable: {test_map[goal] > 0}")

path = astar(test_map, start, goal)
print(f"  Result: {'SUCCESS' if path is not None else 'FAILED'}")
if path is not None:
    print(f"  Path length: {len(path)}")
    print(f"  First 3 steps: {path[:3].tolist()}")
    print(f"  Last 3 steps: {path[-3:].tolist()}")
else:
    print(f"  ❌ A* failed on simple map - THIS IS A BUG!")

# ========== 测试 2: 有障碍物的地图 ==========
print("\n[Test 2] Map with obstacles")
test_map2 = th.ones((10, 10), dtype=th.uint8) * 255
# 添加一些障碍物
test_map2[5, :] = 0  # 横向墙
test_map2[5, 8] = 255  # 留一个缺口
start2 = (2, 2)
goal2 = (8, 8)

print(f"  Start: {start2}, Goal: {goal2}")
path2 = astar(test_map2, start2, goal2)
print(f"  Result: {'SUCCESS' if path2 is not None else 'FAILED'}")
if path2 is not None:
    print(f"  Path length: {len(path2)}")

# ========== 测试 3: 无效的起点/终点 ==========
print("\n[Test 3] Invalid start/goal")
test_map3 = th.ones((10, 10), dtype=th.uint8) * 255
test_map3[5, 5] = 0  # 障碍物
start3 = (5, 5)  # 在障碍物上
goal3 = (8, 8)

print(f"  Start: {start3} (on obstacle), Goal: {goal3}")
print(f"  Start traversable: {test_map3[start3] > 0}")
path3 = astar(test_map3, start3, goal3)
print(f"  Result: {'Path found (unexpected)' if path3 is not None else 'No path (expected)'}")

# ========== 测试 4: 实际场景地图 ==========
print("\n[Test 4] Real scene map from OmniGibson")
try:
    # 加载实际的地图
    map_path = '/home/kamwing/workspace/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes/office_vendor_machine/layout/floor_trav_0.png'
    real_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"  Original map shape: {real_map.shape}")
    
    # 关键：调整到 TraversableMap 使用的尺寸 (437x437)
    # TraversableMap 会 resize 地图到: original_size * (default_res / target_res)
    # default_res = 0.01, target_res = 0.1, 所以是 1/10
    target_size = 437
    real_map_resized = cv2.resize(real_map, (target_size, target_size))
    
    # 二值化：TraversableMap 会把 <255 的都设为 0
    real_map_resized[real_map_resized < 255] = 0
    
    real_map_tensor = th.from_numpy(real_map_resized)
    
    print(f"  Resized map shape: {real_map_tensor.shape}")
    
    # 使用实际的坐标（从调试输出）
    start_real = (218, 218)  # Robot position
    goal_real = (291, 178)   # Target position
    
    print(f"  Start: {start_real}, Goal: {goal_real}")
    print(f"  Start traversable: {real_map_tensor[start_real] > 0}")
    print(f"  Goal traversable: {real_map_tensor[goal_real] > 0}")
    
    # 测试正确尺寸的地图
    print(f"\n  Testing with RESIZED map (437x437, same as TraversableMap)...")
    path_real = astar(real_map_tensor, start_real, goal_real)
    print(f"  Result: {'SUCCESS' if path_real is not None else 'FAILED'}")
    
    if path_real is not None:
        print(f"  ✅ A* SUCCESS! Path length: {len(path_real)}")
        print(f"  Path works on correctly sized map!")
        
        # 可视化路径
        vis_map = cv2.cvtColor(real_map_resized, cv2.COLOR_GRAY2BGR)
        # 画机器人（绿色）
        cv2.circle(vis_map, (start_real[1], start_real[0]), 5, (0, 255, 0), -1)
        # 画目标（红色）
        cv2.circle(vis_map, (goal_real[1], goal_real[0]), 5, (0, 0, 255), -1)
        # 画路径（蓝色）
        for i in range(len(path_real) - 1):
            pt1 = (int(path_real[i][1]), int(path_real[i][0]))
            pt2 = (int(path_real[i+1][1]), int(path_real[i+1][0]))
            cv2.line(vis_map, pt1, pt2, (255, 0, 0), 1)
        cv2.imwrite('/tmp/astar_path_success.png', vis_map)
        print(f"  Saved path visualization to /tmp/astar_path_success.png")
        
    else:
        print(f"  ❌ A* STILL failed on correctly sized map!")
        print(f"  This means the two points are in DISCONNECTED regions")
        
        # 可视化连通性问题
        vis_map = cv2.cvtColor(real_map_resized, cv2.COLOR_GRAY2BGR)
        # 画机器人（绿色）
        cv2.circle(vis_map, (start_real[1], start_real[0]), 8, (0, 255, 0), -1)
        cv2.putText(vis_map, 'Robot', (start_real[1]+10, start_real[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 画目标（红色）
        cv2.circle(vis_map, (goal_real[1], goal_real[0]), 8, (0, 0, 255), -1)
        cv2.putText(vis_map, 'Target', (goal_real[1]+10, goal_real[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # 画直线
        cv2.line(vis_map, (start_real[1], start_real[0]), 
                (goal_real[1], goal_real[0]), (255, 0, 0), 1)
        cv2.imwrite('/tmp/astar_no_path.png', vis_map)
        print(f"  Saved visualization to /tmp/astar_no_path.png")
        
        # 尝试更近的目标
        print(f"\n  Trying closer target...")
        goal_close = (230, 210)
        print(f"  New goal: {goal_close}")
        if 0 <= goal_close[0] < target_size and 0 <= goal_close[1] < target_size:
        print(f"  New goal traversable: {real_map_tensor[goal_close] > 0}")
        path_close = astar(real_map_tensor, start_real, goal_close)
        print(f"  Result: {'SUCCESS' if path_close is not None else 'FAILED'}")
        
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# ========== 测试 5: 检查 g_score 初始化性能 ==========
print("\n[Test 5] Performance test (g_score initialization)")
import time

large_map = th.ones((437, 437), dtype=th.uint8) * 255
start_perf = (100, 100)
goal_perf = (200, 200)

print(f"  Map size: {large_map.shape} (same as real scene)")
t0 = time.time()
path_perf = astar(large_map, start_perf, goal_perf)
t1 = time.time()
print(f"  Time taken: {t1-t0:.3f} seconds")
print(f"  Result: {'SUCCESS' if path_perf is not None else 'FAILED'}")

if t1-t0 > 2.0:
    print(f"  ⚠️  WARNING: A* is very slow! (>2s for simple path)")
    print(f"  This confirms the g_score initialization is inefficient")

print("\n" + "="*70)
print("Test Summary:")
print("  If Test 1 fails: A* has a fundamental bug")
print("  If Test 4 fails: Map connectivity or coordinate issue")
print("  If Test 5 is slow: g_score initialization is the bottleneck")
print("="*70)