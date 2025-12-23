"""
chair_arrangement.py: 椅子摆放实验

实验目标：
1. 加载场景图，筛选椅子和桌子
2. 按桌子分组椅子，在每张桌子周围生成目标位置
3. 使用匈牙利算法最优分配椅子到目标位置
4. 生成任务序列并可视化（使用 TSP 优化执行顺序）

使用方法：
    cd /home/kamwing/catkin_ws/src/phy_plan/phy_plan/experiments
    python chair_arrangement.py --scene scene_graph.json

或者在 Python 中：
    from phy_plan.experiments.chair_arrangement import run_experiment
    run_experiment("scene_graph.json")
"""

import sys
from pathlib import Path

# 添加父目录到路径（必须在导入 phy_plan 之前）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Dict, List, Optional, Tuple
import numpy as np

from phy_plan.input.phy_graph_io import load_scene_graph
from phy_plan.core.scene_graph import SceneGraph, ObjectNode
from phy_plan.core.task import TaskSequence, Action, ActionType, Position
from phy_plan.visualization.arrangement_viz import (
    visualize_scene_2d, 
    visualize_arrangement,
    visualize_task_sequence,
    animate_arrangement
)
from phy_plan.utils.assignment import hungarian_assignment, compute_assignment_with_order, compute_execution_order_tsp
from phy_plan.utils.arrangement import (
    generate_positions_around_table,
    group_objects_by_nearest_anchor
)

import matplotlib.pyplot as plt


# 椅子类别名称（支持多种命名）
CHAIR_CATEGORIES = ["chair", "swivel_chair", "office_chair", "dining_chair"]
# 桌子类别名称
TABLE_CATEGORIES = ["table", "desk", "conference_table", "dining_table"]


def get_chairs(sg: SceneGraph) -> List[ObjectNode]:
    """获取所有椅子类型的物体"""
    chairs = []
    for cat in CHAIR_CATEGORIES:
        chairs.extend(sg.get_objects_by_category(cat))
    return chairs


def get_tables(sg: SceneGraph) -> List[ObjectNode]:
    """获取所有桌子类型的物体"""
    tables = []
    for cat in TABLE_CATEGORIES:
        tables.extend(sg.get_objects_by_category(cat))
    return tables


def group_chairs_by_table(
    chairs: List[ObjectNode],
    tables: List[ObjectNode]
) -> Dict[str, List[ObjectNode]]:
    """
    按桌子分组椅子，尽量使每个桌子分配到的椅子数量相同
    
    使用平衡分配算法：
    1. 计算每个桌子应该分配的基础数量（总数/桌子数）
    2. 使用贪心算法，在考虑距离和当前分配数量的情况下分配
    
    Args:
        chairs: 椅子列表
        tables: 桌子列表
        
    Returns:
        {table_node_id: [chair1, chair2, ...]}
    """
    if not tables:
        return {}
    
    if not chairs:
        return {t.node_id: [] for t in tables}
    
    # 初始化分组
    groups = {t.node_id: [] for t in tables}
    
    # 计算每个桌子应该分配的基础数量
    base_count = len(chairs) // len(tables)  # 每个桌子至少分配这么多
    extra_count = len(chairs) % len(tables)   # 多余的椅子数量
    
    # 计算每个桌子的目标数量（前 extra_count 个桌子多分配一个）
    target_counts = {}
    for i, table in enumerate(tables):
        target_counts[table.node_id] = base_count + (1 if i < extra_count else 0)
    
    # 创建椅子分配状态（是否已分配）
    chair_assigned = [False] * len(chairs)
    
    # 贪心分配：每次选择"最佳"的椅子-桌子对
    # 评分 = 距离权重 - 当前分配数量权重
    # 优先分配给距离近且当前椅子少的桌子
    
    while any(not assigned for assigned in chair_assigned):
        best_chair_idx = None
        best_table_id = None
        best_score = float('inf')
        
        # 找到所有未分配的椅子
        for chair_idx, chair in enumerate(chairs):
            if chair_assigned[chair_idx]:
                continue
            
            chair_pos = np.array(chair.position[:2])
            
            # 对每个桌子计算评分
            for table in tables:
                table_id = table.node_id
                current_count = len(groups[table_id])
                target_count = target_counts[table_id]
                
                # 如果这个桌子已经达到目标数量，跳过
                if current_count >= target_count:
                    continue
                
                # 计算距离
                table_pos = np.array(table.position[:2])
                distance = np.linalg.norm(chair_pos - table_pos)
                
                # 计算评分：距离越小越好，当前分配数量越少越好
                # 使用加权组合：distance_weight * distance - count_weight * (target_count - current_count)
                distance_weight = 1.0
                count_weight = 0.5  # 平衡权重：如果桌子椅子少，降低距离要求
                
                score = distance_weight * distance - count_weight * (target_count - current_count)
                
                if score < best_score:
                    best_score = score
                    best_chair_idx = chair_idx
                    best_table_id = table_id
        
        # 分配最佳匹配
        if best_chair_idx is not None and best_table_id is not None:
            groups[best_table_id].append(chairs[best_chair_idx])
            chair_assigned[best_chair_idx] = True
        else:
            # 如果所有桌子都达到目标数量，将剩余椅子分配给最近的桌子
            for chair_idx, chair in enumerate(chairs):
                if chair_assigned[chair_idx]:
                    continue
                
                chair_pos = np.array(chair.position[:2])
                min_dist = float('inf')
                nearest_table_id = None
                
                for table in tables:
                    table_pos = np.array(table.position[:2])
                    dist = np.linalg.norm(chair_pos - table_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_table_id = table.node_id
                
                if nearest_table_id:
                    groups[nearest_table_id].append(chairs[chair_idx])
                    chair_assigned[chair_idx] = True
    
    return groups


def create_arrangement_task_with_hungarian(
    sg: SceneGraph,
    chairs: List[ObjectNode],
    tables: List[ObjectNode],
    offset: float = 0.6,
    distribution: str = "long_sides",
    agent_start_pos: Optional[List[float]] = None
) -> Tuple[TaskSequence, Dict[str, List[float]], float]:
    """
    使用匈牙利算法创建椅子摆放任务序列，并使用 TSP 优化执行顺序
    
    流程：
    1. 按桌子分组椅子
    2. 为每张桌子生成目标位置
    3. 用匈牙利算法最优分配椅子到目标位置
    4. 使用 TSP 算法优化任务执行顺序（最小化空手移动距离）
    
    Args:
        sg: 场景图
        chairs: 椅子列表
        tables: 桌子列表
        offset: 椅子到桌子边缘的距离
        distribution: 位置分布方式 ("long_sides", "uniform", "one_side")
        agent_start_pos: 机器人起始位置 (可选，默认为 [0,0,0])
        
    Returns:
        (TaskSequence, target_positions, total_travel_dist) 元组
    """
    task_seq = TaskSequence(
        task_name="Arrange Chairs (Hungarian + TSP)",
        metadata={
            "description": "Move chairs to optimal positions around tables",
            "algorithm": "Hungarian + TSP",
            "offset": offset,
            "distribution": distribution
        }
    )
    
    target_positions: Dict[str, List[float]] = {}
    total_assign_cost = 0.0
    
    # 1. 按桌子分组椅子
    table_groups = group_chairs_by_table(chairs, tables)
    
    # 创建桌子查找表
    table_dict = {t.node_id: t for t in tables}
    
    print(f"\n[Grouping] Chairs grouped by nearest table:")
    for table_id, group_chairs in table_groups.items():
        if group_chairs:
            print(f"  - {table_id}: {len(group_chairs)} chairs")
    
    # 收集所有分配结果，用于后续排序
    all_assignments = []
    
    # 2. 对每组椅子进行分配
    for table_id, group_chairs in table_groups.items():
        if not group_chairs:
            continue
        
        table = table_dict.get(table_id)
        if not table:
            continue
        
        # 准备桌子的包围盒
        table_bbox = None
        if table.bounding_box:
            table_bbox = {
                "min": table.bounding_box.min_point,
                "max": table.bounding_box.max_point
            }
        
        # 生成目标位置
        targets = generate_positions_around_table(
            table_bbox=table_bbox,
            table_position=table.position,
            num_positions=len(group_chairs),
            offset=offset,
            distribution=distribution
        )
        
        print(f"\n[Targets] Generated {len(targets)} target positions around {table_id}")
        for i, t in enumerate(targets):
            print(f"    Target {i}: [{t[0]:.2f}, {t[1]:.2f}]")
        
        # 获取椅子当前位置
        chair_positions = [c.position for c in group_chairs]
        
        # 匈牙利算法最优分配
        assignments, group_cost = hungarian_assignment(
            chair_positions, targets, use_2d=True
        )
        total_assign_cost += group_cost
        
        print(f"\n[Assignment] Hungarian algorithm result (cost: {group_cost:.2f}):")
        
        # 收集分配结果
        for assignment in assignments:
            chair_idx = assignment["object_idx"]
            target_idx = assignment["target_idx"]
            cost = assignment["cost"]
            
            chair = group_chairs[chair_idx]
            target = targets[target_idx]
            
            target_positions[chair.node_id] = target
            
            print(f"    {chair.node_id} -> Target {target_idx} (move: {cost:.2f}m)")
            
            all_assignments.append({
                "chair": chair,
                "target": target,
                "cost": cost,
                "table_id": table_id
            })
            
    # ---------------------------------------------------------
    # 优化执行顺序 (TSP)
    # ---------------------------------------------------------
    
    # 准备输入数据给 TSP
    tsp_assignments = []
    tsp_objects = []  # 椅子位置
    tsp_targets = []  # 目标位置
    
    for i, item in enumerate(all_assignments):
        tsp_assignments.append({"object_idx": i, "target_idx": i})
        tsp_objects.append(item["chair"].position)
        tsp_targets.append(item["target"])
    
    # 默认机器人位置
    if agent_start_pos is None:
        agent_start_pos = [0.0, 0.0, 0.0]
        
    print(f"\n[Ordering] Computing optimal execution order for {len(all_assignments)} tasks...")
    
    # 计算最优顺序
    execution_order, total_travel_dist = compute_execution_order_tsp(
        tsp_assignments, 
        tsp_objects, 
        tsp_targets, 
        agent_start_pos=agent_start_pos
    )
    
    print(f"Total travel distance (estimated): {total_travel_dist:.2f} meters")

    # 按优化后的顺序生成任务序列
    for idx in execution_order:
        item = all_assignments[idx]
        chair = item["chair"]
        target = item["target"]
        table_id = item["table_id"]
        
        task_seq.add_move_object(
            object_id=chair.node_id,
            target_position=Position.from_list(target),
            description=f"Move {chair.node_id} to position near {table_id}"
        )
    
    task_seq.metadata["total_assignment_cost"] = total_assign_cost
    task_seq.metadata["total_travel_distance"] = total_travel_dist
    
    return task_seq, target_positions, total_travel_dist


def run_experiment(
    scene_graph_path: str = "scene_graph.json",
    show_plots: bool = True,
    show_animation: bool = False,
    offset: float = 0.6,
    distribution: str = "long_sides",
    agent_pos: Optional[List[float]] = None
) -> Tuple[TaskSequence, Dict[str, List[float]]]:
    """
    运行椅子摆放实验
    
    Args:
        scene_graph_path: 场景图 JSON 文件路径
        show_plots: 是否显示可视化图表
        show_animation: 是否显示动画
        offset: 椅子到桌子边缘的距离
        distribution: 位置分布方式
        agent_pos: 机器人起始位置
        
    Returns:
        (TaskSequence, target_positions) 元组
    """
    # 解析路径（支持相对路径和绝对路径）
    path = Path(scene_graph_path)
    if not path.is_absolute():
        path = Path(__file__).parent / scene_graph_path
    
    print("=" * 60)
    print("Chair Arrangement Experiment (with Hungarian Algorithm + TSP)")
    print("=" * 60)
    print(f"\nLoading scene graph from: {path}")
    
    # 加载场景图
    sg = load_scene_graph(str(path))
    print(f"\n{sg.summary()}")
    
    # 筛选椅子和桌子
    chairs = get_chairs(sg)
    tables = get_tables(sg)
    
    print(f"\n[Objects] Found {len(chairs)} chairs and {len(tables)} tables")
    
    if not chairs:
        print("\nERROR: No chairs found in the scene graph!")
        print(f"  Supported chair categories: {CHAIR_CATEGORIES}")
        return TaskSequence(task_name="Empty"), {}
    
    if not tables:
        print("\nERROR: No tables found in the scene graph!")
        print(f"  Supported table categories: {TABLE_CATEGORIES}")
        return TaskSequence(task_name="Empty"), {}
    
    # 打印椅子信息
    print("\nChairs:")
    for chair in chairs:
        print(f"  - {chair.node_id} ({chair.category}): "
              f"position=[{chair.position[0]:.2f}, {chair.position[1]:.2f}]")
    
    print("\nTables:")
    for table in tables:
        print(f"  - {table.node_id} ({table.category}): "
              f"position=[{table.position[0]:.2f}, {table.position[1]:.2f}]")
    
    # 创建任务序列（使用匈牙利算法 + TSP）
    task_seq, target_positions, total_dist = create_arrangement_task_with_hungarian(
        sg, chairs, tables, 
        offset=offset, 
        distribution=distribution,
        agent_start_pos=agent_pos
    )
    
    print("\n" + "=" * 60)
    print(f"Total travel distance: {total_dist:.2f} meters")
    print("=" * 60)
    print(f"\n{task_seq.summary()}")
    
    # 可视化
    if show_plots:
        output_dir = Path(__file__).parent
        
        # 1. 显示整个场景
        print("\n[1/3] Visualizing entire scene...")
        fig1, ax1 = visualize_scene_2d(sg, show_labels=True)
        fig1.savefig(output_dir / "output_scene.png", dpi=150, bbox_inches='tight')
        
        # 2. 显示椅子摆放
        print("[2/3] Visualizing chair arrangement...")
        fig2, ax2 = visualize_arrangement(
            sg, 
            target_positions,
            categories=CHAIR_CATEGORIES + TABLE_CATEGORIES,
            title=f"Chair Arrangement (TSP dist={total_dist:.2f}m)"
        )
        fig2.savefig(output_dir / "output_arrangement.png", dpi=150, bbox_inches='tight')
        
        # 3. 显示任务序列
        print("[3/3] Visualizing task sequence...")
        fig3, ax3 = visualize_task_sequence(sg, task_seq)
        fig3.savefig(output_dir / "output_task_sequence.png", dpi=150, bbox_inches='tight')
        
        print(f"\nPlots saved to: {output_dir}")
        
        # 4. 动画展示（如果启用）
        if show_animation:
            print("[4/4] Creating animation...")
            anim = animate_arrangement(
                sg,
                task_seq,
                target_positions,
                categories=CHAIR_CATEGORIES + TABLE_CATEGORIES,
                steps_per_action=30,
                interval=50,
                title=f"Chair Arrangement Animation (TSP dist={total_dist:.2f}m)"
            )
            print("Animation created. Close the window to continue.")
            plt.show()
            # 保持引用避免被GC
            _animation_ref = anim
        else:
            plt.show()
    
    return task_seq, target_positions


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chair Arrangement Experiment with Hungarian Algorithm and TSP"
    )
    parser.add_argument(
        "--scene", "-s",
        default="scene_graph.json",
        help="Path to scene graph JSON file (default: scene_graph.json)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable visualization"
    )
    parser.add_argument(
        "--animation", "-a",
        action="store_true",
        help="Show animation of chair movement"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.6,
        help="Distance from chair to table edge (default: 0.6m)"
    )
    parser.add_argument(
        "--distribution",
        choices=["long_sides", "uniform", "one_side", "short_sides"],
        default="long_sides",
        help="Chair distribution pattern (default: long_sides)"
    )
    parser.add_argument(
        "--agent-pos",
        nargs=3,
        type=float,
        default=None,
        help="Agent start position (x y z)"
    )
    
    args = parser.parse_args()
    
    task_seq, targets = run_experiment(
        scene_graph_path=args.scene,
        show_plots=not args.no_plot,
        show_animation=args.animation,
        offset=args.offset,
        distribution=args.distribution,
        agent_pos=args.agent_pos
    )
    
    # 输出任务序列 JSON
    print("\n" + "=" * 60)
    print("Task Sequence JSON:")
    print("=" * 60)
    print(task_seq.to_json())


if __name__ == "__main__":
    main()
