"""
assignment.py: 任务分配算法工具库

提供物体到目标位置的最优分配算法，供 LLM 任务规划调用。

算法：
- hungarian_assignment: 匈牙利算法（全局最优）
- greedy_assignment: 贪心算法（快速近似）
- compute_execution_order_tsp: 计算最优执行顺序（TSP）
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Any, Optional

# 尝试导入 ortools，如果没有则使用贪心算法作为回退
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


def hungarian_assignment(
    objects: List[List[float]],
    targets: List[List[float]],
    use_2d: bool = True
) -> Tuple[List[Dict[str, Any]], float]:
    """
    匈牙利算法最优分配
    
    将物体分配到目标位置，最小化总移动距离。
    
    Args:
        objects: 物体当前位置列表 [[x, y, z], ...]
        targets: 目标位置列表 [[x, y, z], ...]
        use_2d: True 表示只用 x, y 计算欧式距离
        
    Returns:
        (assignments, total_cost) 元组
        - assignments: [{"object_idx": 0, "target_idx": 2, "cost": 1.5}, ...]
        - total_cost: 总移动代价
        
    Example:
        >>> objects = [[0, 0, 0], [1, 1, 0], [2, 2, 0]]
        >>> targets = [[0, 1, 0], [1, 0, 0], [2, 1, 0]]
        >>> assignments, cost = hungarian_assignment(objects, targets)
    """
    if not objects or not targets:
        return [], 0.0
    
    obj_arr = np.array(objects, dtype=float)
    tgt_arr = np.array(targets, dtype=float)
    
    # 确保至少有 2D 坐标
    if obj_arr.ndim == 1:
        obj_arr = obj_arr.reshape(1, -1)
    if tgt_arr.ndim == 1:
        tgt_arr = tgt_arr.reshape(1, -1)
    
    if use_2d:
        obj_arr = obj_arr[:, :2]
        tgt_arr = tgt_arr[:, :2]
    
    # 构建代价矩阵：cost[i, j] = 物体 i 到目标 j 的距离
    cost_matrix = np.linalg.norm(
        obj_arr[:, None, :] - tgt_arr[None, :, :], axis=2
    )
    
    # 处理物体数量与目标数量不等的情况
    n_obj, n_tgt = cost_matrix.shape
    if n_obj != n_tgt:
        # 填充为方阵（使用大代价值填充）
        max_dim = max(n_obj, n_tgt)
        padded_cost = np.full((max_dim, max_dim), 1e9)
        padded_cost[:n_obj, :n_tgt] = cost_matrix
        cost_matrix = padded_cost
    
    # 匈牙利算法求解
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 只保留有效分配（非填充部分）
    assignments = []
    total_cost = 0.0
    for obj_idx, tgt_idx in zip(row_ind, col_ind):
        if obj_idx < n_obj and tgt_idx < n_tgt:
            cost = float(cost_matrix[obj_idx, tgt_idx])
            assignments.append({
                "object_idx": int(obj_idx),
                "target_idx": int(tgt_idx),
                "cost": cost
            })
            total_cost += cost
    
    return assignments, total_cost


def greedy_assignment(
    objects: List[List[float]],
    targets: List[List[float]],
    use_2d: bool = True
) -> Tuple[List[Dict[str, Any]], float]:
    """
    贪心算法分配
    
    每次将最近的物体-目标配对，直到所有物体都分配完毕。
    比匈牙利算法快，但不保证全局最优。
    
    Args:
        objects: 物体当前位置列表
        targets: 目标位置列表
        use_2d: 是否只用 x, y 计算距离
        
    Returns:
        (assignments, total_cost) 元组
    """
    if not objects or not targets:
        return [], 0.0
    
    obj_arr = np.array(objects, dtype=float)
    tgt_arr = np.array(targets, dtype=float)
    
    if use_2d:
        obj_arr = obj_arr[:, :2]
        tgt_arr = tgt_arr[:, :2]
    
    n_obj = len(obj_arr)
    n_tgt = len(tgt_arr)
    
    # 构建代价矩阵
    cost_matrix = np.linalg.norm(
        obj_arr[:, None, :] - tgt_arr[None, :, :], axis=2
    )
    
    assignments = []
    total_cost = 0.0
    used_objects = set()
    used_targets = set()
    
    # 贪心选择：每次选代价最小的配对
    for _ in range(min(n_obj, n_tgt)):
        min_cost = float('inf')
        best_pair = None
        
        for i in range(n_obj):
            if i in used_objects:
                continue
            for j in range(n_tgt):
                if j in used_targets:
                    continue
                if cost_matrix[i, j] < min_cost:
                    min_cost = cost_matrix[i, j]
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        obj_idx, tgt_idx = best_pair
        used_objects.add(obj_idx)
        used_targets.add(tgt_idx)
        
        assignments.append({
            "object_idx": obj_idx,
            "target_idx": tgt_idx,
            "cost": min_cost
        })
        total_cost += min_cost
    
    return assignments, total_cost


def compute_assignment_with_order(
    objects: List[List[float]],
    targets: List[List[float]],
    agent_position: Optional[List[float]] = None,
    use_2d: bool = True
) -> Tuple[List[Dict[str, Any]], List[int], float]:
    """
    计算分配并确定执行顺序
    
    先用匈牙利算法分配，然后计算机器人执行任务的最优顺序。
    
    Args:
        objects: 物体当前位置
        targets: 目标位置
        agent_position: 机器人起始位置（用于计算执行顺序）
        use_2d: 是否只用 2D 距离
        
    Returns:
        (assignments, order, total_cost)
        - assignments: 分配结果
        - order: 执行顺序（任务索引列表）
        - total_cost: 分配代价
    """
    assignments, assign_cost = hungarian_assignment(objects, targets, use_2d)
    
    if not assignments:
        return [], [], 0.0
    
    if agent_position is None:
        # 没有指定机器人位置，按分配顺序执行
        order = list(range(len(assignments)))
        return assignments, order, assign_cost
    
    # 贪心确定执行顺序：每次选离当前位置最近的任务
    agent_pos = np.array(agent_position[:2] if use_2d else agent_position)
    remaining = set(range(len(assignments)))
    order = []
    current_pos = agent_pos.copy()
    
    while remaining:
        min_dist = float('inf')
        next_task = None
        
        for task_idx in remaining:
            # 任务位置：物体当前位置
            obj_idx = assignments[task_idx]["object_idx"]
            obj_pos = np.array(objects[obj_idx][:2] if use_2d else objects[obj_idx])
            dist = np.linalg.norm(current_pos - obj_pos)
            
            if dist < min_dist:
                min_dist = dist
                next_task = task_idx
        
        if next_task is not None:
            order.append(next_task)
            remaining.remove(next_task)
            # 更新当前位置为目标位置
            tgt_idx = assignments[next_task]["target_idx"]
            current_pos = np.array(targets[tgt_idx][:2] if use_2d else targets[tgt_idx])
    
    return assignments, order, assign_cost


def compute_execution_order_tsp(
    assignments: List[Dict[str, Any]],
    objects: List[List[float]],
    targets: List[List[float]],
    agent_start_pos: Optional[List[float]] = None,
    return_to_start: bool = False
) -> Tuple[List[int], float]:
    """
    计算任务的最优执行顺序（旅行商问题 TSP）
    
    考虑从上一个任务完成点到下一个任务开始点的移动代价。
    如果安装了 OR-Tools，使用其 Routing 求解器；否则使用贪心策略。
    
    Args:
        assignments: 分配列表 [{"object_idx": 0, "target_idx": 1}, ...]
        objects: 物体位置列表
        targets: 目标位置列表
        agent_start_pos: 机器人起始位置 (可选)
        return_to_start: 是否需要回到起点
        
    Returns:
        (order, total_distance)
        - order: 任务索引列表 [2, 0, 1, ...]
        - total_distance: 总移动距离
    """
    n_tasks = len(assignments)
    if n_tasks == 0:
        return [], 0.0
    
    # 提取每个任务的起点（椅子位置）和终点（目标位置）
    task_starts = []
    task_ends = []
    
    for assign in assignments:
        obj_idx = assign["object_idx"]
        tgt_idx = assign["target_idx"]
        
        # 确保位置是 np.array
        start = np.array(objects[obj_idx][:2])
        end = np.array(targets[tgt_idx][:2])
        
        task_starts.append(start)
        task_ends.append(end)
        
    # 如果没有指定起点，默认从第一个任务的起点开始（或设为 (0,0)）
    if agent_start_pos is None:
        start_pos = task_starts[0] # 简化处理
    else:
        start_pos = np.array(agent_start_pos[:2])

    if ORTOOLS_AVAILABLE:
        return _solve_tsp_ortools(start_pos, task_starts, task_ends, return_to_start)
    else:
        print("[WARN] OR-Tools not found, falling back to greedy ordering.")
        return _solve_tsp_greedy(start_pos, task_starts, task_ends)


def _solve_tsp_ortools(
    agent_pos: np.ndarray,
    task_starts: List[np.ndarray],
    task_ends: List[np.ndarray],
    return_to_start: bool
) -> Tuple[List[int], float]:
    """使用 OR-Tools 求解 TSP"""
    n_tasks = len(task_starts)
    # 节点 0 是 agent 起点，1..n 是任务
    # 大小为 n_tasks + 1
    size = n_tasks + 1
    
    # 构建代价矩阵
    # M[i, j] 表示从节点 i 到节点 j 的移动代价
    # 节点 i (i>0) 代表第 i-1 个任务，其实际位置是该任务的结束点
    # 节点 j (j>0) 代表第 j-1 个任务，其实际位置是该任务的开始点
    # 也就是：任务 i 完成 -> 移动到任务 j 开始 -> 完成任务 j
    
    # 距离函数
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    manager = pywrapcp.RoutingIndexManager(size, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        cost = 0.0
        
        # 起点 (0) 到 任务 j (j > 0)
        if from_node == 0 and to_node > 0:
            task_idx = to_node - 1
            # agent -> 任务起点 + 任务长度
            cost = dist(agent_pos, task_starts[task_idx]) + dist(task_starts[task_idx], task_ends[task_idx])
            
        # 任务 i (i > 0) 到 任务 j (j > 0)
        elif from_node > 0 and to_node > 0:
            if from_node == to_node:
                return 0
            prev_task = from_node - 1
            curr_task = to_node - 1
            # 上个任务终点 -> 当前任务起点 + 当前任务长度
            cost = dist(task_ends[prev_task], task_starts[curr_task]) + dist(task_starts[curr_task], task_ends[curr_task])
            
        # 任务 i (i > 0) 到 起点 (0)
        elif from_node > 0 and to_node == 0:
            if not return_to_start:
                return 0
            prev_task = from_node - 1
            # 任务终点 -> agent 起点
            cost = dist(task_ends[prev_task], agent_pos)
            
        return int(cost * 1000) # 转换为整数避免精度问题

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    # search_params.time_limit.FromSeconds(1)

    solution = routing.SolveWithParameters(search_params)

    if solution:
        order = []
        index = routing.Start(0)
        total_dist = 0.0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index > 0:
                order.append(node_index - 1)
            
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            # 累加实际浮点距离
            dist_int = distance_callback(previous_index, index)
            total_dist += dist_int / 1000.0
            
        return order, total_dist
    else:
        return list(range(n_tasks)), 0.0

def _solve_tsp_greedy(
    agent_pos: np.ndarray,
    task_starts: List[np.ndarray],
    task_ends: List[np.ndarray]
) -> Tuple[List[int], float]:
    """简单的贪心策略"""
    n_tasks = len(task_starts)
    remaining = set(range(n_tasks))
    order = []
    
    current_pos = agent_pos
    total_dist = 0.0
    
    while remaining:
        best_task = None
        min_dist = float('inf')
        
        for task_idx in remaining:
            # 代价 = 移动到任务起点 + 任务长度
            d_to_start = np.linalg.norm(current_pos - task_starts[task_idx])
            d_task = np.linalg.norm(task_starts[task_idx] - task_ends[task_idx])
            cost = d_to_start + d_task
            
            if cost < min_dist:
                min_dist = cost
                best_task = task_idx
        
        if best_task is not None:
            order.append(best_task)
            remaining.remove(best_task)
            total_dist += min_dist
            current_pos = task_ends[best_task]
            
    return order, total_dist
