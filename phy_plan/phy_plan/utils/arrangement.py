"""
arrangement.py: 物体摆放目标位置生成

根据规则计算物体应该摆放的目标位置，供 LLM 任务规划调用。

功能：
- generate_positions_around_table: 在桌子周围生成椅子位置
- generate_positions_in_grid: 在区域内生成网格位置
- group_objects_by_nearest_anchor: 按最近锚点分组物体
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any


def generate_positions_around_table(
    table_bbox: Dict[str, List[float]],
    table_position: List[float],
    num_positions: int,
    offset: float = 0.6,
    distribution: str = "uniform"
) -> List[List[float]]:
    """
    在桌子周围生成目标位置
    
    Args:
        table_bbox: 桌子包围盒 {"min": [x, y, z], "max": [x, y, z]}
        table_position: 桌子中心位置 [x, y, z]
        num_positions: 需要生成的位置数量
        offset: 到桌子边缘的距离（米）
        distribution: 分布方式
            - "uniform": 均匀分布在四边
            - "long_sides": 只在长边放置（适合会议桌）
            - "short_sides": 只在短边放置
            - "one_side": 只在一边放置
        
    Returns:
        目标位置列表 [[x, y, z], ...]
        
    Example:
        >>> bbox = {"min": [-2, -1, 0.7], "max": [2, 1, 0.8]}
        >>> pos = [0, 0, 0.7]
        >>> targets = generate_positions_around_table(bbox, pos, 6, offset=0.5)
    """
    if not table_bbox or "min" not in table_bbox or "max" not in table_bbox:
        # 没有 bbox，在桌子周围生成圆形分布
        return _generate_circular_positions(table_position, num_positions, radius=1.5)
    
    min_pt = table_bbox["min"]
    max_pt = table_bbox["max"]
    
    min_x, min_y, min_z = min_pt[0], min_pt[1], min_pt[2]
    max_x, max_y, max_z = max_pt[0], max_pt[1], max_pt[2]
    z = table_position[2] if len(table_position) > 2 else min_z
    
    width = max_x - min_x   # x 方向长度
    depth = max_y - min_y   # y 方向长度
    
    # 确定长边和短边
    if width >= depth:
        long_axis = "x"
        long_min, long_max = min_x, max_x
        short_min, short_max = min_y, max_y
        long_len, short_len = width, depth
    else:
        long_axis = "y"
        long_min, long_max = min_y, max_y
        short_min, short_max = min_x, max_x
        long_len, short_len = depth, width
    
    positions = []
    
    if distribution == "long_sides":
        positions = _distribute_on_long_sides(
            long_axis, long_min, long_max, short_min, short_max,
            num_positions, offset, z
        )
    elif distribution == "short_sides":
        positions = _distribute_on_short_sides(
            long_axis, long_min, long_max, short_min, short_max,
            num_positions, offset, z
        )
    elif distribution == "one_side":
        positions = _distribute_on_one_side(
            min_x, max_x, min_y, offset, num_positions, z
        )
    else:  # uniform
        positions = _distribute_uniform(
            min_x, max_x, min_y, max_y, num_positions, offset, z
        )
    
    return positions[:num_positions]


def _generate_circular_positions(
    center: List[float],
    num_positions: int,
    radius: float = 1.5
) -> List[List[float]]:
    """在中心点周围生成圆形分布的位置"""
    cx, cy = center[0], center[1]
    cz = center[2] if len(center) > 2 else 0.0
    
    positions = []
    for i in range(num_positions):
        angle = 2 * np.pi * i / num_positions
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        positions.append([x, y, cz])
    
    return positions


def _distribute_on_long_sides(
    long_axis: str,
    long_min: float, long_max: float,
    short_min: float, short_max: float,
    num_positions: int,
    offset: float,
    z: float
) -> List[List[float]]:
    """
    在长边上对称分布位置
    
    策略：
    1. 计算每边应该放置的数量
    2. 在长边方向对称分布（从中心向两边）
    3. 在两侧同时生成位置对，确保对称
    """
    positions = []
    
    # 计算每边应该放置的数量
    per_side = (num_positions + 1) // 2
    
    # 计算长边中心
    long_center = (long_min + long_max) / 2.0
    long_range = long_max - long_min
    
    # 生成对称的位置
    for i in range(per_side):
        if len(positions) >= num_positions:
            break
            
        # 在长边方向对称分布
        # 使用对称索引：从中心向两边
        if per_side == 1:
            # 只有一个位置时，放在中心
            offset_from_center = 0.0
        else:
            # 多个位置时，对称分布
            # i=0: 中心, i=1: 中心±0.33, i=2: 中心±0.67, ...
            offset_from_center = (i - (per_side - 1) / 2.0) / max(1, per_side - 1) * (long_range / 2.0)
        
        long_val = long_center + offset_from_center
        
        # 在两侧同时生成位置（确保对称）
        if long_axis == "x":
            # 第一侧（短边负方向）
            positions.append([long_val, short_min - offset, z])
            # 第二侧（短边正方向），如果还有位置
            if len(positions) < num_positions:
                positions.append([long_val, short_max + offset, z])
        else:
            # 第一侧（短边负方向）
            positions.append([short_min - offset, long_val, z])
            # 第二侧（短边正方向），如果还有位置
            if len(positions) < num_positions:
                positions.append([short_max + offset, long_val, z])
    
    return positions[:num_positions]


def _distribute_on_short_sides(
    long_axis: str,
    long_min: float, long_max: float,
    short_min: float, short_max: float,
    num_positions: int,
    offset: float,
    z: float
) -> List[List[float]]:
    """在短边上分布位置"""
    positions = []
    per_side = (num_positions + 1) // 2
    
    for side_idx, side_offset in enumerate([-offset, offset]):
        for i in range(per_side):
            if len(positions) >= num_positions:
                break
            
            # 在短边方向均匀分布
            t = (i + 0.5) / per_side
            short_val = short_min + t * (short_max - short_min)
            
            # 长边方向的偏移
            if side_idx == 0:
                long_val = long_min - offset
            else:
                long_val = long_max + offset
            
            if long_axis == "x":
                positions.append([long_val, short_val, z])
            else:
                positions.append([short_val, long_val, z])
    
    return positions


def _distribute_on_one_side(
    min_x: float, max_x: float,
    min_y: float,
    offset: float,
    num_positions: int,
    z: float
) -> List[List[float]]:
    """在一边（下边）分布位置"""
    positions = []
    width = max_x - min_x
    
    for i in range(num_positions):
        t = (i + 0.5) / num_positions
        x = min_x + t * width
        y = min_y - offset
        positions.append([x, y, z])
    
    return positions


def _distribute_uniform(
    min_x: float, max_x: float,
    min_y: float, max_y: float,
    num_positions: int,
    offset: float,
    z: float
) -> List[List[float]]:
    """在四边均匀分布位置"""
    positions = []
    
    # 四条边：下、上、左、右
    sides = [
        ("bottom", min_y - offset, "x", min_x, max_x),
        ("top", max_y + offset, "x", min_x, max_x),
        ("left", min_x - offset, "y", min_y, max_y),
        ("right", max_x + offset, "y", min_y, max_y),
    ]
    
    # 计算每边放几个
    per_side = max(1, (num_positions + 3) // 4)
    
    for side_name, fixed_val, var_axis, var_min, var_max in sides:
        for i in range(per_side):
            if len(positions) >= num_positions:
                break
            
            t = (i + 0.5) / per_side
            var_val = var_min + t * (var_max - var_min)
            
            if var_axis == "x":
                positions.append([var_val, fixed_val, z])
            else:
                positions.append([fixed_val, var_val, z])
    
    return positions[:num_positions]


def generate_positions_in_grid(
    area_min: List[float],
    area_max: List[float],
    num_positions: int,
    z: float = 0.0
) -> List[List[float]]:
    """
    在矩形区域内生成网格分布的位置
    
    Args:
        area_min: 区域最小点 [x, y]
        area_max: 区域最大点 [x, y]
        num_positions: 位置数量
        z: 高度
        
    Returns:
        位置列表 [[x, y, z], ...]
    """
    width = area_max[0] - area_min[0]
    height = area_max[1] - area_min[1]
    
    # 计算网格行列数
    aspect = width / height if height > 0 else 1.0
    cols = max(1, int(np.sqrt(num_positions * aspect)))
    rows = max(1, int(np.ceil(num_positions / cols)))
    
    positions = []
    dx = width / (cols + 1)
    dy = height / (rows + 1)
    
    for row in range(rows):
        for col in range(cols):
            if len(positions) >= num_positions:
                break
            x = area_min[0] + (col + 1) * dx
            y = area_min[1] + (row + 1) * dy
            positions.append([x, y, z])
    
    return positions


def group_objects_by_nearest_anchor(
    objects: List[Dict[str, Any]],
    anchors: List[Dict[str, Any]],
    position_key: str = "position"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    按最近锚点分组物体
    
    将每个物体分配到最近的锚点（如将椅子分配到最近的桌子）。
    
    Args:
        objects: 物体列表，每个物体需包含 position_key 指定的位置字段
        anchors: 锚点列表（如桌子），需包含 "node_id" 和位置字段
        position_key: 位置字段名称
        
    Returns:
        分组结果 {anchor_node_id: [object1, object2, ...]}
        
    Example:
        >>> chairs = [{"node_id": "O(2)", "position": [0, 0, 0]}, ...]
        >>> tables = [{"node_id": "O(0)", "position": [1, 1, 0]}, ...]
        >>> groups = group_objects_by_nearest_anchor(chairs, tables)
    """
    if not anchors:
        return {}
    
    # 初始化分组
    groups = {anchor.get("node_id", str(i)): [] for i, anchor in enumerate(anchors)}
    
    for obj in objects:
        obj_pos = obj.get(position_key, [0, 0, 0])
        if isinstance(obj_pos, dict):
            obj_pos = [obj_pos.get("x", 0), obj_pos.get("y", 0), obj_pos.get("z", 0)]
        obj_pos = np.array(obj_pos[:2])  # 只用 x, y
        
        # 找最近的锚点
        min_dist = float('inf')
        nearest_anchor_id = None
        
        for i, anchor in enumerate(anchors):
            anchor_pos = anchor.get(position_key, [0, 0, 0])
            if isinstance(anchor_pos, dict):
                anchor_pos = [anchor_pos.get("x", 0), anchor_pos.get("y", 0), anchor_pos.get("z", 0)]
            anchor_pos = np.array(anchor_pos[:2])
            
            dist = np.linalg.norm(obj_pos - anchor_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_anchor_id = anchor.get("node_id", str(i))
        
        if nearest_anchor_id is not None:
            groups[nearest_anchor_id].append(obj)
    
    return groups

