"""
utils: 任务规划工具库

提供 LLM 任务规划可调用的算法工具：
- assignment: 最优分配算法（匈牙利算法）
- arrangement: 目标位置生成规则
"""

from .assignment import hungarian_assignment, greedy_assignment
from .arrangement import (
    generate_positions_around_table,
    generate_positions_in_grid,
    group_objects_by_nearest_anchor
)

__all__ = [
    # 分配算法
    "hungarian_assignment",
    "greedy_assignment",
    # 位置生成
    "generate_positions_around_table",
    "generate_positions_in_grid",
    "group_objects_by_nearest_anchor",
]

