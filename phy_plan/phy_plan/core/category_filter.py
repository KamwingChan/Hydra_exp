"""
category_filter.py: Shared category exclusion for scene graph / DSG

Single source of truth for EXCLUDED_CATEGORIES and should_include_object.
Used by: phy_graph_io (load from file), env/tools/dsg_utils (publish DSG from OmniGibson).
"""

from typing import Set


# 不参与规划/发布的对象类别（场景结构、机器人等）
EXCLUDED_CATEGORIES: Set[str] = {
    "walls", "floors", "ceilings", "wall", "floor", "ceiling",
    "agent", "robot", "fetch", "tiago",
}


def should_include_object(category: str, name: str = "") -> bool:
    """
    判断该类别/名称是否应保留（True=保留，False=排除）。

    Args:
        category: 对象类别，如 "chair", "wall"
        name: 可选对象名称，用于额外匹配

    Returns:
        True 若应包含，False 若应排除
    """
    if not category:
        return False
    category_lower = category.lower()
    for excluded in EXCLUDED_CATEGORIES:
        if excluded in category_lower:
            return False
    name_lower = (name or "").lower()
    for excluded in EXCLUDED_CATEGORIES:
        if excluded in name_lower:
            return False
    return True
