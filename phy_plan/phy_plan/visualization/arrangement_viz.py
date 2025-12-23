"""
arrangement_viz.py: 物体摆放可视化

提供 2D matplotlib 可视化，用于展示：
1. 场景图中的物体分布
2. 物体的目标位置
3. 移动路径和任务序列
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np

from ..core.scene_graph import SceneGraph, ObjectNode
from ..core.task import TaskSequence, Action, ActionType, Position


# id-color mapping
CATEGORY_COLORS = {
    "chair": "#4A90D9",      # blue
    "table": "#8B4513",      # brown
    "desk": "#8B4513",       # brown
    "couch": "#9370DB",      # purple
    "computer": "#2E8B57",   # green
    "plant": "#228B22",      # dark green
    "trashcan": "#696969",   # gray
    "painting": "#DAA520",   # gold
    "default": "#A0A0A0",    # default gray
}


def get_category_color(category: str) -> str:
    """get the color of the category"""
    return CATEGORY_COLORS.get(category.lower(), CATEGORY_COLORS["default"])


def visualize_scene_2d(
    sg: SceneGraph,
    categories: Optional[List[str]] = None,
    show_labels: bool = True,
    show_bbox: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    visualize the scene graph in 2D
    
    Args:
        sg: SceneGraph 
        categories: the list of categories to display, None means display all
        show_labels: whether to display the labels of the objects
        show_bbox: whether to display the bounding boxes of the objects
        figsize: the size of the figure
        ax: optional matplotlib axes
        
    Returns:
        (fig, ax) tuple
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    objects = sg.all_objects()
    if categories:
        categories_lower = [c.lower() for c in categories]
        objects = [obj for obj in objects if obj.category.lower() in categories_lower]
    
    # calculate the coordinate range
    if not objects:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    else:
        xs = [obj.position[0] for obj in objects]
        ys = [obj.position[1] for obj in objects]
        margin = 2.0
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    
    # draw the objects
    for obj in objects:
        x, y = obj.position[0], obj.position[1]
        color = get_category_color(obj.category)
        
        if show_bbox and obj.bounding_box:
            # draw the bounding box (top view)
            bbox = obj.bounding_box
            width = bbox.max_point[0] - bbox.min_point[0]
            height = bbox.max_point[1] - bbox.min_point[1]
            rect = patches.Rectangle(
                (bbox.min_point[0], bbox.min_point[1]),
                width, height,
                linewidth=1.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.3
            )
            ax.add_patch(rect)
            # center point
            ax.plot(x, y, 'o', color=color, markersize=6)
        else:
            # only draw the center point
            ax.plot(x, y, 'o', color=color, markersize=10)
        
        if show_labels:
            ax.annotate(
                f"{obj.node_id}\n({obj.category})",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                color='black'
            )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Scene Graph - {len(objects)} objects')
    
    # add the legend
    legend_elements = []
    seen_categories = set()
    for obj in objects:
        if obj.category.lower() not in seen_categories:
            seen_categories.add(obj.category.lower())
            color = get_category_color(obj.category)
            legend_elements.append(
                patches.Patch(facecolor=color, alpha=0.5, label=obj.category)
            )
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    
    return fig, ax


def visualize_arrangement(
    sg: SceneGraph,
    target_positions: Dict[str, List[float]],
    categories: Optional[List[str]] = None,
    show_arrows: bool = True,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    title: str = "Object Arrangement"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    visualize the arrangement: current position -> target position
    
    Args:
        sg: SceneGraph object
        target_positions: the mapping of the target positions {node_id: [x, y, z]}
        categories: the list of categories to display
        show_arrows: whether to display the arrows of the movements
        show_labels: whether to display the labels of the objects
        figsize: the size of the figure
        title: the title of the figure
        
    Returns:
        (fig, ax) tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    objects = sg.all_objects()
    if categories:
        categories_lower = [c.lower() for c in categories]
        objects = [obj for obj in objects if obj.category.lower() in categories_lower]
    
    # separate the tables and chairs (for drawing the bbox)
    table_categories = ["table", "desk", "conference_table", "dining_table"]
    tables = [obj for obj in objects if obj.category.lower() in [tc.lower() for tc in table_categories]]
    other_objects = [obj for obj in objects if obj not in tables]
    
    # collect all the coordinate points (including the target positions)
    all_xs = [obj.position[0] for obj in objects]
    all_ys = [obj.position[1] for obj in objects]
    for pos in target_positions.values():
        all_xs.append(pos[0])
        all_ys.append(pos[1])
    
    if all_xs and all_ys:
        margin = 2.0
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    
    # 先绘制桌子的 bbox（作为背景层）
    for table in tables:
        if table.bounding_box:
            bbox = table.bounding_box
            width = bbox.max_point[0] - bbox.min_point[0]
            height = bbox.max_point[1] - bbox.min_point[1]
            color = get_category_color(table.category)
            rect = patches.Rectangle(
                (bbox.min_point[0], bbox.min_point[1]),
                width, height,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.2,  # 半透明，不遮挡其他物体
                zorder=1  # 底层
            )
            ax.add_patch(rect)
    
    # 绘制物体和移动
    for obj in objects:
        x, y = obj.position[0], obj.position[1]
        color = get_category_color(obj.category)
        
        # 当前位置（实心圆）
        ax.plot(x, y, 'o', color=color, markersize=12, label=obj.category if obj == objects[0] else "", zorder=3)
        
        if obj.node_id in target_positions:
            target = target_positions[obj.node_id]
            tx, ty = target[0], target[1]
            
            # 目标位置（空心圆）
            ax.plot(tx, ty, 'o', color=color, markersize=12, 
                   markerfacecolor='white', markeredgewidth=2, zorder=3)
            
            # 移动箭头
            if show_arrows:
                ax.annotate(
                    '',
                    xy=(tx, ty),
                    xytext=(x, y),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        lw=2,
                        connectionstyle='arc3,rad=0.1'
                    ),
                    zorder=2
                )
        
        if show_labels:
            ax.annotate(
                obj.node_id,
                (x, y),
                textcoords="offset points",
                xytext=(0, 15),
                ha='center',
                fontsize=9,
                fontweight='bold',
                zorder=4
            )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    
    # 添加图例说明
    ax.plot([], [], 'o', color='gray', markersize=10, label='Current position')
    ax.plot([], [], 'o', color='gray', markersize=10, 
           markerfacecolor='white', markeredgewidth=2, label='Target position')
    ax.legend(loc='upper right')
    
    return fig, ax


def visualize_task_sequence(
    sg: SceneGraph,
    task_seq: TaskSequence,
    show_order: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    可视化任务序列
    
    Args:
        sg: SceneGraph 对象
        task_seq: TaskSequence 对象
        show_order: 是否显示执行顺序
        figsize: 图像大小
        
    Returns:
        (fig, ax) 元组
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 收集所有相关物体
    target_positions: Dict[str, List[float]] = {}
    involved_objects = set()
    
    for action in task_seq.actions:
        if action.target_object:
            involved_objects.add(action.target_object)
        if action.target_position and action.target_object:
            target_positions[action.target_object] = action.target_position.to_list()
    
    # 获取相关物体
    objects = [sg.get_object(oid) for oid in involved_objects if sg.get_object(oid)]
    
    # 获取所有桌子（用于显示 bbox）
    all_objects = sg.all_objects()
    table_categories = ["table", "desk", "conference_table", "dining_table"]
    tables = [obj for obj in all_objects 
              if obj.category.lower() in [tc.lower() for tc in table_categories]]
    
    # 计算坐标范围
    all_xs = [obj.position[0] for obj in objects if obj]
    all_ys = [obj.position[1] for obj in objects if obj]
    for pos in target_positions.values():
        all_xs.append(pos[0])
        all_ys.append(pos[1])
    
    # 包含桌子的 bbox 范围
    for table in tables:
        if table.bounding_box:
            all_xs.extend([table.bounding_box.min_point[0], table.bounding_box.max_point[0]])
            all_ys.extend([table.bounding_box.min_point[1], table.bounding_box.max_point[1]])
    
    if all_xs and all_ys:
        margin = 3.0
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    
    # 先绘制桌子的 bbox（作为背景层）
    for table in tables:
        if table.bounding_box:
            bbox = table.bounding_box
            width = bbox.max_point[0] - bbox.min_point[0]
            height = bbox.max_point[1] - bbox.min_point[1]
            color = get_category_color(table.category)
            rect = patches.Rectangle(
                (bbox.min_point[0], bbox.min_point[1]),
                width, height,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.2,  # 半透明
                zorder=1  # 底层
            )
            ax.add_patch(rect)
    
    # 绘制物体和任务
    for i, action in enumerate(task_seq.actions, 1):
        if not action.target_object:
            continue
            
        obj = sg.get_object(action.target_object)
        if not obj:
            continue
        
        x, y = obj.position[0], obj.position[1]
        color = get_category_color(obj.category)
        
        # 当前位置
        ax.plot(x, y, 'o', color=color, markersize=14, zorder=3)
        
        # 执行顺序标记
        if show_order:
            ax.annotate(
                str(i),
                (x, y),
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white',
                zorder=4
            )
        
        # 目标位置和箭头
        if action.target_position:
            tx = action.target_position.x
            ty = action.target_position.y
            
            ax.plot(tx, ty, 's', color=color, markersize=12,
                   markerfacecolor='white', markeredgewidth=2, zorder=3)
            
            ax.annotate(
                '',
                xy=(tx, ty),
                xytext=(x, y),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=2,
                    mutation_scale=15
                ),
                zorder=2
            )
            
            # 目标位置序号
            if show_order:
                ax.annotate(
                    str(i),
                    (tx, ty),
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color=color,
                    zorder=4
                )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Task Sequence: {task_seq.task_name} ({len(task_seq)} actions)')
    
    # 添加任务列表
    task_text = "\n".join([
        f"{i}. {a.description or a.action_type.value}"
        for i, a in enumerate(task_seq.actions, 1)
    ])
    ax.text(
        1.02, 0.98, task_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    return fig, ax


def interpolate_along_waypoints(
    waypoints: List[List[float]], 
    current_step: int, 
    steps_per_segment: int
) -> List[float]:
    """
    沿着路径点进行分段插值
    
    Args:
        waypoints: 路径点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        current_step: 当前步数
        steps_per_segment: 每个路径段使用的步数
        
    Returns:
        当前步数对应的插值位置 [x, y, z]
    """
    if len(waypoints) < 2:
        return waypoints[0] if waypoints else [0.0, 0.0, 0.0]
    
    # 计算当前在哪一段
    total_segments = len(waypoints) - 1
    segment_idx = min(current_step // steps_per_segment, total_segments - 1)
    local_step = current_step % steps_per_segment
    
    # 在该段内插值
    start = waypoints[segment_idx]
    end = waypoints[segment_idx + 1]
    t = local_step / steps_per_segment if steps_per_segment > 0 else 1.0
    
    # 确保维度一致
    dim = min(len(start), len(end))
    return [
        start[i] + (end[i] - start[i]) * t 
        for i in range(dim)
    ]


def animate_arrangement(
    sg: SceneGraph,
    task_seq: TaskSequence,
    target_positions: Dict[str, List[float]],
    categories: Optional[List[str]] = None,
    steps_per_action: int = 30,
    interval: int = 50,
    title: str = "Object Arrangement Animation",
    trajectories: Optional[Dict[str, List[List[float]]]] = None
) -> "matplotlib.animation.FuncAnimation":
    """
    创建物体摆放动画，展示物体按任务序列顺序移动到目标位置
    
    Args:
        sg: SceneGraph 对象
        task_seq: TaskSequence 对象（包含动作序列）
        target_positions: 目标位置映射 {node_id: [x, y, z]}
        categories: 要显示的类别列表，None 表示显示所有
        steps_per_action: 每个动作的动画帧数
        interval: 帧间隔（毫秒）
        title: 动画标题
        trajectories: 可选的路径点字典
            {node_id: [[x1,y1,z1], [x2,y2,z2], ..., [x_end, y_end, z_end]]}
            如果提供，物体将沿着这些路径点移动
            如果不提供（None），使用起点到终点的直线插值
            
    Returns:
        matplotlib.animation.FuncAnimation 动画对象
    """
    from matplotlib.animation import FuncAnimation
    
    # 1. 初始化：创建图形、筛选物体
    fig, ax = plt.subplots(figsize=(12, 10))
    
    objects = sg.all_objects()
    if categories:
        categories_lower = [c.lower() for c in categories]
        objects = [obj for obj in objects if obj.category.lower() in categories_lower]
    
    # 分离桌子和椅子（用于绘制 bbox）
    table_categories = ["table", "desk", "conference_table", "dining_table"]
    tables = [obj for obj in objects if obj.category.lower() in [tc.lower() for tc in table_categories]]
    
    # 收集所有坐标点（包括目标位置）
    all_xs = [obj.position[0] for obj in objects]
    all_ys = [obj.position[1] for obj in objects]
    for pos in target_positions.values():
        all_xs.append(pos[0])
        all_ys.append(pos[1])
    
    # 如果有路径点，也包含进去
    if trajectories:
        for waypoints in trajectories.values():
            for wp in waypoints:
                if len(wp) >= 2:
                    all_xs.append(wp[0])
                    all_ys.append(wp[1])
    
    if all_xs and all_ys:
        margin = 2.0
        ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
        ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
    # 2. 初始化动画状态
    # 记录每个物体的初始位置
    initial_positions: Dict[str, List[float]] = {}
    for obj in objects:
        initial_positions[obj.node_id] = [obj.position[0], obj.position[1], obj.position[2] if len(obj.position) > 2 else 0.0]
    
    # 动画过程中物体的当前位置（会动态更新）
    animated_positions: Dict[str, List[float]] = initial_positions.copy()
    
    # 3. 定义动画更新函数
    def animate(frame: int):
        nonlocal animated_positions
        
        ax.clear()
        
        # 重新绘制静态元素：桌子的 bbox（背景层）
        for table in tables:
            if table.bounding_box:
                bbox = table.bounding_box
                width = bbox.max_point[0] - bbox.min_point[0]
                height = bbox.max_point[1] - bbox.min_point[1]
                color = get_category_color(table.category)
                rect = patches.Rectangle(
                    (bbox.min_point[0], bbox.min_point[1]),
                    width, height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.2,
                    zorder=1
                )
                ax.add_patch(rect)
        
        # 计算当前应该执行的动作和进度
        total_frames = len(task_seq.actions) * steps_per_action
        if frame >= total_frames:
            frame = total_frames - 1
        
        current_action_idx = frame // steps_per_action
        current_step = frame % steps_per_action
        
        # 更新已完成动作的物体位置（保持在目标位置）
        for i in range(current_action_idx):
            if i < len(task_seq.actions):
                action = task_seq.actions[i]
                if action.target_object and action.target_object in target_positions:
                    animated_positions[action.target_object] = target_positions[action.target_object]
        
        # 更新正在执行的物体的位置（插值）
        if current_action_idx < len(task_seq.actions):
            action = task_seq.actions[current_action_idx]
            if action.target_object and action.target_object in target_positions:
                obj_id = action.target_object
                
                # 获取起点和终点
                if obj_id in animated_positions:
                    start_pos = animated_positions[obj_id]
                else:
                    obj = sg.get_object(obj_id)
                    if obj:
                        start_pos = [obj.position[0], obj.position[1], obj.position[2] if len(obj.position) > 2 else 0.0]
                    else:
                        start_pos = [0.0, 0.0, 0.0]
                
                end_pos = target_positions[obj_id]
                
                # 如果有路径点，使用路径插值；否则使用直线插值
                if trajectories and obj_id in trajectories:
                    waypoints = trajectories[obj_id]
                    # 确保起点和终点在路径中
                    full_waypoints = [start_pos] + waypoints
                    if full_waypoints[-1] != end_pos:
                        full_waypoints.append(end_pos)
                    # 计算每段的步数（均匀分配）
                    steps_per_segment = max(1, steps_per_action // (len(full_waypoints) - 1))
                    current_pos = interpolate_along_waypoints(full_waypoints, current_step, steps_per_segment)
                else:
                    # 直线插值
                    t = current_step / (steps_per_action - 1) if steps_per_action > 1 else 1.0
                    current_pos = [
                        start_pos[i] + (end_pos[i] - start_pos[i]) * t
                        for i in range(min(len(start_pos), len(end_pos)))
                    ]
                
                animated_positions[obj_id] = current_pos
        
        # 绘制所有目标位置（空心圆标记）
        for obj_id, target_pos in target_positions.items():
            obj = sg.get_object(obj_id)
            if obj:
                color = get_category_color(obj.category)
                ax.plot(target_pos[0], target_pos[1], 'o', 
                       color=color, markersize=12,
                       markerfacecolor='white', markeredgewidth=2, zorder=3)
        
        # 绘制所有物体
        for obj in objects:
            if obj.node_id not in animated_positions:
                continue
                
            current_pos = animated_positions[obj.node_id]
            x, y = current_pos[0], current_pos[1]
            color = get_category_color(obj.category)
            
            # 判断物体状态
            is_completed = False
            is_active = False
            if current_action_idx < len(task_seq.actions):
                action = task_seq.actions[current_action_idx]
                if action.target_object == obj.node_id:
                    is_active = True
            # 检查是否已完成（位置接近目标位置）
            if obj.node_id in target_positions:
                target = target_positions[obj.node_id]
                dist = np.sqrt((x - target[0])**2 + (y - target[1])**2)
                if dist < 0.01:  # 接近目标位置
                    is_completed = True
            
            # 绘制物体（正在移动的用不同标记）
            if is_active:
                # 正在移动的物体：用大一点的标记，带边框
                ax.plot(x, y, 'o', color=color, markersize=14,
                       markeredgecolor='black', markeredgewidth=2, zorder=4)
            elif is_completed:
                # 已完成的物体：实心圆
                ax.plot(x, y, 'o', color=color, markersize=12, zorder=3)
            else:
                # 未开始的物体：小一点的标记
                ax.plot(x, y, 'o', color=color, markersize=10, alpha=0.6, zorder=3)
            
            # 显示标签
            ax.annotate(
                obj.node_id,
                (x, y),
                textcoords="offset points",
                xytext=(0, 15),
                ha='center',
                fontsize=9,
                fontweight='bold' if is_active else 'normal',
                zorder=5
            )
        
        # 显示当前执行的动作信息
        if current_action_idx < len(task_seq.actions):
            action = task_seq.actions[current_action_idx]
            status_text = f"Action {current_action_idx + 1}/{len(task_seq.actions)}: {action.target_object}"
            if action.description:
                status_text += f" - {action.description}"
        else:
            status_text = "All actions completed"
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title}\n{status_text}")
        
        # 添加图例
        ax.plot([], [], 'o', color='gray', markersize=10, label='Pending', alpha=0.6)
        ax.plot([], [], 'o', color='gray', markersize=12, label='Completed')
        ax.plot([], [], 'o', color='gray', markersize=14, 
               markeredgecolor='black', markeredgewidth=2, label='In progress')
        ax.plot([], [], 'o', color='gray', markersize=12, 
               markerfacecolor='white', markeredgewidth=2, label='Target')
        ax.legend(loc='upper right', fontsize=8)
        
        return []
    
    # 4. 计算总帧数
    total_frames = len(task_seq.actions) * steps_per_action
    
    # 5. 创建动画对象
    anim = FuncAnimation(
        fig, animate,
        frames=total_frames,
        interval=interval,
        repeat=False,
        blit=False
    )
    
    return anim