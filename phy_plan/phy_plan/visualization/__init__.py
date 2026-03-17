"""
visualization: 可视化模块

- arrangement_viz: 摆放结果可视化（2D matplotlib）
- task_recorder: 任务执行录制（零外部依赖，纯 JSON）
- rerun_scene_graph: Rerun 3D 回放（仅回放机需 rerun-sdk）
"""

from .arrangement_viz import visualize_arrangement, visualize_scene_2d
from .task_recorder import TaskRecorder

