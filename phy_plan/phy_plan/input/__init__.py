"""
input: 输入处理模块

- phy_graph_io: 加载 phy_graph 输出的 JSON
- hydra_io: 加载 Hydra DSG（含 place 节点、mesh）
"""

from .phy_graph_io import load_scene_graph, load_scene_graph_from_dict

