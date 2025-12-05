# conda your phyplan environment
import pathlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import spark_dsg as dsg
from spark_dsg.open3d_visualization import DsgVisualizer as SparkDsgVisualizer, render_to_open3d,OPEN3D_VISUALIZER_ENABLED
import open3d as o3d
import tempfile
import os
import json
import traceback
from spark_dsg import NodeSymbol
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ==================== Helper Functions ====================

def _get_all_dsg_layer_ids():
    """dynamically get all unique DsgLayers IDs"""
    layer_ids = set()
    for name in dir(dsg.DsgLayers):
        if not name.startswith('_'):
            try:
                layer_id = getattr(dsg.DsgLayers, name)
                if isinstance(layer_id, int):
                    layer_ids.add(layer_id)
            except:
                pass
    return sorted(list(layer_ids))


def _get_layer_name(layer_id: int) -> str:
    """get layer name by layer ID (reverse lookup)"""
    for name in dir(dsg.DsgLayers):
        if not name.startswith('_'):
            try:
                if getattr(dsg.DsgLayers, name) == layer_id:
                    return name
            except:
                pass
    return f"Layer_{layer_id}"


# ==================== Definitions ====================

@dataclass
class QueryFilter:
    layer: Optional[int] = None  
    layer_name: Optional[str] = None  
    semantic_label: Optional[int] = None  
    node_name: Optional[str] = None  
    min_position: Optional[np.ndarray] = None  # 
    max_position: Optional[np.ndarray] = None  #
    bounding_box_type: Optional[str] = None
    node_id: Optional[int] = None  #
    distance: Optional[float] = None


@dataclass
class NodeInfo:
    node_id: int
    layer: str
    position: np.ndarray
    semantic_label: Optional[int] = None
    name: Optional[str] = None
    bounding_box: Optional[Dict] = None
    mesh_connections: Optional[List[int]] = None
    attributes: Optional[Dict] = None
    distance: Optional[float] = None

class DsgGraphReader:
    
    def __init__(self, dsg_path: str, ply_path: Optional[str] = None):
        """
        Initialize the reader
        """
        self.dsg_path = pathlib.Path(dsg_path).resolve()
        if not self.dsg_path.exists():
            raise FileNotFoundError(f"DSG file not found: {self.dsg_path}")
        
        self.graph = dsg.DynamicSceneGraph.load(str(self.dsg_path))
        if self.graph is None:
            raise ValueError("Failed to load scene graph, the file may be corrupted")
        
        if ply_path:
            self.load_mesh(ply_path)
    
    def load_mesh(self, ply_path: str):

        if self.graph.has_mesh():
            mesh = self.graph.mesh()
            if mesh and mesh.num_vertices() > 0:
                print(f"scene graph contained ply with {mesh.num_vertices()} vertices, skip loading ply file")
                self.mesh = mesh
                return mesh
        
        ply_path = pathlib.Path(ply_path).resolve()
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        mesh_o3d = o3d.io.read_triangle_mesh(str(ply_path))
        
        if len(mesh_o3d.vertices) == 0:
            raise ValueError("PLY error")
        
        num_vertices = len(mesh_o3d.vertices)
        num_faces = len(mesh_o3d.triangles)

        self.mesh_o3d = mesh_o3d
        
        print(f"success to load mesh with {num_vertices} vertices and {num_faces} faces")
        
        return mesh_o3d


class DsgQuery:
    
    def __init__(self, graph: dsg.DynamicSceneGraph):
        self.graph = graph
    
    def get_layer_nodes(self, layer_id: int) -> List[NodeInfo]:
        if not self.graph.has_layer(layer_id):
            return []
        
        layer = self.graph.get_layer(layer_id)
        nodes = []
        
        for node in layer.nodes:
            node_info = self._extract_node_info(node, layer_id)
            if node_info:
                nodes.append(node_info)
        
        return nodes
    
    def get_objects(self, 
                   semantic_label: Optional[int] = None,
                   name: Optional[str] = None,
                   node_id: Optional[int] = None,
                   min_pos: Optional[np.ndarray] = None,
                   max_pos: Optional[np.ndarray] = None) -> List[NodeInfo]:

        if not self.graph.has_layer(dsg.DsgLayers.OBJECTS):
            return []
        
        layer = self.graph.get_layer(dsg.DsgLayers.OBJECTS)
        objects = []
        
        for node in layer.nodes:
            attrs = node.attributes
            
            if node_id is not None:
                if node.id.value == node_id:
                    node_info = self._extract_node_info(node, dsg.DsgLayers.OBJECTS)
                    return node_info
            else:
                if semantic_label is not None:
                    if hasattr(attrs, 'semantic_label') and attrs.semantic_label != semantic_label:
                        continue
                
                if name is not None:
                    if hasattr(attrs, 'name') and attrs.name != name:
                        continue
                
                # Handle position - it might be a numpy array or an object with .x, .y, .z
                if isinstance(attrs.position, np.ndarray):
                    pos = attrs.position
                else:
                    pos = np.array([attrs.position.x, attrs.position.y, attrs.position.z])
                
                if min_pos is not None and np.any(pos < min_pos):
                    continue
                if max_pos is not None and np.any(pos > max_pos):
                    continue
                
                node_info = self._extract_node_info(node, dsg.DsgLayers.OBJECTS)
                if node_info:
                    objects.append(node_info)

        
        return objects
    
    def get_places(self) -> List[NodeInfo]:
        # 使用 PLACES 层（PlaceNodeAttributes，有 distance 属性）
        if not self.graph.has_layer(dsg.DsgLayers.PLACES):
            return []
        
        layer = self.graph.get_layer(dsg.DsgLayers.PLACES)
        places = []
        
        for node in layer.nodes:
            node_info = self._extract_node_info(node, dsg.DsgLayers.PLACES)
            if node_info:
                places.append(node_info)
        
        return places
    
    def get_place_edges(self) -> List[Dict]:
        """
        获取所有 place 节点之间的边
        
        Returns:
            List[Dict]: 边的列表，每个边包含：
                - source: 源节点 ID
                - target: 目标节点 ID
                - weight: 边的权重（Clearance，即边上最窄处的宽度）
        """
        if not self.graph.has_layer(dsg.DsgLayers.PLACES):
            return []
        
        layer = self.graph.get_layer(dsg.DsgLayers.PLACES)
        edges = []
        edge_set = set()  # 用于去重（因为是无向图）
        
        # 方法1: 尝试直接遍历 layer 的 edges
        try:
            if hasattr(layer, 'edges'):
                # edges 是属性，返回迭代器，不是方法
                edge_count = 0
                for edge_item in layer.edges:
                    edge_count += 1
                    # 调试：只打印前几个 edge_item 的格式
                    if edge_count <= 3:
                        print(f"  调试: edge_item 类型: {type(edge_item)}, 值: {edge_item}")
                    
                    # edge_item 可能是 (key, edge_info) 或 edge_info 或其他格式
                    if isinstance(edge_item, tuple) and len(edge_item) == 2:
                        edge_key, edge_info = edge_item
                        # 获取 source 和 target
                        if hasattr(edge_key, 'source') and hasattr(edge_key, 'target'):
                            source_id = edge_key.source
                            target_id = edge_key.target
                        elif isinstance(edge_key, (list, tuple)) and len(edge_key) >= 2:
                            source_id = edge_key[0]
                            target_id = edge_key[1]
                        else:
                            if edge_count <= 3:
                                print(f"    无法提取 source/target from edge_key: {edge_key}")
                            continue
                    else:
                        # 可能 edge_item 本身就是 edge_key，或者有其他格式
                        if edge_count <= 3:
                            print(f"    edge_item 不是 tuple，尝试其他方式...")
                        # 尝试直接访问属性
                        if hasattr(edge_item, 'source') and hasattr(edge_item, 'target'):
                            source_id = edge_item.source
                            target_id = edge_item.target
                            edge_info = edge_item
                        else:
                            continue
                    
                    # 创建边的唯一标识（避免重复）
                    edge_key_tuple = tuple(sorted([source_id, target_id]))
                    if edge_key_tuple in edge_set:
                        continue
                    edge_set.add(edge_key_tuple)
                    
                    # 获取 weight
                    weight = None
                    if hasattr(edge_info, 'info') and hasattr(edge_info.info, 'weight'):
                        weight = edge_info.info.weight
                    elif hasattr(edge_info, 'weight'):
                        weight = edge_info.weight
                    
                    edges.append({
                        'source': source_id,
                        'target': target_id,
                        'weight': weight
                    })
                
                if edge_count > 0 and len(edges) == 0:
                    print(f"  警告: 遍历了 {edge_count} 个 edge_item，但没有成功提取任何边")
        except Exception as e:
            print(f"Warning: Failed to get edges from layer.edges: {e}")
            import traceback
            traceback.print_exc()
        
        # 方法2: 如果方法1失败，通过节点的 siblings 获取边
        if len(edges) == 0:
            print("  尝试通过节点的 siblings 获取边...")
            for node in layer.nodes:
                node_id = node.id.value
                try:
                    siblings = list(node.siblings())
                    for sibling in siblings:
                        edge_key_tuple = tuple(sorted([node_id, sibling]))
                        if edge_key_tuple in edge_set:
                            continue
                        edge_set.add(edge_key_tuple)
                        
                        # 尝试获取边的信息
                        weight = None
                        try:
                            # 尝试通过 graph 获取边
                            if hasattr(self.graph, 'getEdge'):
                                edge = self.graph.getEdge(node_id, sibling)
                            elif hasattr(layer, 'getEdge'):
                                edge = layer.getEdge(node_id, sibling)
                            else:
                                edge = None
                            
                            if edge:
                                if hasattr(edge, 'info') and hasattr(edge.info, 'weight'):
                                    weight = edge.info.weight
                                elif hasattr(edge, 'weight'):
                                    weight = edge.weight
                        except Exception:
                            pass
                        
                        edges.append({
                            'source': node_id,
                            'target': sibling,
                            'weight': weight
                        })
                except Exception as e:
                    continue
        
        return edges
    
    def get_rooms(self) -> List[NodeInfo]:
        if not self.graph.has_layer(dsg.DsgLayers.ROOMS):
            return []
        
        layer = self.graph.get_layer(dsg.DsgLayers.ROOMS)
        rooms = []
        
        for node in layer.nodes:
            node_info = self._extract_node_info(node, dsg.DsgLayers.ROOMS)
            if node_info:
                rooms.append(node_info)
        
        return rooms
    
    def get_buildings(self) -> List[NodeInfo]:
        if not self.graph.has_layer(dsg.DsgLayers.BUILDINGS):
            return []
        
        layer = self.graph.get_layer(dsg.DsgLayers.BUILDINGS)
        buildings = []
        
        for node in layer.nodes:
            node_info = self._extract_node_info(node, dsg.DsgLayers.BUILDINGS)
            if node_info:
                buildings.append(node_info)
        
        return buildings
    
    def get_agents(self, agent_prefix: str = "a") -> List[NodeInfo]:
        try:
            agents_layer = self.graph.get_dynamic_layer(dsg.DsgLayers.AGENTS, agent_prefix)
            agents = []
            
            for node in agents_layer.nodes:
                node_info = NodeInfo(
                    node_id=node.id.value,
                    layer=f"AGENTS_{agent_prefix}",
                    position=np.array([node.attributes.position.x, 
                                     node.attributes.position.y, 
                                     node.attributes.position.z]),
                    semantic_label=None,
                    name=None
                )
                agents.append(node_info)
            
            return agents
        except Exception:
            return []
    
    def query_by_filter(self, filter: QueryFilter) -> List[NodeInfo]:
        results = []
        
        if filter.layer is not None:
            results = self.get_layer_nodes(filter.layer)
        elif filter.layer_name == "objects":
            results = self.get_objects(
                semantic_label=filter.semantic_label,
                name=filter.node_name,
                min_pos=filter.min_position,
                max_pos=filter.max_position
            )
        elif filter.layer_name == "places":
            results = self.get_places()
        elif filter.layer_name == "rooms":
            results = self.get_rooms()
        elif filter.layer_name == "buildings":
            results = self.get_buildings()
        else:
            # query all layers
            for layer_id in _get_all_dsg_layer_ids():
                if self.graph.has_layer(layer_id):
                    results.extend(self.get_layer_nodes(layer_id))
        
        # apply additional filters
        if filter.semantic_label is not None:
            results = [n for n in results if n.semantic_label == filter.semantic_label]
        if filter.node_name is not None:
            results = [n for n in results if n.name == filter.node_name]
        
        return results
    
    def _vector_to_array(self, vec) -> Optional[np.ndarray]:
        """
        将各种格式的向量转换为 numpy 数组（3D）
        
        支持格式：
        - numpy 数组
        - list/tuple
        - 支持索引访问的对象 (vec[0], vec[1], vec[2])
        - 有 x, y, z 属性的对象 (vec.x, vec.y, vec.z)
        """
        # 方法1: 直接尝试转换为 numpy 数组
        try:
            arr = np.array(vec, dtype=float)
            if arr.size >= 3:
                return arr.flatten()[:3]
        except (TypeError, ValueError):
            pass
        
        # 方法2: 如果是数组类型
        if isinstance(vec, (list, tuple, np.ndarray)):
            try:
                arr = np.array(vec, dtype=float)
                if len(arr) >= 3:
                    return arr[:3]
            except (TypeError, ValueError):
                pass
        
        # 方法3: 尝试通过索引访问
        if hasattr(vec, '__getitem__'):
            try:
                return np.array([vec[0], vec[1], vec[2]], dtype=float)
            except (IndexError, TypeError, ValueError):
                pass
        
        # 方法4: 尝试访问 x, y, z 属性
        if hasattr(vec, 'x') and hasattr(vec, 'y') and hasattr(vec, 'z'):
            try:
                return np.array([float(vec.x), float(vec.y), float(vec.z)])
            except (AttributeError, ValueError, TypeError):
                pass
        
        return None
    
    def _extract_node_info(self, node, layer_id: int) -> Optional[NodeInfo]:
        """extract node information"""
        try:
            # Use node.attributes (property) instead of node.attributes() (method)
            # This matches the usage in get_objects method
            attrs = node.attributes
            
            # Handle position - it might be a numpy array or an object with .x, .y, .z
            if isinstance(attrs.position, np.ndarray):
                pos = attrs.position
            else:
                pos = np.array([attrs.position.x, attrs.position.y, attrs.position.z])
            
            node_info = NodeInfo(
                node_id=node.id.value,
                layer=_get_layer_name(layer_id),
                position=pos
            )
                
            # extract semantic label
            if hasattr(attrs, 'semantic_label'):
                node_info.semantic_label = attrs.semantic_label
            
            # extract name
            if hasattr(attrs, 'name'):
                node_info.name = attrs.name
            
            # extract bounding box
            if hasattr(attrs, 'bounding_box'):
                bbox = attrs.bounding_box
                try:
                    bbox_dict = {
                        'type': str(bbox.type) if hasattr(bbox, 'type') else 'AABB',
                    }
                    # 格式1: 有 pos 和 dim（中心位置 + 尺寸）
                    # if hasattr(bbox, 'pos') and hasattr(bbox, 'dim'):
                    #     pos_array = self._vector_to_array(bbox.pos)
                    #     dim_array = self._vector_to_array(bbox.dim)
                        
                    #     if pos_array is not None and dim_array is not None:
                    #         half_dim = dim_array / 2.0
                    #         bbox_dict['min'] = (-half_dim).tolist()
                    #         bbox_dict['max'] = half_dim.tolist()
                    #         bbox_dict['dimensions'] = dim_array.tolist()
                    #         bbox_dict['center'] = pos_array.tolist()
                    
                    # 格式2: 有 min 和 max（直接使用）
                    if hasattr(bbox, 'min') and hasattr(bbox, 'max'):
                        min_array = self._vector_to_array(bbox.min)
                        max_array = self._vector_to_array(bbox.max)
                        
                        if min_array is not None and max_array is not None:
                            bbox_dict['min'] = min_array.tolist()
                            bbox_dict['max'] = max_array.tolist()
                    
                    # # 格式3: 有 dimensions（尺寸）
                    # elif hasattr(bbox, 'dimensions'):
                    #     dim_array = self._vector_to_array(bbox.dimensions)
                        
                    #     if dim_array is not None:
                    #         half_dim = dim_array / 2.0
                    #         bbox_dict['min'] = (-half_dim).tolist()
                    #         bbox_dict['max'] = half_dim.tolist()
                    #         bbox_dict['dimensions'] = dim_array.tolist()
                    
                    # 如果成功提取了数据，保存到 node_info
                    if 'min' in bbox_dict and 'max' in bbox_dict:
                        node_info.bounding_box = bbox_dict
                    else:
                        # 如果都没有，设置为 None
                        node_info.bounding_box = None
                        
                except Exception as e:
                    # 如果提取失败，打印错误信息（用于调试）
                    print(f"Warning: Failed to extract bounding box for node {node.id.value}: {e}")
                    import traceback
                    traceback.print_exc()  # 打印完整堆栈跟踪
                    node_info.bounding_box = None
            
            # extract mesh connections
            if hasattr(attrs, 'mesh_connections'):
                node_info.mesh_connections = list(attrs.mesh_connections)
            elif hasattr(attrs, 'pcl_mesh_connections'):
                node_info.mesh_connections = list(attrs.pcl_mesh_connections)
            
            # extract distance - 对于 Place 节点特别重要
            if hasattr(attrs, 'distance'):
                try:
                    distance_value = attrs.distance
                    # 检查 distance 是否是有效的数值
                    if distance_value is not None:
                        # 尝试转换为 float
                        if isinstance(distance_value, (int, float)):
                            node_info.distance = float(distance_value)
                        else:
                            # 尝试从对象中提取
                            try:
                                node_info.distance = float(distance_value)
                            except (ValueError, TypeError):
                                node_info.distance = None
                    else:
                        node_info.distance = None
                except Exception as e:
                    # 如果提取失败，设置为 None
                    node_info.distance = None
            
            return node_info
        except Exception as e:
            # Print error for debugging
            print(f"Error in _extract_node_info: {e}")
            traceback.print_exc()
            return None


class DsgStatistics:
    """DSG statistics tool"""
    
    def __init__(self, graph: dsg.DynamicSceneGraph):
        self.graph = graph
    
    def get_summary(self) -> Dict[str, Any]:
        """获取场景图摘要统计"""
        stats = {
            'total_nodes': self.graph.num_nodes(),
            'total_edges': self.graph.num_edges(),
            'has_mesh': self.graph.has_mesh(),
            'layers': {},
            'dynamic_layers': {}
        }
        
        if stats['has_mesh']:
            mesh = self.graph.mesh()
            stats['mesh'] = {
                'vertices': mesh.num_vertices(),
                'faces': mesh.num_faces(),
                'has_colors': getattr(mesh, 'has_colors', False),
                'has_labels': getattr(mesh, 'has_labels', False)
            }
        
        # 统计各静态层
        # define static layer name mapping (to avoid confusion with dynamic layers)
        # note: some layer IDs may correspond to multiple names (e.g. OBJECTS/AGENTS are both 2, PLACES/STRUCTURE are both 3)
        # here we explicitly specify the names of static layers
        static_layer_names = {
            dsg.DsgLayers.OBJECTS: "OBJECTS",  # explicitly specify, to avoid confusion with AGENTS
            dsg.DsgLayers.PLACES: "PLACES",    # 明确指定，避免与 STRUCTURE 混淆
            dsg.DsgLayers.ROOMS: "ROOMS",
            dsg.DsgLayers.BUILDINGS: "BUILDINGS",
            dsg.DsgLayers.MESH_PLACES: "MESH_PLACES",
            dsg.DsgLayers.SEGMENTS: "SEGMENTS",
            dsg.DsgLayers.STRUCTURE: "PLACES",  # STRUCTURE 和 PLACES 共享层 ID 3，统一使用 PLACES
        }
        
        for layer_id in _get_all_dsg_layer_ids():
            # 对于层 ID 2，需要区分静态的 OBJECTS 层和动态的 AGENTS 层
            # has_layer(2) 检查的是静态层，如果返回 True，说明有静态的 OBJECTS 层
            if layer_id == dsg.DsgLayers.AGENTS:
                # 层 ID 2 可能是静态的 OBJECTS 层或动态的 AGENTS 层
                # 先检查是否有静态的 OBJECTS 层
                if self.graph.has_layer(layer_id):
                    layer = self.graph.get_layer(layer_id)
                    stats['layers']["OBJECTS"] = {
                        'num_nodes': layer.num_nodes(),
                        'num_edges': layer.num_edges()
                    }
                else:
                    # 没有静态的 OBJECTS 层，记录为 0
                    stats['layers']["OBJECTS"] = {
                        'num_nodes': 0,
                        'num_edges': 0
                    }
                # 动态的 AGENTS 层会在后面单独统计
                continue
                
            if self.graph.has_layer(layer_id):
                layer = self.graph.get_layer(layer_id)
                # 使用优先名称映射，如果没有则使用反向查找
                layer_name = static_layer_names.get(layer_id, _get_layer_name(layer_id))
                stats['layers'][layer_name] = {
                    'num_nodes': layer.num_nodes(),
                    'num_edges': layer.num_edges()
                }
            else:
                # 即使层不存在，也记录为 0
                layer_name = static_layer_names.get(layer_id, _get_layer_name(layer_id))
                stats['layers'][layer_name] = {
                    'num_nodes': 0,
                    'num_edges': 0
                }
        
        # 统计动态层
        try:
            dynamic_layers = self.graph.get_dynamic_layer_names()
            for layer_name in dynamic_layers:
                layer = self.graph.get_dynamic_layer(dsg.DsgLayers.AGENTS, layer_name)
                stats['dynamic_layers'][layer_name] = {
                    'num_nodes': layer.num_nodes()
                }
        except Exception:
            pass
        
        return stats
    
    def print_summary(self):
        """打印统计摘要
        
        Args:
            detail: 如果为 True，输出每个节点的 name
        """
        stats = self.get_summary()
        
        print("\n" + "="*60)
        print("Scene graph statistics")
        print("="*60)
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print(f"Has mesh: {stats['has_mesh']}")
        
        if stats['has_mesh']:
            mesh_info = stats['mesh']
            print(f"\n网格信息:")
            print(f"  Vertices: {mesh_info['vertices']}")
            print(f"  Faces: {mesh_info['faces']}")
            print(f"  Has colors: {mesh_info['has_colors']}")
            print(f"  Has labels: {mesh_info['has_labels']}")
        
        print("\n各层节点统计Layer statistics:")
        for layer_name, layer_stats in stats['layers'].items():
            # 显示所有层，包括节点数为 0 的层
            print(f"  {layer_name}: {layer_stats['num_nodes']} nodes, "
                  f"{layer_stats['num_edges']} edges")
        
        if stats['dynamic_layers']:
            print("\nDynamic layers:")
            for layer_name, layer_stats in stats['dynamic_layers'].items():
                print(f"  {layer_name}: {layer_stats['num_nodes']} nodes")
        print("="*60 + "\n")


class DsgExporter:
    """DSG exporter tool"""
    
    def __init__(self, graph: dsg.DynamicSceneGraph):
        self.graph = graph
    
    def export_mesh(self, output_path: str, format: str = 'ply'):
        """导出网格到文件"""
        if not self.graph.has_mesh():
            raise ValueError("场景图没有网格")

        
        mesh = self.graph.mesh()
        
        # 转换为 open3d mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        
        # 顶点 - 使用 get_vertices() 或 pos(i) 方法
        num_vertices = mesh.num_vertices()
        vertices = []
        for i in range(num_vertices):
            pos = mesh.pos(i)
            vertices.append([pos[0], pos[1], pos[2]])
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        
        # 面片 - 使用 get_faces() 或 face(i) 方法
        num_faces = mesh.num_faces()
        faces = []
        for i in range(num_faces):
            face = mesh.face(i)
            faces.append([face.v0, face.v1, face.v2])
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(faces))
        
        # 颜色 - 使用 color(i) 方法
        colors = []
        for i in range(num_vertices):
            try:
                c = mesh.color(i)
                colors.append([c.r/255.0, c.g/255.0, c.b/255.0])
            except Exception:
                colors.append([0.5, 0.5, 0.5])  # 默认灰色
        if colors:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # 保存
        output_path = pathlib.Path(output_path)
        if format.lower() == 'ply':
            o3d.io.write_triangle_mesh(str(output_path), mesh_o3d)
        elif format.lower() == 'obj':
            o3d.io.write_triangle_mesh(str(output_path), mesh_o3d)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def export_nodes_json(self, nodes: List[NodeInfo], output_path: str):
        """导出节点信息到 JSON"""

        data = []
        for node in nodes:
            node_dict = {
                'node_id': node.node_id,
                'layer': node.layer,
                'position': node.position.tolist(),
                'semantic_label': node.semantic_label,
                'name': node.name,
                'bounding_box': node.bounding_box,
                'mesh_connections': node.mesh_connections
            }
            data.append(node_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def extract_object_mesh(self, node: NodeInfo, output_path: str):
        """提取单个物体的网格"""
        if not self.graph.has_mesh():
            raise ValueError("场景图没有网格")
        
        if node.mesh_connections is None or len(node.mesh_connections) == 0:
            raise ValueError(f"节点 {node.node_id} 没有网格连接")

        
        mesh = self.graph.mesh()
        
        # 获取连接的顶点索引
        vertex_indices = set(node.mesh_connections)
        
        # 找到包含这些顶点的面片
        faces_to_keep = []
        vertex_map = {}  # 旧索引 -> 新索引
        new_vertex_idx = 0
        
        num_faces = mesh.num_faces()
        for i in range(num_faces):
            face = mesh.face(i)
            if (face.v0 in vertex_indices and 
                face.v1 in vertex_indices and 
                face.v2 in vertex_indices):
                # 映射顶点索引
                for v_idx in [face.v0, face.v1, face.v2]:
                    if v_idx not in vertex_map:
                        vertex_map[v_idx] = new_vertex_idx
                        new_vertex_idx += 1
                
                faces_to_keep.append((i, face))
        
        if len(faces_to_keep) == 0:
            raise ValueError(f"节点 {node.node_id} 没有有效的面片")
        
        # 创建新的 mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        
        # 顶点
        vertices = []
        colors = []
        for old_idx, new_idx in sorted(vertex_map.items(), key=lambda x: x[1]):
            pos = mesh.pos(old_idx)
            vertices.append([pos[0], pos[1], pos[2]])
            try:
                c = mesh.color(old_idx)
                colors.append([c.r/255.0, c.g/255.0, c.b/255.0])
            except Exception:
                colors.append([0.5, 0.5, 0.5])  # 默认灰色
        
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        if colors:
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # 面片
        faces = []
        for face_idx, face in faces_to_keep:
            faces.append([vertex_map[face.v0], 
                         vertex_map[face.v1], 
                         vertex_map[face.v2]])
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(faces))
        
        # 保存
        o3d.io.write_triangle_mesh(str(output_path), mesh_o3d)


class DsgVisualizer:
    """
    DSG 可视化工具
    
    提供基于 Open3D 的交互式 3D 场景图可视化功能。
    支持静态显示已保存的场景图文件。
    """
    
    def __init__(self, graph: dsg.DynamicSceneGraph):
        """
        初始化可视化器
        
        Args:
            graph: DynamicSceneGraph 对象
        """
        self.graph = graph
        self.visualizer = None

    def _prepare_graph_for_visualization(self, show_node_ids: bool = False) -> dsg.DynamicSceneGraph:
        """
        Prepare graph for visualization.
        
        If show_node_ids is True, creates a copy of the graph and sets node names to their IDs.
        Otherwise, returns the original graph.
        """
        if not show_node_ids:
            return self.graph
            
        # Clone the graph to avoid modifying the original
        # We use binary serialization for deep copy as it is reliable for C++ bindings
        
        # Create a temporary file to save and load the graph
        fd, temp_path = tempfile.mkstemp(suffix='.sparkdsg')
        os.close(fd)
            
        try:
            self.graph.save(temp_path)
            viz_graph = dsg.DynamicSceneGraph.load(temp_path)
            
            # If load failed, fall back to original
            if viz_graph is None:
                print("Warning: Failed to clone graph for visualization, using original graph.")
                return self.graph
                
        except Exception as e:
            print(f"Warning: Error during graph cloning: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return self.graph
            
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Update node names with IDs for OBJECTS layer only
        # This helps identify objects in visualization (e.g., "O(11)")
        layers_to_check = [
            dsg.DsgLayers.OBJECTS, 
        ]
        
        for layer_id in layers_to_check:
            if viz_graph.has_layer(layer_id):
                layer = viz_graph.get_layer(layer_id)
                for node in layer.nodes:  # layer.nodes is an iterator, not a method
                    # Set name to NodeSymbol label (e.g., "O(11)")
                    try:
                        # Use NodeSymbol to get formatted label like "O(11)"
                        # In C++: NodeSymbol(node.id).getLabel()
                        # In Python: NodeSymbol(node.id).get_label() or similar
                        node_label = str(dsg.NodeSymbol(node.id.value))
                    except Exception as e:
                        print(f"Error in _prepare_graph_for_visualization: {e}")
                    # Update name attribute
                    # Note: This assumes attributes are mutable and reflected in C++
                    node.attributes.name = node.attributes.name + node_label
                    
        return viz_graph
    
    def visualize(self, start_remote: bool = True, show_node_ids: bool = False):
        """
        Start visualization.
        
        Args:
            start_remote: Whether to use remote mode (default True, automatically starts visualizer window).
            show_node_ids: Whether to show node IDs (default False, useful for debugging).
                          If True, creates a graph copy and sets node IDs as names.
        """
        try:
            # Check if open3d visualization is enabled
            if not OPEN3D_VISUALIZER_ENABLED:
                raise ImportError("Open3D visualization not enabled, check dependencies (open3d, seaborn, scipy, zmq)")
            
            # Prepare graph for visualization
            if show_node_ids:
                print("Preparing graph for visualization (setting node IDs as names)...")
            viz_graph = self._prepare_graph_for_visualization(show_node_ids=show_node_ids)
            
            print("Starting visualizer window...")
            print("Hint: Close the visualization window to exit")
            
            # Use render_to_open3d function
            print(f"Using render_to_open3d (start_remote={start_remote})...")
            render_to_open3d(
                viz_graph,  # Use the prepared graph
                block=True,  # Block until window is closed
                start_remote=start_remote
            )
            print("Visualization window closed")
        except KeyboardInterrupt:
            print("\nUser interrupted, exiting visualization...")
            raise
        except Exception as e:
            print(f"Error during visualization: {e}")
            traceback.print_exc()
            raise
    
    def stop(self):
        """Stop visualization"""
        if self.visualizer:
            self.visualizer.stop()
    
    @staticmethod
    def visualize_static(dsg_path: str, ply_path: Optional[str] = None, start_remote: bool = True, show_node_ids: bool = False):
        """
        Start static visualization of a saved scene graph (class method).
        
        This is a convenience method for quickly loading and visualizing a saved scene graph file.
        Suitable for viewing built scene graphs, no real-time updates.
        
        Args:
            dsg_path: Path to DSG JSON file
            ply_path: Optional path to PLY mesh file
            start_remote: Whether to use remote mode (default True, automatically starts visualizer window)
                         If False, will not start a window, requires manually starting RemoteVisualizer
            show_node_ids: Whether to show node IDs (default False, useful for debugging)
        
        Example:
            >>> # Visualize only dsg.json
            >>> DsgVisualizer.visualize_static("path/to/dsg.json")
            >>> 
            >>> # Visualize dsg.json and mesh.ply, showing node IDs
            >>> DsgVisualizer.visualize_static("path/to/dsg.json", "path/to/mesh.ply", show_node_ids=True)
        """
        print(f"Loading scene graph: {dsg_path}")
        if ply_path:
            print(f"Loading mesh: {ply_path}")
        
        try:
            graph = load_dsg(dsg_path, ply_path)
            print("Scene graph loaded successfully!")
            print("Starting visualizer...")
            print("Hint: Close the visualization window to exit")
            
            viz = DsgVisualizer(graph)
            try:
                viz.visualize(start_remote=start_remote, show_node_ids=show_node_ids)
            except KeyboardInterrupt:
                print("\nUser interrupted, exiting...")
            finally:
                viz.stop()
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()


# ==================== Convenience Functions ====================

def load_dsg(dsg_path: str, ply_path: Optional[str] = None) -> dsg.DynamicSceneGraph:
    """
    Args:
        dsg_path: 
        ply_path: 
    
    Returns:
        DynamicSceneGraph
    """
    reader = DsgGraphReader(dsg_path, ply_path)
    return reader.graph


def get_graph_summary(graph: dsg.DynamicSceneGraph) -> Dict[str, Any]:
    """
    Get graph summary
    
    Args:
        graph: DynamicSceneGraph 对象
    
    Returns:
        dictionary
    """
    stats = DsgStatistics(graph)
    return stats.get_summary()


def visualize_static(dsg_path: str, ply_path: Optional[str] = None, 
    start_remote: bool = False, show_node_ids: bool = False):
    """
    Static visualization of saved scene graph
    Args:
        dsg_path: 
        ply_path: 
        start_remote: 
    
    Example:
        >>> from graphReader import visualize_static
        >>> visualize_static("dsg.json", "mesh.ply")
    """
    DsgVisualizer.visualize_static(dsg_path, ply_path, start_remote, show_node_ids)


def visualize_gvd_graph(query: DsgQuery, 
                        robot_radius: float = 0.3,
                        min_clearance: float = 0.0,
                        show_spheres: bool = True,
                        show_edges: bool = True,
                        ax=None):
    """
    可视化 GVD 图（Generalized Voronoi Diagram Graph）
    
    通过 place 节点绘制 GVD 图：
    - 每个 Place 节点是 GVD 骨架上的采样点
    - 节点的 distance 表示离最近障碍物的距离（节点大小）
    - 边的 weight 表示 Clearance（通达度/安全宽度），不是距离
    
    Args:
        query: DsgQuery 对象
        robot_radius: 机器人半径（米），用于过滤边。如果 edge.weight < robot_radius，不显示该边
        min_clearance: 最小 clearance 阈值（米），用于过滤边
        show_spheres: 是否根据 distance 显示节点大小（用球体大小表示）
        show_edges: 是否显示边
        ax: 可选的 matplotlib 3D axes，如果提供则在该 axes 上绘制
    
    Returns:
        matplotlib figure 和 axes（如果 ax 为 None）
    
    Example:
        >>> from graphReader import load_dsg, DsgQuery, visualize_gvd_graph
        >>> graph = load_dsg("dsg.json", "mesh.ply")
        >>> query = DsgQuery(graph)
        >>> visualize_gvd_graph(query, robot_radius=0.3, min_clearance=0.0)
    """

    
    places = query.get_places()
    edges = query.get_place_edges()
    
    if len(places) == 0:
        print("Warning: No place nodes found")
        return None, None
    
    # 调试：检查 distance 提取情况
    places_with_distance = [p for p in places if p.distance is not None]
    places_without_distance = [p for p in places if p.distance is None]
    print(f"\nDistance 提取情况: {len(places_with_distance)} 个节点有 distance, {len(places_without_distance)} 个节点没有 distance")
    
    # 如果大部分节点都没有 distance，尝试调试第一个节点
    if len(places_without_distance) > len(places) * 0.5 and len(places) > 0:
        print("调试: 检查第一个节点的 attributes...")
        try:
            if query.graph.has_layer(dsg.DsgLayers.PLACES):
                layer = query.graph.get_layer(dsg.DsgLayers.PLACES)
                for node in layer.nodes:
                    attrs = node.attributes
                    print(f"  节点 {node.id.value} 的 attributes 类型: {type(attrs)}")
                    print(f"  是否有 distance 属性: {hasattr(attrs, 'distance')}")
                    if hasattr(attrs, 'distance'):
                        print(f"  distance 值: {attrs.distance}, 类型: {type(attrs.distance)}")
                    break
        except Exception as e:
            print(f"  调试失败: {e}")
    
    # 统计 distance 信息
    distances = [place.distance for place in places if place.distance is not None]
    if len(distances) > 0:
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        print(f"\nDistance 统计信息:")
        print(f"  平均值: {avg_distance:.3f} m")
        print(f"  最小值: {min_distance:.3f} m")
        print(f"  最大值: {max_distance:.3f} m")
        print(f"  标准差: {std_distance:.3f} m")
    else:
        print("Warning: No distance information found")
        avg_distance = 0.5  # 默认值
        min_distance = 0.0
        max_distance = 1.0
    
    # 创建图形（如果未提供 ax）
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True
    else:
        fig = None
        created_fig = False
    
    # 创建节点 ID 到索引的映射
    node_id_to_idx = {place.node_id: i for i, place in enumerate(places)}
    
    # 提取位置
    positions = np.array([place.position for place in places])
    
    # 绘制节点
    if show_spheres:
        # 根据 distance 设置节点大小和颜色
        distances = [place.distance if place.distance is not None else avg_distance for place in places]
        if len(distances) > 0 and max(distances) > 0:
            # 节点大小：根据 distance 缩放，使用更大的缩放因子让节点更明显
            base_size = 200  # 基础大小
            size_scale = 300  # 缩放因子（增加这个值让节点更大）
            sizes = [base_size + d * size_scale for d in distances]
            
            # 节点颜色：根据 distance 使用颜色映射
            colors = plt.cm.viridis([d / max(distances) for d in distances])
            
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidths=0.5,
                      label='Place Nodes (size = distance)')
        else:
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      s=200, c='blue', alpha=0.7, label='Place Nodes')
    else:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  s=200, c='blue', alpha=0.7, label='Place Nodes')
    
    # 绘制边
    valid_edges = 0
    filtered_edges = 0
    
    if show_edges:
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            weight = edge.get('weight')
            
            # 安全性检查：如果 clearance 太小，不显示这条边
            if weight is not None:
                if weight < robot_radius:
                    filtered_edges += 1
                    continue
                if weight < min_clearance:
                    filtered_edges += 1
                    continue
            
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                source_idx = node_id_to_idx[source_id]
                target_idx = node_id_to_idx[target_id]
                
                # 绘制线段 - 增加可见性
                if weight is not None:
                    # 根据 clearance 设置颜色：越大越绿，越小越红
                    normalized_weight = min(weight / 2.0, 1.0)  # 假设最大 clearance 为 2.0m
                    color = plt.cm.RdYlGn(normalized_weight)  # 红-黄-绿
                    linewidth = max(1.5, weight * 2)  # 线宽根据 clearance 调整
                else:
                    color = 'blue'
                    linewidth = 1.5
                
                ax.plot([positions[source_idx, 0], positions[target_idx, 0]],
                       [positions[source_idx, 1], positions[target_idx, 1]],
                       [positions[source_idx, 2], positions[target_idx, 2]],
                       color=color, alpha=0.7, linewidth=linewidth)
                valid_edges += 1
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # 更新标题，包含 distance 统计信息
    title = f'GVD Graph Visualization\n'
    title += f'Places: {len(places)}, Edges: {len(edges)}, Valid edges: {valid_edges}, Filtered: {filtered_edges}\n'
    if len(distances) > 0:
        title += f'Distance: avg={avg_distance:.2f}m, min={min_distance:.2f}m, max={max_distance:.2f}m\n'
    title += f'Robot radius: {robot_radius}m, Min clearance: {min_clearance}m'
    ax.set_title(title)
    ax.legend()
    
    if created_fig:
        plt.tight_layout()
        plt.show()
    
    return fig, ax
