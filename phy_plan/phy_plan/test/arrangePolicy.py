from graphReader import DsgQuery, NodeInfo
import numpy as np
import spark_dsg as dsg
import open3d as o3d
import time
from typing import List, Tuple, Optional, Dict
import heapq
from collections import defaultdict
import tempfile
import os


class AStarPathPlanner:
    """
    A* path planner for 3D space planning from start to goal
    Consider mesh surface and collision detection
    """
    
    def __init__(self, graph, resolution=0.1, collision_radius=0.3):
        """
        Args:
            graph: DynamicSceneGraph对象
            resolution: 
            collision_radius: 
        """
        self.graph = graph
        self.resolution = resolution
        self.collision_radius = collision_radius
        self.query = DsgQuery(graph)
        
    def heuristic(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Using Euclidean distance as the heuristic function"""
        return np.linalg.norm(pos1 - pos2)
    
    def get_neighbors(self, pos: np.ndarray) -> List[np.ndarray]:
        """
        Get the neighbors of the current position
        
        Supports 18 directions (for ground movement with diagonal moves)
        This allows smoother paths compared to 6-direction movement
        """
        neighbors = []
        
        # 18方向：支持对角线移动（适合地面物体）
        # 去掉上下方向，只在地面移动
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                # 只在地面移动（z方向不变）
                offsets.append([dx * self.resolution, dy * self.resolution, 0])
        
        for offset in offsets:
            neighbor = pos + np.array(offset)
            neighbors.append(neighbor)
        return neighbors
    
    def _get_bbox_from_node_info(self, obj: NodeInfo) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get bounding box min and max from NodeInfo (直接使用已解析的字典，使用缓存避免重复转换)
        
        Returns:
            (min_corner, max_corner) or None
        """
        if obj.bounding_box is None:
            return None
        
        bbox = obj.bounding_box
        if 'min' in bbox and 'max' in bbox and bbox['min'] and bbox['max']:
            # 使用对象级缓存，避免每次重复转换 numpy 数组
            if not hasattr(obj, '_bbox_array_cache'):
                obj._bbox_array_cache = (
                    np.array(bbox['min']),
                    np.array(bbox['max'])
                )
            return obj._bbox_array_cache
        
        return None
    
    def _check_bbox_collision(self, pos1: np.ndarray, bbox1_min: np.ndarray, bbox1_max: np.ndarray,
                              pos2: np.ndarray, bbox2_min: np.ndarray, bbox2_max: np.ndarray) -> bool:
        """
        Check if two bounding boxes collide (AABB detection)
        
        Args:
            pos1, pos2: Center positions of two objects
            bbox1_min, bbox1_max: First object's bbox (relative to center)
            bbox2_min, bbox2_max: Second object's bbox (relative to center)
        """
        # Calculate world coordinates of bboxes
        bbox1_world_min = pos1 + bbox1_min
        bbox1_world_max = pos1 + bbox1_max
        bbox2_world_min = pos2 + bbox2_min
        bbox2_world_max = pos2 + bbox2_max
        
        # AABB collision detection: check for overlap
        return not (bbox1_world_max[0] < bbox2_world_min[0] or 
                   bbox1_world_min[0] > bbox2_world_max[0] or
                   bbox1_world_max[1] < bbox2_world_min[1] or 
                   bbox1_world_min[1] > bbox2_world_max[1] or
                   bbox1_world_max[2] < bbox2_world_min[2] or 
                   bbox1_world_min[2] > bbox2_world_max[2])
    
    def _check_mesh_collision(self, pos: np.ndarray, obj: NodeInfo, 
                             moving_obj_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        """
        检查位置是否与物体的 mesh 碰撞
        
        Args:
            pos: 要检查的位置（移动物体的中心）
            obj: 其他物体信息
            moving_obj_bbox: 移动物体的 bounding box（用于更精确的检测）
        
        Returns:
            True if collision detected with mesh
        """
        if not self.graph.has_mesh():
            return False
        
        if obj.mesh_connections is None or len(obj.mesh_connections) == 0:
            return False
        
        mesh = self.graph.mesh()
        vertex_indices = set(obj.mesh_connections)
        
        # 方法1: 检查移动物体的 bbox 是否与物体的 mesh 顶点重叠
        if moving_obj_bbox is not None:
            moving_min, moving_max = moving_obj_bbox
            moving_world_min = pos + np.array(moving_min)
            moving_world_max = pos + np.array(moving_max)
            
            # 检查物体的 mesh 顶点是否在移动物体的 bbox 内
            for v_idx in vertex_indices:
                try:
                    vertex_pos = mesh.pos(v_idx)
                    vertex_array = np.array([vertex_pos[0], vertex_pos[1], vertex_pos[2]])
                    
                    # 检查顶点是否在移动物体的 bbox 内
                    if np.all(vertex_array >= moving_world_min) and np.all(vertex_array <= moving_world_max):
                        return True
                except (IndexError, AttributeError):
                    continue
        
        # 方法2: 检查位置是否在物体的 mesh 顶点附近（使用碰撞半径）
        else:
            for v_idx in vertex_indices:
                try:
                    vertex_pos = mesh.pos(v_idx)
                    vertex_array = np.array([vertex_pos[0], vertex_pos[1], vertex_pos[2]])
                    
                    dist = np.linalg.norm(pos - vertex_array)
                    if dist < self.collision_radius:
                        return True
                except (IndexError, AttributeError):
                    continue
        
        return False
    
    def check_collision(self, pos: np.ndarray, exclude_node_id: Optional[int] = None,
                       moving_obj_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        """
        Check if the position is colliding with other objects
        
        同时考虑 bounding box 和 mesh 碰撞检测
        
        Args:
            pos: Position to check (center of moving object)
            exclude_node_id: Node ID to exclude (the moving object itself)
            moving_obj_bbox: Moving object's bounding box (min, max), if provided uses bbox detection
        
        Returns:
            True if collision detected
        """
        all_objects = self.query.get_objects()
        
        for obj in all_objects:
            if exclude_node_id is not None and obj.node_id == exclude_node_id:
                continue
            
            # 1. 检查 bounding box 碰撞
            obj_bbox = self._get_bbox_from_node_info(obj)
            
            bbox_collision = False
            if moving_obj_bbox is not None and obj_bbox is not None:
                # Both objects have bboxes: use AABB collision detection
                moving_min, moving_max = moving_obj_bbox
                obj_min, obj_max = obj_bbox
                
                if self._check_bbox_collision(pos, np.array(moving_min), np.array(moving_max),
                                             obj.position, np.array(obj_min), np.array(obj_max)):
                    bbox_collision = True
            elif obj_bbox is not None:
                # Other object has bbox, check if point is inside
                obj_min, obj_max = obj_bbox
                obj_world_min = obj.position + np.array(obj_min)
                obj_world_max = obj.position + np.array(obj_max)
                
                # Check if pos is inside other object's bbox
                if np.all(pos >= obj_world_min) and np.all(pos <= obj_world_max):
                    bbox_collision = True
            else:
                # Fall back to point distance detection
                dist = np.linalg.norm(pos - obj.position)
                if dist < self.collision_radius:
                    bbox_collision = True
            
            if bbox_collision:
                return True
            
            # 2. 检查 mesh 碰撞
            if self._check_mesh_collision(pos, obj, moving_obj_bbox):
                return True
        
        return False
    
    def project_to_mesh_surface(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """Project the position to the mesh surface (simplified version: keep z coordinate, assume on a plane)"""
        # if mesh is available, find the nearest mesh vertex
        if not self.graph.has_mesh():
            return pos
        return pos
    
    def plan_path(self, start: np.ndarray, goal: np.ndarray, 
                  exclude_node_id: Optional[int] = None,
                  moving_obj_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Optional[List[np.ndarray]]:
        """
        Use A* algorithm to plan the path
        
        Args:
            start: The start position [x, y, z]
            goal: The goal position [x, y, z]
            exclude_node_id: The ID of the node to exclude (the moving object itself)
            moving_obj_bbox: Moving object's bounding box (min, max) for precise collision detection
        
        Returns:
            List of path points, if no path is found, return None
        """
        # project the position to the mesh surface
        start = self.project_to_mesh_surface(start)
        goal = self.project_to_mesh_surface(goal)
        
        # A* algorithm
        open_set = [(0, tuple(start))]  # (f_score, position)
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[tuple(start)] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[tuple(start)] = self.heuristic(start, goal)
        
        visited = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            current_pos = np.array(current)
            
            if tuple(current) in visited:
                continue
            visited.add(tuple(current))
            
            # check if the goal is reached (allow some error)
            if np.linalg.norm(current_pos - goal) < self.resolution * 2:
                # reconstruct the path
                path = [goal]
                while tuple(current) in came_from:
                    current = came_from[tuple(current)]
                    path.append(np.array(current))
                path.append(start)
                path.reverse()
                return path
            
            # check the neighbors
            neighbors = self.get_neighbors(current_pos)
            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                
                if neighbor_tuple in visited:
                    continue
                
                # collision detection
                if self.check_collision(neighbor, exclude_node_id, moving_obj_bbox):
                    continue
                
                # calculate the g_score
                # For diagonal moves, distance is resolution * sqrt(2), for straight moves it's resolution
                offset = neighbor - current_pos
                move_distance = np.linalg.norm(offset)
                tentative_g = g_score[tuple(current)] + move_distance
                
                if tentative_g < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = tuple(current)
                    g_score[neighbor_tuple] = tentative_g
                    f_score[neighbor_tuple] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))
        
        # cannot find the path
        return None


class AnimatedVisualizer:
    """
    Animated visualizer, for displaying the object movement process
    """
    
    def __init__(self, graph, mesh_file: Optional[str] = None):
        self.graph = graph
        self.mesh_file = mesh_file
        self.visualizer = None
        self.geometries = []
        
        # if the graph has no mesh but has mesh_file, save the mesh_file path for later use
        if not graph.has_mesh() and mesh_file:
            self.mesh_file = mesh_file
        
    def setup_visualization(self):
        """Set up the visualization environment"""
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name="Object Movement Animation - Mesh & Path", width=1920, height=1080)
        
        # add the mesh (prioritize from the graph, if not, load from the file)
        mesh_added = False
        if self.graph.has_mesh():
            try:
                mesh = self.graph.mesh()
                mesh_o3d = self._convert_mesh_to_open3d(mesh)
                self.visualizer.add_geometry(mesh_o3d)
                self.geometries.append(mesh_o3d)
                mesh_added = True
                print(f"Loaded mesh from graph: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} faces")
            except Exception as e:
                print(f"Warning: Failed to load mesh from graph: {e}")
        
        if not mesh_added and self.mesh_file:
            try:
                if os.path.exists(self.mesh_file):
                    mesh_o3d = o3d.io.read_triangle_mesh(self.mesh_file)
                    if len(mesh_o3d.vertices) > 0:
                        self.visualizer.add_geometry(mesh_o3d)
                        self.geometries.append(mesh_o3d)
                        mesh_added = True
                        print(f"Loaded mesh from file: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} faces")
                    else:
                        print(f"Warning: Mesh file is empty: {self.mesh_file}")
                else:
                    print(f"Warning: Mesh file not found: {self.mesh_file}")
            except Exception as e:
                print(f"Warning: Failed to load mesh from file: {e}")
        
        if not mesh_added:
            print("Warning: No mesh loaded, only path and markers will be displayed")
        
        # 设置视角
        ctr = self.visualizer.get_view_control()
        ctr.set_zoom(0.7)
        
    def _convert_mesh_to_open3d(self, mesh) -> o3d.geometry.TriangleMesh:
        """Convert the DSG mesh to the Open3D mesh"""
        mesh_o3d = o3d.geometry.TriangleMesh()
        
        # vertices
        num_vertices = mesh.num_vertices()
        vertices = []
        for i in range(num_vertices):
            pos = mesh.pos(i)
            vertices.append([pos[0], pos[1], pos[2]])
        mesh_o3d.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        
        # triangles
        num_faces = mesh.num_faces()
        faces = []
        for i in range(num_faces):
            face = mesh.face(i)
            faces.append([face.v0, face.v1, face.v2])
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(faces))
        
        # colors (if available)
        try:
            colors = []
            for i in range(num_vertices):
                c = mesh.color(i)
                colors.append([c.r/255.0, c.g/255.0, c.b/255.0])
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
        except:
            pass
        
        return mesh_o3d
    
    def add_object_marker(self, position: np.ndarray, color: List[float] = [1.0, 0.0, 0.0], 
                         radius: float = 0.2) -> o3d.geometry.TriangleMesh:
        """Add an object marker (sphere)"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        self.visualizer.add_geometry(sphere)
        self.geometries.append(sphere)
        return sphere
    
    def update_object_position(self, sphere: o3d.geometry.TriangleMesh, new_position: np.ndarray):
        """Update the object position"""
        # calculate the displacement
        current_center = np.mean(np.asarray(sphere.vertices), axis=0)
        translation = new_position - current_center
        sphere.translate(translation)
        self.visualizer.update_geometry(sphere)
    
    def add_path_line(self, path: List[np.ndarray], color: List[float] = [0.0, 1.0, 0.0]):
        """Add the path line"""
        if len(path) < 2:
            return None
        
        points = np.array(path)
        lines = [[i, i+1] for i in range(len(path)-1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        
        self.visualizer.add_geometry(line_set)
        self.geometries.append(line_set)
        return line_set
    
    def add_bounding_box(self, center: np.ndarray, bbox: Dict, 
                        color: List[float] = [1.0, 0.0, 0.0]) -> Optional[o3d.geometry.LineSet]:
        """
        Add bounding box visualization
        
        Args:
            center: Center position of the bounding box
            bbox: Bounding box dictionary containing 'min' and 'max' or 'type'
            color: Line color
        
        Returns:
            LineSet object or None
        """
        if bbox is None:
            return None
        
        # Get bounding box dimensions
        if 'min' in bbox and 'max' in bbox and bbox['min'] and bbox['max']:
            min_corner = np.array(bbox['min'])
            max_corner = np.array(bbox['max'])
            
            # Calculate 8 vertices of the bounding box
            # Offset relative to center
            size = max_corner - min_corner
            half_size = size / 2.0
            
            # 8 vertices relative positions
            vertices = np.array([
                [-half_size[0], -half_size[1], -half_size[2]],  # 0
                [ half_size[0], -half_size[1], -half_size[2]],  # 1
                [ half_size[0],  half_size[1], -half_size[2]],  # 2
                [-half_size[0],  half_size[1], -half_size[2]],  # 3
                [-half_size[0], -half_size[1],  half_size[2]],  # 4
                [ half_size[0], -half_size[1],  half_size[2]],  # 5
                [ half_size[0],  half_size[1],  half_size[2]],  # 6
                [-half_size[0],  half_size[1],  half_size[2]],  # 7
            ])
            
            # Convert to world coordinates
            vertices = vertices + center
            
            # Define 12 edges (12 edges of a cube)
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
            ]
            
            # Create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            line_set.paint_uniform_color(color)
            
            self.visualizer.add_geometry(line_set)
            self.geometries.append(line_set)
            
            return line_set
        else:
            # If no min/max, use default size
            default_size = 0.3  # default 0.3 meters
            half_size = default_size / 2.0
            
            vertices = np.array([
                [-half_size, -half_size, -half_size],
                [ half_size, -half_size, -half_size],
                [ half_size,  half_size, -half_size],
                [-half_size,  half_size, -half_size],
                [-half_size, -half_size,  half_size],
                [ half_size, -half_size,  half_size],
                [ half_size,  half_size,  half_size],
                [-half_size,  half_size,  half_size],
            ]) + center
            
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(edges)
            line_set.paint_uniform_color(color)
            
            self.visualizer.add_geometry(line_set)
            self.geometries.append(line_set)
            
            return line_set
    
    def update_bounding_box_position(self, bbox_lineset: o3d.geometry.LineSet, 
                                    new_center: np.ndarray):
        """Update bounding box position"""
        if bbox_lineset is None:
            return
        
        # Get current bounding box size (calculate from first two vertices)
        vertices = np.asarray(bbox_lineset.points)
        if len(vertices) >= 2:
            # Calculate bounding box size (from difference between vertex 0 and vertex 1)
            size = np.abs(vertices[1] - vertices[0]) * 2
            
            # Calculate new 8 vertex positions
            half_size = size / 2.0
            new_vertices = np.array([
                [-half_size[0], -half_size[1], -half_size[2]],
                [ half_size[0], -half_size[1], -half_size[2]],
                [ half_size[0],  half_size[1], -half_size[2]],
                [-half_size[0],  half_size[1], -half_size[2]],
                [-half_size[0], -half_size[1],  half_size[2]],
                [ half_size[0], -half_size[1],  half_size[2]],
                [ half_size[0],  half_size[1],  half_size[2]],
                [-half_size[0],  half_size[1],  half_size[2]],
            ]) + new_center
            
            bbox_lineset.points = o3d.utility.Vector3dVector(new_vertices)
            self.visualizer.update_geometry(bbox_lineset)
    
    def animate_movement(self, path: List[np.ndarray], object_sphere: o3d.geometry.TriangleMesh,
                        bbox_lineset: Optional[o3d.geometry.LineSet] = None,
                        step_delay: float = 0.1):
        """
        Animate the movement process
        
        Args:
            path: List of path positions
            object_sphere: The sphere marker representing the object
            bbox_lineset: Optional bounding box LineSet to update during movement
            step_delay: Delay between steps (seconds)
        """
        for i, position in enumerate(path):
            self.update_object_position(object_sphere, position)
            
            # If bounding box is provided, update its position
            if bbox_lineset is not None:
                self.update_bounding_box_position(bbox_lineset, position)
            
            self.visualizer.poll_events()
            self.visualizer.update_renderer()
            time.sleep(step_delay)
    
    def close(self):
        """Close the visualization window"""
        if self.visualizer:
            self.visualizer.destroy_window()


class GraphMeshAnimatedVisualizer:
    """
    Animated visualizer based on Graph and Mesh
    使用spark_dsg的render_to_open3d来显示完整的graph和mesh结构
    Display the object movement on the graph level, not just simple geometries
    """
    
    def __init__(self, graph, mesh_file: Optional[str] = None):
        self.original_graph = graph
        self.mesh_file = mesh_file
        
    def _clone_graph(self) -> dsg.DynamicSceneGraph:
        """Create a copy of the graph for visualization (without modifying the original graph)"""
        fd, temp_path = tempfile.mkstemp(suffix='.sparkdsg')
        os.close(fd)
        
        try:
            self.original_graph.save(temp_path)
            cloned_graph = dsg.DynamicSceneGraph.load(temp_path)
            
            if cloned_graph is None:
                print("Warning: Failed to clone graph, using original graph.")
                return self.original_graph
                
        except Exception as e:
            print(f"Warning: Error during graph cloning: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return self.original_graph
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return cloned_graph
    
    def _update_node_position(self, graph: dsg.DynamicSceneGraph, node_id: int, new_position: np.ndarray) -> bool:
        """
        Update the position of the node in the graph
        
        Note: Since the attributes in the Python binding may be read-only, if it cannot be modified, it will return False
        But the visualization will still display the original graph structure
        """
        if not graph.has_layer(dsg.DsgLayers.OBJECTS):
            return False
        
        layer = graph.get_layer(dsg.DsgLayers.OBJECTS)
        
        for node in layer.nodes:
            if node.id.value == node_id:
                attrs = node.attributes
                
                # try to update the position (may fail because of the read-only attribute)
                try:
                    if isinstance(attrs.position, np.ndarray):
                        # try to assign directly instead of slicing assignment
                        attrs.position = np.array([new_position[0], new_position[1], new_position[2]], dtype=np.float64)
                    elif hasattr(attrs.position, 'x'):
                        attrs.position.x = float(new_position[0])
                        attrs.position.y = float(new_position[1])
                        attrs.position.z = float(new_position[2])
                    else:
                        return False
                    return True
                except (ValueError, AttributeError, TypeError) as e:
                    # if it cannot be modified (read-only attribute), return False but do not report an error
                    # the visualization will still display the original graph structure
                    return False
        
        return False
    
    def animate_movement_in_graph(self, path: List[np.ndarray], node_id: int,
                                  step_delay: float = 0.1, 
                                  start_remote: bool = True,
                                  key_frames_only: bool = True):
        """
        Animate the movement process on the graph and mesh level
        
        Since render_to_open3d is blocking, we use the keyframe scheme:
        1. Select several key positions (start, middle, end)
        2. 对每个关键帧，更新graph并渲染
        3. Continue to the next frame after the user closes the window
        
        Args:
            path: The list of path points
            node_id: The ID of the node to move
            step_delay: The delay per step (seconds, only used for non-keyframe mode)
            start_remote: Whether to use remote mode
            key_frames_only: If True, only display the keyframes; if False, try to display all frames (may be slow)
        """
        from spark_dsg.open3d_visualization import render_to_open3d, OPEN3D_VISUALIZER_ENABLED
        
        if not OPEN3D_VISUALIZER_ENABLED:
            raise ImportError("Open3D visualization not enabled, check dependencies")
        
        print("Creating graph copy for animation...")
        viz_graph = self._clone_graph()
        
        if key_frames_only:
            # the keyframe scheme: only display several key positions
            key_frames = []
            if len(path) > 0:
                key_frames.append(0)  # 起点
                if len(path) > 4:
                    # add several key points in the middle
                    num_key_frames = min(5, len(path))  # 最多5个关键帧
                    step = len(path) // (num_key_frames - 1)
                    for i in range(1, num_key_frames - 1):
                        key_frames.append(i * step)
                key_frames.append(len(path) - 1)  # 终点
            
            print(f"Showing {len(key_frames)} key frames of the path...")
            print("Note: Close each window to continue to the next frame")
            
            for i, frame_idx in enumerate(key_frames):
                position = path[frame_idx]
                
                print(f"\n{'='*60}")
                print(f"Key Frame {i+1}/{len(key_frames)}: Position {position}")
                print(f"Progress: {frame_idx+1}/{len(path)} points")
                print("Close the visualization window to continue to next frame...")
                print(f"{'='*60}")
                
                # 重新创建graph副本用于渲染（因为render_to_open3d可能会修改graph）
                frame_graph = self._clone_graph()
                
                # 尝试更新节点位置（如果失败，将显示原始位置）
                position_updated = self._update_node_position(frame_graph, node_id, position)
                if not position_updated:
                    print(f"Note: Cannot modify node position (read-only), showing original graph position")
                    print(f"      Target position: {position}")
                
                try:
                    render_to_open3d(
                        frame_graph,
                        block=True,  # 阻塞直到窗口关闭
                        start_remote=start_remote
                    )
                except Exception as e:
                    print(f"Error rendering frame {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        else:
            # try to display all frames (will be very slow, because each time needs to be re-rendered)
            print(f"Warning: Showing all {len(path)} frames will be very slow!")
            print("Consider using key_frames_only=True for better performance")
            
            for i, position in enumerate(path):
                self._update_node_position(viz_graph, node_id, position)
                
                if i % 10 == 0:
                    print(f"Frame {i+1}/{len(path)}: Position {position}")
                
                # recreate the graph copy
                frame_graph = self._clone_graph()
                self._update_node_position(frame_graph, node_id, position)
                
                try:
                    # non-blocking mode (if supported)
                    render_to_open3d(
                        frame_graph,
                        block=False,  # try to use non-blocking
                        start_remote=start_remote
                    )
                    time.sleep(step_delay)
                except (TypeError, ValueError):
                    # if non-blocking is not supported, fall back to the keyframe scheme
                    print("Non-blocking mode not supported, falling back to key frames...")
                    self.animate_movement_in_graph(path, node_id, step_delay, start_remote, key_frames_only=True)
                    return
                except Exception as e:
                    print(f"Error rendering frame {i+1}: {e}")
                    break


def arrange_objects_with_path_planning(graph, node_info: NodeInfo, target_position: np.ndarray,
                                       visualize: bool = True, step_delay: float = 0.1,
                                       resolution: float = 0.2, collision_radius: float = 0.5):
    """
    Use A* path planning to move the object, and visualize the movement process
    
    Args:
        graph: DynamicSceneGraph对象
        node_info: 要移动的物体信息
        target_position: 目标位置 [x, y, z]
        visualize: 是否可视化移动过程
        step_delay: 动画每步延迟（秒）
        resolution: 路径规划分辨率（米）
        collision_radius: 碰撞检测半径（米）
    
    Returns:
        bool: 是否成功
        path: 规划的路径（如果成功）
    """
    print(f"\nStart planning the path: from {node_info.position} to {target_position}")
    
    # Get moving object's bounding box for collision detection
    moving_obj_bbox = None
    if node_info.bounding_box:
        bbox = node_info.bounding_box
        if 'min' in bbox and 'max' in bbox and bbox['min'] and bbox['max']:
            # Note: bbox min/max might be relative to object center or absolute
            # We'll assume they're relative to center for now
            moving_obj_bbox = (np.array(bbox['min']), np.array(bbox['max']))
            print(f"Using bounding box for collision detection: min={bbox['min']}, max={bbox['max']}")
    
    # 创建路径规划器
    planner = AStarPathPlanner(graph, resolution=resolution, collision_radius=collision_radius)
    
    # 规划路径
    path = planner.plan_path(node_info.position, target_position, 
                            exclude_node_id=node_info.node_id,
                            moving_obj_bbox=moving_obj_bbox)
    
    if path is None:
        print("Cannot find the path!")
        return False, None
    
    print(f"Path planning successful! The path contains {len(path)} points")
    
    # 可视化
    if visualize:
        # 使用AnimatedVisualizer显示mesh和路径动画
        try:
            print("\nVisualize the mesh and path animation...")
            print("This will display the mesh scene, path lines and the moving object")
            
            # if the graph has no mesh, try to load from the mesh_file
            mesh_file = None
            if not graph.has_mesh():
                # try the default path
                default_mesh = "backend/mesh.ply"
                if os.path.exists(default_mesh):
                    mesh_file = default_mesh
                    print(f"Graph has no mesh, will load mesh from: {mesh_file}")
            
            animator = AnimatedVisualizer(graph, mesh_file=mesh_file)
            animator.setup_visualization()
            
            # add the start and end markers
            start_sphere = animator.add_object_marker(path[0], color=[0.0, 1.0, 0.0], radius=0.15)
            end_sphere = animator.add_object_marker(path[-1], color=[0.0, 0.0, 1.0], radius=0.15)
            
            # add the path line (yellow)
            path_line = animator.add_path_line(path, color=[1.0, 1.0, 0.0])
            
            # add the moving object (red sphere)
            object_sphere = animator.add_object_marker(path[0], color=[1.0, 0.0, 0.0], radius=0.2)
            
            # add bounding box (if node_info has bounding box information)
            bbox_lineset = None
            if node_info.bounding_box:
                bbox_lineset = animator.add_bounding_box(
                    path[0], 
                    node_info.bounding_box,
                    color=[1.0, 0.5, 0.0],  # Orange
                )
                print("Added bounding box visualization")
            
            # animate the movement
            print("Start the animation...")
            print(f"The path contains {len(path)} points, the object will move along the path")
            animator.animate_movement(path, object_sphere, step_delay=step_delay, 
                                     bbox_lineset=bbox_lineset)
            
            # keep the window open
            print("\nAnimation completed!")
            print("The window will stay open, you can rotate the view to see...")
            print("Press Enter to close the window...")
            input()
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'animator' in locals():
                animator.close()
    
    # Note: The original graph is not modified, only for visualization demonstration
    print("\nNote: The original graph is not modified, only for visualization demonstration")
    
    return True, path

