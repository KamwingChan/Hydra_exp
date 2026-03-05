#!/usr/bin/env python3
"""
dsg_utils.py: DSG 发布相关工具函数

封装 spark_dsg 构建和发布逻辑，避免 behavior_ros.py 过于臃肿。
用于实时模式下从 OmniGibson 场景构建并发布 DSG 消息。

NOTE: 
- 语义标签 (semantic_label) 对 phy_graph 不重要，phy_graph 通过 category 名称过滤
- 如果 Hydra 需要语义映射，参考 hydra/config/datasets/behavior.yaml
- 实时模式直接从 OmniGibson 对象获取 aabb_extent/aabb_center，不需要硬编码尺寸
"""

from typing import Dict, List, Optional, Set
import numpy as np

try:
    import spark_dsg as dsg
    SPARK_DSG_AVAILABLE = True
except ImportError:
    SPARK_DSG_AVAILABLE = False
    dsg = None

try:
    import rospy
    import hydra_msgs.msg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None

from phy_plan.core.category_filter import should_include_object as _should_include_category


def should_include_object(obj) -> bool:
    """判断是否应该包含该对象（委托到 phy_plan.core.category_filter）。"""
    return _should_include_category(
        getattr(obj, "category", ""),
        getattr(obj, "name", ""),
    )


class DsgPublisher:
    """
    封装 DSG 构建和发布逻辑
    
    用于实时模式下从 OmniGibson 场景构建并发布 spark_dsg.DynamicSceneGraph
    """
    
    def __init__(
        self, 
        topic: str = "/hydra_ros_node/backend/dsg",
        include_rooms: bool = True,
        category_filter: Optional[List[str]] = None,
        collect_gt_physics: bool = False
    ):
        """
        初始化 DSG 发布器
        
        Args:
            topic: DSG 发布的 ROS 话题
            include_rooms: 是否包含房间节点
            category_filter: 可选的类别过滤列表，None 表示包含所有
            collect_gt_physics: 是否收集真值物理属性（用于对比实验）
        """
        if not SPARK_DSG_AVAILABLE:
            raise ImportError("spark_dsg is not available. Please install it first.")
        if not ROS_AVAILABLE:
            raise ImportError("rospy/hydra_msgs not available. Please source ROS workspace.")
        
        self.topic = topic
        self.include_rooms = include_rooms
        self.category_filter = set(c.lower() for c in category_filter) if category_filter else None
        self.collect_gt_physics = collect_gt_physics
        
        self.pub = rospy.Publisher(topic, hydra_msgs.msg.DsgUpdate, queue_size=1)
        self.object_counter = 0
        self.room_counter = 0
        
        # 缓存：对象名称到节点 ID 的映射
        self._obj_name_to_node_id: Dict[str, int] = {}
        
        # GT 物理属性缓存（如果启用）
        self._gt_physics_cache: Dict[str, dict] = {}
        
        rospy.loginfo(f"DsgPublisher initialized, publishing to {topic}")
        if collect_gt_physics:
            rospy.loginfo("GT physics collection enabled")
    
    def build_and_publish(self, og_scene, timestamp=None):
        """
        从 OmniGibson 场景构建并发布 DSG
        
        Args:
            og_scene: OmniGibson 场景对象 (env.scene)
            timestamp: 可选的时间戳，默认使用当前时间
        """
        G = self._build_dsg(og_scene)
        self._publish(G, timestamp)
    
    def get_gt_physics_cache(self) -> Dict[str, dict]:
        """
        获取收集的 GT 物理属性缓存
        
        Returns:
            字典：对象名称 -> 物理属性字典
        """
        return self._gt_physics_cache.copy()
    
    def _build_dsg(self, og_scene) -> "dsg.DynamicSceneGraph":
        """
        从 OmniGibson 场景构建 DSG
        
        Args:
            og_scene: OmniGibson 场景对象
            
        Returns:
            构建好的 DynamicSceneGraph
        """
        G = dsg.DynamicSceneGraph()
        self.object_counter = 0
        self.room_counter = 0
        self._obj_name_to_node_id.clear()
        
        # 收集房间信息（如果有）
        room_objects: Dict[str, List[str]] = {}  # room_name -> [obj_name, ...]
        
        # 添加所有对象节点
        for obj in og_scene.objects:
            if not should_include_object(obj):
                continue
            
            category = obj.category.lower() if obj.category else ""
            
            # 应用类别过滤
            if self.category_filter and category not in self.category_filter:
                continue
            
            node_id = self._add_object_node(G, obj)
            
            # 收集 GT 物理属性（如果启用）
            if self.collect_gt_physics:
                self._gt_physics_cache[obj.name] = get_gt_physics_from_og_object(obj)
            
            # 收集房间信息
            if self.include_rooms and hasattr(obj, 'in_rooms'):
                in_rooms = obj.in_rooms if obj.in_rooms else []
                if isinstance(in_rooms, str):
                    in_rooms = [in_rooms] if in_rooms else []
                for room_name in in_rooms:
                    if room_name not in room_objects:
                        room_objects[room_name] = []
                    room_objects[room_name].append(obj.name)
        
        # 添加房间节点和边
        if self.include_rooms and room_objects:
            for room_name, obj_names in room_objects.items():
                room_node_id = self._add_room_node(G, room_name, obj_names)
                
                # 添加 object -> room 边
                for obj_name in obj_names:
                    if obj_name in self._obj_name_to_node_id:
                        obj_node_id = self._obj_name_to_node_id[obj_name]
                        try:
                            G.insert_edge(obj_node_id, room_node_id)
                        except Exception as e:
                            rospy.logwarn_throttle(10.0, f"Failed to add edge: {e}")
        
        return G
    
    def _add_object_node(self, G: "dsg.DynamicSceneGraph", obj) -> int:
        """
        添加 Object 节点
        
        Args:
            G: DynamicSceneGraph
            obj: OmniGibson 对象
            
        Returns:
            节点 ID
        """
        attrs = dsg.ObjectNodeAttributes()
        
        # 位置
        try:
            pos, orn = obj.get_position_orientation()
            pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.array(pos)
            attrs.position = pos_np
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"Failed to get position for {obj.name}: {e}")
            attrs.position = np.array([0.0, 0.0, 0.0])
        
        # 类别和名称
        # NOTE: semantic_label 对 phy_graph 不重要，统一用 0
        # phy_graph 通过 category 名称（attrs.name）来过滤和处理
        attrs.name = obj.category if obj.category else obj.name
        attrs.semantic_label = 0
        
        # 边界框 - 直接从 OmniGibson 获取真实尺寸
        try:
            # OmniGibson 对象直接提供 AABB
            bbox_center = obj.aabb_center
            bbox_extent = obj.aabb_extent
            
            # 转换为 numpy float32（BoundingBox 构造函数需要 float32）
            if hasattr(bbox_center, 'cpu'):
                bbox_center = bbox_center.cpu().numpy()
            if hasattr(bbox_extent, 'cpu'):
                bbox_extent = bbox_extent.cpu().numpy()
            
            center = np.array(bbox_center, dtype=np.float32)
            dimensions = np.array(bbox_extent, dtype=np.float32)
            
            # 使用简单构造函数 BoundingBox(dimensions, center)
            if np.all(dimensions > 1e-6):
                attrs.bounding_box = dsg.BoundingBox(dimensions, center)
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"Failed to get bbox for {obj.name}: {e}")
        
        # 时间戳
        if ROS_AVAILABLE:
            attrs.last_update_time_ns = int(rospy.Time.now().to_nsec())
        else:
            import time
            attrs.last_update_time_ns = int(time.time() * 1e9)
        
        attrs.is_active = True
        
        # 创建节点 ID
        node_symbol = dsg.NodeSymbol('O', self.object_counter)
        node_id = node_symbol.value
        self.object_counter += 1
        
        # 缓存映射
        self._obj_name_to_node_id[obj.name] = node_id
        
        # 添加节点
        G.add_node(dsg.DsgLayers.OBJECTS, node_id, attrs)
        
        return node_id
    
    def _add_room_node(self, G: "dsg.DynamicSceneGraph", room_name: str, obj_names: List[str]) -> int:
        """
        添加 Room 节点
        
        Args:
            G: DynamicSceneGraph
            room_name: 房间名称
            obj_names: 房间内的对象名称列表
            
        Returns:
            节点 ID
        """
        attrs = dsg.RoomNodeAttributes()
        attrs.name = room_name
        # NOTE: semantic_label 对 phy_graph 不重要，统一用 0
        attrs.semantic_label = 0
        
        # 时间戳
        if ROS_AVAILABLE:
            attrs.last_update_time_ns = int(rospy.Time.now().to_nsec())
        else:
            import time
            attrs.last_update_time_ns = int(time.time() * 1e9)
        
        # 创建节点 ID
        node_symbol = dsg.NodeSymbol('R', self.room_counter)
        node_id = node_symbol.value
        self.room_counter += 1
        
        # 添加节点
        G.add_node(dsg.DsgLayers.ROOMS, node_id, attrs)
        
        return node_id
    
    def _publish(self, G: "dsg.DynamicSceneGraph", timestamp=None):
        """
        发布 DSG 消息
        
        Args:
            G: 要发布的 DynamicSceneGraph
            timestamp: 可选的时间戳
        """
        msg = hydra_msgs.msg.DsgUpdate()
        msg.header.stamp = timestamp if timestamp else rospy.Time.now()
        msg.header.frame_id = "world"
        msg.layer_contents = G.to_binary(False)  # False = don't include mesh
        msg.full_update = True
        msg.sequence_number = 0
        
        self.pub.publish(msg)


def mass_to_weight_level(mass_kg: float) -> int:
    """
    将质量转换为权重级别
    
    Args:
        mass_kg: 质量（千克）
        
    Returns:
        权重级别: 0 (轻), 1 (中), 2 (重)
    """
    if mass_kg < 2.0:
        return 0  # 轻
    elif mass_kg < 15.0:
        return 1  # 中
    else:
        return 2  # 重


def get_gt_physics_from_og_object(obj) -> dict:
    """
    从 OmniGibson 对象获取真值物理属性
    
    Args:
        obj: OmniGibson 对象
        
    Returns:
        物理属性字典
    """
    try:
        mass = obj.mass if hasattr(obj, 'mass') else 1.0
    except Exception:
        mass = 1.0
    
    return {
        "weight_level": mass_to_weight_level(mass),
        "estimated_weight_kg": f"{mass:.1f}",
        "pushable": mass < 20.0,
        "friction_level": 1,  # 默认中等摩擦
        "source": "behavior_gt"
    }
