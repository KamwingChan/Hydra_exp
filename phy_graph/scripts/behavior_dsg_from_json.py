#!/usr/bin/env python3
"""
behavior_dsg_from_json.py: 从 BEHAVIOR JSON 文件发布 DSG

用于离线模式（rosbag 回放）：
1. 从 BEHAVIOR JSON 文件构建 DSG
2. 周期性发布 DSG 消息到 /hydra_ros_node/backend/dsg
3. phy_graph C++ 节点订阅该消息进行物理推断

Usage:
    rosrun phy_graph behavior_dsg_from_json.py --json /path/to/scene.json
    
    # 或者通过 launch 文件启动
    roslaunch phy_graph behavior_offline.launch json_path:=/path/to/scene.json
"""

import argparse
import sys
from pathlib import Path

# 添加 phy_graph_lib 到路径
_script_dir = Path(__file__).resolve().parent
_lib_dir = _script_dir.parent / "src" / "phy_graph_lib"
if str(_lib_dir.parent) not in sys.path:
    sys.path.insert(0, str(_lib_dir.parent))

try:
    import rospy
    import hydra_msgs.msg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[ERROR] rospy or hydra_msgs not available. Please source ROS workspace.")
    sys.exit(1)

try:
    import spark_dsg as dsg
    from phy_graph_lib.dsg_builder import BehaviorDsgBuilder, build_dsg_from_behavior_json
    DSG_BUILDER_AVAILABLE = True
except ImportError as e:
    DSG_BUILDER_AVAILABLE = False
    print(f"[ERROR] Failed to import dsg_builder or spark_dsg: {e}")
    print("[ERROR] Make sure spark_dsg is installed.")
    sys.exit(1)


class JsonDsgPublisher:
    """
    从 JSON 文件发布 DSG 消息
    
    用于 rosbag 回放场景，提供静态场景图给 phy_graph
    """
    
    def __init__(
        self, 
        json_path: str, 
        rate: float = 2.0,
        topic: str = "/hydra_ros_node/backend/dsg",
        target_categories: list = None
    ):
        """
        初始化 JSON DSG 发布器
        
        Args:
            json_path: BEHAVIOR JSON 文件路径
            rate: 发布频率 (Hz)
            topic: DSG 发布话题
            target_categories: 要提取的类别列表，None 表示提取所有
        """
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        
        self.rate_hz = rate
        self.topic = topic
        
        # 构建 DSG（只需要构建一次，因为是静态场景）
        rospy.loginfo(f"Building DSG from: {self.json_path}")
        self.builder = BehaviorDsgBuilder(target_categories)
        self.G = self.builder.build_from_json(str(self.json_path))
        
        # 统计节点数量（nodes 是属性，不是方法）
        num_objects = 0
        num_rooms = 0
        if self.G.has_layer(dsg.DsgLayers.OBJECTS):
            objects_layer = self.G.get_layer(dsg.DsgLayers.OBJECTS)
            num_objects = sum(1 for _ in objects_layer.nodes)
        if self.G.has_layer(dsg.DsgLayers.ROOMS):
            rooms_layer = self.G.get_layer(dsg.DsgLayers.ROOMS)
            num_rooms = sum(1 for _ in rooms_layer.nodes)
        rospy.loginfo(f"DSG built: {num_objects} objects, {num_rooms} rooms")
        
        # 创建发布器
        self.pub = rospy.Publisher(topic, hydra_msgs.msg.DsgUpdate, queue_size=1)
        rospy.loginfo(f"Publishing DSG to: {topic} at {rate} Hz")
        
    def run(self):
        """主循环：周期性发布 DSG"""
        rate = rospy.Rate(self.rate_hz)
        sequence_number = 0
        
        rospy.loginfo("JsonDsgPublisher running. Press Ctrl+C to stop.")
        
        while not rospy.is_shutdown():
            current_ros_time = rospy.Time.now()
            current_time_ns = int(current_ros_time.to_nsec())
            
            # 每次发布时都更新时间戳为当前 ROS 时间
            # 这样物体节点的时间戳能持续匹配 keyframe 数据库的时间窗口
            # 在 rosbag 回放时，ROS 时间会跟随 rosbag 时间
            if self.G.has_layer(dsg.DsgLayers.OBJECTS):
                objects_layer = self.G.get_layer(dsg.DsgLayers.OBJECTS)
                for node in objects_layer.nodes:
                    attrs = node.attributes
                    attrs.last_update_time_ns = current_time_ns
            
            if self.G.has_layer(dsg.DsgLayers.ROOMS):
                rooms_layer = self.G.get_layer(dsg.DsgLayers.ROOMS)
                for node in rooms_layer.nodes:
                    attrs = node.attributes
                    attrs.last_update_time_ns = current_time_ns
            
            msg = hydra_msgs.msg.DsgUpdate()
            msg.header.stamp = current_ros_time
            msg.header.frame_id = "world"
            msg.layer_contents = self.G.to_binary(False)  # False = don't include mesh
            msg.full_update = True
            msg.sequence_number = sequence_number
            
            self.pub.publish(msg)
            sequence_number += 1
            
            if sequence_number == 1:
                rospy.loginfo(f"First DSG published with timestamp: {current_ros_time}")
            
            rate.sleep()
        
        rospy.loginfo("JsonDsgPublisher stopped.")


def main():
    # 先初始化 ROS 节点（这样可以从参数服务器读取）
    rospy.init_node("behavior_dsg_from_json", anonymous=True)
    
    parser = argparse.ArgumentParser(
        description="Publish DSG from BEHAVIOR JSON file for rosbag playback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  rosrun phy_graph behavior_dsg_from_json.py --json /path/to/scene.json
  
  # With specific categories
  rosrun phy_graph behavior_dsg_from_json.py --json /path/to/scene.json \\
      --categories swivel_chair conference_table
  
  # Custom rate
  rosrun phy_graph behavior_dsg_from_json.py --json /path/to/scene.json --rate 1.0
        """
    )
    parser.add_argument(
        "--json", "-j",
        required=False,  # 改为 False，因为可以通过 ROS param 传递
        default=None,
        help="Path to BEHAVIOR JSON file (can also be provided via ~json_path ROS parameter)"
    )
    parser.add_argument(
        "--rate", "-r",
        type=float,
        default=2.0,
        help="Publishing rate in Hz (default: 2.0)"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default="/hydra_ros_node/backend/dsg",
        help="DSG topic to publish to (default: /hydra_ros_node/backend/dsg)"
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="+",
        default=None,
        help="Categories to extract (default: all categories)"
    )
    
    # 解析命令行参数（过滤掉 ROS 特有的参数）
    args_to_parse = [arg for arg in sys.argv[1:] if not arg.startswith('__') and ':=' not in arg]
    args = parser.parse_args(args_to_parse)
    
    # 从 ROS 参数服务器获取参数（优先级高于命令行）
    # 尝试多个可能的参数名称
    json_path = None
    if rospy.has_param("~json_path"):
        json_path = rospy.get_param("~json_path")
    elif rospy.has_param("json_path"):
        json_path = rospy.get_param("json_path")
    elif args.json:
        json_path = args.json
    
    if json_path is None:
        rospy.logerr("JSON path is required. Provide via --json argument or ~json_path ROS parameter.")
        sys.exit(1)
    
    # 获取其他参数（优先从 ROS 参数服务器读取）
    rate = rospy.get_param("~rate", None)
    if rate is None:
        rate = rospy.get_param("rate", args.rate)
    
    topic = rospy.get_param("~topic", None)
    if topic is None:
        topic = rospy.get_param("topic", args.topic)
    
    # 处理 categories 参数（可能是字符串或列表）
    categories = None
    if rospy.has_param("~categories"):
        categories_param = rospy.get_param("~categories")
    elif rospy.has_param("categories"):
        categories_param = rospy.get_param("categories")
    else:
        categories_param = args.categories
    
    if categories_param is not None:
        if isinstance(categories_param, str):
            # 如果是字符串，按空格分割（支持 launch 文件中的空格分隔字符串）
            # 过滤掉空字符串
            categories = [c.strip() for c in categories_param.split() if c.strip()]
            if not categories:  # 如果分割后为空，设为 None
                categories = None
        elif isinstance(categories_param, list):
            # 过滤掉空字符串
            categories = [c for c in categories_param if c]
            if not categories:  # 如果列表为空，设为 None
                categories = None
        else:
            categories = None
    
    rospy.loginfo(f"Configuration: json_path={json_path}, rate={rate}, topic={topic}, categories={categories}")
    
    try:
        publisher = JsonDsgPublisher(
            json_path=json_path,
            rate=rate,
            topic=topic,
            target_categories=categories
        )
        publisher.run()
    except FileNotFoundError as e:
        rospy.logerr(str(e))
        sys.exit(1)
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
