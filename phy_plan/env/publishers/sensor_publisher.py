"""
Sensor data publisher (RGB, Depth, Semantic).
"""
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock

import sys
import os
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from utils.id_mapper import IDMapper


class SensorPublisher:
    """Publishes RGB, depth, and semantic segmentation images."""
    
    def __init__(self, eyes_sensor, semantic_segmentation=True, id_mapper=None):
        """
        Initialize sensor publisher.
        
        Args:
            eyes_sensor: Robot's head camera sensor
            semantic_segmentation: Whether to publish semantic segmentation
            id_mapper: Optional IDMapper for semantic ID remapping
        """
        self.eyes_sensor = eyes_sensor
        self.semantic_segmentation = semantic_segmentation
        self.id_mapper = id_mapper or IDMapper()
        self.bridge = CvBridge()
        
        # ROS publishers
        self.pub_rgb = None
        self.pub_depth = None
        self.pub_sem = None
        self.pub_clock = None
        
        # Topics
        self.rgb_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/depth/image_raw"
        self.semantic_topic = "/camera/semantic/image_raw"
        self.clock_topic = "/clock"
        
        # DSG publisher (for phy_graph)
        self.dsg_publisher = None
        self.dsg_publish_counter = 0
        self.dsg_publish_interval = 5  # 每 5 帧发布一次 DSG
    
    def setup_ros(self):
        """Setup ROS publishers."""
        self.pub_rgb = rospy.Publisher(self.rgb_topic, Image, queue_size=1)
        self.pub_depth = rospy.Publisher(self.depth_topic, Image, queue_size=1)
        if self.semantic_segmentation:
            self.pub_sem = rospy.Publisher(self.semantic_topic, Image, queue_size=1)
        self.pub_clock = rospy.Publisher(self.clock_topic, Clock, queue_size=1)
    
    def setup_dsg_publisher(self, enabled=False):
        """Setup DSG publisher if enabled."""
        if not enabled:
            return
        
        try:
            import sys
            import os
            # Add parent directory to path for imports
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from util.dsg_utils import DsgPublisher
            self.dsg_publisher = DsgPublisher()
            rospy.loginfo("DSG publishing enabled for phy_graph")
        except ImportError as e:
            rospy.logwarn(f"Failed to import DsgPublisher: {e}")
            rospy.logwarn("DSG publishing disabled. Make sure spark_dsg and hydra_msgs are available.")
            self.dsg_publisher = None
        except Exception as e:
            rospy.logerr(f"Failed to initialize DsgPublisher: {e}")
            self.dsg_publisher = None
    
    def publish(self, current_time, frame_id="camera_optical_frame", rosbag_manager=None, env_scene=None):
        """
        Get sensor data and publish to ROS topics.
        
        Args:
            current_time: rospy.Time for message timestamps
            frame_id: Frame ID for camera messages
            rosbag_manager: Optional RosbagManager for recording
            env_scene: Optional environment scene for DSG publishing
        """
        # Publish clock
        clock_msg = Clock()
        clock_msg.clock = current_time
        self.pub_clock.publish(clock_msg)
        if rosbag_manager:
            rosbag_manager.record(self.clock_topic, clock_msg, current_time)
        
        if self.eyes_sensor is None:
            rospy.logwarn("No eyes sensor found for robot")
            return
        
        # Get sensor observations
        try:
            obs, info = self.eyes_sensor.get_obs()
        except Exception as e:
            rospy.logwarn(f"Failed to get sensor obs: {e}")
            return
        
        # Publish RGB image
        if "rgb" in obs:
            rgb_data = obs["rgb"]
            if hasattr(rgb_data, 'cpu'):
                rgb_data = rgb_data.cpu().numpy()
            if rgb_data.shape[-1] == 4:
                rgb_data = rgb_data[:, :, :3]
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_data.astype(np.uint8), "rgb8")
            rgb_msg.header.stamp = current_time
            rgb_msg.header.frame_id = frame_id
            self.pub_rgb.publish(rgb_msg)
            if rosbag_manager:
                rosbag_manager.record(self.rgb_topic, rgb_msg, current_time)
        
        # Publish depth image
        if "depth" in obs:
            depth_data = obs["depth"]
            if hasattr(depth_data, 'cpu'):
                depth_data = depth_data.cpu().numpy()
            if len(depth_data.shape) == 3:
                depth_data = depth_data[:, :, 0]
            depth_msg = self.bridge.cv2_to_imgmsg(depth_data.astype(np.float32), "32FC1")
            depth_msg.header.stamp = current_time
            depth_msg.header.frame_id = frame_id
            self.pub_depth.publish(depth_msg)
            if rosbag_manager:
                rosbag_manager.record(self.depth_topic, depth_msg, current_time)
        
        # Publish semantic segmentation image
        if self.semantic_segmentation and "seg_semantic" in obs:
            sem_data = obs["seg_semantic"]
            if hasattr(sem_data, 'cpu'):
                sem_data = sem_data.cpu().numpy()
            if len(sem_data.shape) == 3:
                sem_data = sem_data[:, :, 0]
            
            # Apply ID mapping: large ID -> small ID, and convert to uint16
            sem_data_remapped = self.id_mapper.remap(sem_data.astype(np.uint32))
            
            # Use 16UC1 instead of 32UC1 to avoid cv_bridge segfault
            sem_msg = self.bridge.cv2_to_imgmsg(sem_data_remapped, "16UC1")
            sem_msg.header.stamp = current_time
            sem_msg.header.frame_id = frame_id
            self.pub_sem.publish(sem_msg)
            if rosbag_manager:
                rosbag_manager.record(self.semantic_topic, sem_msg, current_time)
        
        # Publish DSG (for phy_graph) - at lower frequency
        if self.dsg_publisher and env_scene:
            self.dsg_publish_counter += 1
            if self.dsg_publish_counter >= self.dsg_publish_interval:
                self.dsg_publish_counter = 0
                try:
                    self.dsg_publisher.build_and_publish(env_scene, current_time)
                except Exception as e:
                    rospy.logwarn_throttle(10.0, f"Failed to publish DSG: {e}")
