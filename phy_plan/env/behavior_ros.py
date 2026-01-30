# ROS1 Publisher for Behavior Simulation
# For Hydra and Phy_Graph and Phy_Plan to get the scene graph
# import sys
# import os

# # 找到 behavior 的 omnigibson 路径
# og_dir = os.path.expanduser("~/workspace/BEHAVIOR-1K/OmniGibson")  # 包含 omnigibson/ 的主目录

# # 加入 Python 搜索路径
# if og_dir not in sys.path:
#     sys.path.append(og_dir)

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy
import rospy
import rosbag
import tf
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from camera_util import CameraMover

# SCENE_NAME = "office_vendor_machine"
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.DEFAULT_VIEWER_WIDTH = 640
gm.DEFAULT_VIEWER_HEIGHT = 480

def choose_scene(scene_name, scene_file):
    cfg = {
        "render": {
            "viewer_width": gm.DEFAULT_VIEWER_WIDTH,
            "viewer_height": gm.DEFAULT_VIEWER_HEIGHT,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_name,
            "scene_file": scene_file,
        },
    }
    return cfg


class ROSBehavior:
    def __init__(self, env, record_rosbag=False, publish_dsg=False):
        self.env = env
        self.sensor = og.sim.viewer_camera
        self.camera_mover = None 
        self.bridge = CvBridge()
        self.rate = 15  # 发布频率 Hz
        self.is_running = True
        self.record_rosbag = record_rosbag
        self.publish_dsg = publish_dsg
        
        # resolution
        self.width = 640
        self.height = 480
        
        # ros topics
        self.rgb_topic = "/camera/color/image_raw"
        self.camera_info_topic = "/camera/camera_info"
        self.depth_topic = "/camera/depth/image_raw"
        self.semantic_topic = "/camera/semantic/image_raw"
        self.pose_topic = "/camera/pose"
        self.clock_topic = "/clock"
        
        # ros publishers
        self.pub_rgb = None
        self.pub_depth = None
        self.pub_sem = None
        self.pub_info = None
        self.pub_pose = None
        self.pub_clock = None
        self.pub_tf = None
        
        # rosbag related
        self.bag = None
        self.bag_name = None
        
        # DSG publisher (for phy_graph)
        self.dsg_publisher = None
        self.dsg_publish_counter = 0
        self.dsg_publish_interval = 5  # 每 5 帧发布一次 DSG
        
        # ID 映射相关
        self.id_mapping = {}  # 大ID -> 小ID 的字典
        self._load_id_mapping()
        
        # start ros
        self._setup_ros()
        self._setup_camera_mover()
        self._setup_rosbag()
        self._setup_dsg_publisher()

    def _setup_camera_mover(self):
        viewer_cam = self.sensor
        viewer_cam.set_position_orientation(position=[1.35, 4.63, 1.93], orientation=[0.19, 0.62, 0.71, 0.22])
        viewer_cam.add_modality('depth')
        viewer_cam.add_modality('seg_semantic')
        self.camera_mover = CameraMover(viewer_cam)
        return self.camera_mover

    def _setup_ros(self):
        try:
            if not rospy.core.is_initialized():
                rospy.init_node("behavior_ros", anonymous=True)
                rospy.loginfo("ROS node initialized successfully.")
            else:
                rospy.loginfo("ROS node already initialized.")
        except rospy.ROSException as e:
            rospy.loginfo(f"ROS node init failed:{e}")
        
        # Check if ROS is actually running
        if rospy.is_shutdown():
            rospy.logerr("ROS is not running. Please launch ROS Master (roscore).")
            raise RuntimeError("ROS is not running. Please launch ROS Master (roscore).")
        
        rospy.loginfo("ROS is running. Initializing publishers.")
        # setup publishers
        self.pub_rgb = rospy.Publisher(self.rgb_topic, Image, queue_size=1)
        self.pub_depth = rospy.Publisher(self.depth_topic, Image, queue_size=1)
        self.pub_sem = rospy.Publisher(self.semantic_topic, Image, queue_size=1)
        self.pub_info = rospy.Publisher(self.camera_info_topic, CameraInfo, queue_size=1)
        self.pub_pose = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.pub_clock = rospy.Publisher(self.clock_topic, Clock, queue_size=1)
        self.pub_tf = tf.TransformBroadcaster()

    def _setup_rosbag(self):
        """Initialize rosbag if recording is enabled."""
        if not self.record_rosbag:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bag_name = f"behavior_ros_{timestamp}.bag"
        try:
            self.bag = rosbag.Bag(self.bag_name, "w")
            rospy.loginfo(f"Recording rosbag to {self.bag_name}")
        except Exception as e:
            rospy.logerr(f"Unable to create rosbag file: {e}")
            self.record_rosbag = False
            self.bag = None
            self.bag_name = None
    
    def _setup_dsg_publisher(self):
        """Initialize DSG publisher if enabled."""
        if not self.publish_dsg:
            return
        
        try:
            from util.dsg_utils import DsgPublisher
            self.dsg_publisher = DsgPublisher()
            rospy.loginfo("DSG publishing enabled for phy_graph")
        except ImportError as e:
            rospy.logwarn(f"Failed to import DsgPublisher: {e}")
            rospy.logwarn("DSG publishing disabled. Make sure spark_dsg and hydra_msgs are available.")
            self.publish_dsg = False
            self.dsg_publisher = None
        except Exception as e:
            rospy.logerr(f"Failed to initialize DsgPublisher: {e}")
            self.publish_dsg = False
            self.dsg_publisher = None

    def _record_if_needed(self, topic, msg, stamp):
        """Record message into rosbag if recording is active."""
        if not self.bag:
            return
        try:
            self.bag.write(topic, msg, t=stamp)
        except Exception as e:
            rospy.logwarn(f"Rosbag write failed for {topic}: {e}")

    def _close_rosbag(self):
        """Close rosbag file if opened."""
        if not self.bag:
            return
        try:
            self.bag.close()
            rospy.loginfo(f"Rosbag saved: {self.bag_name}")
        finally:
            self.bag = None
            self.bag_name = None

    def _load_id_mapping(self):
        """
        加载 ID 映射表（大ID -> 小ID）
        从 env/behavior_remap.yaml 加载
        """
        # 获取当前文件所在目录
        env_dir = Path(__file__).parent
        local_remap = env_dir / "behavior_remap.yaml"
        
        if not local_remap.exists():
            rospy.logerr(f"Remap file not found: {local_remap}")
            rospy.logerr("Please copy behavior.yaml to env/behavior_remap.yaml")
            # 至少映射 0 -> 0 (unknown)，避免完全崩溃
            self.id_mapping[0] = 0
            return
        
        try:
            with open(local_remap, 'r') as f:
                remap_data = yaml.safe_load(f)
                for item in remap_data:
                    big_id = item['sub_id']
                    small_id = item['super_id']
                    self.id_mapping[big_id] = small_id
            rospy.loginfo(f"Loaded {len(self.id_mapping)} ID mappings from: {local_remap}")
        except Exception as e:
            rospy.logerr(f"Failed to load remap yaml: {e}")
            # 至少映射 0 -> 0 (unknown)，避免完全崩溃
            self.id_mapping[0] = 0

    def _remap_semantic_ids(self, sem_data):
        """
        将语义图像中的大ID映射为小ID
        优化：使用 np.unique + inverse 避免多次遍历全图
        """
        if not self.id_mapping:
            return sem_data.astype(np.uint16)
        
        # 转换为 uint32 确保能容纳大ID
        sem_uint32 = sem_data.astype(np.uint32)
        
        # 1. 获取唯一值和反向索引
        # unique_ids: 图像中出现的大ID (sorted)
        # inverse: 原始图像拍平后，每个像素对应在 unique_ids 中的下标
        unique_ids, inverse = np.unique(sem_uint32, return_inverse=True)
        
        # 2. 向量化查找：只对图像中出现的少量唯一ID进行字典查找
        # 相比对全图每个像素查找，或者对每个ID做全图掩码，效率极高
        mapped_vals = np.array([self.id_mapping.get(int(uid), 0) for uid in unique_ids], dtype=np.uint16)
        
        # 3. 重构图像：利用 numpy 高级索引直接生成结果
        # mapped_vals[inverse] 会根据索引一次性生成新的像素值数组
        remapped = mapped_vals[inverse].reshape(sem_data.shape)
        
        return remapped

    def _get_camera_info(self):
        """
        get camera intrinsic parameters
        Returns:
            fx, fy, cx, cy: camera intrinsic parameters
        """
        try:
            intrinsics = self.sensor.intrinsic_matrix
            if intrinsics is not None:
                intrinsics = intrinsics.cpu().numpy()
                fx = float(intrinsics[0, 0])
                fy = float(intrinsics[1, 1])
                cx = float(intrinsics[0, 2])
                cy = float(intrinsics[1, 2])
                return fx, fy, cx, cy
        except Exception:
            pass
        # default value (assume 60 degree FOV)
        fov_rad = np.deg2rad(60)
        fx = self.width / (2 * np.tan(fov_rad / 2))
        fy = fx
        cx, cy = self.width / 2.0, self.height / 2.0
        return fx, fy, cx, cy

    def _publish_data(self):
        """
        get sensor data and publish to ROS topics
        """
        current_time = rospy.Time.now()
        frame_id = "camera_optical_frame"
        
        # publish clock
        clock_msg = Clock()
        clock_msg.clock = current_time
        self.pub_clock.publish(clock_msg)
        self._record_if_needed(self.clock_topic, clock_msg, current_time)
        
        # 1. get camera pose
        pos, orn = self.sensor.get_position_orientation()
        pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.array(pos)
        orn_np = orn.cpu().numpy() if hasattr(orn, 'cpu') else np.array(orn)
        
        # 应用 ROS 相机坐标系转换
        # OmniGibson 的四元数格式是 (x, y, z, w)，scipy 也使用 (x, y, z, w) 格式，可以直接使用
        r_cam = Rotation.from_quat(orn_np)  # orn_np 已经是 (x, y, z, w) 格式
        
        # 应用旋转修正
        # 之前解决"地面像左边的墙壁"用的是: [-90, 0, -90]
        # 现在解决"点云上下颠倒"，尝试绕 X 轴旋转 180 度
        # 可以尝试不同的角度组合：
        # - [180, 0, 0] - 绕 X 轴旋转 180 度（翻转 Y 和 Z）
        # - [0, 180, 0] - 绕 Y 轴旋转 180 度
        # - [0, 0, 180] - 绕 Z 轴旋转 180 度
        # - [-90, 0, -90] - 之前的组合（解决地面像墙壁的问题）
        r_correction = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
        r_new = r_cam * r_correction
        
        # 转回 (x, y, z, w) 格式用于 ROS（ROS 也使用 x, y, z, w 格式）
        ros_quat = r_new.as_quat()  # (x, y, z, w)
        
        # publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(pos_np[0])
        pose_msg.pose.position.y = float(pos_np[1])
        pose_msg.pose.position.z = float(pos_np[2])
        pose_msg.pose.orientation.x = float(ros_quat[0])  # x
        pose_msg.pose.orientation.y = float(ros_quat[1])  # y
        pose_msg.pose.orientation.z = float(ros_quat[2])  # z
        pose_msg.pose.orientation.w = float(ros_quat[3])  # w
        self.pub_pose.publish(pose_msg)
        self._record_if_needed(self.pose_topic, pose_msg, current_time)
        
        # publish TF
        self.pub_tf.sendTransform(
            (float(pos_np[0]), float(pos_np[1]), float(pos_np[2])),
            (float(ros_quat[0]), float(ros_quat[1]), float(ros_quat[2]), float(ros_quat[3])),  # (x, y, z, w)
            current_time,
            "camera_link",
            "world"
        )
        transform_msg = TransformStamped()
        transform_msg.header.stamp = current_time
        transform_msg.header.frame_id = "world"
        transform_msg.child_frame_id = "camera_link"
        transform_msg.transform.translation.x = float(pos_np[0])
        transform_msg.transform.translation.y = float(pos_np[1])
        transform_msg.transform.translation.z = float(pos_np[2])
        transform_msg.transform.rotation.x = float(ros_quat[0])
        transform_msg.transform.rotation.y = float(ros_quat[1])
        transform_msg.transform.rotation.z = float(ros_quat[2])
        transform_msg.transform.rotation.w = float(ros_quat[3])
        tf_msg = TFMessage([transform_msg])
        self._record_if_needed("/tf", tf_msg, current_time)
        
        # 2. publish CameraInfo
        fx, fy, cx, cy = self._get_camera_info()
        info_msg = CameraInfo()
        info_msg.header.stamp = current_time
        info_msg.header.frame_id = frame_id
        info_msg.width = self.width
        info_msg.height = self.height
        info_msg.distortion_model = "plumb_bob"
        info_msg.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]
        self.pub_info.publish(info_msg)
        self._record_if_needed(self.camera_info_topic, info_msg, current_time)
        
        # 3. get sensor observations
        try:
            obs, info = self.sensor.get_obs()
        except Exception as e:
            rospy.logwarn(f"Failed to get sensor obs: {e}")
            return
        
        # 4. publish RGB image
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
            self._record_if_needed(self.rgb_topic, rgb_msg, current_time)
        
        # 5. publish depth image
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
            self._record_if_needed(self.depth_topic, depth_msg, current_time)
        
        # 6. publish semantic segmentation image
        if "seg_semantic" in obs:
            sem_data = obs["seg_semantic"]
            if hasattr(sem_data, 'cpu'):
                sem_data = sem_data.cpu().numpy()
            if len(sem_data.shape) == 3:
                sem_data = sem_data[:, :, 0]
            
            # 应用 ID 映射：大ID -> 小ID，并转换为 uint16
            sem_data_remapped = self._remap_semantic_ids(sem_data.astype(np.uint32))
            
            # 使用 16UC1 而不是 32UC1，避免 cv_bridge 段错误
            sem_msg = self.bridge.cv2_to_imgmsg(sem_data_remapped, "16UC1")
            sem_msg.header.stamp = current_time
            sem_msg.header.frame_id = frame_id
            self.pub_sem.publish(sem_msg)
            self._record_if_needed(self.semantic_topic, sem_msg, current_time)
        
        # 7. publish DSG (for phy_graph) - at lower frequency
        if self.dsg_publisher:
            self.dsg_publish_counter += 1
            if self.dsg_publish_counter >= self.dsg_publish_interval:
                self.dsg_publish_counter = 0
                try:
                    self.dsg_publisher.build_and_publish(self.env.scene, current_time)
                except Exception as e:
                    rospy.logwarn_throttle(10.0, f"Failed to publish DSG: {e}")

    def run(self):
        """
        main loop: execute simulation step and publish data
        """
        rospy.loginfo("ROSBehavior is running. Press ESC to quit.")
        self.camera_mover.print_info()
        
        rate = rospy.Rate(self.rate)
        last_update_time = rospy.Time.now()
        
        while not rospy.is_shutdown() and self.is_running:
            try:
                # 更新相机平滑移动
                current_time_ros = rospy.Time.now()
                dt = (current_time_ros - last_update_time).to_sec()
                last_update_time = current_time_ros
                if self.camera_mover:
                    self.camera_mover.update(dt)
                
                # omnigibson simulation step
                self.env.step([])
                
                # publish sensor data
                self._publish_data()
                
                # control publish frequency
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                break
        
        rospy.loginfo("ROSBehavior main loop ended.")
    
    def stop(self):
        """stop running"""
        self.is_running = False
        self._close_rosbag()


def main():
    parser = argparse.ArgumentParser(
        description="ROS1 Publisher for BEHAVIOR/OmniGibson Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python behavior_ros.py --scene office_vendor_machine
  
  # With DSG publishing for phy_graph
  python behavior_ros.py --scene office_vendor_machine --publish_dsg true
  
  # Record rosbag
  python behavior_ros.py --scene office_vendor_machine --rosbag true
        """
    )
    parser.add_argument("--scene", type=str, default="office_vendor_machine",
                        help="Scene name (default: office_vendor_machine)")
    parser.add_argument("--scene_file", type=str, 
                        default="/home/kamwing/catkin_ws/src/phy_plan/env/office_vendor_machine_0.json",
                        help="Path to scene JSON file")
    parser.add_argument(
        "--rosbag",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="Record all topics to rosbag (including TF and clock) when true",
    )
    parser.add_argument(
        "--publish_dsg",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="Publish DSG messages for phy_graph (requires spark_dsg and hydra_msgs)",
    )
    args = parser.parse_args()
    
    env = og.Environment(choose_scene(args.scene, args.scene_file))
    ros_behavior = ROSBehavior(env, record_rosbag=args.rosbag, publish_dsg=args.publish_dsg)
    
    def shutdown():
        """clean up and exit"""
        rospy.loginfo("Shutting down...")
        ros_behavior.stop()
        if ros_behavior.camera_mover:
            ros_behavior.camera_mover.clear()
        og.shutdown()
    
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=shutdown,
    )
    
    print("Press ESC to quit")
    
    try:
        ros_behavior.run()
    finally:
        ros_behavior.stop()
        if ros_behavior.camera_mover:
            ros_behavior.camera_mover.clear()


if __name__ == "__main__":
    main()
