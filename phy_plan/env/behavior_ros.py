# ROS1 Publisher for Behavior Simulation
# For Hydra and Phy_Graph and Phy_Plan to get the scene graph
# import sys
# import os

# # 找到 behavior 的 omnigibson 路径
# og_dir = os.path.expanduser("~/workspace/BEHAVIOR-1K/OmniGibson")  # 包含 omnigibson/ 的主目录

# # 加入 Python 搜索路径
# if og_dir not in sys.path:
#     sys.path.append(og_dir)

import rospy
import argparse
import numpy as np
import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf
from camera_util import CameraMover

SCENE_NAME = "Beechwood_0_int"
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False


def choose_scene(scene_name):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_name,
        },
    }
    return cfg


class ROSBehavior:
    def __init__(self, env):
        self.env = env
        self.sensor = og.sim.viewer_camera
        self.camera_mover = None
        self.bridge = CvBridge()
        self.rate = 15  # 发布频率 Hz
        self.is_running = True
        
        # 图像分辨率
        self.width = 640
        self.height = 480
        
        # ros topics
        self.rgb_topic = "/camera/color/image_raw"
        self.camera_info_topic = "/camera/camera_info"
        self.depth_topic = "/camera/depth/image_raw"
        self.semantic_topic = "/camera/semantic/image_raw"
        self.pose_topic = "/camera/pose"
        
        # ros publishers
        self.pub_rgb = None
        self.pub_depth = None
        self.pub_sem = None
        self.pub_tf = None
        self.pub_info = None
        self.pub_pose = None
        
        # 元神启动
        self._setup_ros()
        self._setup_camera_mover()

    def _setup_camera_mover(self):
        viewer_cam = self.sensor
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
        self.pub_tf = tf.TransformBroadcaster()

    def _get_camera_info(self):
        """
        获取相机内参
        Returns:
            fx, fy, cx, cy: 相机内参
        """
        try:
            intrinsics = self.sensor.intrinsic_matrix
            if intrinsics is not None:
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
                return float(fx), float(fy), float(cx), float(cy)
        except Exception:
            pass
        # 默认值（假设 60 度 FOV）
        fov_rad = np.deg2rad(60)
        fx = self.width / (2 * np.tan(fov_rad / 2))
        fy = fx
        cx, cy = self.width / 2.0, self.height / 2.0
        return fx, fy, cx, cy

    def _publish_data(self):
        """
        获取传感器数据并发布到 ROS topics
        """
        current_time = rospy.Time.now()
        frame_id = "camera_optical_frame"
        
        # 1. 获取相机位姿
        pos, orn = self.sensor.get_position_orientation()
        pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.array(pos)
        orn_np = orn.cpu().numpy() if hasattr(orn, 'cpu') else np.array(orn)
        # omnigibson 四元数 wxyz -> ROS xyzw
        orn_xyzw = [orn_np[1], orn_np[2], orn_np[3], orn_np[0]]
        
        # 发布 PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = pos_np[0]
        pose_msg.pose.position.y = pos_np[1]
        pose_msg.pose.position.z = pos_np[2]
        pose_msg.pose.orientation.x = orn_xyzw[0]
        pose_msg.pose.orientation.y = orn_xyzw[1]
        pose_msg.pose.orientation.z = orn_xyzw[2]
        pose_msg.pose.orientation.w = orn_xyzw[3]
        self.pub_pose.publish(pose_msg)
        
        # 发布 TF
        self.pub_tf.sendTransform(
            (pos_np[0], pos_np[1], pos_np[2]),
            orn_xyzw,
            current_time,
            "camera_link",
            "world"
        )
        
        # 2. 发布 CameraInfo
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
        
        # 3. 获取传感器观测数据
        try:
            obs, info = self.sensor.get_obs()
        except Exception as e:
            rospy.logwarn(f"Failed to get sensor obs: {e}")
            return
        
        # 4. 发布 RGB 图像
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
        
        # 5. 发布深度图像
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
        
        # 6. 发布语义分割图像
        if "seg_semantic" in obs:
            sem_data = obs["seg_semantic"]
            if hasattr(sem_data, 'cpu'):
                sem_data = sem_data.cpu().numpy()
            if len(sem_data.shape) == 3:
                sem_data = sem_data[:, :, 0]
            sem_msg = self.bridge.cv2_to_imgmsg(sem_data.astype(np.uint16), "16UC1")
            sem_msg.header.stamp = current_time
            sem_msg.header.frame_id = frame_id
            self.pub_sem.publish(sem_msg)

    def run(self):
        """
        主循环：执行仿真步进并发布数据
        """
        rospy.loginfo("ROSBehavior is running. Press ESC to quit.")
        self.camera_mover.print_info()
        
        rate = rospy.Rate(self.rate)
        
        while not rospy.is_shutdown() and self.is_running:
            try:
                # omnigibson 仿真步进
                self.env.step([])
                
                # 发布传感器数据
                self._publish_data()
                
                # 控制发布频率
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                break
        
        rospy.loginfo("ROSBehavior main loop ended.")
    
    def stop(self):
        """停止运行"""
        self.is_running = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="Beechwood_0_int")
    args = parser.parse_args()
    
    env = og.Environment(choose_scene(args.scene))
    ros_behavior = ROSBehavior(env)
    
    def shutdown():
        """清理并退出"""
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
        if ros_behavior.camera_mover:
            ros_behavior.camera_mover.clear()


if __name__ == "__main__":
    main()
