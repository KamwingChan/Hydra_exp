"""
Camera info and pose publisher.
"""
import rospy
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from scipy.spatial.transform import Rotation
import tf
from tf2_msgs.msg import TFMessage


class CameraInfoPublisher:
    """Publishes camera intrinsic parameters and pose."""
    
    def __init__(self, eyes_sensor, width=640, height=480):
        """
        Initialize camera info publisher.
        
        Args:
            eyes_sensor: Robot's head camera sensor
            width: Image width
            height: Image height
        """
        self.eyes_sensor = eyes_sensor
        self.width = width
        self.height = height
        
        # ROS publishers
        self.pub_info = None
        self.pub_pose = None
        self.pub_tf = None
        
        # Topics
        self.camera_info_topic = "/camera/camera_info"
        self.pose_topic = "/camera/pose"
    
    def setup_ros(self):
        """Setup ROS publishers."""
        self.pub_info = rospy.Publisher(self.camera_info_topic, CameraInfo, queue_size=1)
        self.pub_pose = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.pub_tf = tf.TransformBroadcaster()
    
    def _get_camera_info(self):
        """
        Get camera intrinsic parameters.
        
        Returns:
            fx, fy, cx, cy: camera intrinsic parameters
        """
        try:
            if self.eyes_sensor is not None and hasattr(self.eyes_sensor, "intrinsic_matrix"):
                intrinsics = self.eyes_sensor.intrinsic_matrix
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
    
    def publish(self, current_time, frame_id="camera_optical_frame", rosbag_manager=None):
        """
        Publish camera info and pose.
        
        Args:
            current_time: rospy.Time for message timestamps
            frame_id: Frame ID for camera messages
            rosbag_manager: Optional RosbagManager for recording
        """
        if self.eyes_sensor is None:
            return
        
        # 1. Get camera pose
        pos, orn = self.eyes_sensor.get_position_orientation()
        pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.array(pos)
        orn_np = orn.cpu().numpy() if hasattr(orn, 'cpu') else np.array(orn)
        
        # Apply ROS camera coordinate system transformation
        r_cam = Rotation.from_quat(orn_np)  # orn_np is already (x, y, z, w) format
        
        # Apply rotation correction (fix point cloud orientation)
        r_correction = Rotation.from_euler('xyz', [180, 0, 0], degrees=True)
        r_new = r_cam * r_correction
        
        # Convert back to (x, y, z, w) format for ROS
        ros_quat = r_new.as_quat()  # (x, y, z, w)
        
        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(pos_np[0])
        pose_msg.pose.position.y = float(pos_np[1])
        pose_msg.pose.position.z = float(pos_np[2])
        pose_msg.pose.orientation.x = float(ros_quat[0])
        pose_msg.pose.orientation.y = float(ros_quat[1])
        pose_msg.pose.orientation.z = float(ros_quat[2])
        pose_msg.pose.orientation.w = float(ros_quat[3])
        self.pub_pose.publish(pose_msg)
        if rosbag_manager:
            rosbag_manager.record(self.pose_topic, pose_msg, current_time)
        
        # Publish TF
        self.pub_tf.sendTransform(
            (float(pos_np[0]), float(pos_np[1]), float(pos_np[2])),
            (float(ros_quat[0]), float(ros_quat[1]), float(ros_quat[2]), float(ros_quat[3])),
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
        if rosbag_manager:
            rosbag_manager.record("/tf", tf_msg, current_time)
        
        # 2. Publish CameraInfo
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
        if rosbag_manager:
            rosbag_manager.record(self.camera_info_topic, info_msg, current_time)
