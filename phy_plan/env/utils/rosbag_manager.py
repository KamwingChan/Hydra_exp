"""
Rosbag recording manager.
"""
import rospy
import rosbag
from datetime import datetime


class RosbagManager:
    """Manages rosbag recording."""
    
    def __init__(self, enabled=False):
        """
        Initialize rosbag manager.
        
        Args:
            enabled: Whether to enable rosbag recording.
        """
        self.enabled = enabled
        self.bag = None
        self.bag_name = None
        
        if enabled:
            self._setup_rosbag()
    
    def _setup_rosbag(self):
        """Initialize rosbag if recording is enabled."""
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bag_name = f"behavior_ros_{timestamp}.bag"
        try:
            self.bag = rosbag.Bag(self.bag_name, "w")
            rospy.loginfo(f"Recording rosbag to {self.bag_name}")
        except Exception as e:
            rospy.logerr(f"Unable to create rosbag file: {e}")
            self.enabled = False
            self.bag = None
            self.bag_name = None
    
    def record(self, topic, msg, stamp):
        """Record message into rosbag if recording is active."""
        if not self.bag:
            return
        try:
            self.bag.write(topic, msg, t=stamp)
        except Exception as e:
            rospy.logwarn(f"Rosbag write failed for {topic}: {e}")
    
    def close(self):
        """Close rosbag file if opened."""
        if not self.bag:
            return
        try:
            self.bag.close()
            rospy.loginfo(f"Rosbag saved: {self.bag_name}")
        finally:
            self.bag = None
            self.bag_name = None
