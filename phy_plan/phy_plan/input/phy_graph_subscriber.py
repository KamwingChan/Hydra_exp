"""
phy_graph_subscriber.py: ROS subscriber, for receiving real-time scene graph

supports two modes:
1. ROS real-time subscription mode: from /phy_graph/scene_graph_full receive data
2. file fallback mode: load static scene graph from JSON file
"""

import json
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[Warning] rospy not available. Only file-based mode will work.")

from ..core.scene_graph import SceneGraph
from .phy_graph_io import load_scene_graph_from_dict


class SceneGraphSubscriber:
    """
    scene graph subscriber
    
    subscribe to ROS topic or load static scene graph from JSON file.
    """
    
    def __init__(self, topic: str = "/phy_graph/scene_graph_full", use_ros: bool = True):
        """
        initialize subscriber
        
        Args:
            topic: ROS topic name
            use_ros: whether to use ROS subscription (False only supports file mode)
        """
        self._scene_graph: Optional[SceneGraph] = None
        self._last_update_time: Optional[datetime] = None
        self._lock = threading.Lock()  # thread safe
        self._has_new_data = False
        self._use_ros = use_ros and ROS_AVAILABLE
        
        if self._use_ros:
            self._topic = topic
            self._subscriber = rospy.Subscriber(
                topic, 
                String, 
                self._callback,
                queue_size=1
            )
            rospy.loginfo(f"[SceneGraphSubscriber] Subscribed to {topic}")
        else:
            rospy.loginfo("[SceneGraphSubscriber] Initialized in file-only mode")
    
    def _callback(self, msg: String):
        """
        ROS callback function
        
        Args:
            msg: ROS String message (contains JSON)
        """
        try:
            # parse JSON
            data = json.loads(msg.data)
            
            # convert to SceneGraph object
            scene_graph = load_scene_graph_from_dict(data)
            
            # thread safe update
            with self._lock:
                self._scene_graph = scene_graph
                self._last_update_time = datetime.now()
                self._has_new_data = True
            
            rospy.loginfo(f"[SceneGraphSubscriber] Received scene graph: "
                         f"{len(scene_graph.rooms)} rooms, {len(scene_graph.objects)} objects")
        
        except json.JSONDecodeError as e:
            rospy.logerr(f"[SceneGraphSubscriber] Failed to parse JSON: {e}")
        except Exception as e:
            rospy.logerr(f"[SceneGraphSubscriber] Error in callback: {e}")
    
    def get_latest(self) -> Optional[SceneGraph]:
        """
        get latest scene graph
        
        Returns:
            latest SceneGraph object, None if no data
        """
        with self._lock:
            self._has_new_data = False  # reset flag after reading
            return self._scene_graph
    
    def has_update(self) -> bool:
        """
        check if there is new data
        
        Returns:
            True if there is new data since last get_latest() call
        """
        with self._lock:
            return self._has_new_data
    
    def get_last_update_time(self) -> Optional[datetime]:
        """
        get last update time
        
        Returns:
            last update time
        """
        with self._lock:
            return self._last_update_time
    
    @classmethod
    def from_file(cls, file_path: str) -> "SceneGraphSubscriber":
        """
        load scene graph from file (fallback mode)
        
        for offline testing or environments without ROS.
        
        Args:
            file_path: JSON file path
            
        Returns:
            SceneGraphSubscriber object with preloaded scene graph
            
        Example:
            >>> subscriber = SceneGraphSubscriber.from_file("data/test_scene.json")
            >>> sg = subscriber.get_latest()
        """
        # create an instance that does not use ROS
        instance = cls(use_ros=False)
        
        # load file
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Scene graph file not found: {file_path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # convert to SceneGraph
        scene_graph = load_scene_graph_from_dict(data)
        
        # set data
        instance._scene_graph = scene_graph
        instance._last_update_time = datetime.now()
        instance._has_new_data = True
        
        print(f"[SceneGraphSubscriber] Loaded scene graph from file: {file_path}")
        print(f"  - {len(scene_graph.rooms)} rooms")
        print(f"  - {len(scene_graph.objects)} objects")
        
        return instance
    
    def shutdown(self):
        """shutdown subscriber"""
        if self._use_ros and hasattr(self, '_subscriber'):
            self._subscriber.unregister()
            rospy.loginfo("[SceneGraphSubscriber] Unsubscribed")
