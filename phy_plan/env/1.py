import rospy
import omnigibson as og
from config.scene_config import choose_scene
from utils.dsg_utils import DsgPublisher

if not rospy.core.is_initialized():
    rospy.init_node("dsg_test", anonymous=True)

cfg = choose_scene("Rs_int", "/home/kamwing/catkin_ws/src/phy_plan/env/config/scene_configs/Rs_int_T4.json")
env = og.Environment(cfg)

pub = DsgPublisher(topic="/hydra_ros_node/backend/dsg")
pub.build_and_publish(env.scene)
print("build_and_publish ok")

# 手动清理后再 shutdown
pub.pub.unregister()
del pub
import gc; gc.collect()

og.shutdown()
print("shutdown ok")