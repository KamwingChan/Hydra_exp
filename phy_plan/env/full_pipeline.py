# ROS1 Publisher for Behavior Simulation
# For Hydra and Phy_Graph and Phy_Plan to get the scene graph
import sys
import os

# 设置 PyTorch CUDA 内存分配策略，减少内存碎片
# 这有助于在 GPU 内存有限的情况下运行 curobo
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置 CUDA 架构列表，避免编译时的警告
# RTX 4060 的 compute capability 是 8.9
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# 添加 phy_plan 到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
phy_plan_root = os.path.dirname(current_dir)  # 从 env/ 到 phy_plan/
if phy_plan_root not in sys.path:
    sys.path.insert(0, phy_plan_root)

# 添加当前目录到路径，以便导入子模块
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import omnigibson as og
import omnigibson.lazy as lazy
import rospy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler
from std_msgs.msg import String

from camera_util import CameraMover
from config.scene_config import choose_scene
from controllers.robot_controller import RobotController
from publishers.sensor_publisher import SensorPublisher
from publishers.camera_info_publisher import CameraInfoPublisher
from utils.id_mapper import IDMapper
from utils.rosbag_manager import RosbagManager
from utils.trav_map_utils import inject_custom_trav_map

# Global configuration
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.DEFAULT_VIEWER_WIDTH = 640
gm.DEFAULT_VIEWER_HEIGHT = 480


class ExperimentBehavior:
    """Main class for ExperimentBehavior."""
    
    def __init__(self, env, scene_name,record_rosbag=False, publish_dsg=False, semantic_segmentation=True, execution_mode=None, scene_graph_file=None, online_mode=False):
        """
        Initialize ExperimentBehavior.
        """
        self.env = env
        self.sensor = og.sim.viewer_camera
        self.rate = 15  # 发布频率 Hz
        self.is_running = True
        self.semantic_segmentation = semantic_segmentation
        self.width = 640
        self.height = 480
        self.scene_name = scene_name
        self.scene_graph_file = scene_graph_file
        self.online_mode = online_mode
        # Robot and sensor setup
        self.robot = self.env.robots[0]
        pos, orn = self.robot.get_position_orientation()
        print(f"Robot initial pose: pos={pos}, orn={orn}")
        
        # Find head camera sensor
        self.eyes_sensor = None
        self.eyes_sensor_name = None
        if hasattr(self.robot, "_sensors"):
            print("=== Robot sensors (actually loaded) ===")
            for name, sensor in self.robot._sensors.items():
                print(f"  {name}: {type(sensor).__name__}")
                if "eyes" in name:
                    self.eyes_sensor = sensor
                    self.eyes_sensor_name = name
                    print(f"Use robot head sensor: {self.eyes_sensor_name}")
            print(f"Total sensors loaded: {len(self.robot._sensors)}")
        if self.eyes_sensor is None:
            rospy.logwarn("No head sensor (eyes) found for robot")
        
        # Initialize modules (composition pattern)
        self.id_mapper = IDMapper()
        self.rosbag_manager = RosbagManager(enabled=record_rosbag)
        self.robot_controller = RobotController(self.robot, self.env, curobo_batch_size=1, execution_mode=execution_mode)
        self.sensor_publisher = SensorPublisher(
            self.eyes_sensor,
            semantic_segmentation=semantic_segmentation,
            id_mapper=self.id_mapper
        )
        self.camera_info_publisher = CameraInfoPublisher(
            self.eyes_sensor,
            width=self.width,
            height=self.height
        )
        
        # Setup ROS
        self._setup_ros()
        
        # Setup camera mover
        self._setup_camera_mover()
        
        # Setup DSG publisher
        self.sensor_publisher.setup_dsg_publisher(enabled=publish_dsg)
        
        # Register pipeline factory for lazy init (P key triggers creation)
        self._register_pipeline_factory()
    
    def _setup_ros(self):
        """Setup ROS node, publishers, and subscribers."""
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
        
        # Setup publishers in modules
        self.sensor_publisher.setup_ros()
        self.camera_info_publisher.setup_ros()
        
        # Setup subscribers
        self.mode_sub = rospy.Subscriber(
            "/behavior/control_mode",
            String,
            self._on_mode_change,
            queue_size=1,
        )
        rospy.loginfo("Subscribed to /behavior/control_mode for hot-switching control modes")
    
    def _on_mode_change(self, msg: String):
        """
        ROS topic callback: hot-switch control mode.
        
        Args:
            msg: std_msgs/String, content: "idle" / "teleop" / "primitive"
        """
        mode = msg.data.strip().lower()
        self.robot_controller.set_mode(mode)
        
        # enable/disable camera_mover keyboard listening based on mode
        if mode == "teleop":
            # switch to teleop: disable camera_mover keyboard, avoid conflict
            if self.camera_mover:
                self.camera_mover.disable()
                rospy.loginfo("Camera mover keyboard disabled (teleop mode)")
        else:
            # switch to idle/primitive: re-enable camera_mover keyboard
            if self.camera_mover:
                self.camera_mover.enable()
                rospy.loginfo("Camera mover keyboard enabled")
    
    def _setup_camera_mover(self):
        """Setup camera mover for viewer camera control."""
        viewer_cam = self.sensor
        if self.scene_name == "office_vendor_machine":
            viewer_cam.set_position_orientation(
                position=[1.35, 4.63, 1.93],
                orientation=[0.19, 0.62, 0.71, 0.22]
            )
        elif self.scene_name == "Rs_int":
            viewer_cam.set_position_orientation(
                position=[0.8253, -2.6221,  1.6581],
                orientation=[0.6108, 0.1980, 0.2341, 0.7300]
            )
        self.camera_mover = CameraMover(viewer_cam)
        # if default is teleop mode, disable camera_mover keyboard listening
        if self.robot_controller.control_mode == "teleop":
            self.camera_mover.disable()
            rospy.loginfo("Camera mover keyboard disabled (teleop mode active)")
        return self.camera_mover
    
    def _register_pipeline_factory(self):
        """
        Register a factory for lazy pipeline initialization.
        
        Pipeline is NOT created at startup. Instead, the factory is called
        on first P key press, which:
        1. Creates SceneGraphSubscriber (ROS real-time scene graph)
        2. Creates LLMPlannerPipeline (interactive planning with full features)
        3. Creates DynamicPlannerPipeline (replanning + change detection)
        4. Creates PhyPlanPipeline (non-blocking adapter)
        5. Returns the PhyPlanPipeline instance
        
        This keeps startup fast and avoids unnecessary resource allocation
        when only using G key experiments.
        """
        def _create_pipeline():
            """Factory: creates the full pipeline chain on first P key press."""
            from phy_plan.input.phy_graph_subscriber import SceneGraphSubscriber
            from phy_plan.planner.llm_planner_pipeline import LLMPlannerPipeline
            from phy_plan.planner.dynamic_planner import DynamicPlannerPipeline
            from phy_plan.core.pipeline import PhyPlanPipeline
            
            rospy.loginfo("[Pipeline Factory] Creating full planning pipeline...")
            
            # 1. Scene graph source (ROS real-time subscription)
            
            if not self.online_mode and self.scene_graph_file is not None:
                subscriber = SceneGraphSubscriber.from_file(
                    file_path=self.scene_graph_file
                )
            else:
                subscriber = SceneGraphSubscriber(topic="/phy_graph/scene_graph_full")
            
            # 2. LLM planning pipeline (interactive: info_request,
            #    spatial resolution, physics validation, user confirmation)
            llm_pipeline = LLMPlannerPipeline(
                model="gpt-4o",
                subscriber=subscriber,
                enable_physics_validation=True,
                enable_spatial_resolver=True
            )
            
            # 3. Dynamic planning pipeline (replanning + change detection)
            dynamic_pipeline = DynamicPlannerPipeline(
                pipeline=llm_pipeline,
                subscriber=subscriber,
                executor=None,  # PhyPlanPipeline handles execution
            )
            
            # 4. Non-blocking pipeline adapter
            phy_pipeline = PhyPlanPipeline(
                primitive_controller=self.robot_controller.primitive_controller,
                dynamic_pipeline=dynamic_pipeline
            )
            
            rospy.loginfo("[Pipeline Factory] Pipeline ready:")
            rospy.loginfo("  - SceneGraphSubscriber listening on /phy_graph/scene_graph_full")
            rospy.loginfo("  - LLMPlannerPipeline (gpt-4o-mini, physics+spatial)")
            rospy.loginfo("  - DynamicPlannerPipeline (replanning enabled)")
            
            return phy_pipeline
        
        def _on_mode_switch(mode):
            """Callback: auto-switch control mode when P key triggers pipeline."""
            self.robot_controller.set_mode(mode)
        
        # Register factory + mode switch callback on PrimitiveController
        self.robot_controller.primitive_controller.set_pipeline_factory(
            factory=_create_pipeline,
            mode_switch_callback=_on_mode_switch
        )
        
        rospy.loginfo("[ROSBehavior] Pipeline factory registered:")
        rospy.loginfo("  P key -> lazy-init pipeline + interactive planning")
        rospy.loginfo("  G key -> hardcoded experiment (no pipeline needed)")
    
    def run(self):
        """Main loop: execute simulation step and publish data."""
        rospy.loginfo("ROSBehavior is running. Press ESC to quit.")
        self.camera_mover.print_info()
        
        rate = rospy.Rate(self.rate)
        last_update_time = rospy.Time.now()
        
        while not rospy.is_shutdown() and self.is_running:
            try:
                # update camera smooth movement (only in non-teleop mode, avoid keyboard conflict)
                current_time_ros = rospy.Time.now()
                dt = (current_time_ros - last_update_time).to_sec()
                last_update_time = current_time_ros
                if self.camera_mover and self.robot_controller.control_mode != "teleop":
                    self.camera_mover.update(dt)
                
                # decide action for this frame based on current control mode
                action = self.robot_controller.get_action()
                
                # OmniGibson simulation step
                self.env.step(action)
                
                # Publish sensor data
                current_time = rospy.Time.now()
                self.sensor_publisher.publish(
                    current_time,
                    rosbag_manager=self.rosbag_manager,
                    env_scene=self.env.scene if self.sensor_publisher.dsg_publisher else None
                )
                self.camera_info_publisher.publish(
                    current_time,
                    rosbag_manager=self.rosbag_manager
                )
                
                # Control publish frequency
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                break
        
        rospy.loginfo("ROSBehavior main loop ended.")
    
    def stop(self):
        """Stop running and cleanup."""
        self.is_running = False
        if self.rosbag_manager:
            self.rosbag_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="ROS1 Publisher for BEHAVIOR/OmniGibson Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python behavior_ros_robot.py --scene office_vendor_machine
  
  # With DSG publishing for phy_graph
  python behavior_ros_robot.py --scene office_vendor_machine --publish_dsg true
  
  # Record rosbag
  python behavior_ros_robot.py --scene office_vendor_machine --rosbag true
  
  # Use SYMBOLIC mode (no CuRobo, saves GPU memory)
  python behavior_ros_robot.py --scene office_vendor_machine --execution_mode symbolic
        """
    )
    parser.add_argument("--scene", type=str, default="office_vendor_machine",
                        help="Scene name (default: office_vendor_machine)")
    parser.add_argument("--scene_file", type=str, 
                        default="/home/kamwing/catkin_ws/src/phy_plan/env/config/scene_configs/office_vendor_machine_0.json",
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
    parser.add_argument("--scene_graph_file", type=str, 
                        default="scene_graph_enhanced.json",
                        help="Path to scene graph JSON file")
    parser.add_argument("--online_mode", type=bool, 
                        default=False,
                        help="Use online mode: true (real-time scene graph) or false (offline scene graph)")
    parser.add_argument("--semantic_segmentation", 
                        type=bool,
                        default=False,
                        help="Enable semantic segmentation")
    # Robot executor
    parser.add_argument(
        "--execution_mode",
        type=str,
        choices=["full", "symbolic"],
        default="symbolic",
        help="Execution mode: 'full' (CuRobo motion planning, GPU intensive) or 'symbolic' (teleport + physics, GPU efficient). Default: full"
    )
    parser.add_argument(
        "--trav_map",
        type=str,
        default=None,
        help="Path to custom traversability map (e.g. from generate_trav_map_accurate.py). If set, navigation uses this map instead of scene layout."
    )

    args = parser.parse_args()
    
    # parse execution mode
    from phy_plan.executor.behavior_action_api import ExecutionMode
    execution_mode = ExecutionMode.SYMBOLIC if args.execution_mode == "symbolic" else ExecutionMode.FULL
    
    env = og.Environment(choose_scene(
        args.scene, args.scene_file, args.semantic_segmentation,
        trav_map_path=args.trav_map, trav_map_resolution=0.01,
    ))
    if args.trav_map:
        try:
            inject_custom_trav_map(env, args.trav_map, custom_resolution=0.01)
            rospy.loginfo(f"Using custom trav map: {args.trav_map}")
        except Exception as e:
            rospy.logerr(f"Failed to inject custom trav map: {e}")
            raise
    experiment_behavior = ExperimentBehavior(
        env,
        scene_name=args.scene,
        record_rosbag=args.rosbag,
        publish_dsg=args.publish_dsg,
        semantic_segmentation=args.semantic_segmentation,
        execution_mode=execution_mode,
        scene_graph_file=args.scene_graph_file,
        online_mode=args.online_mode
    )
    
    def shutdown():
        """Clean up and exit."""
        rospy.loginfo("Shutting down...")
        experiment_behavior.stop()
        if experiment_behavior.camera_mover:
            experiment_behavior.camera_mover.clear()
        og.shutdown()
    
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=shutdown,
    )
    
    print("Press ESC to quit")
    
    try:
        experiment_behavior.run()
    finally:
        experiment_behavior.stop()
        if experiment_behavior.camera_mover:
            experiment_behavior.camera_mover.clear()


if __name__ == "__main__":
    main()
