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
# 当前文件在: .../phy_plan/env/behavior_ros_robot.py
# 需要添加: .../phy_plan 到路径，以便导入 phy_plan.executor
current_dir = os.path.dirname(os.path.abspath(__file__))
phy_plan_root = os.path.dirname(current_dir)  # 从 env/ 到 phy_plan/
if phy_plan_root not in sys.path:
    sys.path.insert(0, phy_plan_root)

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
from omnigibson.utils.ui_utils import KeyboardEventHandler, KeyboardRobotController
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from camera_util import CameraMover

# SCENE_NAME = "office_vendor_machine"
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.DEFAULT_VIEWER_WIDTH = 640
gm.DEFAULT_VIEWER_HEIGHT = 480

def choose_scene(scene_name, scene_file, semantic_segmentation=True):
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
        "robots": [
            {
                "type": "R1Pro",
                "name": "robot_r1",  # R1Pro 的默认名字
                "obs_modalities": ["rgb", "depth", "seg_semantic"] if semantic_segmentation else ["rgb", "depth"],
                "action_type": "continuous",
                "action_normalize": True,
                # camera config
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": 480,
                            "image_width": 640,
                        }
                    }
                },
                "include_sensor_names": ["zed_link"],  # R1Pro 的头相机关键字
                "exclude_sensor_names": ["realsense"],
            }
        ],
    }
    return cfg


class ROSBehavior:
    def __init__(self, env, record_rosbag=False, publish_dsg=False, semantic_segmentation=True):
        self.env = env
        self.sensor = og.sim.viewer_camera
        self.camera_mover = None 
        self.bridge = CvBridge()
        self.rate = 15  # 发布频率 Hz
        self.is_running = True
        self.record_rosbag = record_rosbag
        self.publish_dsg = publish_dsg
        self.semantic_segmentation = semantic_segmentation
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

        # robot & onboard camera (eyes)
        self.robot = self.env.robots[0]
        pos, orn = self.robot.get_position_orientation()
        print(f"Robot initial pose: pos={pos}, orn={orn}")
        # 控制模式：先默认键盘控制（teleop）
        self.control_mode = "primitive"  # "idle" / "teleop" / "primitive"
        self._idle_action = None
        # 键盘控制器（参考 omnigibson/examples/robots/robot_control_example.py）
        try:
            self.teleop_controller = KeyboardRobotController(robot=self.robot)
            self.teleop_controller.print_keyboard_teleop_info()
        except Exception as e:
            rospy.logwarn(f"Failed to create KeyboardRobotController: {e}")
            self.teleop_controller = None
        self.eyes_sensor = None
        self.eyes_sensor_name = None
        # 通过内部 _sensors 查找名字里包含 "zed_link" 的相机（R1Pro 的头相机）
        if hasattr(self.robot, "_sensors"):
            # 调试：打印所有实际加载的传感器
            print("=== Robot sensors (actually loaded) ===")
            for name, sensor in self.robot._sensors.items():
                print(f"  {name}: {type(sensor).__name__}")
                if "zed_link" in name:
                    self.eyes_sensor = sensor
                    self.eyes_sensor_name = name
                    print(f"Use robot head sensor: {self.eyes_sensor_name}")
            print(f"Total sensors loaded: {len(self.robot._sensors)}")
        if self.eyes_sensor is None:
            rospy.logwarn("No head sensor (zed_link) found for robot")
        
        # primitive 控制相关
        self._primitive_api = None
        self._primitive_controller = None  # generator for primitive actions
        self._g_key_pressed = False  # G 键按下标志
        
        # 注册 G 键回调（使用 KeyboardEventHandler）
        try:
            from omnigibson.utils.ui_utils import KeyboardEventHandler
            KeyboardEventHandler.add_keyboard_callback(
                lazy.carb.input.KeyboardInput.G,
                self._on_g_key_pressed
            )
            rospy.loginfo("Registered G key callback for primitive control")
        except Exception as e:
            rospy.logwarn(f"Failed to register G key callback: {e}")

        self._load_id_mapping()
        
        # start ros
        self._setup_ros()
        self._setup_camera_mover()
        self._setup_rosbag()
        self._setup_dsg_publisher()
    
    def _get_idle_action(self):
        """
        生成一个“静止不动”的占位动作（全 0），用于 idle / 控制器失败时兜底。
        """
        if self._idle_action is None:
            try:
                sample = self.robot.action_space.sample()
                import numpy as np
                self._idle_action = np.zeros_like(sample)
            except Exception:
                # 兜底：如果 action_space 不工作，就返回标量 0
                self._idle_action = 0.0
        return self._idle_action

    def _get_teleop_action(self):
        """
        从键盘控制器获取动作；如果不可用或失败，退回 idle 动作。
        同时提供 R1Pro base 的简化键位映射（I/K/J/L）。
        """
        if self.teleop_controller is None:
            return self._get_idle_action()
        try:
            action = self.teleop_controller.get_teleop_action()
            if action is None:
                # 没有按键输入时，有些实现会返回 None，这里退回 idle
                return self._get_idle_action()
            
            # R1Pro 简化键位：I/K/J/L 直接控制 base（如果当前按的是这些键）
            if hasattr(self.teleop_controller, 'current_keypress') and self.teleop_controller.current_keypress is not None:
                # 找到 base 控制器在 action 向量里的起始索引
                base_start_idx = None
                for component, info in self.teleop_controller.controller_info.items():
                    if component == "base" and "HolonomicBaseJointController" in info["name"]:
                        base_start_idx = info["start_idx"]
                        break
                
                if base_start_idx is not None:
                    key = self.teleop_controller.current_keypress
                    # I/K: 前进/后退 (x), J/L: 左转/右转 (rz)
                    if key == lazy.carb.input.KeyboardInput.I:
                        action[base_start_idx + 0] = 0.3  # x 前进
                    elif key == lazy.carb.input.KeyboardInput.K:
                        action[base_start_idx + 0] = -0.3  # x 后退
                    elif key == lazy.carb.input.KeyboardInput.J:
                        action[base_start_idx + 2] = 0.3  # rz 左转
                    elif key == lazy.carb.input.KeyboardInput.L:
                        action[base_start_idx + 2] = -0.3  # rz 右转
            
            return action
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Keyboard teleop failed, fallback to idle: {e}")
            return self._get_idle_action()
    
    def _on_g_key_pressed(self):
        """G 键按下时的回调函数"""
        rospy.loginfo(f"G key callback triggered, control_mode={self.control_mode}, primitive_controller={self._primitive_controller is not None}")
        if self.control_mode == "primitive" and self._primitive_controller is None:
            self._g_key_pressed = True
            rospy.loginfo("G key pressed - will start primitive in next frame")
        else:
            if self.control_mode != "primitive":
                rospy.logwarn(f"G key pressed but control_mode is '{self.control_mode}', not 'primitive'. Switch to primitive mode first!")
            if self._primitive_controller is not None:
                rospy.logwarn("G key pressed but primitive is already running")

    def _get_primitive_action(self):
        """
        测试用：简单的 primitive 操作
        按 G 键：执行 navigate_to coffee cup
        """
        # 延迟初始化 BehaviorActionAPI（只在第一次调用时创建）
        if self._primitive_api is None:
            try:
                # 在初始化前清理 GPU 缓存和 Python 垃圾，尝试释放一些内存
                import gc
                import torch
                gc.collect()  # Python 垃圾回收
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    rospy.loginfo(f"GPU memory before curobo init: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                    rospy.loginfo(f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
                
                from phy_plan.executor import BehaviorActionAPI
                from omnigibson.action_primitives.starter_semantic_action_primitives import (
                    StarterSemanticActionPrimitiveSet,
                )
                # 使用 batch_size=1 来减少 GPU 内存使用（默认是 3，对于 8GB GPU 可能不够）
                rospy.loginfo("Initializing BehaviorActionAPI with curobo_batch_size=1...")
                self._primitive_api = BehaviorActionAPI(
                    self.env, 
                    self.robot,
                    curobo_batch_size=1
                )
                rospy.loginfo("BehaviorActionAPI initialized successfully (curobo_batch_size=1)")
                if torch.cuda.is_available():
                    rospy.loginfo(f"GPU memory after curobo init: {torch.cuda.memory_allocated()/1024**3:.2f} GB / {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    rospy.logerr(f"GPU out of memory when initializing curobo. Current GPU usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                    rospy.logerr("Suggestions:")
                    rospy.logerr("  1. Close other GPU-intensive applications")
                    rospy.logerr("  2. Reduce scene complexity")
                    rospy.logerr("  3. Restart the simulation to free GPU memory")
                    import traceback
                    rospy.logerr(traceback.format_exc())
                else:
                    rospy.logerr(f"Failed to init BehaviorActionAPI: {e}")
                    import traceback
                    rospy.logerr(traceback.format_exc())
                self._primitive_api = None
                return self._get_idle_action()
            except Exception as e:
                rospy.logerr(f"Failed to init BehaviorActionAPI: {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                self._primitive_api = None
                return self._get_idle_action()
        
        # 如果按了 G 键，开始执行 navigate_to coffee cup
        if self._g_key_pressed and self._primitive_controller is None:
            print("G key pressed, starting navigate_to primitive...")
            self._g_key_pressed = False  # 重置标志（在日志之后）
            # 查找场景中的 coffee cup
            try:
                from omnigibson.action_primitives.starter_semantic_action_primitives import (
                    StarterSemanticActionPrimitiveSet,
                )
                
                # 方法1: 通过名字查找
                target = None
                for obj in self.env.scene.objects:
                    if hasattr(obj, 'name') and 'coffee' in obj.name.lower():
                        target = obj
                        rospy.loginfo(f"Found coffee cup by name: {obj.name}")
                        break
                
                # 方法2: 如果没找到，通过 category 查找
                if target is None:
                    for obj in self.env.scene.objects:
                        if hasattr(obj, 'category') and 'coffee' in str(obj.category).lower():
                            target = obj
                            rospy.loginfo(f"Found coffee cup by category: {obj.name} (category: {obj.category})")
                            break
                
                # 方法3: 如果还是没找到，列出所有对象让用户选择
                if target is None:
                    rospy.logwarn("Coffee cup not found. Available objects:")
                    for obj in self.env.scene.objects:
                        if hasattr(obj, 'name'):
                            rospy.logwarn(f"  - {obj.name} (category: {getattr(obj, 'category', 'N/A')})")
                    return self._get_idle_action()
                
                # 启动 primitive（返回一个 generator）
                rospy.loginfo(f"Starting navigate_to: {target.name}")
                try:
                    # 在创建 generator 时暂停仿真（因为可能涉及初始化计算）
                    with og.sim.paused():
                        rospy.loginfo("Creating primitive generator (simulation paused)...")
                        self._primitive_controller = self._primitive_api.controller.apply_ref(
                            StarterSemanticActionPrimitiveSet.NAVIGATE_TO,
                            target,
                            attempts=3
                        )
                        rospy.loginfo("Primitive generator created successfully")
                except Exception as e:
                    rospy.logerr(f"Failed to create primitive generator: {e}")
                    import traceback
                    rospy.logerr(traceback.format_exc())
                    self._primitive_controller = None
            except Exception as e:
                rospy.logwarn(f"Failed to start navigate_to: {e}")
                import traceback
                rospy.logwarn(traceback.format_exc())
                self._primitive_controller = None
        
        # 如果 primitive 正在执行，每帧从 generator 取一个 action
        if self._primitive_controller is not None:
            try:
                import time
                import threading
                import psutil
                import torch
                
                # 记录开始时间
                start_time = time.time()
                
                # 创建进度信息字典（内存占用很小，只有几个数值）
                progress_info = {
                    'active': True,
                    'start_time': start_time,
                    'last_gpu_memory': 0.0
                }
                
                # 在后台线程中定期输出详细进度
                def show_progress():
                    interval = 2.0  # 每2秒输出一次
                    
                    while progress_info['active']:
                        elapsed = time.time() - progress_info['start_time']
                        
                        if elapsed > 1.0:  # 超过1秒才开始显示
                            # 获取系统资源使用
                            try:
                                cpu_percent = psutil.cpu_percent(interval=0.1)
                            except:
                                cpu_percent = 0.0
                            
                            # 获取 GPU 使用（如果可用）
                            gpu_info = ""
                            stage_hint = ""
                            if torch.cuda.is_available():
                                try:
                                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                                    gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                                    gpu_info = f"GPU: {gpu_memory:.2f}GB/{gpu_reserved:.2f}GB"
                                    
                                    # 检测 GPU 内存变化来判断是否在计算
                                    if abs(gpu_memory - progress_info['last_gpu_memory']) > 0.01:
                                        stage_hint = " (GPU active)"
                                        progress_info['last_gpu_memory'] = gpu_memory
                                except:
                                    gpu_info = "GPU: N/A"
                            
                            # 估算阶段（基于时间）
                            if elapsed < 5:
                                stage_guess = "Initializing"
                            elif elapsed < 15:
                                stage_guess = "IK solving / Trajectory optimization"
                            elif elapsed < 30:
                                stage_guess = "Collision checking / Refinement"
                            else:
                                stage_guess = "Long computation"
                                stage_hint += " ⚠️"
                            
                            # 组合进度信息
                            progress_msg = (
                                f"[Motion Planning] {stage_guess}{stage_hint} | "
                                f"Time: {elapsed:.1f}s | "
                                f"CPU: {cpu_percent:.1f}%"
                            )
                            
                            if gpu_info:
                                progress_msg += f" | {gpu_info}"
                            
                            rospy.loginfo(progress_msg)
                        
                        time.sleep(interval)
                
                progress_thread = threading.Thread(target=show_progress, daemon=True)
                progress_thread.start()
                
                try:
                    # 在获取 action 时暂停仿真（因为可能涉及运动规划计算）
                    rospy.loginfo("Getting next action from primitive generator (motion planning may take time, simulation paused)...")
                    with og.sim.paused():
                        action = next(self._primitive_controller)
                    
                    # 停止进度提示
                    progress_info['active'] = False
                    elapsed = time.time() - start_time
                    
                    # 最终统计
                    try:
                        final_cpu = psutil.cpu_percent(interval=0.1)
                    except:
                        final_cpu = 0.0
                    
                    final_info = f"Completed in {elapsed:.2f}s | CPU: {final_cpu:.1f}%"
                    if torch.cuda.is_available():
                        try:
                            final_gpu = torch.cuda.memory_allocated() / 1024**3
                            final_info += f" | GPU: {final_gpu:.2f}GB"
                        except:
                            pass
                    
                    rospy.loginfo(f"[Motion Planning] ✅ {final_info} (simulation was paused during computation)")
                    
                    return action
                    
                except Exception as e:
                    # 停止进度提示
                    progress_info['active'] = False
                    elapsed = time.time() - start_time
                    rospy.logwarn(f"[Motion Planning] ❌ Failed after {elapsed:.2f}s: {e}")
                    raise
                    
            except StopIteration:
                # primitive 执行完成
                rospy.loginfo("Primitive execution completed")
                self._primitive_controller = None
                return self._get_idle_action()
            except Exception as e:
                rospy.logwarn(f"Primitive execution error: {e}")
                import traceback
                rospy.logwarn(traceback.format_exc())
                self._primitive_controller = None
                return self._get_idle_action()
        
        return self._get_idle_action()

    def _on_mode_change(self, msg: String):
        """
        ROS topic 回调：热切换控制模式
        
        Args:
            msg: std_msgs/String，内容为 "idle" / "teleop" / "primitive"
        """
        mode = msg.data.strip().lower()
        if mode in ("idle", "teleop", "primitive"):
            old_mode = self.control_mode
            self.control_mode = mode
            rospy.loginfo(f"Switch control_mode: {old_mode} -> {mode}")
            
            # 根据模式启用/禁用 camera_mover 的键盘监听
            if mode == "teleop":
                # 切换到 teleop：禁用 camera_mover 键盘，避免冲突
                if self.camera_mover:
                    self.camera_mover.disable()
                    rospy.loginfo("Camera mover keyboard disabled (teleop mode)")
            else:
                # 切换到 idle/primitive：重新启用 camera_mover 键盘
                # (primitive 模式使用 G 键，不会和 CameraMover 的 P 键冲突)
                if self.camera_mover:
                    self.camera_mover.enable()
                    rospy.loginfo("Camera mover keyboard enabled")
        else:
            rospy.logwarn(f"Unknown control_mode: {mode}, valid modes are: idle, teleop, primitive")

    def _setup_camera_mover(self):
        viewer_cam = self.sensor
        viewer_cam.set_position_orientation(position=[1.35, 4.63, 1.93], orientation=[0.19, 0.62, 0.71, 0.22])
        self.camera_mover = CameraMover(viewer_cam)
        # 如果默认是 teleop 模式，禁用 camera_mover 的键盘监听，避免和机器人 teleop 冲突
        if self.control_mode == "teleop":
            self.camera_mover.disable()
            rospy.loginfo("Camera mover keyboard disabled (teleop mode active)")
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
        
        # setup subscribers
        self.mode_sub = rospy.Subscriber(
            "/behavior/control_mode",
            String,
            self._on_mode_change,
            queue_size=1,
        )
        rospy.loginfo("Subscribed to /behavior/control_mode for hot-switching control modes")

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
        
        if self.eyes_sensor is None:
            rospy.logwarn("No eyes sensor found for robot")
            return
        # 1. get camera pose
        pos, orn = self.eyes_sensor.get_position_orientation()
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
            obs, info = self.eyes_sensor.get_obs()
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
        if self.semantic_segmentation and "seg_semantic" in obs:
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
                # 更新相机平滑移动（只在非 teleop 模式下，避免键盘冲突）
                current_time_ros = rospy.Time.now()
                dt = (current_time_ros - last_update_time).to_sec()
                last_update_time = current_time_ros
                if self.camera_mover and self.control_mode != "teleop":
                    self.camera_mover.update(dt)
                
                # 根据当前控制模式决定这一帧的 action
                if self.control_mode == "teleop":
                    action = self._get_teleop_action()
                elif self.control_mode == "primitive":
                    action = self._get_primitive_action()
                else:
                    action = self._get_idle_action()

                # omnigibson simulation step
                self.env.step(action)
                
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
    parser.add_argument("--semantic_segmentation", 
                        type=bool,
                        default=False,
                        help="Enable semantic segmentation")

    args = parser.parse_args()
    
    env = og.Environment(choose_scene(args.scene, args.scene_file, args.semantic_segmentation))
    ros_behavior = ROSBehavior(env, record_rosbag=args.rosbag, publish_dsg=args.publish_dsg, semantic_segmentation=args.semantic_segmentation)
    
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
