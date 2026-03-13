"""
Main robot controller that manages different control modes.

Control modes:
    - idle: robot holds current pose (no drop)
    - teleop: keyboard teleoperation
    - primitive: semantic action primitives (G key: hardcoded/JSON, P key: LLM pipeline)
"""
import rospy
import numpy as np

from .teleop_controller import TeleopController
from .primitive_controller import PrimitiveController


class RobotController:
    """Manages robot control modes: idle, teleop, primitive."""
    
    def __init__(self, robot, env, curobo_batch_size=1, execution_mode=None):
        """
        Initialize robot controller.
        
        Args:
            robot: Robot instance
            env: OmniGibson environment
            curobo_batch_size: Batch size for CuRobo (default 1 for 8GB GPU)
            execution_mode: ExecutionMode.FULL or ExecutionMode.SYMBOLIC (default: FULL)
        """
        self.robot = robot
        self.env = env
        self.control_mode = "primitive"  # "idle" / "teleop" / "primitive"
        
        # Initialize sub-controllers
        self.teleop_controller = TeleopController(robot)
        self.primitive_controller = PrimitiveController(env, robot, curobo_batch_size, execution_mode)
        
        # Idle action cache
        self._idle_action = None
    
    def set_pipeline(self, pipeline):
        """
        Inject PhyPlanPipeline into PrimitiveController.
        
        Called by behavior_ros_robot.py after creating the full pipeline chain.
        Enables P key (LLM planning) and G key (JSON loading) on PrimitiveController.
        
        Args:
            pipeline: PhyPlanPipeline instance (created externally)
        """
        self.primitive_controller.set_pipeline(pipeline)
        rospy.loginfo("[RobotController] PhyPlanPipeline injected (P/G key enabled)")
    
    def _get_idle_action(self):
        """
        Generate idle action that holds current pose (joints stay, base stops).
        Prevents arm from dropping when switching to idle.
        
        Returns:
            action: Action array (numpy) with no-op for joints and zero velocity for base.
        """
        try:
            control_dict = self.robot.get_control_dict()
            action = np.zeros(self.robot.action_dim, dtype=np.float32)
            for name, controller in self.robot._controllers.items():
                action_idx = self.robot.controller_action_idx[name]
                if name == "base":
                    action[action_idx] = np.zeros(controller.command_dim, dtype=np.float32)
                else:
                    no_op = controller.compute_no_op_action(control_dict)
                    if hasattr(no_op, "cpu"):
                        no_op = no_op.cpu().numpy()
                    action[action_idx] = np.asarray(no_op, dtype=np.float32)
            return action
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"Idle hold pose failed, fallback to zeros: {e}")
            if self._idle_action is None:
                try:
                    sample = self.robot.action_space.sample()
                    self._idle_action = np.zeros_like(sample)
                except Exception:
                    self._idle_action = np.zeros(self.robot.action_dim, dtype=np.float32)
            return self._idle_action
    
    def set_mode(self, mode):
        """
        Set control mode.
        
        Args:
            mode: "idle", "teleop", or "primitive"
        """
        if mode not in ("idle", "teleop", "primitive"):
            rospy.logwarn(f"Unknown control_mode: {mode}, valid modes are: idle, teleop, primitive")
            return
        
        old_mode = self.control_mode
        self.control_mode = mode
        rospy.loginfo(f"Switch control_mode: {old_mode} -> {mode}")
        
        # Update primitive controller's mode (for G key callback)
        self.primitive_controller.set_mode(mode)
    
    def get_action(self):
        """
        Get action based on current control mode.
        
        Returns:
            action: Robot action array
        """
        idle_action = self._get_idle_action()
        
        if self.control_mode == "teleop":
            return self.teleop_controller.get_action(idle_action)
        elif self.control_mode == "primitive":
            return self.primitive_controller.get_action(idle_action)
        else:  # idle
            return idle_action
