"""
Main robot controller that manages different control modes.
"""
import rospy
import numpy as np

from .teleop_controller import TeleopController
from .primitive_controller import PrimitiveController


class RobotController:
    """Manages robot control modes: idle, teleop, primitive."""
    
    def __init__(self, robot, env, curobo_batch_size=1):
        """
        Initialize robot controller.
        
        Args:
            robot: Robot instance
            env: OmniGibson environment
            curobo_batch_size: Batch size for CuRobo (default 1 for 8GB GPU)
        """
        self.robot = robot
        self.env = env
        self.control_mode = "primitive"  # "idle" / "teleop" / "primitive"
        
        # Initialize sub-controllers
        self.teleop_controller = TeleopController(robot)
        self.primitive_controller = PrimitiveController(env, robot, curobo_batch_size)
        
        # Idle action cache
        self._idle_action = None
    
    def _get_idle_action(self):
        """
        Generate idle action (all zeros).
        
        Returns:
            action: Zero action array
        """
        if self._idle_action is None:
            try:
                sample = self.robot.action_space.sample()
                self._idle_action = np.zeros_like(sample)
            except Exception:
                # 兜底：如果 action_space 不工作，就返回标量 0
                self._idle_action = 0.0
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
