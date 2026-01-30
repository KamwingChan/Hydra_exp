"""
Teleoperation (keyboard) controller.
"""
import rospy
import numpy as np
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardRobotController


class TeleopController:
    """Handles keyboard teleoperation control."""
    
    def __init__(self, robot):
        """
        Initialize teleop controller.
        
        Args:
            robot: Robot instance
        """
        self.robot = robot
        self.teleop_controller = None
        
        try:
            self.teleop_controller = KeyboardRobotController(robot=self.robot)
            self.teleop_controller.print_keyboard_teleop_info()
        except Exception as e:
            rospy.logwarn(f"Failed to create KeyboardRobotController: {e}")
            self.teleop_controller = None
    
    def get_action(self, idle_action):
        """
        Get teleop action from keyboard controller.
        
        Args:
            idle_action: Fallback action if teleop fails
            
        Returns:
            action: Robot action array
        """
        if self.teleop_controller is None:
            return idle_action
        
        try:
            action = self.teleop_controller.get_teleop_action()
            if action is None:
                # 没有按键输入时，有些实现会返回 None，这里退回 idle
                return idle_action
            
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
            return idle_action
