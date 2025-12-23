import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
import torch as th
import contextlib
import datetime
import logging
import math
import random
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class CameraMover:
    """
    A helper class for manipulating a camera via the keyboard. Utilizes carb keyboard callbacks to move
    the camera around.

    Args:
        cam (VisionSensor): The camera vision sensor to manipulate via the keyboard
        delta (float): Base linear speed in m/s when moving the camera
        save_dir (str): Absolute path to where recorded images should be stored. Default is <OMNIGIBSON_PATH>/imgs
    """

    def __init__(self, cam, delta=0.8, save_dir=None):
        if save_dir is None:
            save_dir = f"{og.root_path}/../images"

        self.cam = cam
        # 将 delta 视作连续移动的速度（m/s），方便后续按帧更新
        self.delta = delta
        self.fast_delta = delta * 3.0  # Shift加速
        self.slow_delta = delta * 0.2  # Ctrl减速
        # 原始 0.02 rad/keypress （约30Hz）≈0.6 rad/s，这里直接使用角速度
        self.rot_speed = 0.6  # 旋转速度 (rad/s)
        self.light_val = gm.FORCE_LIGHT_INTENSITY
        self.save_dir = save_dir
        
        # 保存初始位姿用于重置
        self.initial_position, self.initial_orientation = self.cam.get_position_orientation()
        
        # 键盘状态追踪（用于连续移动）
        self.key_state = {}

        self._appwindow = lazy.omni.appwindow.get_default_app_window()
        self._input = lazy.carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    def clear(self):
        """
        Clears this camera mover. After this is called, the camera mover cannot be used.
        """
        # 防止重复清理
        if self._sub_keyboard is None:
            return
            
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
        self._sub_keyboard = None
        self.key_state.clear()  # 清空键盘状态
        og.log.info("CameraMover keyboard subscription cleared.")

    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.clear()
        return False
    
    def __del__(self):
        """析构时尝试清理（作为最后保障）"""
        try:
            self.clear()
        except Exception:
            pass

    def set_save_dir(self, save_dir):
        """
        Sets the absolute path corresponding to the image directory where recorded images from this CameraMover
        should be saved

        Args:
            save_dir (str): Absolute path to where recorded images should be stored
        """
        self.save_dir = save_dir

    def change_light(self, delta):
        self.light_val += delta
        self.set_lights(self.light_val)

    def set_lights(self, intensity):
        world = lazy.isaacsim.core.utils.prims.get_prim_at_path("/World")
        for prim in world.GetChildren():
            for prim_child in prim.GetChildren():
                for prim_child_child in prim_child.GetChildren():
                    if "Light" in prim_child_child.GetPrimTypeInfo().GetTypeName():
                        prim_child_child.GetAttribute("inputs:intensity").Set(intensity)

    def print_info(self):
        """
        Prints keyboard command info out to the user
        """
        print("*" * 40)
        print("CameraMover! Commands:")
        print()
        print("\t Right Click + Drag: Rotate camera")
        print("\t W / S : Move camera forward / backward")
        print("\t A / D : Move camera left / right")
        print("\t Q / E : Move camera up / down")
        print("\t Arrow Keys : Rotate camera (Up/Down: pitch, Left/Right: yaw)")
        print("\t Shift : Fast movement")
        print("\t Ctrl : Slow movement")
        print("\t Space : Reset camera to initial position")
        print("\t 9 / 0 : Increase / decrease the lights")
        print("\t P : Print current camera pose")
        print("\t O : Save the current camera view as an image")
        print("*" * 40)
    
    def reset_camera(self):
        """
        Resets the camera to its initial position and orientation.
        """
        self.cam.set_position_orientation(
            position=self.initial_position, 
            orientation=self.initial_orientation
        )
        og.log.info("Camera reset to initial position.")

    def print_cam_pose(self):
        """
        Prints out the camera pose as (position, quaternion) in the world frame
        """
        print(f"cam pose: {self.cam.get_position_orientation()}")

    def get_image(self):
        """
        Helper function for quickly grabbing the currently viewed RGB image

        Returns:
            th.tensor: (H, W, 3) sized RGB image array
        """
        return self.cam.get_obs()[0]["rgb"][:, :, :-1]

    def record_image(self, fpath=None):
        """
        Saves the currently viewed image and writes it to disk

        Args:
            fpath (None or str): If specified, the absolute fpath to the image save location. Default is located in
                self.save_dir
        """
        og.log.info("Recording image...")

        # Use default fpath if not specified
        if fpath is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fpath = f"{self.save_dir}/og_{timestamp}.png"

        # Make sure save path directory exists, and then save the image to that location
        Path(Path(fpath).parent).mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.get_image()).save(fpath)
        og.log.info(f"Saved current viewer camera image to {fpath}.")

    def record_trajectory(self, poses, fps, steps_per_frame=1, fpath=None):
        """
        Moves the viewer camera through the poses specified by @poses and records the resulting trajectory to an mp4
        video file on disk.

        Args:
            poses (list of 2-tuple): List of global (position, quaternion) values to set the viewer camera to defining
                this trajectory
            fps (int): Frames per second when recording this video
            steps_per_frame (int): How many sim steps should occur between each frame being recorded. Minimum and
                default is 1.
            fpath (None or str): If specified, the absolute fpath to the video save location. Default is located in
                self.save_dir
        """
        og.log.info("Recording trajectory...")

        # Use default fpath if not specified
        if fpath is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fpath = f"{self.save_dir}/og_{timestamp}.mp4"

        # Make sure save path directory exists, and then create the video writer
        Path(Path(fpath).parent).mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(fpath, fps=fps)

        # Iterate through all desired poses, and record the trajectory
        for i, (pos, quat) in enumerate(poses):
            self.cam.set_position_orientation(position=pos, orientation=quat)
            og.sim.step()
            if i % steps_per_frame == 0:
                video_writer.append_data(self.get_image())

        # Close writer
        video_writer.close()
        og.log.info(f"Saved camera trajectory video to {fpath}.")

    def record_trajectory_from_waypoints(self, waypoints, per_step_distance, fps, steps_per_frame=1, fpath=None):
        """
        Moves the viewer camera through the waypoints specified by @waypoints and records the resulting trajectory to
        an mp4 video file on disk.

        Args:
            waypoints (th.tensor): (n, 3) global position waypoint values to set the viewer camera to defining this trajectory
            per_step_distance (float): How much distance (in m) should be approximately covered per trajectory step.
                This will determine the path length between individual waypoints
            fps (int): Frames per second when recording this video
            steps_per_frame (int): How many sim steps should occur between each frame being recorded. Minimum and
                default is 1.
            fpath (None or str): If specified, the absolute fpath to the video save location. Default is located in
                self.save_dir
        """
        # Create splines and their derivatives
        n_waypoints = len(waypoints)
        if n_waypoints < 3:
            og.log.error("Cannot generate trajectory from waypoints with less than 3 waypoints!")
            return

        splines = [CubicSpline(range(n_waypoints), waypoints[:, i], bc_type="clamped") for i in range(3)]
        dsplines = [spline.derivative() for spline in splines]

        # Function help get arc derivative
        def arc_derivative(u):
            return th.sqrt(th.sum([dspline(u) ** 2 for dspline in dsplines]))

        # Function to help get interpolated positions
        def get_interpolated_positions(step):
            assert step < n_waypoints - 1
            dist = quad(func=arc_derivative, a=step, b=step + 1)[0]
            path_length = int(dist / per_step_distance)
            interpolated_points = th.zeros((path_length, 3))
            for i in range(path_length):
                curr_step = step + (i / path_length)
                interpolated_points[i, :] = th.tensor([spline(curr_step) for spline in splines])
            return interpolated_points

        # Iterate over all waypoints and infer the resulting trajectory, recording the resulting poses
        poses = []
        for i in range(n_waypoints - 1):
            positions = get_interpolated_positions(step=i)
            for j in range(len(positions) - 1):
                # Get direction vector from the current to the following point
                direction = positions[j + 1] - positions[j]
                direction = direction / th.norm(direction)
                # Infer tilt and pan angles from this direction
                xy_direction = direction[:2] / th.norm(direction[:2])
                z = direction[2]
                pan_angle = th.arctan2(-xy_direction[0], xy_direction[1])
                tilt_angle = th.arcsin(z)
                # Infer global quat orientation from these angles
                quat = T.euler2quat([math.pi / 2 + tilt_angle, 0.0, pan_angle])
                poses.append([positions[j], quat])

        # Record the generated trajectory
        self.record_trajectory(poses=poses, fps=fps, steps_per_frame=steps_per_frame, fpath=fpath)

    def set_delta(self, delta):
        """
        Sets the base linear speed (m/s) for this CameraMover

        Args:
            delta (float): Base linear speed in m/s when moving the camera
        """
        self.delta = delta

    def set_cam(self, cam):
        """
        Sets the active camera sensor for this CameraMover

        Args:
            cam (VisionSensor): The camera vision sensor to manipulate via the keyboard
        """
        self.cam = cam

    @property
    def input_to_function(self):
        """
        Returns:
            dict: Mapping from relevant keypresses to corresponding function call to use
        """
        return {
            lazy.carb.input.KeyboardInput.O: lambda: self.record_image(fpath=None),
            lazy.carb.input.KeyboardInput.P: lambda: self.print_cam_pose(),
            lazy.carb.input.KeyboardInput.KEY_9: lambda: self.change_light(delta=-2e4),
            lazy.carb.input.KeyboardInput.KEY_0: lambda: self.change_light(delta=2e4),
            lazy.carb.input.KeyboardInput.SPACE: lambda: self.reset_camera(),
        }

    def get_current_delta(self):
        """
        根据是否按下 Shift/Ctrl 返回当前移动速度
        """
        if self.key_state.get(lazy.carb.input.KeyboardInput.LEFT_SHIFT, False):
            return self.fast_delta
        elif self.key_state.get(lazy.carb.input.KeyboardInput.LEFT_CONTROL, False):
            return self.slow_delta
        return self.delta

    @property
    def input_to_command(self):
        """
        Returns:
            dict: Mapping from relevant keypresses to corresponding direction vectors in camera frame
        """
        return {
            lazy.carb.input.KeyboardInput.D: th.tensor([1.0, 0.0, 0.0]),
            lazy.carb.input.KeyboardInput.A: th.tensor([-1.0, 0.0, 0.0]),
            lazy.carb.input.KeyboardInput.W: th.tensor([0.0, 0.0, -1.0]),
            lazy.carb.input.KeyboardInput.S: th.tensor([0.0, 0.0, 1.0]),
            lazy.carb.input.KeyboardInput.Q: th.tensor([0.0, 1.0, 0.0]),
            lazy.carb.input.KeyboardInput.E: th.tensor([0.0, -1.0, 0.0]),
        }
    
    @property
    def input_to_rotation(self):
        """
        Returns:
            dict: Mapping from arrow keys to rotation axis and direction
        """
        return {
            lazy.carb.input.KeyboardInput.UP: ('pitch', -1.0),
            lazy.carb.input.KeyboardInput.DOWN: ('pitch', 1.0),
            lazy.carb.input.KeyboardInput.LEFT: ('yaw', 1.0),
            lazy.carb.input.KeyboardInput.RIGHT: ('yaw', -1.0),
        }

    def _movement_direction(self):
        """
        Returns normalized movement direction in camera frame if any keys are active.
        """
        direction = th.zeros(3)
        for key, vec in self.input_to_command.items():
            if self.key_state.get(key, False):
                direction += vec
        norm = th.norm(direction)
        if norm.item() == 0:
            return None
        return direction / norm

    def _rotation_delta(self, dt):
        """
        Returns (pitch_delta, yaw_delta) in radians for this frame.
        """
        pitch = 0.0
        yaw = 0.0
        for key, (axis, direction) in self.input_to_rotation.items():
            if not self.key_state.get(key, False):
                continue
            if axis == 'pitch':
                pitch += direction * self.rot_speed * dt
            elif axis == 'yaw':
                yaw += direction * self.rot_speed * dt
        if abs(pitch) < 1e-6 and abs(yaw) < 1e-6:
            return None
        return pitch, yaw

    def update(self, dt):
        """
        Continuous update for smooth camera motion. Should be called every frame with elapsed time (seconds).
        """
        if dt is None or dt <= 0 or self.cam is None:
            return

        direction = self._movement_direction()
        rotation = self._rotation_delta(dt)
        if direction is None and rotation is None:
            return

        pos, orn = self.cam.get_position_orientation()
        pos = pos if isinstance(pos, th.Tensor) else th.tensor(pos, dtype=th.float32)
        orn = orn if isinstance(orn, th.Tensor) else th.tensor(orn, dtype=th.float32)

        updated = False

        if direction is not None:
            speed = self.get_current_delta()
            if speed > 0:
                delta_local = direction * speed * dt
                transform = T.quat2mat(orn)
                delta_global = transform @ delta_local
                pos = pos + delta_global
                updated = True

        if rotation is not None:
            pitch_delta, yaw_delta = rotation
            if pitch_delta or yaw_delta:
                orn_tensor = orn if isinstance(orn, th.Tensor) else th.tensor(orn, dtype=th.float32)
                orn_xyzw = th.stack([orn_tensor[1], orn_tensor[2], orn_tensor[3], orn_tensor[0]])
                euler = R.from_quat(orn_xyzw.detach().cpu().numpy()).as_euler('xyz')
                euler[2] += pitch_delta
                euler[0] += yaw_delta
                new_quat_xyzw = R.from_euler('xyz', euler).as_quat()
                orn = th.tensor(
                    [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]],
                    dtype=orn.dtype,
                )
                updated = True

        if updated:
            self.cam.set_position_orientation(position=pos, orientation=orn)

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """
        Handle keyboard events. Note: The signature is pulled directly from omni.

        Args:
            event (int): keyboard event type
        """
        # 更新键盘状态（用于 Shift/Ctrl 与连续移动检测）
        if event.type == lazy.carb.input.KeyboardEventType.KEY_PRESS:
            self.key_state[event.input] = True
            if event.input in self.input_to_function:
                self.input_to_function[event.input]()
        elif event.type == lazy.carb.input.KeyboardEventType.KEY_RELEASE:
            self.key_state[event.input] = False
        elif event.type == lazy.carb.input.KeyboardEventType.KEY_REPEAT:
            self.key_state[event.input] = True
        return True

