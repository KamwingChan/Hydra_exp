"""
Scene configuration for OmniGibson.
"""
from omnigibson.macros import gm


def choose_scene(scene_name, scene_file, semantic_segmentation=True):
    """
    Create scene configuration for OmniGibson.
    
    Args:
        scene_name: Name of the scene model
        scene_file: Path to scene JSON file
        semantic_segmentation: Whether to enable semantic segmentation
        
    Returns:
        cfg: Scene configuration dictionary
    """
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
                "type": "Stretch",
                "name": "robot_stretch",  # Stretch 机器人名称
                "obs_modalities": ["rgb", "depth", "seg_semantic"] if semantic_segmentation else ["rgb", "depth"],
                "action_type": "continuous",
                "action_normalize": True,
                # High friction on drive wheels to prevent slipping during velocity control navigation
                "link_physics_materials": {
                    "link_right_wheel": {
                        "static_friction": 10.0,
                        "dynamic_friction": 10.0,
                        "restitution": 0.0,
                    },
                    "link_left_wheel": {
                        "static_friction": 10.0,
                        "dynamic_friction": 10.0,
                        "restitution": 0.0,
                    },
                },
                # Controller config for SYMBOLIC mode with velocity control for smooth navigation
                # Stretch uses DifferentialDriveController (default), so we don't override base
                "controller_config": {
                    "arm_0": {
                        "name": "JointController",
                        "use_delta_commands": False,
                    },
                    "gripper_0": {
                        "name": "JointController",
                        "use_delta_commands": False,
                    },
                    "camera": {
                        "name": "JointController",
                        "use_delta_commands": False,
                    },
                },
                # camera config
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_height": 480,
                            "image_width": 640,
                        }
                    }
                },
                "include_sensor_names": ["eyes"],  # Stretch 的头部相机
                "exclude_sensor_names": [],
            }
        ],
    }
    return cfg
