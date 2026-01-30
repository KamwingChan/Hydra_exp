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
