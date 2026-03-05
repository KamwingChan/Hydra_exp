import omnigibson as og
import torch as th
import numpy as np
import os
import sys
import json
import time
from omnigibson.utils.python_utils import create_object_from_init_info
from omnigibson.scenes.scene_base import Scene

# 启动 simulator
og.launch()

try:
    # 从配置创建物体
    obj_info = {
        "class_module": "omnigibson.objects.dataset_object",
        "class_name": "DatasetObject",
        "args": {
        "name": "swivel_chair_xvpywp_0",
            "category": "swivel_chair",
            "model": "xvpywp",
            "scale": [
                0.9999871253967285,
                0.9999610781669617,
                0.9999590516090393
            ],
            "fixed_base": False,
            "visual_only": False,
        }
    }

    obj = create_object_from_init_info(obj_info)

    # 创建临时场景并导入物体
    scene = Scene(use_floor_plane=False)
    og.sim.import_scene(scene)
    scene.add_object(obj)

    # 获取尺寸
    bbox_size = obj.aabb_extent
    native_size = obj.native_bbox
    actual_size = native_size * obj.scale

    # 转成 numpy
    bbox_size_np = bbox_size.cpu().numpy() if hasattr(bbox_size, 'cpu') else np.array(bbox_size)
    native_size_np = native_size.cpu().numpy() if hasattr(native_size, 'cpu') else np.array(native_size)
    actual_size_np = actual_size.cpu().numpy() if hasattr(actual_size, 'cpu') else np.array(actual_size)

    # 保存到文件（确保数据不丢失）
    result = {
        "bbox_size": bbox_size_np.tolist(),
        "native_size": native_size_np.tolist(),
        "actual_size": actual_size_np.tolist(),
    }
    with open("bbox_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # 打印结果（使用 flush=True 确保立即输出）
    print(f"物体尺寸: {bbox_size_np}", flush=True)
    print(f"原生尺寸: {native_size_np}", flush=True)
    print(f"实际尺寸: {actual_size_np}", flush=True)
    print("结果已保存到 bbox_result.json", flush=True)

except Exception as e:
    print(f"错误: {e}", flush=True)
    import traceback
    traceback.print_exc()
finally:
    # 确保输出完成
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.2)  # 给输出一点时间
    os._exit(0)