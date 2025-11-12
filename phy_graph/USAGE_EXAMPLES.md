# Physical Inference Launch Configuration Examples

本文档展示如何使用 `inference.launch` 的不同配置。

## 默认配置

使用默认的 Hydra topics：

```bash
roslaunch physical_inference inference.launch
```

默认订阅：
- RGB图像: `/hydra_ros_node/input/left_cam/rgb/image_raw`
