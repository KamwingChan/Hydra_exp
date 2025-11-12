# 图像缓存问题修复总结

## 问题描述

在使用 `use_sim_time=true` 播放 rosbag 时，系统出现以下警告：

```
[WARN] No cached images in time range [22.655, 32.655]
[WARN] Failed to extract image for object couch, skipping...
```

## 根本原因

在 `use_sim_time=true` 环境下，系统存在以下时序问题：

1. **时间窗口策略不当**：原代码使用物体的 `last_update_time_ns` 来定义搜索窗口（物体时间 ± 5秒）
2. **消息到达顺序**：DSG 更新到达时，对应时间段的 RGB 图像可能还未从 bag 播放
3. **缓存容量不足**：原始缓存仅 100 帧，约 33 秒数据，处理延迟时可能溢出

### 典型场景流程

```
t=20s: RGB 图像开始到达并缓存
t=25s: DSG 更新到达，包含 last_update_time=27.655s 的物体
       系统尝试查找 [22.655, 32.655] 范围的图像
       但这些图像还未播放，导致失败
```

## 修复方案

### 1. 智能时间窗口选择 (`imageProcessor.cpp`)

改进了 `extractBestObjectImage()` 函数，增加了三种时间窗口策略：

```cpp
// 情况1：物体时间戳远早于当前时间（正常播放）
if ((current_time - object_time).toSec() > 5.0) {
    search_center = current_time;  // 使用当前时间往回搜索
}

// 情况2：物体时间戳远晚于当前时间（bag 还没播放到）
else if ((object_time - current_time).toSec() > 5.0) {
    search_center = recent_images.back().timestamp;  // 使用最新缓存图像
}

// 情况3：时间接近，使用物体时间
else {
    search_center = object_time;
}
```

**关键改进**：
- 将时间窗口从 ±5 秒扩大到 ±10 秒
- 优先使用当前系统时间而非物体时间戳
- 自动适应 bag 播放延迟

### 2. 增加诊断日志

添加了详细的时间对比和缓存状态日志：

```cpp
ROS_DEBUG("Object '%s' time: %.3f, Current time: %.3f, Delta: %.3f s", ...);
ROS_DEBUG("Cache status: %zu images available", cache_size);

// 当找不到图像时，显示实际缓存范围
if (!all_images.empty()) {
    ROS_WARN("But cache contains %zu images in range [%.3f, %.3f]", ...);
}
```

### 3. 增加缓存容量 (`nodeSub.cpp`)

```cpp
// 从 100 帧增加到 300 帧
image_cache_ = std::make_shared<ImageCache>(300);  // 约 100 秒@3Hz
```

### 4. 改进 TF 查询 (`imageProcessor.cpp`)

```cpp
// 增加超时时间以应对 use_sim_time 场景
transform = tf_buffer_->lookupTransform(
    "world", camera_frame_, 
    msg->header.stamp,
    ros::Duration(0.5)  // 从 0.1s 增加到 0.5s
);

// 增加时间戳信息
ROS_WARN_THROTTLE(5.0, "TF lookup failed at t=%.3f: %s", ...);
```

## 修改的文件

1. `../physical_inference/src/imageProcessor.cpp`
   - 修改 `extractBestObjectImage()` - 智能时间窗口选择
   - 修改 `rgbCallback()` - 改进 TF 查询
   - 添加详细诊断日志

2. `../physical_inference/src/nodeSub.cpp`
   - 增加图像缓存容量（100 → 300 帧）

## 预期效果

修复后的系统将能够：

1. ✅ 在 `use_sim_time=true` 环境下正确找到缓存图像
2. ✅ 自动适应不同的 bag 播放速率
3. ✅ 提供详细的诊断信息帮助调试
4. ✅ 更稳健地处理时间同步问题

## 测试建议

运行修复后的代码时，建议：

1. 检查日志中的时间对比信息
2. 确认缓存范围是否覆盖物体时间
3. 观察是否还有 "No cached images" 警告
4. 如果仍有问题，可以启用 DEBUG 日志查看详细信息：

```bash
rosrun physical_inference physical_inference_node --log-level debug
```

