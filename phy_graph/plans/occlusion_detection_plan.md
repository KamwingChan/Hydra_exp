# Phy_graph 遮挡检测 & Mesh优化 实施计划

## 📋 项目概述

本计划实现两个功能：
1. **遮挡检测**：在图像评分系统中加入基于深度图的遮挡评估
2. **Mesh订阅优化**：在 `hydra_bbox` 模式下禁用mesh订阅，节省资源

**决策记录：**
- 深度图存储方案：方案A - 同步存储到KeyframeDatabase
- 深度图线程处理：共享RGB线程（复用 `rgb_queue_`）
- 内存预算：~500-650MB（可接受）

---

## 🎯 功能一：遮挡检测

### 1.1 设计思路

在图像评分系统中新增 **Occlusion Score** 维度：
- 将物体3D点投影到图像平面
- 比较物体深度与深度图像素值
- 如果 `object_depth > depth_image_value + threshold`，则该点被遮挡
- 计算遮挡比例，转换为评分

### 1.2 评分系统修改

当前评分分布（100分满分）：
| 维度 | 当前权重 |
|------|----------|
| Visibility | 35分 |
| Coverage | 40分 |
| Center | 10分 |
| Margin | 15分 |

修改后（加入遮挡检测）：
| 维度 | 新权重 | 说明 |
|------|--------|------|
| Visibility | 30分 | 物体有多少顶点在视野内 |
| Coverage | 35分 | 物体在图像中的大小 |
| **Occlusion** | **15分** | 物体未被遮挡的比例（新增） |
| Center | 8分 | 物体距离中心的距离 |
| Margin | 12分 | 物体是否完全在画面内 |

### 1.3 新增配置参数

在 `config/inference_config.yaml` 中添加：

```yaml
# === 遮挡检测参数 ===
occlusion:
  # 是否启用遮挡检测
  enable: true
  # 深度比较阈值（米），超过此值认为被遮挡
  depth_threshold: 0.1
  # 遮挡检测采样点数（从bbox/mesh中采样，减少计算量）
  sample_points: 50
  # 遮挡评分权重（0-15分）
  max_score: 15
```

### 1.4 需要修改的文件

#### 1.4.1 `include/phy_graph/keyframe_database.h`

```cpp
struct Keyframe {
    // 现有成员...
    
    // 新增：深度图相关
    std::vector<uchar> depth_buffer;  // PNG压缩的深度图
    bool has_depth;                    // 深度图是否可用
    
    // 解码方法
    cv::Mat decode() const;             // RGB图（已有）
    cv::Mat decodeDepth() const;        // 深度图（新增）
};
```

#### 1.4.2 `src/keyframe_database.cpp`

- 修改 `addImage` 方法：增加深度图参数
- 新增 `decodeDepth` 方法
- 修改 `maintainMemoryLimit`：处理深度图的冷热分层

#### 1.4.3 `include/phy_graph/physical_inference_node.h`

```cpp
// 新增深度图订阅（共享RGB线程队列）
ros::Subscriber depth_sub_;

// 新增回调
void depthCallback(const sensor_msgs::ImageConstPtr& msg);

// 深度图缓存（与RGB近似同步）
sensor_msgs::ImageConstPtr latest_depth_;
std::mutex depth_mutex_;

// 遮挡检测配置标志
bool occlusion_enabled_;
```

#### 1.4.4 `src/nodeSub.cpp`

**深度图订阅（共享RGB线程）：**
```cpp
// 在构造函数中，复用 rgb_queue_
if (cfg.occlusion.enable) {
    ros::SubscribeOptions depth_ops = ros::SubscribeOptions::create<sensor_msgs::Image>(
        "depth_image", 10,
        boost::bind(&PhysicalInferenceNode::depthCallback, this, _1),
        ros::VoidPtr(),
        &rgb_queue_   // 关键：复用RGB的专用队列
    );
    depth_ops.transport_hints = ros::TransportHints().tcpNoDelay();
    depth_sub_ = nh_.subscribe(depth_ops);
    occlusion_enabled_ = true;
    ROS_INFO("Occlusion detection enabled, depth subscribed to rgb_queue_");
}
```

**深度回调（只缓存最新）：**
```cpp
void PhysicalInferenceNode::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    latest_depth_ = msg;
}
```

**修改 rgbCallback：**
```cpp
void PhysicalInferenceNode::rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (!camera_info_received_) return;
    // 帧抽样逻辑...
    
    // 获取当前最近的深度图（如果启用遮挡检测）
    sensor_msgs::ImageConstPtr depth_msg;
    if (occlusion_enabled_) {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        depth_msg = latest_depth_;
    }
    
    // 添加到数据库（RGB必选，深度可选）
    keyframe_db_->addImage(msg, world_T_camera, depth_msg);
}
```

#### 1.4.5 `src/imageProcessor.cpp`

新增遮挡评估函数：

```cpp
double PhysicalInferenceNode::calculateOcclusionScore(
    const hydra::ObjectNodeAttributes& attrs,
    const cv::Mat& depth_image,
    const Eigen::Isometry3d& world_T_camera,
    const cv::Size& image_size);
```

修改 `projectObjectToImage`：
- 如果深度图可用，计算遮挡分数
- 集成到总评分中

#### 1.4.6 `include/phy_graph/inference_config.h`

```cpp
struct InferenceConfig {
    // 新增
    struct Occlusion {
        bool enable = true;
        double depth_threshold = 0.1;
        int sample_points = 50;
        int max_score = 15;
    } occlusion;
};
```

#### 1.4.7 `config/inference_config.yaml`

添加遮挡检测配置节

#### 1.4.8 `launch/inference.launch`

新增深度图topic参数：

```xml
<arg name="depth_topic" default="/camera/aligned_depth_to_color/image_raw" />
<remap from="depth_image" to="$(arg depth_topic)" />
```

---

## 🎯 功能二：Mesh订阅优化

### 2.1 设计思路

当 `projection_mode: hydra_bbox` 时：
- 不订阅mesh topic（节省带宽和内存）
- 所有涉及mesh的函数参数改为可选指针

### 2.2 优化好处

| 方面 | 优化效果 |
|------|----------|
| **网络带宽** | 减少mesh数据传输（mesh通常较大） |
| **内存占用** | 不存储 `latest_mesh_` |
| **CPU开销** | 减少mesh回调和数据拷贝 |
| **代码清晰度** | 明确mesh的可选性，避免传递无用数据 |

### 2.3 需要修改的文件

#### 2.3.1 `include/phy_graph/physical_inference_node.h`

将mesh参数改为可选指针：

```cpp
// 修改前
void enqueueForInference(const SceneGraphNode& node, const KimeraPgmoMesh& mesh);
ProjectionResult projectObjectToImage(..., const KimeraPgmoMesh& mesh, ...);
std::pair<cv::Mat, double> extractBestObjectImage(..., const KimeraPgmoMesh& mesh);
std::vector<ScoredImage> scoreCandidateImages(..., const KimeraPgmoMesh& mesh);

// 修改后
void enqueueForInference(const SceneGraphNode& node, const KimeraPgmoMesh* mesh = nullptr);
ProjectionResult projectObjectToImage(..., const KimeraPgmoMesh* mesh = nullptr, ...);
std::pair<cv::Mat, double> extractBestObjectImage(..., const KimeraPgmoMesh* mesh = nullptr);
std::vector<ScoredImage> scoreCandidateImages(..., const KimeraPgmoMesh* mesh = nullptr);
```

#### 2.3.2 `src/nodeSub.cpp`

- 根据 `projection_mode` 有条件地订阅mesh
- 传递mesh指针时检查是否为nullptr

```cpp
// 构造函数中
if (cfg.image.projection_mode == "mesh_vertices") {
    mesh_sub_ = nh_.subscribe("input_mesh", 1, &PhysicalInferenceNode::meshCallback, this);
}

// processDsg中
const KimeraPgmoMesh* mesh_ptr = nullptr;
if (latest_mesh_) {
    mesh_ptr = latest_mesh_.get();
}
enqueueForInference(node, mesh_ptr);
```

#### 2.3.3 `src/imageProcessor.cpp`

- `projectMeshVertices` 添加空指针检查
- `projectObjectToImage` 根据mesh可用性选择投影模式

---

## 📅 实施顺序

建议按以下顺序实施：

### Phase 1: 基础设施（遮挡检测准备）
- [ ] 修改 `inference_config.h/cpp`：添加遮挡检测配置
- [ ] 修改 `inference_config.yaml`：添加配置节
- [ ] 修改 `keyframe_database.h/cpp`：支持深度图存储

### Phase 2: 深度图订阅
- [ ] 修改 `physical_inference_node.h`：添加深度订阅成员
- [ ] 修改 `nodeSub.cpp`：实现深度订阅和回调
- [ ] 修改 `launch/inference.launch`：添加深度topic参数

### Phase 3: 遮挡评分集成
- [ ] 修改 `imageProcessor.cpp`：实现 `calculateOcclusionScore`
- [ ] 修改 `projectObjectToImage`：集成遮挡分数到总评分

### Phase 4: Mesh优化
- [ ] 修改函数签名：mesh参数改为可选指针
- [ ] 修改 `nodeSub.cpp`：有条件订阅mesh
- [ ] 更新所有调用点

### Phase 5: 测试与验证
- [ ] 单元测试：验证遮挡检测逻辑
- [ ] 集成测试：验证整体评分系统
- [ ] 性能测试：验证内存占用在预期范围内

---

## 📊 预期效果

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 图像评分准确性 | 不考虑遮挡 | 考虑遮挡，更准确 |
| 内存占用（RGB模式） | ~200MB | ~500-650MB |
| Mesh带宽（hydra_bbox模式） | 全量订阅 | 零订阅 |

---

## ⚠️ 注意事项

1. **深度图时间同步**：RGB和深度需要在同一时刻采集，使用相同的timestamp匹配
2. **深度图编码**：使用PNG无损压缩，保持16位精度
3. **向后兼容**：遮挡检测默认启用，但可通过配置禁用
4. **降级处理**：如果深度图不可用，跳过遮挡评分（满分15分自动获得）
