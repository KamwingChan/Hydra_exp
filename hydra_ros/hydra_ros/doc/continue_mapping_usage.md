# Continue Mapping 使用说明

## 功能概述

Continue Mapping允许Hydra从之前保存的地图继续建图，而不是每次都从零开始。

## 实现原理

### 核心机制

1. **DeltaCompression初始化**：
   - Frontend的DeltaCompression从已加载的mesh初始化
   - 设置`num_archived_vertices_`为loaded mesh的顶点数
   - 重建voxel映射（vertices_map_）
   - 新mesh会追加到loaded mesh之后，而非替换

2. **Backend同步**：
   - 初始化`original_vertices_`从loaded mesh
   - 设置`num_archived_vertices_`匹配loaded mesh大小
   - 支持对loaded mesh进行deformation

3. **Object Nodes保留**：
   - Loaded object nodes的mesh_connections保持有效
   - 新检测的objects可以与旧objects合并
   - 支持动态场景（物体移动会被检测为新object）

## 使用方法

### 步骤1：第一次运行（创建地图）

```bash
# 正常运行Hydra
roslaunch hydra_ros hydra.launch

# 地图会保存到输出目录，例如：
# output/2024-01-15_10-30-45/backend/dsg.json
# output/2024-01-15_10-30-45/backend/mesh.ply
```

### 步骤2：第二次运行（继续建图）

#### 方法A：通过launch文件参数

```bash
# 使用现有的launch文件，传递参数
roslaunch hydra_ros isaacsim.launch \
  continue_mapping:=true \
  map_load_path:=/path/to/saved/map
```

#### 方法B：在launch文件中直接设置参数

```xml
<launch>
  <!-- 加载正常配置 -->
  <rosparam command="load" file="$(find hydra_ros)/config/datasets/isaacsim.yaml" />
  
  <!-- 添加continue mapping参数 -->
  <param name="continue_mapping" value="true" />
  <param name="map_load_path" value="/path/to/saved/map" />
  
  <!-- Hydra节点 -->
  <node name="hydra_node" pkg="hydra_ros" type="hydra_node" output="screen" />
</launch>
```

#### 方法C：通过YAML配置文件

创建配置文件 `my_continue_mapping.yaml`：

```yaml
continue_mapping: true
map_load_path: "/path/to/saved/map"
```

然后在launch文件中加载：

```xml
<rosparam command="load" file="$(find hydra_ros)/config/my_continue_mapping.yaml" />
```

## 配置参数

### `continue_mapping`
- **类型**：bool
- **默认值**：false
- **说明**：是否启用continue mapping模式

### `map_load_path`
- **类型**：string
- **默认值**：""
- **说明**：已保存地图的路径，应包含`backend/dsg.json`和`backend/mesh.ply`

## 预期行为

### 成功启动时的日志

```
[INFO] [Hydra ROS] Continue mapping enabled, loading map from: /path/to/map
[INFO] [Hydra Load] Starting map loading process from: /path/to/map
[INFO] [Hydra Load] Scene graph loaded successfully.
[INFO] [Hydra Load] PCL mesh loaded: 4522869 points, 8031263 polygons
[INFO] [Hydra Load] Map loading and synchronization complete!
[INFO] [Hydra Frontend] Continue mapping: loaded mesh with 4522869 vertices
[INFO] [DeltaCompression] Initializing from loaded mesh with 4522869 vertices and 8031263 faces
[INFO] [DeltaCompression] Successfully initialized 3245678 voxel mappings
[INFO] [Hydra Frontend] DeltaCompression initialized successfully for continue mapping
[INFO] [Hydra Backend] Continue mapping: preserving existing mesh with 4522869 vertices
[INFO] [Hydra Backend] Initialized deformation tracking for 4522869 loaded vertices
```

### Mesh更新行为

第一个新MeshDelta：
- `vertex_start = 4522869`（从loaded mesh之后开始）
- 新顶点索引：4522869, 4522870, ...
- 总mesh大小：4522869 + 新顶点数

## 故障排查

### 问题1：崩溃或mesh索引越界

**症状**：
```
vector::_M_range_check: __n (which is 4495738) >= this->size() (which is 2354)
```

**原因**：DeltaCompression未正确初始化

**解决**：
1. 确认kimera_pgmo包含`initializeFromLoadedMesh`方法
2. 检查Frontend初始化日志
3. 如果初始化失败，查看错误日志并检查mesh文件完整性

### 问题2：无法加载地图

**症状**：
```
[ERROR] [Hydra Load] Scene graph file not found at: /path/to/backend/dsg.json
```

**解决**：
1. 确认`map_load_path`路径正确
2. 确认路径下有`backend/dsg.json`和`backend/mesh.ply`
3. 检查文件权限

### 问题3：Object nodes重复

**症状**：同一个物体显示两次（旧位置和新位置）

**解决**：
- 这是预期行为（动态场景支持）
- Backend的UpdateObjectsFunctor会尝试合并重叠的objects
- 如果不希望保留旧objects，设置`preserve_objects: false`（需要实现方案B）

## 限制和注意事项

1. **Pose Graph要求**：
   - 目前只支持`use_gt_frame: true`模式（PoseGraphFromOdom）
   - Visual relocalization尚未实现

2. **Object Layer**：
   - Object nodes的mesh_connections依赖于mesh索引连续性
   - 如果DeltaCompression初始化失败，会回退到空mesh模式

3. **性能考虑**：
   - 初始化时需要为每个loaded vertex重建voxel映射
   - 大型mesh（>500万顶点）可能需要几秒钟初始化时间

4. **兼容性**：
   - 向后兼容：`enabled: false`时完全不影响现有功能
   - 需要kimera_pgmo支持（已添加`initializeFromLoadedMesh`）

## 测试验证

### 基本功能测试

1. 运行第一次建图并保存
2. 启用continue mapping重新运行
3. 检查：
   - ✅ 地图正确加载
   - ✅ Mesh顶点数连续增长
   - ✅ Object nodes可访问
   - ✅ 无崩溃或越界错误

### 动态场景测试

1. 第一次建图时记录物体位置
2. 移动场景中的物体
3. Continue mapping重新建图
4. 验证：
   - ✅ 新位置的物体被检测
   - ✅ Backend尝试合并或创建新nodes

## 未来改进

- [ ] 实现visual relocalization支持
- [ ] 添加方案B（清理object layer选项）
- [ ] 支持增量式Places/Rooms更新
- [ ] 优化大型mesh的初始化性能

