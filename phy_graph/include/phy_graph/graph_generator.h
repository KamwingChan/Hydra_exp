#pragma once

#include <ros/ros.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <hydra/common/dsg_types.h>
#include <hydra_ros/utils/dsg_streaming_interface.h>
#include <kimera_pgmo_msgs/KimeraPgmoMesh.h>

#include <boost/filesystem.hpp>
#include <Eigen/Geometry>

namespace phy_graph {

// ============ 数据结构定义 ============

/**
 * @brief 物理属性结构
 * 存储从JSON文件中解析的物理推理结果
 */
struct PhysicalProperties {
    int friction_level;      // 摩擦等级 (0-3)
    bool pushable;           // 是否可推动
    int weight_level;        // 重量等级 (0-3)
    std::string description; // VLM生成的描述
    
    PhysicalProperties() 
        : friction_level(-1), pushable(false), weight_level(-1), description("") {}
};

/**
 * @brief 增强的对象节点
 * 包含DSG对象信息和物理属性
 */
struct EnhancedObjectNode {
    hydra::NodeId node_id;
    std::string category;
    PhysicalProperties properties;
    
    // 位置信息
    Eigen::Vector3d position;
    
    // 边界框信息
    Eigen::Vector3d bbox_min;
    Eigen::Vector3d bbox_max;
    
    EnhancedObjectNode() : node_id(0), category(""), position(0, 0, 0), 
                           bbox_min(0, 0, 0), bbox_max(0, 0, 0) {}
};

/**
 * @brief 房间节点
 * 包含房间信息和包含的对象列表
 */
struct RoomNode {
    hydra::NodeId room_id;
    std::string category;              // TODO (Future): 待实现房间类型推理
    std::string description;           // TODO (Future): 待实现房间描述生成
    std::vector<hydra::NodeId> object_ids;
    
    // Position and Bounding Box
    Eigen::Vector3d position; // Centroid
    Eigen::Vector3d bbox_min;
    Eigen::Vector3d bbox_max;
    
    RoomNode() : room_id(0), category("TODO"), description("TODO: Room description to be generated"),
                 position(0,0,0), bbox_min(0,0,0), bbox_max(0,0,0) {}
};

/**
 * @brief 场景图结构
 * 包含所有房间和对象节点
 */
struct SceneGraph {
    std::vector<RoomNode> rooms;
    std::vector<EnhancedObjectNode> objects;
    std::string timestamp;
    
    SceneGraph() : timestamp("") {}
};

// ============ 房间稳定性跟踪结构 ============

enum class RoomState {
    NEW,            // 刚出现，不可信
    MONITORING,     // 正在观察其稳定性
    STABLE,         // 已稳定，可以分类
    CLASSIFIED,     // 已完成分类
    DIRTY           // 发现变化，数据变脏，需要重置
};

struct RoomInfo {
    RoomState state;
    ros::Time last_change_time;     // 上一次拓扑/内容变化的时间
    size_t last_object_hash;        // 上一次的对象列表指纹
    std::string category;           // 最终分类结果
    
    RoomInfo() : state(RoomState::NEW), last_object_hash(0), category("Unclassified") {}
};

// ============ GraphGenerator类 ============

/**
 * @brief 场景图生成器
 * 
 * 功能：
 * 1. 从output目录读取物理推理结果(JSON)
 * 2. 从Hydra DSG提取对象和房间信息
 * 3. 生成包含物理属性的增强场景图
 * 4. 保存为JSON格式
 */
class GraphGenerator {
public:
    GraphGenerator(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~GraphGenerator();
    
    /**
     * @brief 主循环
     */
    void run();

private:
    // ============ 回调函数 ============
    
    /**
     * @brief DSG更新回调
     */
    void processDsg(const hydra::DynamicSceneGraph::Ptr& graph);
    
    // ============ JSON解析模块 ============
    
    /**
     * @brief 从output目录加载所有物理推理结果
     * @return 节点ID -> 物理属性的映射
     */
    std::unordered_map<std::string, PhysicalProperties> loadPhysicalProperties();
    
    /**
     * @brief 解析单个物理属性JSON文件
     * @param filepath JSON文件路径
     * @return 解析后的物理属性
     */
    PhysicalProperties parseObjectJson(const std::string& filepath);
    
    // ============ DSG处理模块 ============
    
    /**
     * @brief 从DSG提取对象节点
     * @param graph 动态场景图
     * @param physical_props 物理属性映射
     * @return 增强的对象节点列表
     */
    std::vector<EnhancedObjectNode> extractObjectsFromDSG(
        const hydra::DynamicSceneGraph::Ptr& graph,
        const std::unordered_map<std::string, PhysicalProperties>& physical_props);
    
    /**
     * @brief 从DSG提取房间节点
     * @param graph 动态场景图
     * @return 房间节点列表
     */
    std::vector<RoomNode> extractRoomsFromDSG(
        const hydra::DynamicSceneGraph::Ptr& graph);
    
    // ============ 房间分析模块 ============
    
    /**
     * @brief 推理房间类型
     * 
     * 使用稳定性追踪和基于规则的推理:
     * 1. 只有当房间在一定时间内保持稳定时才进行推理
     * 2. 基于房间内对象组合推理 (e.g., bed+nightstand -> bedroom)
     * 
     * @param room 房间节点
     * @param objects 该房间内的所有对象
     * @return 房间类型字符串
     */
    std::string inferRoomCategory(
        const RoomNode& room,
        const std::vector<EnhancedObjectNode>& objects);
    
    /**
     * @brief 清理过期的房间状态跟踪器
     * @param current_rooms 当前存在的房间列表
     */
    void cleanupStaleTrackers(const std::vector<RoomNode>& current_rooms);
    
    /**
     * @brief 生成房间描述
     * 
     * TODO (Future): 实现房间描述生成
     * 
     * @param room 房间节点
     * @param objects 该房间内的所有对象
     * @return 房间描述字符串
     */
    std::string generateRoomDescription(
        const RoomNode& room,
        const std::vector<EnhancedObjectNode>& objects);
    
    // ============ 场景图生成模块 ============
    
    /**
     * @brief 构建完整的场景图
     * @param graph DSG图
     * @return 增强的场景图
     */
    SceneGraph buildSceneGraph(const hydra::DynamicSceneGraph::Ptr& graph);
    
    /**
     * @brief 保存场景图为JSON文件
     * @param scene_graph 场景图
     * @return 保存的文件路径
     */
    std::string saveSceneGraphJson(const SceneGraph& scene_graph);
    
    // ============ 辅助函数 ============
    
    /**
     * @brief 设置输出目录
     */
    void setupOutputDirectory();
    
    /**
     * @brief 节点ID转字符串
     */
    std::string nodeIdToString(const hydra::NodeId& node_id) const;
    
    /**
     * @brief Vector3d转JSON字符串
     */
    std::string vector3dToJson(const Eigen::Vector3d& vec) const;
    
    // ============ 成员变量 ============
    
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    
    // DSG接收器
    std::unique_ptr<hydra::DsgReceiver> dsg_receiver_;
    
    // 输出目录
    std::string output_dir_;
    
    // 已处理的DSG时间戳，避免重复处理
    std::unordered_set<uint64_t> processed_dsg_timestamps_;

    // 房间状态跟踪器 (RoomId -> RoomInfo)
    std::unordered_map<std::string, RoomInfo> room_states_;
};

} // namespace phy_graph
