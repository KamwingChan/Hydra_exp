#include "phy_graph/graph_generator.h"
#include <ros/ros.h>
#include <map>
#include <algorithm>
#include <hydra/common/global_info.h>

namespace phy_graph {

// ============ Constant Definition ============

// 房间分类规则
const std::map<std::string, std::map<std::string, int>> ROOM_RULES = {
    {"Bedroom",    {{"bed", 10}, {"wardrobe", 5}, {"nightstand", 3}, {"pillow", 2}}},
    {"Bathroom",   {{"toilet", 10}, {"shower", 8}, {"bathtub", 8}, {"sink", 3}, {"towel", 2}}},
    {"Kitchen",    {{"stove", 10}, {"oven", 10}, {"refrigerator", 8}, {"microwave", 5}, {"sink", 3}, {"cabinet", 2}}},
    {"LivingRoom", {{"couch", 8}, {"sofa", 8}, {"tv", 5}, {"coffee_table", 3}, {"lamp", 2}}},
    {"Office",     {{"monitor", 5}, {"keyboard", 3}, {"desk", 5}, {"office_chair", 5}, {"book", 2}}},
    {"DiningRoom", {{"dining_table", 8}, {"chair", 3}, {"plate", 2}}}
};

// Stability timeout (seconds)
const double STABILITY_TIMEOUT = 1.5; 

// ============ Extract DSG module ============

std::vector<EnhancedObjectNode> GraphGenerator::extractObjectsFromDSG(
    const hydra::DynamicSceneGraph::Ptr& graph,
    const std::unordered_map<std::string, PhysicalProperties>& physical_props) {
    
    std::vector<EnhancedObjectNode> objects;
    
    if (!graph->hasLayer(hydra::DsgLayers::OBJECTS)) {
        return objects;
    }
    
    const auto& object_layer = graph->getLayer(hydra::DsgLayers::OBJECTS);
    
    for (const auto& id_node_pair : object_layer.nodes()) {
        const auto& node = *id_node_pair.second;
        const auto& attrs = node.attributes<hydra::ObjectNodeAttributes>();
        
        EnhancedObjectNode obj;
        obj.node_id = node.id;
        obj.category = attrs.name;
        obj.position = node.attributes().position;
        
        // Extract bounding box information
        if (attrs.bounding_box.type == hydra::BoundingBox::Type::AABB) {
            // Convert Eigen::Vector3f to Eigen::Vector3d for storage
            Eigen::Vector3d center = attrs.bounding_box.world_P_center.cast<double>();
            Eigen::Vector3d extents = attrs.bounding_box.dimensions.cast<double>();
            
            obj.bbox_min = center - extents / 2.0;
            obj.bbox_max = center + extents / 2.0;
        }
        
        // Find corresponding physical properties
        std::string node_id_str = nodeIdToString(node.id);
        auto it = physical_props.find(node_id_str);
        if (it != physical_props.end()) {
            obj.properties = it->second;
        } else {
            // ROS_DEBUG("No physical properties found for object %s", node_id_str.c_str());
        }
        
        objects.push_back(obj);
    }
    
    ROS_INFO_ONCE("Extracted %zu objects from DSG", objects.size());
    return objects;
}

std::vector<RoomNode> GraphGenerator::extractRoomsFromDSG(
    const hydra::DynamicSceneGraph::Ptr& graph) {
    
    std::vector<RoomNode> rooms;
    
    if (!graph->hasLayer(hydra::DsgLayers::ROOMS)) {
        ROS_WARN_ONCE("DSG does not have ROOMS layer");
        return rooms;
    }
    
    // ============ 步骤 1: 构建 Place -> Objects 的映射表 ============
    // 很多时候 Object 只是通过边连接到 Place，而不是 Place 的直接子节点
    // 我们遍历所有的层间边来建立这个关系
    std::unordered_map<hydra::NodeId, std::vector<hydra::NodeId>> place_to_objects;
    
    for (const auto& [key, edge] : graph->interlayer_edges()) {
        hydra::NodeId n1 = key.k1;
        hydra::NodeId n2 = key.k2;
        char c1 = hydra::NodeSymbol(n1).category();
        char c2 = hydra::NodeSymbol(n2).category();
        
        // 检查是否是 Place <-> Object 的连接
        // Place 前缀通常是 'p', Object 前缀是 'O' 或 'o'
        if (c1 == 'p' && (c2 == 'O' || c2 == 'o')) {
            place_to_objects[n1].push_back(n2);
        } else if (c2 == 'p' && (c1 == 'O' || c1 == 'o')) {
            place_to_objects[n2].push_back(n1);
        }
    }

    // ============ 步骤 2: 提取房间并关联物体 ============
    const auto& room_layer = graph->getLayer(hydra::DsgLayers::ROOMS);
    
    for (const auto& id_node_pair : room_layer.nodes()) {
        const auto& node = *id_node_pair.second;
        
        RoomNode room;
        room.room_id = node.id;
        
        // Extract position
        room.position = node.attributes().position;
        
        // Extract bounding box (try-catch safety)
        try {
            // Cast to SemanticNodeAttributes to access bounding_box
            const auto& attrs = node.attributes<hydra::SemanticNodeAttributes>();
            
            if (attrs.bounding_box.type == hydra::BoundingBox::Type::AABB) {
                Eigen::Vector3d center = attrs.bounding_box.world_P_center.cast<double>();
                Eigen::Vector3d extents = attrs.bounding_box.dimensions.cast<double>();
                room.bbox_min = center - extents / 2.0;
                room.bbox_max = center + extents / 2.0;
            } else {
                room.bbox_min = room.position;
                room.bbox_max = room.position;
            }
        } catch (...) {
            room.bbox_min = room.position;
            room.bbox_max = room.position;
        }
        
        // 遍历房间的直接子节点 (通常是 Place)
        for (const auto& child_id : node.children()) {
            char category = hydra::NodeSymbol(child_id).category();
            
            // 情况 A: 直接连着物体 (少见，但可能)
            if (category == 'O' || category == 'o') {
                room.object_ids.push_back(child_id);
            }
            // 情况 B: 连着 Place -> 查找 Place 连接的物体
            else if (category == 'p') {
                // 1. 从边关系中查找
                if (place_to_objects.count(child_id)) {
                    const auto& connected_objs = place_to_objects[child_id];
                    room.object_ids.insert(room.object_ids.end(), connected_objs.begin(), connected_objs.end());
                }
                
                // 2. 从父子关系中查找 (双重保险)
                const auto* place_node = graph->findNode(child_id);
                if (place_node) {
                    for (const auto& grand_child_id : place_node->children()) {
                        char grand_cat = hydra::NodeSymbol(grand_child_id).category();
                        if (grand_cat == 'O' || grand_cat == 'o') {
                            room.object_ids.push_back(grand_child_id);
                        }
                    }
                }
            }
        }
        
        // 去重 (因为一个物体可能连接到房间内的多个 Place)
        std::sort(room.object_ids.begin(), room.object_ids.end());
        room.object_ids.erase(
            std::unique(room.object_ids.begin(), room.object_ids.end()), 
            room.object_ids.end()
        );
        
        rooms.push_back(room);
    }
    
    ROS_INFO_ONCE("Extracted %zu rooms from DSG", rooms.size());
    return rooms;
}

// ============ 房间分析模块 ============

std::string GraphGenerator::inferRoomCategory(
    const RoomNode& room,
    const std::vector<EnhancedObjectNode>& objects) {
    
    std::string room_id_str = nodeIdToString(room.room_id);
    ros::Time now = ros::Time::now();

    // 1. 计算当前房间内容的简易 Hash (用于检测变化)
    // 只要对象 ID 列表变了，或者对象数量变了，就算变化
    size_t current_hash = objects.size();
    for(const auto& obj : objects) {
        // 简单的异或 hash，只为了检测变化
        current_hash ^= std::hash<std::string>{}(nodeIdToString(obj.node_id));
    }

    // 2. 初始化或获取状态
    if (room_states_.find(room_id_str) == room_states_.end()) {
        RoomInfo info;
        info.state = RoomState::NEW;
        info.last_change_time = now;
        info.last_object_hash = current_hash;
        info.category = "Unclassified";
        room_states_[room_id_str] = info;
        return "Unclassified (New)";
    }

    RoomInfo& info = room_states_[room_id_str];

    // 3. 检测变化 (Change Detection)
    if (current_hash != info.last_object_hash) {
        info.state = RoomState::DIRTY;
        info.last_change_time = now;
        info.last_object_hash = current_hash;
        // info.category = "Unclassified"; // <--- 删除这行！保留旧的分类
        
        ROS_DEBUG("Room %s changed! Resetting timer.", room_id_str.c_str());
        // 返回旧的分类，并加上标记
        return info.category + " (Updating)"; 
    }

    // 4. 状态流转 (State Transition)
    double time_since_change = (now - info.last_change_time).toSec();

    if (info.state == RoomState::DIRTY || info.state == RoomState::NEW) {
        if (time_since_change > STABILITY_TIMEOUT) {
            info.state = RoomState::STABLE; // 终于稳定了！
        } else {
            return "Unclassified (Waiting)";
        }
    }

    // 5. 执行分类 (Execute Classification)
    if (info.state == RoomState::STABLE) {
        std::map<std::string, int> scores;

        // 遍历房间内的所有对象
        for (const auto& obj : objects) {
            std::string label = obj.category;
            std::transform(label.begin(), label.end(), label.begin(), ::tolower);

            // 匹配规则并累加分数
            for (const auto& [room_type, object_weights] : ROOM_RULES) {
                for (const auto& [keyword, weight] : object_weights) {
                    if (label.find(keyword) != std::string::npos) {
                        scores[room_type] += weight;
                    }
                }
            }
        }

        // 找出得分最高的房间类型
        std::string best_room = "Unknown";
        int max_score = 0;

        for (const auto& [room_type, score] : scores) {
            if (score > max_score) {
                max_score = score;
                best_room = room_type;
            }
        }

        // 门控：分数太低的不算
        if (max_score < 3) {
            best_room = "Unknown";
        }
        
        info.category = best_room;
        info.state = RoomState::CLASSIFIED; // 标记为已完成
        
        ROS_INFO("Room %s finalized as %s after %.1fs stability (Score: %d).", 
                 room_id_str.c_str(), best_room.c_str(), time_since_change, max_score);
    }

    // 6. 返回结果
    if (info.state == RoomState::CLASSIFIED) {
        return info.category;
    }

    return "Unclassified";
}

void GraphGenerator::cleanupStaleTrackers(const std::vector<RoomNode>& current_rooms) {
    std::unordered_set<std::string> current_ids;
    for (const auto& room : current_rooms) {
        current_ids.insert(nodeIdToString(room.room_id));
    }

    // 遍历 tracker map，删除不在 current_ids 里的
    for (auto it = room_states_.begin(); it != room_states_.end(); ) {
        if (current_ids.find(it->first) == current_ids.end()) {
            it = room_states_.erase(it); // 安全删除
        } else {
            ++it;
        }
    }
}

std::string GraphGenerator::generateRoomDescription(
    const RoomNode& room,
    const std::vector<EnhancedObjectNode>& objects) {
    
    // TODO (Future): 实现房间描述生成
    return "TODO: Room description to be generated"; 
}

} // namespace phy_graph

