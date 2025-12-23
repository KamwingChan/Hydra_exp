#include "phy_graph/graph_generator.h"
#include "phy_graph/graph_utils.h"

#include <ros/ros.h>
#include <algorithm>
#include <map>
#include <hydra/common/global_info.h>
#include <unordered_map>
#include <limits>  // for numeric_limits

namespace phy_graph {

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
        
        // Extract position (room centroid)
        room.position = node.attributes().position;
        
        // ===== Extract / compute bounding box =====
        // 优先使用 Hydra 提供的 AABB；如果没有，则用房间内的 Place 节点位置计算一个简单 AABB。
        Eigen::Vector3d bb_min( std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity());
        Eigen::Vector3d bb_max(-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity());
        bool has_bbox = false;

        // 1) 优先尝试 SemanticNodeAttributes 里的 bounding_box
        try {
            const auto& attrs = node.attributes<hydra::SemanticNodeAttributes>();
            if (attrs.bounding_box.type == hydra::BoundingBox::Type::AABB) {
                Eigen::Vector3d center  = attrs.bounding_box.world_P_center.cast<double>();
                Eigen::Vector3d extents = attrs.bounding_box.dimensions.cast<double>();
                bb_min = center - extents / 2.0;
                bb_max = center + extents / 2.0;
                has_bbox = true;
            }
        } catch (...) {
            // ignore and fall back to place-based bbox
        }

        // 2) 如果 Hydra 没有给出可用的 AABB，则基于房间内的 Place 节点位置计算 AABB
        if (!has_bbox) {
            bool has_place = false;
            for (const auto& child_id : node.children()) {
                if (hydra::NodeSymbol(child_id).category() != 'p') {
                    continue;
                }

                const auto* place_node = graph->findNode(child_id);
                if (!place_node) {
                    continue;
                }

                const auto& p = place_node->attributes().position;
                bb_min = bb_min.cwiseMin(p);
                bb_max = bb_max.cwiseMax(p);
                has_place = true;
            }

            if (has_place) {
                has_bbox = true;
            }
        }

        // 3) 最终写回 RoomNode 的 bbox 字段；如果仍然没有 bbox，就退回到房间中心点
        if (has_bbox) {
            room.bbox_min = bb_min;
            room.bbox_max = bb_max;
        } else {
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

std::string GraphGenerator::generateRoomDescription(
    const RoomNode& room,
    const std::vector<EnhancedObjectNode>& objects) {
    
    // TODO (Future): 实现房间描述生成
    return "TODO: Room description to be generated"; 
}

} // namespace phy_graph

