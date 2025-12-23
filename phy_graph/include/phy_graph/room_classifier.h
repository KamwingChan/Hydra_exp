#pragma once

#include <ros/ros.h>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

#include <phy_graph/graph_generator.h>  // for RoomNode, EnhancedObjectNode, RoomInfo

namespace phy_graph {

/**
 * @brief 房间分类器：负责规则/同义词加载、稳定性状态机与分类打分
 */
class RoomClassifier {
public:
    RoomClassifier() = default;

    /**
     * @brief 从参数服务器加载房间分类配置（规则、同义词、稳定时间）
     */
    void loadConfig(const ros::NodeHandle& pnh);

    /**
     * @brief 对房间进行分类（带稳定性判定）
     */
    std::string classify(const RoomNode& room,
                         const std::vector<EnhancedObjectNode>& objects);

    /**
     * @brief 清理不再存在的房间状态
     */
    void cleanup(const std::vector<RoomNode>& current_rooms);

private:
    std::string nodeIdToString(const hydra::NodeId& node_id) const;

    // 配置
    std::map<std::string, std::map<std::string, int>> room_rules_;
    std::unordered_map<std::string, std::string> synonym_map_;
    double stability_timeout_{1.5};

    // 状态跟踪
    std::unordered_map<std::string, RoomInfo> room_states_;
};

}  // namespace phy_graph

