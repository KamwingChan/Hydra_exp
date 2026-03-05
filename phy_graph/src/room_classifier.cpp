#include "phy_graph/room_classifier.h"
#include "phy_graph/graph_utils.h"

#include <XmlRpcValue.h>
#include <hydra/common/global_info.h>
#include <algorithm>
#include <cctype>  // for std::isdigit

namespace phy_graph {

namespace {

std::map<std::string, std::map<std::string, int>> defaultRoomRules() {
    return {
        {"Bedroom",    {{"bed", 10}, {"wardrobe", 5}, {"nightstand", 3}, {"pillow", 2}}},
        {"Bathroom",   {{"toilet", 10}, {"shower", 8}, {"bathtub", 8}, {"sink", 3}, {"towel", 2}}},
        {"Kitchen",    {{"stove", 10}, {"oven", 10}, {"refrigerator", 8}, {"microwave", 5}, {"sink", 3}, {"cabinet", 2}}},
        {"LivingRoom", {{"couch", 8}, {"sofa", 8}, {"tv", 5}, {"coffee_table", 3}, {"lamp", 2}}},
        {"Office",     {{"monitor", 5}, {"keyboard", 3}, {"desk", 5}, {"office_chair", 5}, {"book", 2}}},
        {"DiningRoom", {{"dining_table", 8}, {"chair", 3}, {"plate", 2}}}
    };
}

std::unordered_map<std::string, std::string> defaultSynonyms() {
    return {
        {"sofa", "couch"},
        {"television", "tv"},
        {"coffee_table", "coffee_table"},
        {"dining_table", "dining_table"}
    };
}

bool parseRuleParam(const XmlRpc::XmlRpcValue& xml_rules,
                    std::map<std::string, std::map<std::string, int>>& out_rules) {
    if (xml_rules.getType() != XmlRpc::XmlRpcValue::TypeStruct) return false;

    for (auto it = xml_rules.begin(); it != xml_rules.end(); ++it) {
        std::string room_type = it->first;
        const XmlRpc::XmlRpcValue& keyword_map = it->second;
        if (keyword_map.getType() != XmlRpc::XmlRpcValue::TypeStruct) continue;

        std::map<std::string, int> weights;
        for (auto kt = keyword_map.begin(); kt != keyword_map.end(); ++kt) {
            std::string keyword = utils::normalizeLabel(kt->first);
            if (keyword.empty()) continue;

            const XmlRpc::XmlRpcValue& weight_val = kt->second;
            int weight = 0;
            if (weight_val.getType() == XmlRpc::XmlRpcValue::TypeInt) {
                weight = static_cast<int>(weight_val);
            } else if (weight_val.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
                weight = static_cast<int>(static_cast<double>(weight_val));
            } else {
                continue;
            }
            weights[keyword] = weight;
        }

        if (!weights.empty()) {
            out_rules[room_type] = weights;
        }
    }
    return !out_rules.empty();
}

bool parseSynonymParam(const XmlRpc::XmlRpcValue& xml_syn,
                       std::unordered_map<std::string, std::string>& out_syn) {
    if (xml_syn.getType() != XmlRpc::XmlRpcValue::TypeStruct) return false;

    for (auto it = xml_syn.begin(); it != xml_syn.end(); ++it) {
        std::string key = utils::normalizeLabel(it->first);
        if (key.empty()) continue;

        if (it->second.getType() != XmlRpc::XmlRpcValue::TypeString) continue;
        std::string value = utils::normalizeLabel(static_cast<std::string>(it->second));
        if (value.empty()) continue;

        out_syn[key] = value;
    }
    return !out_syn.empty();
}

std::map<std::string, std::map<std::string, int>> normalizeRules(
    const std::map<std::string, std::map<std::string, int>>& in_rules) {
    std::map<std::string, std::map<std::string, int>> normalized;
    for (const auto& [room_type, keywords] : in_rules) {
        for (const auto& [keyword, weight] : keywords) {
            std::string norm_key = utils::normalizeLabel(keyword);
            if (!norm_key.empty()) {
                normalized[room_type][norm_key] = weight;
            }
        }
    }
    return normalized;
}

}  // namespace

void RoomClassifier::loadConfig(const ros::NodeHandle& pnh) {
    room_rules_ = normalizeRules(defaultRoomRules());
    synonym_map_.clear();
    for (const auto& [k, v] : defaultSynonyms()) {
        std::string key_norm = utils::normalizeLabel(k);
        std::string val_norm = utils::normalizeLabel(v);
        if (!key_norm.empty() && !val_norm.empty()) {
            synonym_map_[key_norm] = val_norm;
        }
    }
    stability_timeout_ = 1.5;

    XmlRpc::XmlRpcValue xml_rules;
    if (pnh.getParam("room_rules", xml_rules)) {
        std::map<std::string, std::map<std::string, int>> parsed;
        if (parseRuleParam(xml_rules, parsed)) {
            room_rules_ = normalizeRules(parsed);
            ROS_INFO("Loaded room_rules from params (%zu types)", room_rules_.size());
        } else {
            ROS_WARN("Failed to parse param room_rules, using defaults.");
        }
    }

    XmlRpc::XmlRpcValue xml_syn;
    if (pnh.getParam("room_synonyms", xml_syn)) {
        std::unordered_map<std::string, std::string> parsed_syn;
        if (parseSynonymParam(xml_syn, parsed_syn)) {
            synonym_map_.swap(parsed_syn);
            ROS_INFO("Loaded room_synonyms from params (%zu entries)", synonym_map_.size());
        } else {
            ROS_WARN("Failed to parse param room_synonyms, using defaults.");
        }
    }

    pnh.param("room_stability_timeout", stability_timeout_, stability_timeout_);
    ROS_INFO("Room config: %zu rule types, %zu synonyms, stability timeout = %.2fs",
             room_rules_.size(), synonym_map_.size(), stability_timeout_);
}

std::string RoomClassifier::classify(const RoomNode& room,
                                     const std::vector<EnhancedObjectNode>& objects) {
    std::string room_id_str = nodeIdToString(room.room_id);
    ros::Time now = ros::Time::now();

    // ===== 检查房间是否已有明确的类别（来自 BEHAVIOR） =====
    // 如果房间类别不是 "TODO" 或 "R(数字)" 格式，说明来自 BEHAVIOR，信任它
    std::string current_category = room.category;
    
    // 判断是否是 Hydra 格式（R(数字)）或未分类状态
    bool is_hydra_format = false;
    if (current_category == "TODO" || current_category.empty()) {
        is_hydra_format = true;
    } else if (current_category.length() >= 3 && 
               current_category[0] == 'R' && 
               current_category[1] == '(' && 
               std::isdigit(current_category[2])) {
        is_hydra_format = true;
    }
    
    // 如果不是 Hydra 格式，说明是 BEHAVIOR 来源的明确类别，直接返回
    if (!is_hydra_format) {
        // 进一步验证：检查是否在已知的房间类型列表中
        bool is_known_type = false;
        for (const auto& [room_type, _] : room_rules_) {
            if (current_category == room_type || 
                current_category.find(room_type) != std::string::npos) {
                is_known_type = true;
                break;
            }
        }
        
        if (is_known_type) {
            ROS_INFO("Room %s already has BEHAVIOR category '%s', preserving it.",
                     room_id_str.c_str(), current_category.c_str());
            return current_category;
        }
        // 如果不是已知类型但也不是 Hydra 格式，仍然保留（可能是自定义类别）
        ROS_INFO("Room %s has non-standard category '%s', preserving it.",
                 room_id_str.c_str(), current_category.c_str());
        return current_category;
    }

    // 1. 内容 hash
    size_t current_hash = objects.size();
    for (const auto& obj : objects) {
        current_hash ^= std::hash<std::string>{}(nodeIdToString(obj.node_id));
    }

    // 2. 初始化状态
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

    // 3. 变化检测
    if (current_hash != info.last_object_hash) {
        info.state = RoomState::DIRTY;
        info.last_change_time = now;
        info.last_object_hash = current_hash;
        return info.category + " (Updating)";
    }

    // 4. 稳定性检查
    double time_since_change = (now - info.last_change_time).toSec();
    if (info.state == RoomState::DIRTY || info.state == RoomState::NEW) {
        if (time_since_change > stability_timeout_) {
            info.state = RoomState::STABLE;
        } else {
            return "Unclassified (Waiting)";
        }
    }

    // 5. 分类打分
    if (info.state == RoomState::STABLE) {
        std::map<std::string, int> scores;
        for (const auto& obj : objects) {
            std::string label = utils::normalizeLabel(obj.category);
            label = utils::applySynonyms(label, synonym_map_);
            for (const auto& [room_type, object_weights] : room_rules_) {
                for (const auto& [keyword, weight] : object_weights) {
                    if (label.find(keyword) != std::string::npos) {
                        scores[room_type] += weight;
                    }
                }
            }
        }

        std::string best_room = "Unknown";
        int max_score = 0;
        for (const auto& [room_type, score] : scores) {
            if (score > max_score) {
                max_score = score;
                best_room = room_type;
            }
        }
        if (max_score < 3) {
            best_room = "Unknown";
        }

        info.category = best_room;
        info.state = RoomState::CLASSIFIED;
        ROS_INFO("Room %s finalized as %s after %.1fs stability (Score: %d).",
                 room_id_str.c_str(), best_room.c_str(), time_since_change, max_score);
    }

    if (info.state == RoomState::CLASSIFIED) {
        return info.category;
    }
    return "Unclassified";
}

void RoomClassifier::cleanup(const std::vector<RoomNode>& current_rooms) {
    std::unordered_set<std::string> current_ids;
    for (const auto& room : current_rooms) {
        current_ids.insert(nodeIdToString(room.room_id));
    }
    for (auto it = room_states_.begin(); it != room_states_.end(); ) {
        if (current_ids.find(it->first) == current_ids.end()) {
            it = room_states_.erase(it);
        } else {
            ++it;
        }
    }
}

std::string RoomClassifier::nodeIdToString(const hydra::NodeId& node_id) const {
    return hydra::NodeSymbol(node_id).getLabel();
}

}  // namespace phy_graph

