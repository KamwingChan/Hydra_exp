#include "phy_graph/graph_generator.h"
#include <ros/package.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <cmath>

namespace phy_graph {

// ============ JSON解析模块 ============

namespace {
inline double round3(double v) { return std::round(v * 1000.0) / 1000.0; }

inline nlohmann::json vec3Json(const Eigen::Vector3d& v) {
    return nlohmann::json{
        {"x", round3(v.x())},
        {"y", round3(v.y())},
        {"z", round3(v.z())},
    };
}

inline void atomicWriteTextFile(const std::string& final_path, const std::string& contents) {
    const std::string tmp_path = final_path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open tmp file: " + tmp_path);
        }
        out << contents;
        out.flush();
        if (!out.good()) {
            throw std::runtime_error("Failed to write tmp file: " + tmp_path);
        }
    }

    std::error_code ec;
    std::filesystem::remove(final_path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        std::remove(final_path.c_str());
        std::rename(tmp_path.c_str(), final_path.c_str());
    }
}
}  // namespace

std::unordered_map<std::string, PhysicalProperties> 
GraphGenerator::loadPhysicalProperties() {
    std::unordered_map<std::string, PhysicalProperties> result;
    
    // 获取output目录路径
    std::string pkg_path = ros::package::getPath("phy_graph");
    std::string output_path = pkg_path + "/output";
    
    if (!std::filesystem::exists(output_path)) {
        ROS_WARN_ONCE("Output directory does not exist: %s", output_path.c_str());
        return result;
    }
    
    // 遍历output目录下的所有子目录，找到最新的一个
    std::vector<std::string> subdirs;
    
    for (const auto& entry : std::filesystem::directory_iterator(output_path)) {
        std::error_code ec;
        if (!entry.is_directory(ec) || ec) {
            continue;
        }
        std::string dir_name = entry.path().filename().string();
            // 过滤掉非时间戳目录 (scene_graphs, keyframes, objects 等)
            if (dir_name == "scene_graphs" || dir_name == "keyframes" || dir_name == "objects") continue;
            // 简单的格式检查：是否包含 '-' (如 12-02_...)
            if (dir_name.find('-') == std::string::npos) continue;

            subdirs.push_back(dir_name);
    }

    if (subdirs.empty()) {
        physical_source_dir_path_.clear();
        physical_source_dir_name_.clear();
        return result;
    }

    // 排序找到最新的目录 (假设目录名格式为日期时间，字符串排序即可)
    std::sort(subdirs.begin(), subdirs.end());
    std::string latest_dir_name = subdirs.back();
    std::filesystem::path latest_dir_path = std::filesystem::path(output_path) / latest_dir_name;

    // record physical source directory for traceability
    physical_source_dir_name_ = latest_dir_name;
    physical_source_dir_path_ = latest_dir_path.string();
    
    ROS_INFO_ONCE("Loading physical properties from latest directory: %s", latest_dir_name.c_str());

    // 遍历最新目录中的JSON文件
    for (const auto& entry : std::filesystem::directory_iterator(latest_dir_path)) {
        std::error_code ec;
        if (!entry.is_regular_file(ec) || ec) {
            continue;
        }
        if (entry.path().extension() != ".json") {
            continue;
        }

        std::string filename = entry.path().filename().string();
            
            // 只处理object_*.json文件
            if (filename.find("object_") == 0) {
                try {
                    PhysicalProperties props = parseObjectJson(entry.path().string());
                    
                    // 从文件名提取节点ID (object_O123_chair.json -> O123)
                    size_t first_underscore = filename.find('_');
                    size_t second_underscore = filename.find('_', first_underscore + 1);
                    if (first_underscore != std::string::npos && 
                        second_underscore != std::string::npos) {
                        std::string node_id = filename.substr(
                            first_underscore + 1, 
                            second_underscore - first_underscore - 1);
                        result[node_id] = props;
                    }
                    
                } catch (const std::exception& e) {
                    ROS_WARN("Failed to parse %s: %s", filename.c_str(), e.what());
                }
            }
    }
    
    ROS_INFO_ONCE("Loaded physical properties for %zu objects", result.size());
    return result;
}

PhysicalProperties GraphGenerator::parseObjectJson(const std::string& filepath) {
    PhysicalProperties props;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    nlohmann::json j;
    file >> j;

    props.friction_level = j.value("friction_level", -1);
    props.pushable = j.value("pushable", false);
    props.weight_level = j.value("weight_level", -1);
    props.description = j.value("description", std::string{});
    props.estimated_weight_kg = j.value("estimated_weight_kg", std::string{});
    props.inference_confidence = j.value("inference_confidence", -1);
    return props;
}

std::string GraphGenerator::saveSceneGraphJson(const SceneGraph& scene_graph) {
    // Stable latest path (for downstream consumers)
    const std::string latest_path = output_dir_ + "/scene_graph_latest.json";

    // Also archive into the physical inference run directory (for traceability)
    std::string archive_path;
    if (!physical_source_dir_path_.empty()) {
        archive_path = physical_source_dir_path_ + "/scene_graph_enhanced.json";
    }

    nlohmann::json root;
    root["schema_version"] = 1;
    root["physical_source_dir"] = physical_source_dir_name_;

    nlohmann::json sg;
    sg["timestamp"] = scene_graph.timestamp;

    nlohmann::json rooms = nlohmann::json::array();
    for (const auto& room : scene_graph.rooms) {
        nlohmann::json r;
        r["room_id"] = nodeIdToString(room.room_id);
        r["category"] = room.category;
        r["description"] = room.description;
        r["centroid"] = vec3Json(room.position);
        r["bounding_box"] = {
            {"min", vec3Json(room.bbox_min)},
            {"max", vec3Json(room.bbox_max)},
        };
        nlohmann::json ids = nlohmann::json::array();
        for (const auto& id : room.object_ids) {
            ids.push_back(nodeIdToString(id));
        }
        r["object_ids"] = std::move(ids);
        rooms.push_back(std::move(r));
    }
    sg["rooms"] = std::move(rooms);

    nlohmann::json objects = nlohmann::json::array();
    for (const auto& obj : scene_graph.objects) {
        nlohmann::json o;
        o["node_id"] = nodeIdToString(obj.node_id);
        o["category"] = obj.category;
        if (!obj.properties.description.empty()) {
            nlohmann::json props = {
                {"friction_level", obj.properties.friction_level},
                {"pushable", obj.properties.pushable},
                {"weight_level", obj.properties.weight_level},
                {"description", obj.properties.description},
            };
            // Add new fields if available
            if (!obj.properties.estimated_weight_kg.empty()) {
                props["estimated_weight_kg"] = obj.properties.estimated_weight_kg;
            }
            if (obj.properties.inference_confidence >= 0) {
                props["inference_confidence"] = obj.properties.inference_confidence;
            }
            o["physical_properties"] = std::move(props);
        }
        o["position"] = vec3Json(obj.position);
        o["bounding_box"] = {
            {"min", vec3Json(obj.bbox_min)},
            {"max", vec3Json(obj.bbox_max)},
        };
        objects.push_back(std::move(o));
    }
    sg["objects"] = std::move(objects);

    root["scene_graph"] = std::move(sg);

    const std::string json = root.dump(2) + "\n";

    // Atomic write to latest
    atomicWriteTextFile(latest_path, json);

    // Best-effort archive write (traceability)
    if (!archive_path.empty()) {
        try {
            atomicWriteTextFile(archive_path, json);
        } catch (const std::exception& e) {
            ROS_WARN("Failed to archive scene graph to %s: %s", archive_path.c_str(), e.what());
        }
    }

    // ============================================================
    // Compact scene graph (for LLM planning)
    // - rooms: room_id, category, object_ids
    // - objects: node_id, category
    // ============================================================
    const std::string compact_latest_path = output_dir_ + "/scene_graph_compact_latest.json";
    std::string compact_archive_path;
    if (!physical_source_dir_path_.empty()) {
        compact_archive_path = physical_source_dir_path_ + "/scene_graph_compact.json";
    }

    nlohmann::json compact_root;
    compact_root["schema_version"] = 1;
    compact_root["physical_source_dir"] = physical_source_dir_name_;

    nlohmann::json compact_sg;
    compact_sg["timestamp"] = scene_graph.timestamp;

    nlohmann::json compact_rooms = nlohmann::json::array();
    for (const auto& room : scene_graph.rooms) {
        nlohmann::json r;
        r["room_id"] = nodeIdToString(room.room_id);
        r["category"] = room.category;

        nlohmann::json ids = nlohmann::json::array();
        for (const auto& id : room.object_ids) {
            ids.push_back(nodeIdToString(id));
        }
        r["object_ids"] = std::move(ids);
        compact_rooms.push_back(std::move(r));
    }
    compact_sg["rooms"] = std::move(compact_rooms);

    nlohmann::json compact_objects = nlohmann::json::array();
    for (const auto& obj : scene_graph.objects) {
        nlohmann::json o;
        o["node_id"] = nodeIdToString(obj.node_id);
        o["category"] = obj.category;
        compact_objects.push_back(std::move(o));
    }
    compact_sg["objects"] = std::move(compact_objects);

    compact_root["scene_graph"] = std::move(compact_sg);
    const std::string compact_json = compact_root.dump(2) + "\n";

    // Atomic write to compact latest
    atomicWriteTextFile(compact_latest_path, compact_json);

    // Best-effort archive compact
    if (!compact_archive_path.empty()) {
        try {
            atomicWriteTextFile(compact_archive_path, compact_json);
        } catch (const std::exception& e) {
            ROS_WARN("Failed to archive compact scene graph to %s: %s",
                     compact_archive_path.c_str(), e.what());
        }
    }

    return latest_path;
}

// ============ Helper functions ============

void GraphGenerator::setupOutputDirectory() {
    try {
        std::string pkg_path = ros::package::getPath("phy_graph");
        output_dir_ = pkg_path + "/output/scene_graphs";
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
        if (ec) {
            throw std::runtime_error("Failed to create output directory: " + output_dir_);
        }
        ROS_INFO("Scene graph output directory: %s", output_dir_.c_str());
    } catch (std::exception& e) {
        ROS_ERROR("Failed to setup output directory: %s", e.what());
        output_dir_ = "/tmp/phy_graph_scene_graphs";
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
    }
}

std::string GraphGenerator::vector3dToJson(const Eigen::Vector3d& vec) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\"x\": " << vec.x() << ", \"y\": " << vec.y() << ", \"z\": " << vec.z() << "}";
    return oss.str();
}

} // namespace phy_graph

