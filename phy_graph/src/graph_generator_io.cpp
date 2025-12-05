#include "phy_graph/graph_generator.h"
#include <ros/package.h>
#include <fstream>
#include <boost/filesystem.hpp>
#include <iomanip>
#include <algorithm>

namespace phy_graph {

// ============ JSON解析模块 ============

std::unordered_map<std::string, PhysicalProperties> 
GraphGenerator::loadPhysicalProperties() {
    std::unordered_map<std::string, PhysicalProperties> result;
    
    // 获取output目录路径
    std::string pkg_path = ros::package::getPath("phy_graph");
    std::string output_path = pkg_path + "/output";
    
    if (!boost::filesystem::exists(output_path)) {
        ROS_WARN_ONCE("Output directory does not exist: %s", output_path.c_str());
        return result;
    }
    
    // 遍历output目录下的所有子目录，找到最新的一个
    boost::filesystem::directory_iterator end_iter;
    std::vector<std::string> subdirs;
    
    for (boost::filesystem::directory_iterator dir_iter(output_path); 
         dir_iter != end_iter; ++dir_iter) {
        if (boost::filesystem::is_directory(dir_iter->status())) {
            std::string dir_name = dir_iter->path().filename().string();
            // 过滤掉非时间戳目录 (scene_graphs, keyframes, objects 等)
            if (dir_name == "scene_graphs" || dir_name == "keyframes" || dir_name == "objects") continue;
            // 简单的格式检查：是否包含 '-' (如 12-02_...)
            if (dir_name.find('-') == std::string::npos) continue;

            subdirs.push_back(dir_name);
        }
    }

    if (subdirs.empty()) {
        return result;
    }

    // 排序找到最新的目录 (假设目录名格式为日期时间，字符串排序即可)
    std::sort(subdirs.begin(), subdirs.end());
    std::string latest_dir_name = subdirs.back();
    boost::filesystem::path latest_dir_path = boost::filesystem::path(output_path) / latest_dir_name;
    
    ROS_INFO_ONCE("Loading physical properties from latest directory: %s", latest_dir_name.c_str());

    // 遍历最新目录中的JSON文件
    for (boost::filesystem::directory_iterator file_iter(latest_dir_path); 
            file_iter != end_iter; ++file_iter) {
        
        if (boost::filesystem::is_regular_file(file_iter->status()) &&
            file_iter->path().extension() == ".json") {
            
            std::string filename = file_iter->path().filename().string();
            
            // 只处理object_*.json文件
            if (filename.find("object_") == 0) {
                try {
                    PhysicalProperties props = parseObjectJson(file_iter->path().string());
                    
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
    
    std::string line;
    while (std::getline(file, line)) {

        // Simple trim implementation
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue; // 空行
        size_t last = line.find_last_not_of(" \t\r\n");
        std::string trimmed_line = line.substr(first, (last - first + 1));

        // Parse friction_level
        if (trimmed_line.find("\"friction_level\"") == 0) {
            size_t colon_pos = trimmed_line.find(':');
            if (colon_pos != std::string::npos) {
                std::string value_str = trimmed_line.substr(colon_pos + 1);
                // Remove possible comma
                size_t comma_pos = value_str.find(',');
                if (comma_pos != std::string::npos) value_str = value_str.substr(0, comma_pos);
                try {
                    props.friction_level = std::stoi(value_str);
                } catch (...) {}
            }
        }
        // Parse pushable
        else if (trimmed_line.find("\"pushable\"") == 0) {
            props.pushable = (trimmed_line.find("true") != std::string::npos);
        }
        // Parse weight_level
        else if (trimmed_line.find("\"weight_level\"") == 0) {
            size_t colon_pos = trimmed_line.find(':');
            if (colon_pos != std::string::npos) {
                std::string value_str = trimmed_line.substr(colon_pos + 1);
                size_t comma_pos = value_str.find(',');
                if (comma_pos != std::string::npos) value_str = value_str.substr(0, comma_pos);
                try {
                    props.weight_level = std::stoi(value_str);
                } catch (...) {}
            }
        }
        // Parse description (关键部分)
        else if (trimmed_line.find("\"description\"") == 0) {
            size_t colon_pos = trimmed_line.find(':');
            if (colon_pos != std::string::npos) {
                // Find first quote (after colon)
                size_t start_quote = trimmed_line.find('"', colon_pos + 1);
                // Find last quote
                size_t end_quote = trimmed_line.rfind('"');
                
                if (start_quote != std::string::npos && end_quote != std::string::npos && end_quote > start_quote) {
                    props.description = trimmed_line.substr(start_quote + 1, end_quote - start_quote - 1);
                }
            }
        }
    }
    
    file.close();
    return props;
}

std::string GraphGenerator::saveSceneGraphJson(const SceneGraph& scene_graph) {
    // Modify: Use a fixed filename for the "latest" graph to avoid disk spam
    // We also save a timestamped one ONLY on shutdown (or we can handle that by caller)
    
    // For now, let's just save to "scene_graph_latest.json"
    std::string filepath = output_dir_ + "/scene_graph_latest.json";
    
    // 打开文件 (Overwrite mode by default for ofstream)
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // 写入JSON内容
    file << "{\n";
    file << "  \"scene_graph\": {\n";
    file << "    \"timestamp\": \"" << scene_graph.timestamp << "\",\n";
    
    // 写入房间列表
    file << "    \"rooms\": [\n";
    for (size_t i = 0; i < scene_graph.rooms.size(); ++i) {
        const auto& room = scene_graph.rooms[i];
        file << "      {\n";
        file << "        \"room_id\": \"" << nodeIdToString(room.room_id) << "\",\n";
        file << "        \"category\": \"" << room.category << "\",\n";
        file << "        \"description\": \"" << room.description << "\",\n";
        file << "        \"centroid\": " << vector3dToJson(room.position) << ",\n";
        file << "        \"bounding_box\": {\n";
        file << "          \"min\": " << vector3dToJson(room.bbox_min) << ",\n";
        file << "          \"max\": " << vector3dToJson(room.bbox_max) << "\n";
        file << "        },\n";
        file << "        \"object_ids\": [";
        for (size_t j = 0; j < room.object_ids.size(); ++j) {
            file << "\"" << nodeIdToString(room.object_ids[j]) << "\"";
            if (j < room.object_ids.size() - 1) file << ", ";
        }
        file << "]\n";
        file << "      }";
        if (i < scene_graph.rooms.size() - 1) file << ",";
        file << "\n";
    }
    file << "    ],\n";
    
    // Write objects list
    file << "    \"objects\": [\n";
    for (size_t i = 0; i < scene_graph.objects.size(); ++i) {
        const auto& obj = scene_graph.objects[i];
        file << "      {\n";
        file << "        \"node_id\": \"" << nodeIdToString(obj.node_id) << "\",\n";
        file << "        \"category\": \"" << obj.category << "\",\n";
        
        // Only show physical properties if description is not empty (meaning it was processed by VLM)
        if (!obj.properties.description.empty()) {
            file << "        \"physical_properties\": {\n";
            file << "          \"friction_level\": " << obj.properties.friction_level << ",\n";
            file << "          \"pushable\": " << (obj.properties.pushable ? "true" : "false") << ",\n";
            file << "          \"weight_level\": " << obj.properties.weight_level << ",\n";
            file << "          \"description\": \"" << obj.properties.description << "\"\n";
            file << "        },\n";
        }
        
        file << "        \"position\": " << vector3dToJson(obj.position) << ",\n";
        file << "        \"bounding_box\": {\n";
        file << "          \"min\": " << vector3dToJson(obj.bbox_min) << ",\n";
        file << "          \"max\": " << vector3dToJson(obj.bbox_max) << "\n";
        file << "        }\n";
        file << "      }";
        if (i < scene_graph.objects.size() - 1) file << ",";
        file << "\n";
    }
    file << "    ]\n";
    
    file << "  }\n";
    file << "}\n";
    
    file.close();
    
    return filepath;
}

// ============ Helper functions ============

void GraphGenerator::setupOutputDirectory() {
    try {
        std::string pkg_path = ros::package::getPath("phy_graph");
        output_dir_ = pkg_path + "/output/scene_graphs";
        boost::filesystem::create_directories(output_dir_);
        ROS_INFO("Scene graph output directory: %s", output_dir_.c_str());
    } catch (std::exception& e) {
        ROS_ERROR("Failed to setup output directory: %s", e.what());
        output_dir_ = "/tmp/phy_graph_scene_graphs";
        boost::filesystem::create_directories(output_dir_);
    }
}

std::string GraphGenerator::vector3dToJson(const Eigen::Vector3d& vec) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\"x\": " << vec.x() << ", \"y\": " << vec.y() << ", \"z\": " << vec.z() << "}";
    return oss.str();
}

} // namespace phy_graph

