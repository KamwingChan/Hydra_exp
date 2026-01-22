#pragma once

#include <ros/ros.h>
#include <string>

namespace phy_graph {

/**
 * @brief 推理配置结构体
 * 
 * 从 inference_config.yaml 加载的所有配置参数
 */
struct InferenceConfig {
    // === 图像处理参数 ===
    struct Image {
        int min_crop_size = 64;
        float padding_factor = 0.3f;
        float score_threshold = 60.0f;
        float high_quality_threshold = 70.0f;
        std::string projection_mode = "mesh_vertices";
    } image;

    // === 关键帧数据库参数 ===
    struct Keyframe {
        int max_memory_frames = 3000;
        double min_translation = 0.2;
        double min_rotation = 0.1;
        double min_time_interval = 0.2;
        double time_window = 10.0;
        int frame_skip = 5;  // Process every N-th frame (e.g., 5 means 15Hz→3Hz)
    } keyframe;

    // === 推理队列参数 ===
    struct Inference {
        int num_workers = 1;
        int max_queue_size = 100;
        int max_defer_count = 5;
        double wait_timeout = 5.0;
        double min_acceptable_score = 50.0;
        double loop_rate = 0.5;
    } inference;

    // === VLM 服务参数 ===
    struct VLM {
        std::string model_name = "openai/gpt-4o-mini";
        int max_retries = 3;
        int timeout_seconds = 30;
        bool dry_run = false;
    } vlm;

    // === 物体成熟度参数 ===
    struct ObjectMaturity {
        double min_age_seconds = 0.5;
        bool enable = true;
    } object_maturity;

    // === 遮挡检测参数 ===
    struct Occlusion {
        bool enable = true;                  // 是否启用遮挡检测
        double depth_threshold = 0.1;        // 深度比较阈值（米），超过此值认为被遮挡
        int sample_points = 50;              // 遮挡检测采样点数
        int max_score = 15;                  // 遮挡评分最大分值
    } occlusion;

    // === 调试参数 ===
    struct Debug {
        bool save_images = false;
        bool verbose = false;
    } debug;

    /**
     * @brief 从 ROS 参数服务器加载配置
     * 
     * @param nh NodeHandle，用于读取参数
     * @param ns 参数命名空间（默认 "inference"）
     */
    void loadFromROS(ros::NodeHandle& nh, const std::string& ns = "inference");

    /**
     * @brief 打印当前配置（用于调试）
     */
    void print() const;
};

/**
 * @brief 全局配置单例
 * 
 * 使用方法：
 *   auto& cfg = InferenceConfigManager::get();
 *   cfg.loadFromROS(nh);
 *   int min_size = cfg.config().image.min_crop_size;
 */
class InferenceConfigManager {
public:
    static InferenceConfigManager& get() {
        static InferenceConfigManager instance;
        return instance;
    }

    void loadFromROS(ros::NodeHandle& nh, const std::string& ns = "inference") {
        config_.loadFromROS(nh, ns);
    }

    const InferenceConfig& config() const { return config_; }
    InferenceConfig& config() { return config_; }

private:
    InferenceConfigManager() = default;
    InferenceConfig config_;
};

} // namespace phy_graph
