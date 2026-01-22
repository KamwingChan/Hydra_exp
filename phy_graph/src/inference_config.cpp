#include "phy_graph/inference_config.h"

namespace phy_graph {

void InferenceConfig::loadFromROS(ros::NodeHandle& nh, const std::string& ns) {
    // === 图像处理参数 ===
    nh.param<int>(ns + "/image/min_crop_size", image.min_crop_size, 64);
    nh.param<float>(ns + "/image/padding_factor", image.padding_factor, 0.3f);
    nh.param<float>(ns + "/image/score_threshold", image.score_threshold, 60.0f);
    nh.param<float>(ns + "/image/high_quality_threshold", image.high_quality_threshold, 70.0f);
    nh.param<std::string>(ns + "/image/projection_mode", image.projection_mode, "mesh_vertices");

    // === 关键帧数据库参数 ===
    nh.param<int>(ns + "/keyframe/max_memory_frames", keyframe.max_memory_frames, 3000);
    nh.param<double>(ns + "/keyframe/min_translation", keyframe.min_translation, 0.2);
    nh.param<double>(ns + "/keyframe/min_rotation", keyframe.min_rotation, 0.1);
    nh.param<double>(ns + "/keyframe/min_time_interval", keyframe.min_time_interval, 0.2);
    nh.param<double>(ns + "/keyframe/time_window", keyframe.time_window, 10.0);
    nh.param<int>(ns + "/keyframe/frame_skip", keyframe.frame_skip, 5);  // Process every N-th frame

    // === 推理队列参数 ===
    nh.param<int>(ns + "/inference/num_workers", inference.num_workers, 1);
    nh.param<int>(ns + "/inference/max_queue_size", inference.max_queue_size, 100);
    nh.param<int>(ns + "/inference/max_defer_count", inference.max_defer_count, 5);
    nh.param<double>(ns + "/inference/wait_timeout", inference.wait_timeout, 5.0);
    nh.param<double>(ns + "/inference/min_acceptable_score", inference.min_acceptable_score, 50.0);
    nh.param<double>(ns + "/inference/loop_rate", inference.loop_rate, 0.5);

    // === VLM 服务参数 ===
    nh.param<std::string>(ns + "/vlm/model_name", vlm.model_name, "openai/gpt-4o-mini");
    nh.param<int>(ns + "/vlm/max_retries", vlm.max_retries, 3);
    nh.param<int>(ns + "/vlm/timeout_seconds", vlm.timeout_seconds, 30);
    nh.param<bool>(ns + "/vlm/dry_run", vlm.dry_run, false);

    // === 物体成熟度参数 ===
    nh.param<double>(ns + "/object_maturity/min_age_seconds", object_maturity.min_age_seconds, 0.5);
    nh.param<bool>(ns + "/object_maturity/enable", object_maturity.enable, true);

    // === 遮挡检测参数 ===
    nh.param<bool>(ns + "/occlusion/enable", occlusion.enable, true);
    nh.param<double>(ns + "/occlusion/depth_threshold", occlusion.depth_threshold, 0.1);
    nh.param<int>(ns + "/occlusion/sample_points", occlusion.sample_points, 50);
    nh.param<int>(ns + "/occlusion/max_score", occlusion.max_score, 15);

    // === 调试参数 ===
    nh.param<bool>(ns + "/debug/save_images", debug.save_images, false);
    nh.param<bool>(ns + "/debug/verbose", debug.verbose, false);
}

void InferenceConfig::print() const {
    ROS_INFO("=== InferenceConfig ===");
    ROS_INFO("Image: min_crop_size=%d, padding_factor=%.2f, score_threshold=%.1f, high_quality_threshold=%.1f, projection_mode=%s",
             image.min_crop_size, image.padding_factor, image.score_threshold, image.high_quality_threshold, image.projection_mode.c_str());
    ROS_INFO("Keyframe: max_memory_frames=%d, min_translation=%.2f, min_rotation=%.2f, time_window=%.1f, frame_skip=%d",
             keyframe.max_memory_frames, keyframe.min_translation, keyframe.min_rotation, keyframe.time_window, keyframe.frame_skip);
    ROS_INFO("Inference: num_workers=%d, max_queue_size=%d, max_defer_count=%d, wait_timeout=%.1f, min_score=%.1f",
             inference.num_workers, inference.max_queue_size, inference.max_defer_count, inference.wait_timeout, inference.min_acceptable_score);
    ROS_INFO("VLM: model_name=%s, max_retries=%d, dry_run=%s",
             vlm.model_name.c_str(), vlm.max_retries, vlm.dry_run ? "true" : "false");
    ROS_INFO("ObjectMaturity: min_age_seconds=%.2f, enable=%s",
             object_maturity.min_age_seconds, object_maturity.enable ? "true" : "false");
    ROS_INFO("Occlusion: enable=%s, depth_threshold=%.2f, sample_points=%d, max_score=%d",
             occlusion.enable ? "true" : "false", occlusion.depth_threshold, occlusion.sample_points, occlusion.max_score);
    ROS_INFO("Debug: save_images=%s, verbose=%s",
             debug.save_images ? "true" : "false", debug.verbose ? "true" : "false");
}

} // namespace phy_graph
