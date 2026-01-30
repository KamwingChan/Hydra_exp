#include <phy_graph/physical_inference_node.h>
#include <phy_graph/inference_config.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <ros/package.h>
#include <hydra/common/dsg_types.h>

namespace {
// 辅助函数：投影单个3D点到2D
inline bool projectPoint(const Eigen::Vector3d& p_world,
                        const Eigen::Isometry3d& camera_T_world,
                        double fx, double fy, double cx, double cy,
                        const cv::Size& image_size,
                        cv::Point2f& out_point) {
    Eigen::Vector3d p_cam = camera_T_world * p_world;
    if (p_cam.z() <= 0.1) return false;
    
    double u = fx * p_cam.x() / p_cam.z() + cx;
    double v = fy * p_cam.y() / p_cam.z() + cy;
    
    if (u >= 0 && u < image_size.width && v >= 0 && v < image_size.height) {
        out_point = cv::Point2f(u, v);
        return true;
    }
    return false;
}

// 模式1: 投影 Hydra bbox（快速）
std::pair<std::vector<cv::Point2f>, int> projectHydraBBox(
    const hydra::ObjectNodeAttributes& attrs,
    const Eigen::Isometry3d& camera_T_world,
    double fx, double fy, double cx, double cy,
    const cv::Size& image_size) {
    
    std::vector<cv::Point2f> points_2d;
    
    if (attrs.bounding_box.type != hydra::BoundingBox::Type::AABB) {
        return {points_2d, 0};
    }
    
    Eigen::Vector3d center = attrs.bounding_box.world_P_center.cast<double>();
    Eigen::Vector3d half_dim = attrs.bounding_box.dimensions.cast<double>() / 2.0;
    
    // 8个角点
    std::vector<Eigen::Vector3d> corners = {
        center + Eigen::Vector3d(-half_dim.x(), -half_dim.y(), -half_dim.z()),
        center + Eigen::Vector3d( half_dim.x(), -half_dim.y(), -half_dim.z()),
        center + Eigen::Vector3d(-half_dim.x(),  half_dim.y(), -half_dim.z()),
        center + Eigen::Vector3d( half_dim.x(),  half_dim.y(), -half_dim.z()),
        center + Eigen::Vector3d(-half_dim.x(), -half_dim.y(),  half_dim.z()),
        center + Eigen::Vector3d( half_dim.x(), -half_dim.y(),  half_dim.z()),
        center + Eigen::Vector3d(-half_dim.x(),  half_dim.y(),  half_dim.z()),
        center + Eigen::Vector3d( half_dim.x(),  half_dim.y(),  half_dim.z())
    };
    
    for (const auto& corner : corners) {
        cv::Point2f pt;
        if (projectPoint(corner, camera_T_world, fx, fy, cx, cy, image_size, pt)) {
            points_2d.push_back(pt);
        }
    }
    
    return {points_2d, 8};  // 返回投影点和总顶点数
}

// 模式2: 投影 mesh 顶点（精确但慢）
// mesh 参数改为指针，nullptr 时返回空结果
std::pair<std::vector<cv::Point2f>, int> projectMeshVertices(
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh* mesh,  // nullable pointer
    const Eigen::Isometry3d& camera_T_world,
    double fx, double fy, double cx, double cy,
    const cv::Size& image_size) {
    
    std::vector<cv::Point2f> points_2d;
    
    // If mesh is null, return empty result
    if (!mesh) {
        return {points_2d, 0};
    }
    
    int total_vertices = attrs.mesh_connections.size();
    
    for (const auto& vertex_idx : attrs.mesh_connections) {
        if (vertex_idx >= mesh->vertices.size()) continue;
        
        const auto& v = mesh->vertices[vertex_idx];
        Eigen::Vector3d p_world(v.x, v.y, v.z);
        cv::Point2f pt;
        if (projectPoint(p_world, camera_T_world, fx, fy, cx, cy, image_size, pt)) {
            points_2d.push_back(pt);
        }
    }
    
    return {points_2d, total_vertices};
}

// Helper: Get 3D sample points from object bbox for occlusion check
std::vector<Eigen::Vector3d> getSamplePointsFromBBox(
    const hydra::ObjectNodeAttributes& attrs,
    int num_samples) {
    
    std::vector<Eigen::Vector3d> sample_points;
    
    if (attrs.bounding_box.type != hydra::BoundingBox::Type::AABB) {
        return sample_points;
    }
    
    Eigen::Vector3d center = attrs.bounding_box.world_P_center.cast<double>();
    Eigen::Vector3d half_dim = attrs.bounding_box.dimensions.cast<double>() / 2.0;
    
    // Always include 8 corners
    sample_points.push_back(center + Eigen::Vector3d(-half_dim.x(), -half_dim.y(), -half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d( half_dim.x(), -half_dim.y(), -half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d(-half_dim.x(),  half_dim.y(), -half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d( half_dim.x(),  half_dim.y(), -half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d(-half_dim.x(), -half_dim.y(),  half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d( half_dim.x(), -half_dim.y(),  half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d(-half_dim.x(),  half_dim.y(),  half_dim.z()));
    sample_points.push_back(center + Eigen::Vector3d( half_dim.x(),  half_dim.y(),  half_dim.z()));
    
    // Add center
    sample_points.push_back(center);
    
    // Add face centers if more samples needed
    if (num_samples > 9) {
        sample_points.push_back(center + Eigen::Vector3d( half_dim.x(), 0, 0));
        sample_points.push_back(center + Eigen::Vector3d(-half_dim.x(), 0, 0));
        sample_points.push_back(center + Eigen::Vector3d(0,  half_dim.y(), 0));
        sample_points.push_back(center + Eigen::Vector3d(0, -half_dim.y(), 0));
        sample_points.push_back(center + Eigen::Vector3d(0, 0,  half_dim.z()));
        sample_points.push_back(center + Eigen::Vector3d(0, 0, -half_dim.z()));
    }
    
    return sample_points;
}

}  // anonymous namespace

// ============ Occlusion Detection ============
double PhysicalInferenceNode::calculateOcclusionScore(
    const hydra::ObjectNodeAttributes& attrs,
    const cv::Mat& depth_image,
    const Eigen::Isometry3d& world_T_camera,
    const cv::Size& image_size) {
    
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    
    if (depth_image.empty() || !cfg.occlusion.enable) {
        return cfg.occlusion.max_score;  // Return max score if no depth available
    }
    
    Eigen::Isometry3d camera_T_world = world_T_camera.inverse();
    
    // Get sample points from object bbox
    auto sample_points = getSamplePointsFromBBox(attrs, cfg.occlusion.sample_points);
    if (sample_points.empty()) {
        return cfg.occlusion.max_score;
    }
    
    int visible_count = 0;
    int checked_count = 0;
    
    for (const auto& p_world : sample_points) {
        // Transform to camera frame
        Eigen::Vector3d p_cam = camera_T_world * p_world;
        
        // Skip points behind camera
        if (p_cam.z() <= 0.1) continue;
        
        // Project to image
        double u = fx_ * p_cam.x() / p_cam.z() + cx_;
        double v = fy_ * p_cam.y() / p_cam.z() + cy_;
        
        // Check if in image bounds
        int pixel_x = static_cast<int>(u);
        int pixel_y = static_cast<int>(v);
        if (pixel_x < 0 || pixel_x >= image_size.width ||
            pixel_y < 0 || pixel_y >= image_size.height) {
            continue;
        }
        
        checked_count++;
        
        // Get depth from depth image (in mm, convert to meters)
        double depth_value_mm;
        if (depth_image.type() == CV_16UC1) {
            depth_value_mm = static_cast<double>(depth_image.at<uint16_t>(pixel_y, pixel_x));
        } else if (depth_image.type() == CV_32FC1) {
            depth_value_mm = static_cast<double>(depth_image.at<float>(pixel_y, pixel_x)) * 1000.0;
        } else {
            continue;
        }
        
        double depth_value_m = depth_value_mm / 1000.0;
        
        // Skip invalid depth (0 or too large)
        if (depth_value_m < 0.1 || depth_value_m > 20.0) {
            visible_count++;  // Assume visible if depth is invalid
            continue;
        }
        
        // Compare object depth with depth image
        double object_depth = p_cam.z();
        
        // If object is closer than or equal to depth image value (within threshold), it's visible
        if (object_depth <= depth_value_m + cfg.occlusion.depth_threshold) {
            visible_count++;
        }
        // Otherwise, the point is occluded
    }
    
    if (checked_count == 0) {
        return cfg.occlusion.max_score;
    }
    
    // Calculate occlusion ratio and convert to score
    double visibility_ratio = static_cast<double>(visible_count) / checked_count;
    double score = visibility_ratio * cfg.occlusion.max_score;
    
    return score;
}

// ============ Benchmark for image ============
PhysicalInferenceNode::ProjectionResult
PhysicalInferenceNode::projectObjectToImage(
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh* mesh,  // nullable pointer for hydra_bbox mode
    const Eigen::Isometry3d& world_T_camera,
    const cv::Size& image_size,
    const cv::Mat& depth_image) {
    
    ProjectionResult result;
    result.score = 0.0;
    result.visible_count = 0;
    result.coverage = 0.0;
    result.occlusion_score = 0.0;
    
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    Eigen::Isometry3d camera_T_world = world_T_camera.inverse();
    
    // === 选择投影模式 ===
    // hydra_bbox mode doesn't need mesh; mesh_vertices mode requires valid mesh pointer
    auto [points_2d, total_vertices] = (cfg.image.projection_mode == "hydra_bbox")
        ? projectHydraBBox(attrs, camera_T_world, fx_, fy_, cx_, cy_, image_size)
        : projectMeshVertices(attrs, mesh, camera_T_world, fx_, fy_, cx_, cy_, image_size);
    
    // 检查最小可见点数
    size_t min_points = (cfg.image.projection_mode == "hydra_bbox") ? 3 : 10;
    if (points_2d.size() < min_points) return result;
    
    result.visible_count = points_2d.size();
    result.bbox = cv::boundingRect(points_2d);
    
    // === Improved Scoring Logic ===
    // With occlusion detection:
    //   Visibility 30%, Coverage 35%, Occlusion 15%, Center 8%, Margin 12%
    // Without occlusion detection (depth unavailable):
    //   Visibility 35%, Coverage 40%, Center 10%, Margin 15%
    
    bool use_occlusion = cfg.occlusion.enable && !depth_image.empty();
    
    // 1. Visibility score (30 or 35 points max) - how much of the object is visible
    double visibility = static_cast<double>(points_2d.size()) / total_vertices;
    double visibility_max = use_occlusion ? 30.0 : 35.0;
    if (visibility > 0.8) result.score += visibility_max;
    else if (visibility > 0.5) result.score += visibility_max * 0.8;
    else if (visibility > 0.3) result.score += visibility_max * 0.57;
    else result.score += visibility_max * 0.29;
    
    // 2. Coverage score (30 or 40 points max) - object size in image
    double bbox_area = result.bbox.width * result.bbox.height;
    double img_area = image_size.width * image_size.height;
    result.coverage = bbox_area / img_area;
    
    double coverage_max = use_occlusion ? 30.0 : 40.0;
    // Prefer coverage between 5% and 50% (sweet spot for object recognition)
    if (result.coverage > 0.1 && result.coverage < 0.5) {
        result.score += coverage_max;  // Optimal range
    } else if (result.coverage > 0.05 && result.coverage < 0.6) {
        result.score += coverage_max * 0.75;
    } else if (result.coverage > 0.02) {
        result.score += coverage_max * 0.375;
    } else {
        result.score += coverage_max * 0.125;  // Very small objects get low score
    }
    
    // 3. Occlusion score (25 points max) - NEW: how much of the object is unoccluded
    if (use_occlusion) {
        result.occlusion_score = calculateOcclusionScore(attrs, depth_image, world_T_camera, image_size);
        result.score += result.occlusion_score;
    } else {
        result.occlusion_score = cfg.occlusion.max_score;  // Full score if not checking
    }
    
    // 4. Center score (5 or 10 points max) - reduced weight, not critical for VLM
    double cx_bbox = result.bbox.x + result.bbox.width / 2.0;
    double cy_bbox = result.bbox.y + result.bbox.height / 2.0;
    double cx_img = image_size.width / 2.0;
    double cy_img = image_size.height / 2.0;
    double dist_to_center = std::sqrt(std::pow(cx_bbox - cx_img, 2) + std::pow(cy_bbox - cy_img, 2));
    double max_dist = std::sqrt(cx_img*cx_img + cy_img*cy_img);
    double center_max = use_occlusion ? 5.0 : 10.0;
    result.score += (1.0 - (dist_to_center / max_dist)) * center_max;
    
    // 5. Margin score (10 or 15 points max) - object fully in frame
    const int margin = 20;
    double margin_max = use_occlusion ? 10.0 : 15.0;
    if (result.bbox.x > margin && result.bbox.y > margin &&
        result.bbox.x + result.bbox.width < image_size.width - margin &&
        result.bbox.y + result.bbox.height < image_size.height - margin) {
        result.score += margin_max;
    }
    
    return result;
}

std::vector<PhysicalInferenceNode::ScoredImage>
PhysicalInferenceNode::scoreCandidateImages(
    const std::vector<phy_graph::KeyframeDatabase::Keyframe>& images,
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh* mesh) {  // nullable for hydra_bbox mode

    std::vector<ScoredImage> scored_images;
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();

    for (const auto& keyframe : images) {
        cv::Mat rgb_image = keyframe.decode(); // Might trigger disk read
        if (rgb_image.empty()) continue;

        // Decode depth image if available and occlusion detection is enabled
        cv::Mat depth_image;
        if (cfg.occlusion.enable && keyframe.has_depth) {
            depth_image = keyframe.decodeDepth();
        }

        auto result = projectObjectToImage(
            attrs, mesh, keyframe.world_T_camera, rgb_image.size(), depth_image);

        if (result.score >= cfg.image.score_threshold) {
            ScoredImage scored;
            scored.original_image = rgb_image;
            scored.bbox = result.bbox;
            scored.score = result.score;
            scored.timestamp = keyframe.timestamp;
            scored.has_depth = keyframe.has_depth;
            scored.depth_image = depth_image;
            scored_images.push_back(scored);
        }
    }
    return scored_images;
}

cv::Rect PhysicalInferenceNode::expandAndClampBbox(const cv::Rect& bbox,
                                                   const cv::Size& image_size,
                                                   float padding_factor) {
    int pad_x = static_cast<int>(bbox.width * padding_factor);
    int pad_y = static_cast<int>(bbox.height * padding_factor);
    int x = std::max(0, bbox.x - pad_x);
    int y = std::max(0, bbox.y - pad_y);
    int width = std::min(image_size.width - x, bbox.width + 2 * pad_x);
    int height = std::min(image_size.height - y, bbox.height + 2 * pad_y);
    return cv::Rect(x, y, width, height);
}

// Optimized: Return cv::Mat directly, optionally save to archive
// mesh is optional pointer - pass nullptr when using hydra_bbox mode
std::pair<cv::Mat, double> PhysicalInferenceNode::extractBestObjectImage(
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh* mesh) {
    
    if (keyframe_db_->size() == 0) return {cv::Mat(), 0.0};
    
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    
    ros::Time object_creation_time;
    object_creation_time.fromNSec(attrs.last_update_time_ns);
    ros::Duration window(cfg.keyframe.time_window);

    auto candidate_images = keyframe_db_->getKeyframesInRange(
        object_creation_time - window,
        object_creation_time + window
    );
    
    if (candidate_images.empty()) return {cv::Mat(), 0.0};
    
    auto scored_images = scoreCandidateImages(candidate_images, attrs, mesh);
    if (scored_images.empty()) return {cv::Mat(), 0.0};
    
    std::sort(scored_images.begin(), scored_images.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });
    
    const auto& best = scored_images[0];
    
    cv::Rect expanded_bbox = expandAndClampBbox(best.bbox, best.original_image.size(), cfg.image.padding_factor);
    
    if (expanded_bbox.width <= 0 || expanded_bbox.height <= 0) {
        return {cv::Mat(), 0.0};
    }
    
    // Check minimum crop size - skip if too small
    const int min_size = cfg.image.min_crop_size;
    if (expanded_bbox.width < min_size || expanded_bbox.height < min_size) {
        ROS_DEBUG("Cropped image too small (%dx%d < %d), skipping", 
                  expanded_bbox.width, expanded_bbox.height, min_size);
        return {cv::Mat(), 0.0};
    }

    cv::Mat final_image = best.original_image(expanded_bbox).clone(); // Clone to be safe

    // Archive image for future reference (dataset creation)
    std::string archive_dir = output_dir_ + "/objects";
    std::error_code ec;
    std::filesystem::create_directories(archive_dir, ec);
    std::stringstream ss;
    ss << archive_dir << "/object_" << attrs.name << ".jpg";
    cv::imwrite(ss.str(), final_image);
    
    // === DEBUG: Save detailed extraction process ===
    if (cfg.debug.save_images) {
        // Use timestamp to create unique folder name
        static int debug_counter = 0;
        std::stringstream debug_folder_name;
        debug_folder_name << std::setfill('0') << std::setw(4) << (++debug_counter) 
                         << "_" << attrs.name;
        std::string debug_dir = output_dir_ + "/debug/" + debug_folder_name.str();
        std::filesystem::create_directories(debug_dir, ec);
        
        // 1. Save original full image
        cv::imwrite(debug_dir + "/01_original.jpg", best.original_image);
        
        // 2. Save annotated image with bbox visualization
        cv::Mat vis = best.original_image.clone();
        cv::rectangle(vis, best.bbox, cv::Scalar(0, 255, 0), 2);           // Green: original bbox
        cv::rectangle(vis, expanded_bbox, cv::Scalar(255, 0, 0), 2);       // Blue: expanded bbox
        
        // Add text annotations
        std::stringstream score_text;
        score_text << "Score: " << std::fixed << std::setprecision(1) << best.score;
        cv::putText(vis, score_text.str(), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        
        std::stringstream bbox_text;
        bbox_text << "BBox: [" << best.bbox.x << "," << best.bbox.y << "," 
                  << best.bbox.width << "x" << best.bbox.height << "]";
        cv::putText(vis, bbox_text.str(), cv::Point(10, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);
        
        cv::imwrite(debug_dir + "/02_bbox_vis.jpg", vis);
        
        // 3. Save cropped image
        cv::imwrite(debug_dir + "/03_cropped.jpg", final_image);
        
        // 4. Save text info
        std::ofstream info_file(debug_dir + "/info.txt");
        info_file << "Object: " << attrs.name << " (Label: " << attrs.semantic_label << ")\n";
        info_file << "Timestamp: " << best.timestamp << "\n";
        info_file << "Score: " << best.score << "\n";
        info_file << "Has Depth: " << (best.has_depth ? "Yes" : "No") << "\n";
        info_file << "Occlusion Detection: " << (cfg.occlusion.enable ? "Enabled" : "Disabled") << "\n\n";
        info_file << "Original BBox: [" << best.bbox.x << ", " << best.bbox.y
                  << ", " << best.bbox.width << ", " << best.bbox.height << "]\n";
        info_file << "Expanded BBox: [" << expanded_bbox.x << ", " << expanded_bbox.y
                  << ", " << expanded_bbox.width << ", " << expanded_bbox.height << "]\n";
        info_file << "Image Size: " << best.original_image.cols << "x"
                  << best.original_image.rows << "\n\n";
        info_file << "Total Candidates: " << scored_images.size() << "\n";
        info_file << "Padding Factor: " << cfg.image.padding_factor << "\n";
        info_file << "Occlusion Threshold: " << cfg.occlusion.depth_threshold << "m\n";
        info_file << "Occlusion Sample Points: " << cfg.occlusion.sample_points << "\n";
        info_file.close();
        
        // 5. Save depth visualization if available
        if (best.has_depth && !best.depth_image.empty()) {
            cv::Mat depth_vis;
            if (best.depth_image.type() == CV_16UC1) {
                // Normalize 16-bit depth to 8-bit for visualization
                best.depth_image.convertTo(depth_vis, CV_8UC1, 255.0 / 10000.0);  // 0-10m range
            } else if (best.depth_image.type() == CV_32FC1) {
                // Normalize float depth to 8-bit for visualization
                best.depth_image.convertTo(depth_vis, CV_8UC1, 255.0 / 10.0);  // 0-10m range
            }
            if (!depth_vis.empty()) {
                cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_JET);
                cv::imwrite(debug_dir + "/04_depth_vis.jpg", depth_vis);
            }
        }
        
        ROS_INFO("Debug images saved to: %s", debug_dir.c_str());
    }
    
    ROS_DEBUG("Extracted best image for %s (Score: %.1f). Archived to %s", 
             attrs.name.c_str(), best.score, ss.str().c_str());

    return {final_image, best.score};
}
