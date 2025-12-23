#include <phy_graph/physical_inference_node.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <ros/package.h>
#include <hydra/common/dsg_types.h>

// ============ Benchmark for image ============
PhysicalInferenceNode::ProjectionResult 
PhysicalInferenceNode::projectObjectToImage(
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh& mesh,
    const Eigen::Isometry3d& world_T_camera,
    const cv::Size& image_size) {
    
    ProjectionResult result;
    result.score = 0.0;
    result.visible_count = 0;
    result.coverage = 0.0;
    
    Eigen::Isometry3d camera_T_world = world_T_camera.inverse();
    std::vector<cv::Point2f> points_2d;
    int total_vertices = attrs.mesh_connections.size();
    
    for (const auto& vertex_idx : attrs.mesh_connections) {
        if (vertex_idx >= mesh.vertices.size()) continue;
        
        const auto& v = mesh.vertices[vertex_idx];
        Eigen::Vector3d p_world(v.x, v.y, v.z);
        Eigen::Vector3d p_cam = camera_T_world * p_world;
        
        if (p_cam.z() <= 0.1) continue;
        
        double u = fx_ * p_cam.x() / p_cam.z() + cx_;
        double v_proj = fy_ * p_cam.y() / p_cam.z() + cy_;
        
        if (u >= 0 && u < image_size.width &&
            v_proj >= 0 && v_proj < image_size.height) {
            points_2d.push_back(cv::Point2f(u, v_proj));
        }
    }
    
    if (points_2d.size() < 10) return result;  
    
    result.visible_count = points_2d.size();
    result.bbox = cv::boundingRect(points_2d);
    
    // Scoring logic
    double visibility = static_cast<double>(points_2d.size()) / total_vertices;
    if (visibility > 0.8) result.score += 40;
    else if (visibility > 0.5) result.score += 30;
    else if (visibility > 0.3) result.score += 20;
    else result.score += 10;
    
    double bbox_area = result.bbox.width * result.bbox.height;
    double img_area = image_size.width * image_size.height;
    result.coverage = bbox_area / img_area;
    
    if (result.coverage > 0.1 && result.coverage < 0.4) result.score += 30;
    else if (result.coverage > 0.05 && result.coverage < 0.6) result.score += 20;
    else if (result.coverage > 0.01) result.score += 10;
    
    double cx_bbox = result.bbox.x + result.bbox.width / 2.0;
    double cy_bbox = result.bbox.y + result.bbox.height / 2.0;
    double cx_img = image_size.width / 2.0;
    double cy_img = image_size.height / 2.0;
    double dist_to_center = std::sqrt(std::pow(cx_bbox - cx_img, 2) + std::pow(cy_bbox - cy_img, 2));
    double max_dist = std::sqrt(cx_img*cx_img + cy_img*cy_img);
    result.score += (1.0 - (dist_to_center / max_dist)) * 15.0;
    
    const int margin = 20;
    if (result.bbox.x > margin && result.bbox.y > margin &&
        result.bbox.x + result.bbox.width < image_size.width - margin &&
        result.bbox.y + result.bbox.height < image_size.height - margin) {
        result.score += 15;
    }
    
    return result;
}

std::vector<PhysicalInferenceNode::ScoredImage>
PhysicalInferenceNode::scoreCandidateImages(
    const std::vector<phy_graph::KeyframeDatabase::Keyframe>& images,
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {

    std::vector<ScoredImage> scored_images;
    int img_index = 0;

    for (const auto& keyframe : images) {
        cv::Mat rgb_image = keyframe.decode(); // Might trigger disk read
        if (rgb_image.empty()) continue;

        auto result = projectObjectToImage(
            attrs, mesh, keyframe.world_T_camera, rgb_image.size());

        if (result.score >= 60.0) {
            ScoredImage scored;
            scored.original_image = rgb_image;
            scored.bbox = result.bbox;
            scored.score = result.score;
            scored.timestamp = keyframe.timestamp;
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
std::pair<cv::Mat, double> PhysicalInferenceNode::extractBestObjectImage(
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {
    
    if (keyframe_db_->size() == 0) return {cv::Mat(), 0.0};
    
    ros::Time object_creation_time;
    object_creation_time.fromNSec(attrs.last_update_time_ns);
    ros::Duration window(10.0);

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
    
    cv::Rect expanded_bbox = expandAndClampBbox(best.bbox, best.original_image.size());
    
    if (expanded_bbox.width <= 0 || expanded_bbox.height <= 0) {
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
    
    ROS_DEBUG("Extracted best image for %s (Score: %.1f). Archived to %s", 
             attrs.name.c_str(), best.score, ss.str().c_str());

    return {final_image, best.score};
}
