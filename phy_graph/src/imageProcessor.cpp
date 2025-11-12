#include <phy_graph/physical_inference_node.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <boost/filesystem.hpp>
#include <ros/package.h>
#include <hydra/common/global_info.h>

// ============ ImageCache ============
void ImageCache::addImage(const sensor_msgs::ImageConstPtr& msg,
                          const Eigen::Isometry3d& world_T_camera) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        
        CachedImage cached;
        cached.timestamp = msg->header.stamp;
        cached.rgb_image = cv_ptr->image.clone();  // 深拷贝
        cached.world_T_camera = world_T_camera;
        
        cache_.push_back(cached);
        
        // keep cache size within max_size_
        while (cache_.size() > max_size_) {
            cache_.pop_front();
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("ImageCache: cv_bridge exception: %s", e.what());
    }
}

std::vector<ImageCache::CachedImage> ImageCache::getImagesInRange(
    ros::Time start, ros::Time end) const {
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    std::vector<CachedImage> result;
    
    for (const auto& img : cache_) {
        if (img.timestamp >= start && img.timestamp <= end) {
            result.push_back(img);
        }
    }
    
    return result;
}

// ============ RGB and CameraInfo ============
void PhysicalInferenceNode::cameraInfoCallback(
    const sensor_msgs::CameraInfoConstPtr& msg) {
    
    if (camera_info_received_) return;
    
    fx_ = msg->K[0];
    fy_ = msg->K[4];
    cx_ = msg->K[2];
    cy_ = msg->K[5];
    camera_frame_ = msg->header.frame_id;
    
    camera_info_received_ = true;
    
    ROS_INFO("Camera info received:");
    ROS_INFO("  fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx_, fy_, cx_, cy_);
    ROS_INFO("  frame_id=%s", camera_frame_.c_str());
    
    // receive once, then unsubscribe
    camera_info_sub_.shutdown();
}

void PhysicalInferenceNode::rgbCallback(
    const sensor_msgs::ImageConstPtr& msg) {
    
    if (!camera_info_received_) {
        ROS_DEBUG_THROTTLE(5.0, "Waiting for camera_info...");
        return;
    }
    
    // skip frames to reduce load
    static int frame_counter = 0;
    if (++frame_counter % 10 != 0) return;
    
    try {
        // Query TF to get world_T_camera
        geometry_msgs::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(
                "world",  
                camera_frame_,  
                msg->header.stamp,
                ros::Duration(0.5)  //override timeout
            );
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(5.0, "TF lookup failed at t=%.3f: %s", 
                            msg->header.stamp.toSec(), ex.what());
            return;
        }
        
        // transfer to Eigen::Isometry3d
        Eigen::Isometry3d world_T_camera = tf2::transformToEigen(transform);
        
        // add to image cache
        image_cache_->addImage(msg, world_T_camera);
        
        ROS_DEBUG("Cached RGB image at t=%.3f, cache size=%zu",
                  msg->header.stamp.toSec(),
                  image_cache_->size());
        
    } catch (std::exception& e) {
        ROS_ERROR("RGB callback exception: %s", e.what());
    }
}

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
    
    // 1. transform world_T_camera to camera_T_world
    Eigen::Isometry3d camera_T_world = world_T_camera.inverse();
    
    // 2. project all mesh vertices
    std::vector<cv::Point2f> points_2d;
    int total_vertices = attrs.mesh_connections.size();
    
    for (const auto& vertex_idx : attrs.mesh_connections) {
        if (vertex_idx >= mesh.vertices.size()) continue;
        
        const auto& v = mesh.vertices[vertex_idx];
        Eigen::Vector3d p_world(v.x, v.y, v.z);
        Eigen::Vector3d p_cam = camera_T_world * p_world;
        
        //check z > 0.1
        if (p_cam.z() <= 0.1) continue;
        
        // pinhole projection
        double u = fx_ * p_cam.x() / p_cam.z() + cx_;
        double v_proj = fy_ * p_cam.y() / p_cam.z() + cy_;
        
        // check within image bounds
        if (u >= 0 && u < image_size.width &&
            v_proj >= 0 && v_proj < image_size.height) {
            points_2d.push_back(cv::Point2f(u, v_proj));
        }
    }
    
    // 3. verify visible points count
    if (points_2d.size() < 10) {
        return result;  
    }
    
    result.visible_count = points_2d.size();
    
    // 4. calculate bounding box    
    result.bbox = cv::boundingRect(points_2d);
    
    // 5. scoring criteria
    
    // 5.1 visibility (0-40)
    double visibility = static_cast<double>(points_2d.size()) / total_vertices;
    if (visibility > 0.8) {
        result.score += 40;
    } else if (visibility > 0.5) {
        result.score += 30;
    } else if (visibility > 0.3) {
        result.score += 20;
    } else {
        result.score += 10;
    }
    
    // 5.2 score for coverage (0-30)
    double bbox_area = result.bbox.width * result.bbox.height;
    double img_area = image_size.width * image_size.height;
    result.coverage = bbox_area / img_area;
    
    if (result.coverage > 0.1 && result.coverage < 0.4) {
        result.score += 30;  // 理想范围
    } else if (result.coverage > 0.05 && result.coverage < 0.6) {
        result.score += 20;  // 可接受范围
    } else if (result.coverage > 0.01) {
        result.score += 10;  // 最低可见
    }
    
    // 5.3 score for centrality (0-15)
    double cx_bbox = result.bbox.x + result.bbox.width / 2.0;
    double cy_bbox = result.bbox.y + result.bbox.height / 2.0;
    double cx_img = image_size.width / 2.0;
    double cy_img = image_size.height / 2.0;
    
    double dist_to_center = std::sqrt(
        std::pow(cx_bbox - cx_img, 2) + 
        std::pow(cy_bbox - cy_img, 2)
    );
    double max_dist = std::sqrt(cx_img*cx_img + cy_img*cy_img);
    double centrality = 1.0 - (dist_to_center / max_dist);
    result.score += centrality * 15.0;
    
    // 5.4 score for margin (0-15)
    const int margin = 20;  // 20像素边距
    if (result.bbox.x > margin &&
        result.bbox.y > margin &&
        result.bbox.x + result.bbox.width < image_size.width - margin &&
        result.bbox.y + result.bbox.height < image_size.height - margin) {
        result.score += 15;
    }
    
    return result;
}

// ============ Best image selection refactored ============

// New helper function to score candidate images
std::vector<PhysicalInferenceNode::ScoredImage>
PhysicalInferenceNode::scoreCandidateImages(
    const std::vector<ImageCache::CachedImage>& images,
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {

    std::vector<ScoredImage> scored_images;
    int img_index = 0;

    // Setup debug directory if needed
    std::string debug_dir;
    if (debug_save_images_) {
        std::string pkg_path = ros::package::getPath("phy_graph");
        debug_dir = pkg_path + "/tmp/debug_" + attrs.name;
        boost::filesystem::create_directories(debug_dir);
        ROS_INFO("Debug mode: will save all candidate images to %s", debug_dir.c_str());
    }

    for (const auto& cached : images) {
        // Project and score the object in the current image
        auto result = projectObjectToImage(
            attrs, mesh, cached.world_T_camera, cached.rgb_image.size());

        // Save debug images if enabled and the object is visible
        if (debug_save_images_ && result.visible_count > 0) {
            cv::Mat bbox_img = cached.rgb_image.clone();
            cv::rectangle(bbox_img, result.bbox, cv::Scalar(0, 255, 0), 3);
            
            std::stringstream label_ss;
            label_ss << "Score: " << std::fixed << std::setprecision(1) << result.score
                     << " Vis: " << result.visible_count
                     << " Cov: " << std::setprecision(1) << (result.coverage * 100) << "%";
            cv::putText(bbox_img, label_ss.str(), 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                       0.8, cv::Scalar(0, 255, 0), 2);
            
            std::stringstream bbox_ss;
            bbox_ss << debug_dir << "/bbox_" << std::setfill('0') << std::setw(3) << img_index
                   << "_t" << std::fixed << std::setprecision(3) << cached.timestamp.toSec()
                   << "_score" << std::setprecision(1) << result.score << ".jpg";
            cv::imwrite(bbox_ss.str(), bbox_img);
            
            img_index++;
        }

        // Keep only high-quality images (score >= 60)
        if (result.score >= 60.0) {
            ScoredImage scored;
            scored.original_image = cached.rgb_image.clone();
            scored.bbox = result.bbox;
            scored.score = result.score;
            scored.timestamp = cached.timestamp;
            scored_images.push_back(scored);
            
            ROS_INFO("yes Image t=%.3f: score=%.1f, visible=%d, coverage=%.1f%%, bbox=%dx%d",
                     cached.timestamp.toSec(), result.score, result.visible_count,
                     result.coverage * 100, result.bbox.width, result.bbox.height);
        } else if (result.visible_count > 0) {
            ROS_INFO("no Image t=%.3f: score=%.1f (too low), visible=%d, coverage=%.1f%%",
                     cached.timestamp.toSec(), result.score, result.visible_count,
                     result.coverage * 100);
        }
    }

    if (debug_save_images_ && img_index > 0) {
        ROS_INFO("Saved %d debug images to: %s", img_index, debug_dir.c_str());
    }

    return scored_images;
}

// New helper function to expand and clamp a bounding box
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

// New helper function to save the final image for inference
std::string PhysicalInferenceNode::saveImageForInference(const cv::Mat& image,
                                                         const std::string& object_name) {
    std::string pkg_path = ros::package::getPath("phy_graph");
    std::string temp_dir = pkg_path + "/tmp";
    boost::filesystem::create_directories(temp_dir);

    std::stringstream ss;
    ss << temp_dir << "/object_" << object_name << "_" << ros::Time::now().toNSec() << ".jpg";
    std::string image_path = ss.str();

    if (!cv::imwrite(image_path, image)) {
        ROS_ERROR("Failed to save image to %s", image_path.c_str());
        return "";
    }

    ROS_INFO("Saved final object image for inference to: %s", image_path.c_str());
    return image_path;
}

// Refactored main function
std::string PhysicalInferenceNode::extractBestObjectImage(
    const hydra::ObjectNodeAttributes& attrs,
    const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {
    
    // 1. Check if image cache is available
    if (image_cache_->size() == 0) {
        ROS_WARN("Image cache is empty for object %s", attrs.name.c_str());
        return "";
    }
    
    ROS_INFO("Extracting best image for object '%s' from %zu cached images",
             attrs.name.c_str(), image_cache_->size());
    
    // 2. Get candidate images from a time window around the object's creation
    ros::Time object_creation_time;
    object_creation_time.fromNSec(attrs.last_update_time_ns);
    ros::Duration window(5.0);  // 5 seconds before and after

    auto candidate_images = image_cache_->getImagesInRange(
        object_creation_time - window,
        object_creation_time + window
    );
    
    if (candidate_images.empty()) {
        ROS_WARN("No cached images found in the ±5s window for object %s", attrs.name.c_str());
        return "";
    }
    
    ROS_INFO("Found %zu candidate images in time range [%.3f, %.3f]",
             candidate_images.size(),
             candidate_images.front().timestamp.toSec(),
             candidate_images.back().timestamp.toSec());
    
    // 3. Score all candidate images
    auto scored_images = scoreCandidateImages(candidate_images, attrs, mesh);
    
    if (scored_images.empty()) {
        ROS_WARN("No high-quality images found (all scores < 60) for %s",
                 attrs.name.c_str());
        return "";
    }
    
    // 4. Sort by score to find the best one
    std::sort(scored_images.begin(), scored_images.end(),
        [](const auto& a, const auto& b) {
            return a.score > b.score;
        });
    
    const auto& best = scored_images[0];
    
    ROS_INFO("Selected best image for %s: score=%.1f, bbox=%dx%d @ t=%.3f",
             attrs.name.c_str(),
             best.score,
             best.bbox.width,
             best.bbox.height,
             best.timestamp.toSec());
    
    // 5. Expand the bounding box and crop the final image for more context
    cv::Rect expanded_bbox = expandAndClampBbox(best.bbox, best.original_image.size());
    cv::Mat final_image = best.original_image(expanded_bbox);

    // 6. Save the final image and return its path
    return saveImageForInference(final_image, attrs.name);
}
