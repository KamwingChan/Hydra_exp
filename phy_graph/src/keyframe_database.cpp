#include "phy_graph/keyframe_database.h"
#include <ros/console.h>
#include <filesystem>
#include <iomanip>

namespace phy_graph {

// Helper: Decode RGB implementation
cv::Mat KeyframeDatabase::Keyframe::decode() const {
    if (!is_on_disk) {
        if (memory_buffer.empty()) return cv::Mat();
        return cv::imdecode(memory_buffer, cv::IMREAD_COLOR);
    } else {
        if (disk_path.empty()) return cv::Mat();
        return cv::imread(disk_path); // Read from disk
    }
}

// Helper: Decode Depth implementation
cv::Mat KeyframeDatabase::Keyframe::decodeDepth() const {
    if (!has_depth) return cv::Mat();
    
    if (!depth_is_on_disk) {
        if (depth_memory_buffer.empty()) return cv::Mat();
        return cv::imdecode(depth_memory_buffer, cv::IMREAD_UNCHANGED);
    } else {
        if (depth_disk_path.empty()) return cv::Mat();
        return cv::imread(depth_disk_path, cv::IMREAD_UNCHANGED); // Read 16-bit PNG
    }
}

KeyframeDatabase::KeyframeDatabase(const std::string& storage_dir,
                                   size_t max_memory_frames,
                                   double min_translation, 
                                   double min_rotation,
                                   double min_time_interval)
    : storage_dir_(storage_dir),
      max_memory_frames_(max_memory_frames),
      min_translation_(min_translation), 
      min_rotation_(min_rotation), 
      min_time_interval_(min_time_interval),
      has_keyframes_(false) {
    
    last_keyframe_pose_ = Eigen::Isometry3d::Identity();
    last_keyframe_time_ = ros::Time(0);
    
    // Ensure storage directory exists
    try {
        if (!storage_dir_.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(storage_dir_, ec);
        }
    } catch (std::exception& e) {
        ROS_ERROR("KeyframeDatabase: Failed to create dir %s: %s", storage_dir_.c_str(), e.what());
    }
}

void KeyframeDatabase::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    has_keyframes_ = false;
}

void KeyframeDatabase::addImage(const sensor_msgs::ImageConstPtr& rgb_msg,
                                const Eigen::Isometry3d& world_T_camera,
                                const sensor_msgs::ImageConstPtr& depth_msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 1. Keyframe Selection Criteria
    if (has_keyframes_) {
        // Time interval check
        if ((rgb_msg->header.stamp - last_keyframe_time_).toSec() < min_time_interval_) {
            return;
        }

        Eigen::Isometry3d delta = last_keyframe_pose_.inverse() * world_T_camera;
        double trans_dist = delta.translation().norm();
        Eigen::AngleAxisd rot(delta.rotation());
        double rot_angle = std::abs(rot.angle());
        
        if (trans_dist < min_translation_ && rot_angle < min_rotation_) {
            return;
        }
    }
    
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
        
        // 2. Create Keyframe (Hot in memory)
        Keyframe kf;
        kf.timestamp = rgb_msg->header.stamp;
        kf.world_T_camera = world_T_camera;
        kf.is_on_disk = false;
        
        // Encode RGB as JPEG
        std::vector<int> jpg_params;
        jpg_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        jpg_params.push_back(85);
        cv::imencode(".jpg", cv_ptr->image, kf.memory_buffer, jpg_params);
        
        // 3. Process depth image if available
        kf.has_depth = false;
        kf.depth_is_on_disk = false;
        if (depth_msg) {
            try {
                cv::Mat depth_image;
                // Handle both 16UC1 and 32FC1 depth formats
                if (depth_msg->encoding == "16UC1") {
                    cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg);
                    depth_image = depth_ptr->image;
                } else if (depth_msg->encoding == "32FC1") {
                    cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg);
                    // Convert float (meters) to uint16 (mm) for storage
                    depth_ptr->image.convertTo(depth_image, CV_16UC1, 1000.0);
                } else {
                    ROS_WARN_THROTTLE(5.0, "Unsupported depth encoding: %s", depth_msg->encoding.c_str());
                }
                
                if (!depth_image.empty()) {
                    // Encode as 16-bit PNG (lossless)
                    std::vector<int> png_params;
                    png_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                    png_params.push_back(3); // Compression level 0-9
                    cv::imencode(".png", depth_image, kf.depth_memory_buffer, png_params);
                    kf.has_depth = true;
                }
            } catch (std::exception& e) {
                ROS_WARN_THROTTLE(5.0, "Failed to process depth image: %s", e.what());
            }
        }
        
        keyframes_.push_back(kf);
        last_keyframe_pose_ = world_T_camera;
        last_keyframe_time_ = rgb_msg->header.stamp;
        has_keyframes_ = true;
        
        // 4. Maintain Memory Limit (Offload cold data)
        maintainMemoryLimit();
        
    } catch (std::exception& e) {
        ROS_ERROR("KeyframeDatabase: Add image failed: %s", e.what());
    }
}

void KeyframeDatabase::maintainMemoryLimit() {
    // Count how many frames are currently in memory
    // Optimization: Since we offload FIFO, we only need to check the front of the deque
    // until we find one that is NOT on disk.
    
    // But to be safe and simple: Scan from oldest to newest.
    // If we have N frames total, and M are on disk.
    // If (N - M) > limit, we need to offload more.
    
    // Actually, a simpler logic:
    // We just need to ensure the number of "hot" frames doesn't exceed limit.
    // Since we append to back, the "hot" frames are at the back.
    // The "cold" candidates are at the front.
    
    // size_t hot_count = 0;
    // // Fast check: if total size is small, no need to do anything
    // if (keyframes_.size() <= max_memory_frames_) return;

    // // We need to offload the oldest frames that are still in memory
    // for (auto& kf : keyframes_) {
    //     if (!kf.is_on_disk) {
    //         // Check if we need to offload this one
    //         // We want to keep only the last `max_memory_frames_` hot.
    //         // So if current index is < (total - max), offload it.
    //         // However, iterating and counting is slow.
            
    //         // Better approach: Since we add 1 frame at a time, we just need to offload
    //         // the oldest *in-memory* frame if total > max.
            
    //         // Wait, total size includes disk frames.
    //         // We want to limit RAM usage.
    //         // Let's count hot frames from the back.
    //         // Actually, simplest LRU:
    //         // Just scan from begin(). If !is_on_disk, offload it.
    //         // Stop when we have offloaded enough.
            
    //         // To properly implement this with a deque:
    //         // We can track the index of the first "hot" frame.
    //         // But let's keep it robust:
            
    //         // Construct filename
    //         std::stringstream ss;
    //         ss << storage_dir_ << "/frame_" << std::fixed << std::setprecision(3) 
    //            << kf.timestamp.toSec() << ".jpg";
    //         kf.disk_path = ss.str();
            
    //         // Write to disk
    //         std::ofstream outfile(kf.disk_path, std::ios::out | std::ios::binary);
    //         if (outfile) {
    //             outfile.write(reinterpret_cast<const char*>(kf.memory_buffer.data()), 
    //                           kf.memory_buffer.size());
    //             outfile.close();
                
    //             // Free memory
    //             std::vector<uchar>().swap(kf.memory_buffer); // Force deallocation
    //             kf.is_on_disk = true;
                
    //             // ROS_DEBUG("Offloaded frame t=%.3f to disk", kf.timestamp.toSec());
    //         } else {
    //             ROS_WARN("Failed to offload frame to %s", kf.disk_path.c_str());
    //             // Keep in memory if disk write fails
    //         }
            
    //         // Since we process one addImage at a time, offloading one old frame is enough
    //         // to maintain the balance (roughly).
    //         // But to be correct: we should only offload if *hot_count* exceeds limit.
    //         // The current loop offloads EVERYTHING from the start until... when?
            
    //         // Let's refine:
    //         // We want to keep the LAST `max_memory_frames_` in memory.
    //         // So any frame with index < (size - max) should be on disk.
    //         size_t retention_start_index = 0;
    //         if (keyframes_.size() > max_memory_frames_) {
    //             retention_start_index = keyframes_.size() - max_memory_frames_;
    //         }
            
    //         // If current frame is within the "keep" zone, stop offloading
    //         // Since we are iterating from start (oldest), once we hit the keep zone,
    //         // all subsequent frames are newer and should be kept.
    //         // Wait, we need the index.
    //         break; // Logic needs index.
    //     }
    // }
    
    // Correct Implementation:
    if (keyframes_.size() <= max_memory_frames_) return;
    
    size_t num_to_offload = keyframes_.size() - max_memory_frames_;
    
    for (size_t i = 0; i < num_to_offload; ++i) {
        auto& kf = keyframes_[i];
        
        // Offload RGB if not already on disk
        if (!kf.is_on_disk) {
            std::stringstream ss;
            ss << storage_dir_ << "/frame_" << std::fixed << std::setprecision(3)
               << kf.timestamp.toSec() << ".jpg";
            kf.disk_path = ss.str();
            
            std::ofstream outfile(kf.disk_path, std::ios::out | std::ios::binary);
            if (outfile) {
                outfile.write(reinterpret_cast<const char*>(kf.memory_buffer.data()),
                              kf.memory_buffer.size());
                outfile.close();
                std::vector<uchar>().swap(kf.memory_buffer);
                kf.is_on_disk = true;
            }
        }
        
        // Offload depth if available and not already on disk
        if (kf.has_depth && !kf.depth_is_on_disk) {
            std::stringstream depth_ss;
            depth_ss << storage_dir_ << "/depth_" << std::fixed << std::setprecision(3)
                     << kf.timestamp.toSec() << ".png";
            kf.depth_disk_path = depth_ss.str();
            
            std::ofstream depth_outfile(kf.depth_disk_path, std::ios::out | std::ios::binary);
            if (depth_outfile) {
                depth_outfile.write(reinterpret_cast<const char*>(kf.depth_memory_buffer.data()),
                                    kf.depth_memory_buffer.size());
                depth_outfile.close();
                std::vector<uchar>().swap(kf.depth_memory_buffer);
                kf.depth_is_on_disk = true;
            }
        }
    }
}

std::vector<KeyframeDatabase::Keyframe> 
KeyframeDatabase::getKeyframesInRange(ros::Time start, ros::Time end) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Keyframe> result;
    
    for (const auto& kf : keyframes_) {
        if (kf.timestamp >= start && kf.timestamp <= end) {
            result.push_back(kf);
        }
    }
    return result;
}

} // namespace phy_graph
