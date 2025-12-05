#pragma once

#include <vector>
#include <mutex>
#include <deque>
#include <string>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <ros/time.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <hydra/common/dsg_types.h>

namespace phy_graph {

/**
 * @brief A hybrid database that stores keyframes in memory (hot) and on disk (cold).
 */
class KeyframeDatabase {
public:
    struct Keyframe {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ros::Time timestamp;
        Eigen::Isometry3d world_T_camera;
        
        // Storage state
        bool is_on_disk;
        std::string disk_path;
        std::vector<uchar> memory_buffer; // Empty if is_on_disk is true
        
        Keyframe() : is_on_disk(false) {}

        // Helper to decode image on demand (from memory or disk)
        cv::Mat decode() const;
    };
    
    /**
     * @brief Construct a new Keyframe Database
     * 
     * @param storage_dir Directory to save offloaded keyframes (e.g. output/keyframes)
     * @param max_memory_frames Max number of frames to keep in RAM (e.g. 3000)
     * @param min_translation Minimum movement (meters) for new keyframe
     * @param min_rotation Minimum rotation (radians) for new keyframe
     */
    KeyframeDatabase(const std::string& storage_dir,
                     size_t max_memory_frames = 3000,
                     double min_translation = 0.2, 
                     double min_rotation = 0.1,
                     double min_time_interval = 0.2);
    
    /**
     * @brief Add an image to the database. Handles compression and offloading.
     */
    void addImage(const sensor_msgs::ImageConstPtr& msg,
                  const Eigen::Isometry3d& world_T_camera);
    
    /**
     * @brief Get all keyframes within a time range.
     */
    std::vector<Keyframe> getKeyframesInRange(ros::Time start, ros::Time end) const;

    size_t size() const { return keyframes_.size(); }
    void clear();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // Helper to move oldest frames to disk
    void maintainMemoryLimit();

    std::deque<Keyframe, Eigen::aligned_allocator<Keyframe>> keyframes_;
    mutable std::mutex mutex_;
    
    std::string storage_dir_;
    size_t max_memory_frames_;
    
    // Keyframe selection criteria
    double min_translation_;
    double min_rotation_;
    double min_time_interval_;
    
    Eigen::Isometry3d last_keyframe_pose_;
    ros::Time last_keyframe_time_;
    bool has_keyframes_;
};

} // namespace phy_graph
