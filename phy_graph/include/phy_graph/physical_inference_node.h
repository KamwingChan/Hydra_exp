#pragma once

#include <ros/ros.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <deque>

#include <kimera_pgmo_msgs/KimeraPgmoMesh.h>
#include <hydra/common/dsg_types.h>
#include <hydra_ros/utils/dsg_streaming_interface.h>

// RGB image
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// TF includes
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <Eigen/Geometry>

// Our custom service
#include <phy_graph/GetProperties.h>

class ImageCache {
public:
    struct CachedImage {
        ros::Time timestamp;
        cv::Mat rgb_image;
        Eigen::Isometry3d world_T_camera;
    };
    
    ImageCache(size_t max_size = 100) : max_size_(max_size) {}
    
    void addImage(const sensor_msgs::ImageConstPtr& msg,
                  const Eigen::Isometry3d& world_T_camera);
    
    std::vector<CachedImage> getImagesInRange(
        ros::Time start, ros::Time end) const;
    
    size_t size() const { return cache_.size(); }
    void clear() { cache_.clear(); }
    
private:
    std::deque<CachedImage> cache_;
    const size_t max_size_;
    mutable std::mutex cache_mutex_;
};

class PhysicalInferenceNode {
public:
    PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~PhysicalInferenceNode(); 

    void run();

private:
    // callback 
    void meshCallback(const kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr& msg);
    void processDsg(const hydra::DynamicSceneGraph::Ptr& graph);
    void callInferenceService(const hydra::SceneGraphNode& object_node, 
                             const kimera_pgmo_msgs::KimeraPgmoMesh& mesh);
    void loadLabelWhitelist();
    
    // New: RGB and CameraInfo callbacks
    void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    
    // New: Projection, scoring, and image extraction
    struct ProjectionResult {
        cv::Rect bbox;
        double score;
        int visible_count;
        double coverage;
    };

    struct ScoredImage {
        cv::Mat image;
        cv::Mat original_image;
        cv::Rect bbox;
        double score;
        ros::Time timestamp;
    };
    
    ProjectionResult projectObjectToImage(
        const hydra::ObjectNodeAttributes& attrs,
        const kimera_pgmo_msgs::KimeraPgmoMesh& mesh,
        const Eigen::Isometry3d& world_T_camera,
        const cv::Size& image_size);
    
    std::string extractBestObjectImage(
        const hydra::ObjectNodeAttributes& attrs,
        const kimera_pgmo_msgs::KimeraPgmoMesh& mesh);

    std::vector<ScoredImage> scoreCandidateImages(
        const std::vector<ImageCache::CachedImage>& images,
        const hydra::ObjectNodeAttributes& attrs,
        const kimera_pgmo_msgs::KimeraPgmoMesh& mesh);

    cv::Rect expandAndClampBbox(const cv::Rect& bbox,
                                const cv::Size& image_size,
                                float padding_factor = 0.3f);

    std::string saveImageForInference(const cv::Mat& image,
                                      const std::string& object_name);
    
    // New: JSON saving
    void saveInferenceResult(
        const hydra::NodeId& node_id,
        const std::string& label,
        const std::string& description,
        int friction_level,
        bool pushable,
        int weight_level,
        double processing_time_ms);
    
    void setupOutputDirectory();

    // ros members
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber mesh_sub_;
    ros::ServiceClient service_client_;
    std::unique_ptr<hydra::DsgReceiver> dsg_receiver_;
    // mesh storage
    std::mutex mesh_mutex_;
    kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr latest_mesh_;

    std::string label_space_;
    std::unordered_set<int> label_whitelist_;
    std::unordered_set<uint64_t> processed_object_ids_;
    
    // image cache members
    ros::Subscriber rgb_sub_;
    ros::Subscriber camera_info_sub_;
    std::shared_ptr<ImageCache> image_cache_;
    
    // TF members
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // camera info members
    bool camera_info_received_;
    double fx_, fy_, cx_, cy_;
    std::string camera_frame_;
    
    // output members
    std::string output_dir_;
    int object_counter_;
    
    // debug members
    bool debug_save_images_;  
};
