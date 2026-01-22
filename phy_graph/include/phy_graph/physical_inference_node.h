#pragma once

#include <ros/ros.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <unordered_map>
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

// New Keyframe Database
#include "phy_graph/keyframe_database.h"

// Async Inference Queue
#include "phy_graph/inference_queue.h"

#include <ros/callback_queue.h> // 必须添加
#include <thread>

class PhysicalInferenceNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~PhysicalInferenceNode(); 

    void run();

private:
    // callback 
    void meshCallback(const kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr& msg);
    void processDsg(const hydra::DynamicSceneGraph::Ptr& graph);
    void callInferenceService(const hydra::SceneGraphNode& object_node, 
                             const kimera_pgmo_msgs::KimeraPgmoMesh& mesh);
    void loadLabelFilters();
    bool shouldProcessLabel(int label_id, const std::string& label_name);
    std::string normalizeName(const std::string& name);
    bool wildcardMatch(const std::string& pattern, const std::string& text);
    
    // New: RGB, Depth and CameraInfo callbacks
    void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    
    // New: Projection, scoring, and image extraction
    struct ProjectionResult {
        cv::Rect bbox;
        double score;
        int visible_count;
        double coverage;
        double occlusion_score;  // NEW: occlusion component score
    };

    struct ScoredImage {
        cv::Mat image;
        cv::Mat original_image;
        cv::Mat depth_image;  // NEW: associated depth image
        cv::Rect bbox;
        double score;
        ros::Time timestamp;
        bool has_depth;       // NEW: whether depth is available
    };
    
    // NOTE: mesh is optional pointer - pass nullptr when using hydra_bbox mode
    ProjectionResult projectObjectToImage(
        const hydra::ObjectNodeAttributes& attrs,
        const kimera_pgmo_msgs::KimeraPgmoMesh* mesh,  // Optional: nullptr for hydra_bbox mode
        const Eigen::Isometry3d& world_T_camera,
        const cv::Size& image_size,
        const cv::Mat& depth_image = cv::Mat());
    
    // Calculate occlusion score based on depth comparison
    double calculateOcclusionScore(
        const hydra::ObjectNodeAttributes& attrs,
        const cv::Mat& depth_image,
        const Eigen::Isometry3d& world_T_camera,
        const cv::Size& image_size);
    
    // Return cv::Mat directly and score; mesh is optional for hydra_bbox mode
    std::pair<cv::Mat, double> extractBestObjectImage(
        const hydra::ObjectNodeAttributes& attrs,
        const kimera_pgmo_msgs::KimeraPgmoMesh* mesh = nullptr);

    std::vector<ScoredImage> scoreCandidateImages(
        const std::vector<phy_graph::KeyframeDatabase::Keyframe>& images,
        const hydra::ObjectNodeAttributes& attrs,
        const kimera_pgmo_msgs::KimeraPgmoMesh* mesh = nullptr);

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
        const std::string& estimated_weight_kg,
        double image_score,
        double processing_time_ms);
    
    void setupOutputDirectory();
    
    // Async queue callbacks
    phy_graph::InferenceResult executeInference(const phy_graph::InferenceTask& task);
    void handleInferenceResult(const phy_graph::InferenceResult& result);
    
    // Enqueue object for async inference (mesh is optional for hydra_bbox mode)
    void enqueueForInference(const hydra::SceneGraphNode& object_node,
                             const kimera_pgmo_msgs::KimeraPgmoMesh* mesh = nullptr);

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
    std::vector<std::string> included_names_;
    std::unordered_set<int> excluded_ids_;
    std::vector<std::string> excluded_names_;
    
    std::unordered_set<uint64_t> processed_object_ids_;
    
    // image cache members
    ros::Subscriber rgb_sub_;
    ros::Subscriber depth_sub_;  // Depth subscriber (shares rgb_queue_)
    ros::Subscriber camera_info_sub_;
    
    // Depth cache (latest depth for pairing with RGB)
    sensor_msgs::ImageConstPtr latest_depth_;
    std::mutex depth_mutex_;
    bool occlusion_enabled_;  // Whether occlusion detection is active
    
    // NEW: Use KeyframeDatabase instead of ImageCache
    std::shared_ptr<phy_graph::KeyframeDatabase> keyframe_db_;
    
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

    // 延迟/抑制状态
    struct DeferState {
        int count = 0;
        bool suppressed = false;
        uint64_t last_update_ns = 0;
        ros::Time last_try_time = ros::Time(0);
    };
    std::unordered_map<uint64_t, DeferState> defer_state_;
    
    // 物体成熟度跟踪
    std::unordered_map<uint64_t, ros::Time> object_first_seen_;

    // 专用回调队列
    ros::CallbackQueue rgb_queue_;
    // 专用 Spinner (使用 unique_ptr 延迟初始化)
    std::unique_ptr<ros::AsyncSpinner> rgb_spinner_;
    
    // Async inference queue
    std::unique_ptr<phy_graph::InferenceQueue> inference_queue_;
    
    // Mutex for result handling (called from worker thread)
    std::mutex result_mutex_;
};
