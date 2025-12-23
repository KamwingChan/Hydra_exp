#include <phy_graph/physical_inference_node.h>
#include <ros/package.h>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdio>
#include <stdexcept>
#include <filesystem>
#include <nlohmann/json.hpp>

// PCL for point cloud manipulation
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ============ PhysicalInferenceNode - 节点订阅管理 ============
PhysicalInferenceNode::PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) 
    : nh_(nh), pnh_(pnh), camera_info_received_(false), object_counter_(0), debug_save_images_(false) {
    
    pnh_.param<std::string>("label_space", label_space_, "");
    pnh_.param<bool>("debug_save_images", debug_save_images_, false);
    
    if (label_space_ == "ade20k" || label_space_ == "uhuman2") {
        loadLabelWhitelist();
    }

    ros::NodeHandle backend_nh("/hydra_ros_node/backend");
    dsg_receiver_ = std::make_unique<hydra::DsgReceiver>(backend_nh);

    mesh_sub_ = nh_.subscribe("input_mesh", 1, &PhysicalInferenceNode::meshCallback, this);
    service_client_ = nh_.serviceClient<phy_graph::GetProperties>("get_physical_properties");
    
    setupOutputDirectory();
    
    // Keyframe Database: Store keyframes in output_dir/keyframes
    // Max 3000 in RAM, others on disk.
    keyframe_db_ = std::make_shared<phy_graph::KeyframeDatabase>(
        output_dir_ + "/keyframes", 
        3000, 0.2, 0.1
    );
    
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // === 关键修改：使用 subscribe options 绑定专用队列 ===
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<sensor_msgs::Image>(
        "rgb_image",                                      // topic name
        10,                                               // queue size
        boost::bind(&PhysicalInferenceNode::rgbCallback, this, _1), // callback
        ros::VoidPtr(),                                   // tracked object
        &rgb_queue_                                       // <--- 绑定到专用队列
    );
    // 允许 TCP 无延迟 (可选，但推荐)
    ops.transport_hints = ros::TransportHints().tcpNoDelay();
    
    rgb_sub_ = nh_.subscribe(ops);

    // 启动专用线程处理 RGB 队列
    // 这个线程只会处理 rgbCallback，永远不会被 processDsg 阻塞
    rgb_spinner_ = std::make_unique<ros::AsyncSpinner>(1, &rgb_queue_);
    rgb_spinner_->start();

    camera_info_sub_ = nh_.subscribe("camera_info", 1, &PhysicalInferenceNode::cameraInfoCallback, this);

    ROS_INFO("Physical Inference Node initialized with label space: '%s'", label_space_.c_str());
    ROS_INFO("Output directory: %s", output_dir_.c_str());
}

PhysicalInferenceNode::~PhysicalInferenceNode() {}

void PhysicalInferenceNode::run() {
    ros::Rate loop_rate(0.5); 
    while (ros::ok()) {
        // 多线程模式下不需要手动 spin
        ros::spinOnce(); 
        
        if (dsg_receiver_->updated()) {
            processDsg(dsg_receiver_->graph());
            dsg_receiver_->clearUpdated();
        }
        loop_rate.sleep();
    }
}

void PhysicalInferenceNode::meshCallback(const kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mesh_mutex_);
    latest_mesh_ = msg;
}

void PhysicalInferenceNode::processDsg(const hydra::DynamicSceneGraph::Ptr& graph) {
    if (!graph || !graph->hasLayer(hydra::DsgLayers::OBJECTS)) return;

    kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr current_mesh;
    {
        std::lock_guard<std::mutex> lock(mesh_mutex_);
        if (!latest_mesh_) return;
        current_mesh = latest_mesh_;
    }

    const auto& object_layer = graph->getLayer(hydra::DsgLayers::OBJECTS);
    ROS_INFO("Recieved DSG with %zu objects.", object_layer.numNodes());

    for (const auto& id_node_pair : object_layer.nodes()) {
        const auto& node = *id_node_pair.second;
        if (processed_object_ids_.count(node.id)) continue;

        const auto& attrs = node.attributes<hydra::ObjectNodeAttributes>();
        if (label_whitelist_.find(attrs.semantic_label) == label_whitelist_.end()) continue;
        
        callInferenceService(node, *current_mesh);
    }
}

void PhysicalInferenceNode::callInferenceService(const hydra::SceneGraphNode& object_node, const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {
    const auto& attrs = object_node.attributes<hydra::ObjectNodeAttributes>();
    
    ROS_INFO("Processing object %s (%s)...", 
             hydra::NodeSymbol(object_node.id).getLabel().c_str(),
             attrs.name.c_str());
    
    try {
        auto start_time = ros::Time::now();
        
        // Optimize: Direct cv::Mat return, no tmp file read/write
        auto result_pair = extractBestObjectImage(attrs, mesh);
        cv::Mat img = result_pair.first;
        double score = result_pair.second;
        
        if (img.empty()) {
            ROS_WARN_THROTTLE(5.0, "No valid image for object %s yet, skipping...", attrs.name.c_str());
            return;
        }

        // === Deferred Inference Logic ===
        // 1. High Quality (Score > 80): Immediate inference
        // 2. Medium Quality (40 < Score < 80) & New Object (< 10s): Wait for better view
        // 3. Low Quality / Old Object: Force inference (Best Effort)
        // 4. 延迟上限与抑制：超过次数后暂时跳过，直到有新的观测时间戳
        
        // 延迟/抑制状态
        auto& defer = defer_state_[object_node.id];
        const int MAX_DEFER = 5;
        const double REOBSERVE_RESET_SEC = 1.0;

        if (defer.suppressed) {
            if (attrs.last_update_time_ns > defer.last_update_ns &&
                (attrs.last_update_time_ns - defer.last_update_ns) * 1e-9 > REOBSERVE_RESET_SEC) {
                defer.suppressed = false;
                defer.count = 0;
            } else {
                ROS_DEBUG("Suppressed inference for %s, waiting for new observation.", attrs.name.c_str());
                return;
            }
        }
        
        // Use last_update_time_ns as creation time proxy for now
        ros::Time creation_time;
        creation_time.fromNSec(attrs.last_update_time_ns);
        double age_seconds = std::max(0.0, (ros::Time::now() - creation_time).toSec());
        
        const double HIGH_QUALITY_THRESHOLD = 70.0;
        const double WAIT_TIMEOUT_SECONDS = 5.0;

        if (score < HIGH_QUALITY_THRESHOLD) {
            if (age_seconds < WAIT_TIMEOUT_SECONDS) {
                defer.count++;
                defer.last_update_ns = attrs.last_update_time_ns;
                defer.last_try_time = ros::Time::now();
                if (defer.count >= MAX_DEFER) {
                    defer.suppressed = true;
                    ROS_INFO("Deferring inference for %s reached limit (%d). Suppressing until new observation.",
                             attrs.name.c_str(), defer.count);
                } else {
                    ROS_INFO("Deferring inference for %s (Score: %.1f, Age: %.1fs, Count: %d). Waiting for better view...", 
                             attrs.name.c_str(), score, age_seconds, defer.count);
                }
                return; // SKIP without marking as processed
            } else {
                ROS_WARN("Timeout reached for %s (Score: %.1f, Age: %.1fs). Forcing inference with suboptimal view.", 
                         attrs.name.c_str(), score, age_seconds);
            }
        }
        
        cv_bridge::CvImage cv_image;
        cv_image.image = img;
        cv_image.encoding = "bgr8";
        sensor_msgs::Image img_msg = *cv_image.toImageMsg();
        
        phy_graph::GetProperties srv;
        srv.request.label = attrs.name;
        srv.request.image = img_msg;

        ROS_INFO("Calling VLM service for %s...", attrs.name.c_str());
        
        if (service_client_.call(srv) && !srv.response.description.empty()) {
            double processing_time_ms = (ros::Time::now() - start_time).toSec() * 1000.0;
            saveInferenceResult(
                object_node.id, attrs.name,
                srv.response.description,
                srv.response.friction_level,
                srv.response.pushable,
                srv.response.weight_level,
                processing_time_ms
            );
            processed_object_ids_.insert(object_node.id);
            defer_state_.erase(object_node.id); // 清理延迟状态
            ROS_INFO("✓ VLM Success for %s (%.0f ms)", attrs.name.c_str(), processing_time_ms);
        } else {
            ROS_ERROR("VLM service failed or returned empty response for %s", attrs.name.c_str());
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception processing %s: %s", attrs.name.c_str(), e.what());
    }
}

void PhysicalInferenceNode::loadLabelWhitelist() {
    XmlRpc::XmlRpcValue label_list;
    if (pnh_.getParam("object_labels", label_list) && 
        label_list.getType() == XmlRpc::XmlRpcValue::TypeArray) {
        for (int i = 0; i < label_list.size(); ++i) {
            if (label_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt)
                label_whitelist_.insert(static_cast<int>(label_list[i]));
        }
    }
}

void PhysicalInferenceNode::setupOutputDirectory() {
    try {
        std::string pkg_path = ros::package::getPath("phy_graph");
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        // Include year and seconds to avoid collisions across years/runs and make sorting stable
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        output_dir_ = pkg_path + "/output/" + oss.str();
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
        if (ec) {
            throw std::runtime_error("Failed to create output directory: " + output_dir_);
        }
        ROS_INFO("Output directory created: %s", output_dir_.c_str());
    } catch (std::exception& e) {
        output_dir_ = "/tmp/phy_graph_output";
        std::error_code ec;
        std::filesystem::create_directories(output_dir_, ec);
    }
}

namespace {
inline void atomicWriteTextFile(const std::string& final_path, const std::string& contents) {
    const std::string tmp_path = final_path + ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open tmp file: " + tmp_path);
        }
        out << contents;
        out.flush();
        if (!out.good()) {
            throw std::runtime_error("Failed to write tmp file: " + tmp_path);
        }
    }

    // std::filesystem::rename does not overwrite, so remove first (best-effort)
    std::error_code ec;
    std::filesystem::remove(final_path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        // Fallback to std::rename (best effort)
        std::remove(final_path.c_str());
        std::rename(tmp_path.c_str(), final_path.c_str());
    }
}
}  // namespace

void PhysicalInferenceNode::saveInferenceResult(
    const hydra::NodeId& node_id, const std::string& label,
    const std::string& description, int friction_level,
    bool pushable, int weight_level, double processing_time_ms) {
    
    try {
        object_counter_++;
        std::string node_id_str = hydra::NodeSymbol(node_id).getLabel();
        std::ostringstream filename_ss;
        filename_ss << "object_" << node_id_str << "_" << label << ".json";
        std::string filepath = output_dir_ + "/" + filename_ss.str();

        nlohmann::json j;
        j["object_id"] = node_id_str;
        j["label"] = label;
        j["description"] = description;
        j["friction_level"] = friction_level;
        j["pushable"] = pushable;
        j["weight_level"] = weight_level;
        j["processing_time_ms"] = static_cast<int>(processing_time_ms);
        atomicWriteTextFile(filepath, j.dump(2) + "\n");
    } catch (...) {}
}

void PhysicalInferenceNode::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    if (camera_info_received_) return;
    fx_ = msg->K[0]; fy_ = msg->K[4]; cx_ = msg->K[2]; cy_ = msg->K[5];
    camera_frame_ = msg->header.frame_id;
    camera_info_received_ = true;
    camera_info_sub_.shutdown();
}

void PhysicalInferenceNode::rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (!camera_info_received_) return;
    static int frame_counter = 0;
    if (++frame_counter % 10 != 0) return;
    
    try {
        geometry_msgs::TransformStamped transform = tf_buffer_->lookupTransform(
            "world", camera_frame_, msg->header.stamp, ros::Duration(0.5));
        Eigen::Isometry3d world_T_camera = tf2::transformToEigen(transform);
        keyframe_db_->addImage(msg, world_T_camera);
    } catch (...) {}
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "phy_graph_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    
    // 不需要全局 AsyncSpinner 了，因为 rgb 已经有了自己的 Spinner
    
    PhysicalInferenceNode node(nh, pnh);
    node.run();
    
    return 0;
}
