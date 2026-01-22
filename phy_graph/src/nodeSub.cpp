#include <phy_graph/physical_inference_node.h>
#include <phy_graph/inference_config.h>
#include <ros/package.h>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdio>
#include <stdexcept>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fnmatch.h>
#include <boost/algorithm/string.hpp>

// PCL for point cloud manipulation
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ============ PhysicalInferenceNode - 节点订阅管理 ============
PhysicalInferenceNode::PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), camera_info_received_(false), object_counter_(0), debug_save_images_(false), occlusion_enabled_(false) {
    
    // Load inference configuration from parameter server
    auto& cfg_mgr = phy_graph::InferenceConfigManager::get();
    cfg_mgr.loadFromROS(pnh_);
    const auto& cfg = cfg_mgr.config();
    cfg.print();  // Print loaded configuration
    
    pnh_.param<std::string>("label_space", label_space_, "");
    debug_save_images_ = cfg.debug.save_images;
    
    loadLabelFilters();

    ros::NodeHandle backend_nh("/hydra_ros_node/backend");
    dsg_receiver_ = std::make_unique<hydra::DsgReceiver>(backend_nh);

    // Only subscribe to mesh if using mesh_vertices projection mode
    if (cfg.image.projection_mode != "hydra_bbox") {
        mesh_sub_ = nh_.subscribe("input_mesh", 1, &PhysicalInferenceNode::meshCallback, this);
        ROS_INFO("Mesh subscription enabled (projection_mode: %s)", cfg.image.projection_mode.c_str());
    } else {
        ROS_INFO("Mesh subscription disabled (projection_mode: hydra_bbox)");
    }
    service_client_ = nh_.serviceClient<phy_graph::GetProperties>("get_physical_properties");
    
    setupOutputDirectory();
    
    // Keyframe Database: Use configuration parameters
    keyframe_db_ = std::make_shared<phy_graph::KeyframeDatabase>(
        output_dir_ + "/keyframes", 
        cfg.keyframe.max_memory_frames,
        cfg.keyframe.min_translation,
        cfg.keyframe.min_rotation,
        cfg.keyframe.min_time_interval
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

    // === 深度图订阅（如果遮挡检测启用）===
    if (cfg.occlusion.enable) {
        ros::SubscribeOptions depth_ops = ros::SubscribeOptions::create<sensor_msgs::Image>(
            "depth_image",                                      // topic name
            10,                                                 // queue size
            boost::bind(&PhysicalInferenceNode::depthCallback, this, _1),
            ros::VoidPtr(),
            &rgb_queue_                                         // 复用 RGB 队列
        );
        depth_ops.transport_hints = ros::TransportHints().tcpNoDelay();
        depth_sub_ = nh_.subscribe(depth_ops);
        occlusion_enabled_ = true;
        ROS_INFO("Occlusion detection enabled, depth subscribed to rgb_queue_");
    }

    // 启动专用线程处理 RGB/Depth 队列
    // 这个线程只会处理 rgbCallback 和 depthCallback，永远不会被 processDsg 阻塞
    rgb_spinner_ = std::make_unique<ros::AsyncSpinner>(1, &rgb_queue_);
    rgb_spinner_->start();

    camera_info_sub_ = nh_.subscribe("camera_info", 1, &PhysicalInferenceNode::cameraInfoCallback, this);

    // Initialize async inference queue
    inference_queue_ = std::make_unique<phy_graph::InferenceQueue>(
        cfg.inference.num_workers,
        cfg.inference.max_queue_size
    );
    
    // Start the queue with callbacks
    inference_queue_->start(
        std::bind(&PhysicalInferenceNode::executeInference, this, std::placeholders::_1),
        std::bind(&PhysicalInferenceNode::handleInferenceResult, this, std::placeholders::_1)
    );

    ROS_INFO("Physical Inference Node initialized with label space: '%s'", label_space_.c_str());
    ROS_INFO("Output directory: %s", output_dir_.c_str());
    ROS_INFO("Inference queue started with %d workers", cfg.inference.num_workers);
}

PhysicalInferenceNode::~PhysicalInferenceNode() {
    if (inference_queue_) {
        inference_queue_->stop();
    }
}

void PhysicalInferenceNode::run() {
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    ros::Rate loop_rate(cfg.inference.loop_rate); 
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

    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    
    // Get mesh pointer - may be null if using hydra_bbox mode
    kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr current_mesh;
    bool need_mesh = (cfg.image.projection_mode != "hydra_bbox");
    
    if (need_mesh) {
        std::lock_guard<std::mutex> lock(mesh_mutex_);
        if (!latest_mesh_) return;  // Wait for mesh when needed
        current_mesh = latest_mesh_;
    }

    const auto& object_layer = graph->getLayer(hydra::DsgLayers::OBJECTS);
    ROS_INFO("Received DSG with %zu objects. Queue size: %zu, Processing: %zu",
             object_layer.numNodes(),
             inference_queue_->size(),
             inference_queue_->processingCount());

    // Get mesh pointer for enqueueForInference (may be null)
    const kimera_pgmo_msgs::KimeraPgmoMesh* mesh_ptr = current_mesh ? current_mesh.get() : nullptr;

    for (const auto& id_node_pair : object_layer.nodes()) {
        const auto& node = *id_node_pair.second;
        if (processed_object_ids_.count(node.id)) continue;

        const auto& attrs = node.attributes<hydra::ObjectNodeAttributes>();
        if (!shouldProcessLabel(attrs.semantic_label, attrs.name)) continue;
        
        // === Object Maturity Check ===
        if (cfg.object_maturity.enable) {
            // Record first seen time
            if (object_first_seen_.find(node.id) == object_first_seen_.end()) {
                object_first_seen_[node.id] = ros::Time::now();
                // 第一次见到，静默跳过
                continue;
            }
            
            // Check if object is mature enough
            double age = (ros::Time::now() - object_first_seen_[node.id]).toSec();
            if (age < cfg.object_maturity.min_age_seconds) {
                // 太年轻，静默跳过（不输出日志避免刷屏）
                continue;
            }
            
            // 快速检查是否有候选图像
            ros::Time object_time;
            object_time.fromNSec(attrs.last_update_time_ns);
            ros::Duration window(cfg.keyframe.time_window);
            auto candidates = keyframe_db_->getKeyframesInRange(
                object_time - window,
                object_time + window
            );
            
            if (candidates.empty()) {
                // mature 但没有候选图像，静默跳过（避免刷屏）
                continue;
            }
            
            // 只有两者都满足才输出INFO
            ROS_INFO_THROTTLE(10.0, "Object %s mature (%.2fs) with %zu candidate images, proceeding...",
                     attrs.name.c_str(), age, candidates.size());
        }
        
        // Use async queue instead of blocking call (mesh_ptr may be null for hydra_bbox mode)
        enqueueForInference(node, mesh_ptr);
    }
}

void PhysicalInferenceNode::enqueueForInference(const hydra::SceneGraphNode& object_node,
                                                  const kimera_pgmo_msgs::KimeraPgmoMesh* mesh) {
    const auto& attrs = object_node.attributes<hydra::ObjectNodeAttributes>();
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    
    // Skip if already in queue
    if (inference_queue_->contains(object_node.id)) {
        return;
    }
    
    // Extract best image (mesh may be nullptr for hydra_bbox mode)
    auto result_pair = extractBestObjectImage(attrs, mesh);
    cv::Mat img = result_pair.first;
    double score = result_pair.second;
    
    if (img.empty()) {
        ROS_DEBUG_THROTTLE(5.0, "No valid image for object %s yet, skipping...", attrs.name.c_str());
        return;
    }
    
    // Check defer state
    auto& defer = defer_state_[object_node.id];
    const int MAX_DEFER = cfg.inference.max_defer_count;
    const double REOBSERVE_RESET_SEC = 1.0;

    if (defer.suppressed) {
        if (attrs.last_update_time_ns > defer.last_update_ns &&
            (attrs.last_update_time_ns - defer.last_update_ns) * 1e-9 > REOBSERVE_RESET_SEC) {
            defer.suppressed = false;
            defer.count = 0;
        } else {
            return;  // Still suppressed
        }
    }
    
    ros::Time creation_time;
    creation_time.fromNSec(attrs.last_update_time_ns);
    double age_seconds = std::max(0.0, (ros::Time::now() - creation_time).toSec());
    
    const double HIGH_QUALITY_THRESHOLD = cfg.image.high_quality_threshold;
    const double WAIT_TIMEOUT_SECONDS = cfg.inference.wait_timeout;
    const double MIN_ACCEPTABLE_SCORE = cfg.inference.min_acceptable_score;

    // Decide whether to enqueue
    if (score >= HIGH_QUALITY_THRESHOLD || (age_seconds >= WAIT_TIMEOUT_SECONDS && score > MIN_ACCEPTABLE_SCORE)) {
        // High quality or timeout (with minimum score) - enqueue immediately
        phy_graph::InferenceTask task;
        task.node_id = object_node.id;
        task.label = attrs.name;
        task.image = img.clone();
        task.image_score = score;
        task.last_update_ns = attrs.last_update_time_ns;
        task.defer_count = defer.count;
        
        if (inference_queue_->enqueue(std::move(task))) {
            ROS_INFO("Enqueued %s for inference (Score: %.1f, Age: %.1fs)", 
                     attrs.name.c_str(), score, age_seconds);
        }
    } else {
        // Defer
        defer.count++;
        defer.last_update_ns = attrs.last_update_time_ns;
        defer.last_try_time = ros::Time::now();
        
        if (defer.count >= MAX_DEFER) {
            defer.suppressed = true;
            ROS_INFO("Deferring for %s reached limit (%d). Suppressing.", attrs.name.c_str(), defer.count);
        } else {
            ROS_DEBUG("Deferring %s (Score: %.1f, Count: %d)", attrs.name.c_str(), score, defer.count);
        }
    }
}

phy_graph::InferenceResult PhysicalInferenceNode::executeInference(const phy_graph::InferenceTask& task) {
    phy_graph::InferenceResult result;
    result.node_id = task.node_id;
    result.label = task.label;
    result.image_score = task.image_score;
    result.success = false;
    
    auto start_time = ros::Time::now();
    
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    
    // Check for dry-run mode
    if (cfg.vlm.dry_run) {
        ROS_INFO("[DRY-RUN] Simulating inference for %s", task.label.c_str());
        result.description = "[DRY-RUN] Simulated " + task.label;
        result.friction_level = 1;
        result.pushable = true;
        result.weight_level = 1;
        result.estimated_weight_kg = "1-5";
        result.processing_time_ms = 100.0;
        result.success = true;
        return result;
    }
    
    try {
        cv_bridge::CvImage cv_image;
        cv_image.image = task.image;
        cv_image.encoding = "bgr8";
        sensor_msgs::Image img_msg = *cv_image.toImageMsg();
        
        phy_graph::GetProperties srv;
        srv.request.label = task.label;
        srv.request.image = img_msg;

        ROS_INFO("Calling VLM service for %s...", task.label.c_str());
        
        if (service_client_.call(srv) && !srv.response.description.empty()) {
            result.description = srv.response.description;
            result.friction_level = srv.response.friction_level;
            result.pushable = srv.response.pushable;
            result.weight_level = srv.response.weight_level;
            result.estimated_weight_kg = srv.response.estimated_weight_kg;
            result.processing_time_ms = (ros::Time::now() - start_time).toSec() * 1000.0;
            result.success = true;
            
            ROS_INFO("✓ VLM Success for %s (%.0f ms, weight: %s)", 
                     task.label.c_str(), result.processing_time_ms, result.estimated_weight_kg.c_str());
        } else {
            ROS_ERROR("VLM service failed for %s", task.label.c_str());
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in inference for %s: %s", task.label.c_str(), e.what());
    }
    
    return result;
}

void PhysicalInferenceNode::handleInferenceResult(const phy_graph::InferenceResult& result) {
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    if (result.success) {
        saveInferenceResult(
            result.node_id,
            result.label,
            result.description,
            result.friction_level,
            result.pushable,
            result.weight_level,
            result.estimated_weight_kg,
            result.image_score,
            result.processing_time_ms
        );
        processed_object_ids_.insert(result.node_id);
        defer_state_.erase(result.node_id);
    }
}

// Legacy function - kept for compatibility but no longer used directly
void PhysicalInferenceNode::callInferenceService(const hydra::SceneGraphNode& object_node,
                                                  const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {
    // Redirect to async queue (pass pointer to mesh)
    enqueueForInference(object_node, &mesh);
}

void PhysicalInferenceNode::loadLabelFilters() {
    auto load_to_sets = [&](const std::string& param_name, 
                           std::unordered_set<int>& id_set, 
                           std::vector<std::string>& name_patterns) {
        XmlRpc::XmlRpcValue list;
        if (pnh_.getParam(param_name, list) && list.getType() == XmlRpc::XmlRpcValue::TypeArray) {
            for (int i = 0; i < list.size(); ++i) {
                if (list[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
                    id_set.insert(static_cast<int>(list[i]));
                } else if (list[i].getType() == XmlRpc::XmlRpcValue::TypeString) {
                    name_patterns.push_back(normalizeName(static_cast<std::string>(list[i])));
                }
            }
        }
    };

    // 1. 加载包含列表 (白名单)
    // 兼容原有的 object_labels，同时支持新的 included_labels
    load_to_sets("object_labels", label_whitelist_, included_names_);
    load_to_sets("included_labels", label_whitelist_, included_names_);

    // 2. 加载排除列表 (黑名单)
    load_to_sets("excluded_labels", excluded_ids_, excluded_names_);

    if (!label_whitelist_.empty() || !included_names_.empty()) {
        ROS_INFO("Label filters loaded: %zu IDs and %zu name patterns included.", 
                 label_whitelist_.size(), included_names_.size());
    }
    if (!excluded_ids_.empty() || !excluded_names_.empty()) {
        ROS_INFO("Label filters loaded: %zu IDs and %zu name patterns excluded.", 
                 excluded_ids_.size(), excluded_names_.size());
    }
}

std::string PhysicalInferenceNode::normalizeName(const std::string& name) {
    std::string normalized = boost::algorithm::to_lower_copy(name);
    boost::algorithm::replace_all(normalized, "_", " ");
    boost::algorithm::trim(normalized);
    return normalized;
}

bool PhysicalInferenceNode::wildcardMatch(const std::string& pattern, const std::string& text) {
    // 使用 fnmatch 处理通配符，FNM_CASEFOLD 虽可忽略大小写，但我们已经手动转过小写了
    return fnmatch(pattern.c_str(), text.c_str(), 0) == 0;
}

bool PhysicalInferenceNode::shouldProcessLabel(int label_id, const std::string& label_name) {
    std::string normalized_name = normalizeName(label_name);

    // 1. 检查包含列表 (最高优先级)
    bool is_included = (label_whitelist_.count(label_id) > 0);
    if (!is_included) {
        for (const auto& pattern : included_names_) {
            if (wildcardMatch(pattern, normalized_name)) {
                is_included = true;
                break;
            }
        }
    }

    if (is_included) return true; // 白名单命中，直接准许

    // 2. 检查排除列表
    if (excluded_ids_.count(label_id) > 0) return false;
    for (const auto& pattern : excluded_names_) {
        if (wildcardMatch(pattern, normalized_name)) return false;
    }

    // 3. 默认判定
    // 如果设置了任何包含列表，则默认拒绝；否则默认准许
    bool has_include_list = !label_whitelist_.empty() || !included_names_.empty();
    return !has_include_list;
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
    bool pushable, int weight_level, 
    const std::string& estimated_weight_kg,
    double image_score, double processing_time_ms) {
    
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
        if (!estimated_weight_kg.empty()) {
            j["estimated_weight_kg"] = estimated_weight_kg;
        }
        j["inference_confidence"] = static_cast<int>(image_score);
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

void PhysicalInferenceNode::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    latest_depth_ = msg;
}

void PhysicalInferenceNode::rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (!camera_info_received_) return;
    
    // Use configurable frame skip (default 5 for 15Hz→3Hz)
    const auto& cfg = phy_graph::InferenceConfigManager::get().config();
    static int frame_counter = 0;
    if (++frame_counter % cfg.keyframe.frame_skip != 0) return;
    
    try {
        geometry_msgs::TransformStamped transform = tf_buffer_->lookupTransform(
            "world", camera_frame_, msg->header.stamp, ros::Duration(0.5));
        Eigen::Isometry3d world_T_camera = tf2::transformToEigen(transform);
        
        // Get depth image if occlusion detection is enabled
        sensor_msgs::ImageConstPtr depth_msg;
        if (occlusion_enabled_) {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            depth_msg = latest_depth_;
        }
        
        // Add to database with optional depth
        keyframe_db_->addImage(msg, world_T_camera, depth_msg);
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
