#include <phy_graph/physical_inference_node.h>
#include <ros/package.h>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

// PCL for point cloud manipulation
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ============ PhysicalInferenceNode - 节点订阅管理 ============
PhysicalInferenceNode::PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) 
    : nh_(nh), pnh_(pnh), camera_info_received_(false), object_counter_(0), debug_save_images_(false) {
    // Load parameters
    pnh_.param<std::string>("label_space", label_space_, "");
    pnh_.param<bool>("debug_save_images", debug_save_images_, false);  // 默认关闭调试图像保存
    
    if (label_space_ == "ade20k" || label_space_ == "uhuman2") {
        loadLabelWhitelist();
    }

    // Initialize DSG receiver
    ros::NodeHandle backend_nh("/hydra_ros_node/backend");
    dsg_receiver_ = std::make_unique<hydra::DsgReceiver>(backend_nh);

    // Subscriber for the mesh, using a relative topic name
    mesh_sub_ = nh_.subscribe("input_mesh", 1, &PhysicalInferenceNode::meshCallback, this);

    // Service client to call the Python node
    service_client_ = nh_.serviceClient<phy_graph::GetProperties>("get_physical_properties");
    
    // RGB_cache - 增加到300帧以应对use_sim_time场景
    image_cache_ = std::make_shared<ImageCache>(300);  // 缓存300帧（约100秒@3Hz）
    
    // RF TF listener
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    // ros subscriber for RGB images
    rgb_sub_ = nh_.subscribe("rgb_image", 10, 
                            &PhysicalInferenceNode::rgbCallback, this);
    
    // ros subscriber for camera info
    camera_info_sub_ = nh_.subscribe("camera_info", 1,
                                     &PhysicalInferenceNode::cameraInfoCallback, this);

    // set output directory
    setupOutputDirectory();
    
    ROS_INFO("Physical Inference Node initialized with label space: '%s'", label_space_.c_str());
    ROS_INFO("Output directory: %s", output_dir_.c_str());
    ROS_INFO("Waiting for camera info...");
}

PhysicalInferenceNode::~PhysicalInferenceNode() {}

void PhysicalInferenceNode::run() {
    ros::Rate loop_rate(0.5); // 0.5 Hz, check every 2 seconds
    while (ros::ok()) {
        // Callbacks are handled by the AsyncSpinner, so we just check for DSG updates here
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
    if (!graph || !graph->hasLayer(hydra::DsgLayers::OBJECTS)) {
        return;
    }

    kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr current_mesh;
    {
        std::lock_guard<std::mutex> lock(mesh_mutex_);
        if (!latest_mesh_) {
            ROS_WARN("DSG received, but no mesh has been cached yet. Skipping.");
            return;
        }
        current_mesh = latest_mesh_;
    }

    const auto& object_layer = graph->getLayer(hydra::DsgLayers::OBJECTS);
    ROS_INFO("Recieved DSG with %zu objects.", object_layer.numNodes());

    for (const auto& id_node_pair : object_layer.nodes()) {
        const auto& node = *id_node_pair.second;

        // Skip if this object has been processed before
        if (processed_object_ids_.count(node.id)) {
            continue;
        }

        const auto& attrs = node.attributes<hydra::ObjectNodeAttributes>();

        if ((label_space_ == "ade20k" || label_space_ == "uhuman2") && label_whitelist_.find(attrs.semantic_label) == label_whitelist_.end()) {
            continue;
        }
        
        callInferenceService(node, *current_mesh);
    }
}

void PhysicalInferenceNode::callInferenceService(const hydra::SceneGraphNode& object_node, const kimera_pgmo_msgs::KimeraPgmoMesh& mesh) {
    const auto& attrs = object_node.attributes<hydra::ObjectNodeAttributes>();
    
    ROS_INFO("Processing object %s (%s)...", 
             hydra::NodeSymbol(object_node.id).getLabel().c_str(),
             attrs.name.c_str());
    
    try {
        // record start time
        auto start_time = ros::Time::now();
        
        // extract best object image
        std::string image_path = extractBestObjectImage(attrs, mesh);
        
        if (image_path.empty()) {
            ROS_WARN("Failed to extract image for object %s, skipping...", attrs.name.c_str());
            return;
        }
        
        // convert image to cv::Mat
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            ROS_ERROR("Failed to read extracted image: %s", image_path.c_str());
            return;
        }
        
        // convert cv::Mat to sensor_msgs::Image
        cv_bridge::CvImage cv_image;
        cv_image.image = img;
        cv_image.encoding = "bgr8";
        sensor_msgs::Image img_msg = *cv_image.toImageMsg();
        
        // send service to python node
        phy_graph::GetProperties srv;
        srv.request.label = attrs.name;
        srv.request.image = img_msg;

        ROS_INFO("Calling VLM service for %s with image size %dx%d...", 
                 attrs.name.c_str(), img.cols, img.rows);
        
        bool service_success = false;
        try {
            service_success = service_client_.call(srv);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception during service call for %s: %s", 
                     attrs.name.c_str(), e.what());
            return;
        }
        
        if (service_success) {
            // 检查响应是否包含有效数据（描述不为空表示成功）
            if (!srv.response.description.empty()) {
                // count processing time
                double processing_time_ms = (ros::Time::now() - start_time).toSec() * 1000.0;
                
                // save result to JSON
                saveInferenceResult(
                    object_node.id,
                    attrs.name,
                    srv.response.description,
                    srv.response.friction_level,
                    srv.response.pushable,
                    srv.response.weight_level,
                    processing_time_ms
                );
                
                // Label the object as processed
                processed_object_ids_.insert(object_node.id);
                
                ROS_INFO("✓ VLM Success for %s (%.0f ms)", 
                         hydra::NodeSymbol(object_node.id).getLabel().c_str(),
                         processing_time_ms);
                ROS_INFO("  Description: %s", srv.response.description.c_str());
                ROS_INFO("  Friction: %d", srv.response.friction_level);
                ROS_INFO("  Pushable: %s", srv.response.pushable ? "Yes" : "No");
                ROS_INFO("  Weight: %d", srv.response.weight_level);
            } else {
                ROS_ERROR("VLM service returned empty response for %s (possibly API error), skipping...",
                         attrs.name.c_str());
            }
        } else {
            ROS_ERROR("Failed to call VLM service for object %s, skipping...", 
                      hydra::NodeSymbol(object_node.id).getLabel().c_str());
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Unexpected exception while processing object %s: %s", 
                 attrs.name.c_str(), e.what());
        ROS_ERROR("Skipping this object and continuing...");
    } catch (...) {
        ROS_ERROR("Unknown exception while processing object %s", attrs.name.c_str());
        ROS_ERROR("Skipping this object and continuing...");
    }
}

void PhysicalInferenceNode::loadLabelWhitelist() {
    XmlRpc::XmlRpcValue label_list;
    if (!pnh_.getParam("object_labels", label_list)) {
        ROS_ERROR("Failed to get 'object_labels' from parameter server.");
        return;
    }

    if (label_list.getType() != XmlRpc::XmlRpcValue::TypeArray) {
        ROS_ERROR("'object_labels' is not an array.");
        return;
    }

    for (int i = 0; i < label_list.size(); ++i) {
        if (label_list[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
            label_whitelist_.insert(static_cast<int>(label_list[i]));
        }
    }
    ROS_INFO("Loaded %zu labels into the whitelist.", label_whitelist_.size());
}


// ============ output directory ============
void PhysicalInferenceNode::setupOutputDirectory() {
    try {
    // get package path
    std::string pkg_path = ros::package::getPath("phy_graph");
        
        // folder name with timestamp
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%m-%d_%H-%M");
        std::string timestamp = oss.str();
        
        output_dir_ = pkg_path + "/output/" + timestamp;
        
        // make directories
        boost::filesystem::create_directories(output_dir_);
        
        ROS_INFO("Output directory created: %s", output_dir_.c_str());
        
    } catch (std::exception& e) {
        ROS_ERROR("Failed to setup output directory: %s", e.what());
        output_dir_ = "/tmp/phy_graph_output";
        boost::filesystem::create_directories(output_dir_);
    }
}

void PhysicalInferenceNode::saveInferenceResult(
    const hydra::NodeId& node_id,
    const std::string& label,
    const std::string& description,
    int friction_level,
    bool pushable,
    int weight_level,
    double processing_time_ms) {
    
    try {
        // 增加计数器
        object_counter_++;
        
        // 获取节点ID字符串
        std::string node_id_str = hydra::NodeSymbol(node_id).getLabel();
        
        // 构建文件名
        std::ostringstream filename_ss;
        filename_ss << "object_" << node_id_str << "_" << label << ".json";
        std::string filepath = output_dir_ + "/" + filename_ss.str();
        
        // 构建JSON内容
        std::ostringstream json_ss;
        json_ss << "{\n";
        json_ss << "  \"object_id\": \"" << node_id_str << "\",\n";
        json_ss << "  \"label\": \"" << label << "\",\n";
        json_ss << "  \"description\": \"" << description << "\",\n";
        json_ss << "  \"friction_level\": " << friction_level << ",\n";
        json_ss << "  \"pushable\": " << (pushable ? "true" : "false") << ",\n";
        json_ss << "  \"weight_level\": " << weight_level << ",\n";
        json_ss << "  \"processing_time_ms\": " << static_cast<int>(processing_time_ms) << "\n";
        json_ss << "}\n";
        
        // 写入文件
        std::ofstream outfile(filepath);
        if (!outfile.is_open()) {
            ROS_ERROR("Failed to open file for writing: %s", filepath.c_str());
            return;
        }
        
        outfile << json_ss.str();
        outfile.close();
        
        ROS_INFO("Saved inference result to: %s", filepath.c_str());
        
    } catch (std::exception& e) {
        ROS_ERROR("Failed to save inference result: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "phy_graph_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Use an async spinner to handle callbacks in the background
    // This allows the main loop to run at a slow rate without affecting subscribers
    ros::AsyncSpinner spinner(2); // Use 2 threads
    spinner.start();

    PhysicalInferenceNode node(nh, pnh);
    node.run();

    ros::waitForShutdown();
    return 0;
}
