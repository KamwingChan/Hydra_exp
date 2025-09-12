#include <physical_inference/physical_inference_node.h>

#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

// PCL for point cloud manipulation
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

PhysicalInferenceNode::PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh) {
    // Load parameters
    pnh_.param<std::string>("label_space", label_space_, "");
    if (label_space_ == "ade20k" || label_space_ == "uhuman2") {
        loadLabelWhitelist();
    }

    // Initialize DSG receiver
    ros::NodeHandle backend_nh("/hydra_ros_node/backend");
    dsg_receiver_ = std::make_unique<hydra::DsgReceiver>(backend_nh);

    // Subscriber for the mesh, using a relative topic name
    mesh_sub_ = nh_.subscribe("input_mesh", 1, &PhysicalInferenceNode::meshCallback, this);

    // Service client to call the Python node
    service_client_ = nh_.serviceClient<physical_inference::GetProperties>("get_physical_properties");

    ROS_INFO("Physical Inference Node initialized with label space: '%s'", label_space_.c_str());
}

PhysicalInferenceNode::~PhysicalInferenceNode() {}

void PhysicalInferenceNode::run() {
    ros::Rate loop_rate(2); // 2 Hz
    while (ros::ok()) {
        ros::spinOnce();
        if (dsg_receiver_->updated()) {
            processDsg(dsg_receiver_->graph());
            dsg_receiver_->clearUpdated();
        }
        else{ 
            ROS_DEBUG("waiting update.");
            loop_rate.sleep();
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
    
    // Create a set of vertex indices for quick lookup
    std::unordered_set<uint32_t> object_vertex_indices(attrs.mesh_connections.begin(), attrs.mesh_connections.end());
    
    // Build the object mesh
    kimera_pgmo_msgs::KimeraPgmoMesh object_mesh;
    object_mesh.header = mesh.header;
    object_mesh.ns = mesh.ns + "_object_" + std::to_string(object_node.id);
    
    // Create mapping from original vertex indices to new indices
    std::unordered_map<uint32_t, uint32_t> vertex_index_map;
    uint32_t new_vertex_index = 0;
    
    // Extract vertices and colors for this object
    const auto& all_vertices = mesh.vertices;
    const auto& all_colors = mesh.vertex_colors;
    
    for (const auto& original_index : attrs.mesh_connections) {
        if (original_index < all_vertices.size()) {
            // Add vertex
            object_mesh.vertices.push_back(all_vertices[original_index]);
            
            // Add color if available
            if (original_index < all_colors.size()) {
                object_mesh.vertex_colors.push_back(all_colors[original_index]);
            } else {
                std_msgs::ColorRGBA default_color;
                default_color.r = 1.0;
                default_color.g = 1.0;
                default_color.b = 1.0;
                default_color.a = 1.0;
                object_mesh.vertex_colors.push_back(default_color);
            }
            
            // Add vertex index mapping
            vertex_index_map[original_index] = new_vertex_index++;
        }
    }
    
    // Extract triangles that belong to this object
    for (const auto& triangle : mesh.triangles) {
        // Check if all three vertices of this triangle belong to the object
        bool triangle_belongs_to_object = true;
        for (int i = 0; i < 3; ++i) {
            if (object_vertex_indices.find(triangle.vertex_indices[i]) == object_vertex_indices.end()) {
                triangle_belongs_to_object = false;
                break;
            }
        }
        
        if (triangle_belongs_to_object) {
            // Create new triangle with remapped indices
            kimera_pgmo_msgs::TriangleIndices new_triangle;
            for (int i = 0; i < 3; ++i) {
                auto it = vertex_index_map.find(triangle.vertex_indices[i]);
                if (it != vertex_index_map.end()) {
                    new_triangle.vertex_indices[i] = it->second;
                } else {
                    // This shouldn't happen, but handle gracefully
                    ROS_WARN("Vertex index mapping not found for triangle vertex");
                    triangle_belongs_to_object = false;
                    break;
                }
            }
            
            if (triangle_belongs_to_object) {
                object_mesh.triangles.push_back(new_triangle);
            }
        }
    }
    
    if (object_mesh.vertices.empty()) {
        ROS_WARN("Object %s has no vertices in the mesh.", hydra::NodeSymbol(object_node.id).getLabel().c_str());
        return;
    }
    
    ROS_INFO("Extracted mesh for object %s: %zu vertices, %zu triangles", 
             attrs.name.c_str(), object_mesh.vertices.size(), object_mesh.triangles.size());

    physical_inference::GetProperties srv;
    srv.request.label = attrs.name;
    srv.request.object_mesh = object_mesh;

    ROS_INFO("Calling service for object %s...", attrs.name.c_str());
    if (service_client_.call(srv)) {
        processed_object_ids_.insert(object_node.id); // Mark as processed
        ROS_INFO("Success! Object: %s", hydra::NodeSymbol(object_node.id).getLabel().c_str());
        ROS_INFO("  Description: %s", srv.response.description.c_str());
        ROS_INFO("  Friction: %d", srv.response.friction_level);
        ROS_INFO("  Pushable: %s", srv.response.pushable ? "Yes" : "No");
    } else {
        ROS_ERROR("Failed to call service for object %s", hydra::NodeSymbol(object_node.id).getLabel().c_str());
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

int main(int argc, char** argv) {
    ros::init(argc, argv, "physical_inference_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    PhysicalInferenceNode node(nh, pnh);
    node.run();
    return 0;
}
