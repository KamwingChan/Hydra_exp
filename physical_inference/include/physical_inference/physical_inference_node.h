#pragma once

#include <ros/ros.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>

#include <kimera_pgmo_msgs/KimeraPgmoMesh.h>
#include <hydra/common/dsg_types.h> // Use full definition instead of forward declaration
#include <hydra_ros/utils/dsg_streaming_interface.h> // For DsgReceiver

// Our custom service
#include <physical_inference/GetProperties.h>

class PhysicalInferenceNode {
public:
    PhysicalInferenceNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~PhysicalInferenceNode(); 

    void run();

private:
    void meshCallback(const kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr& msg);
    void processDsg(const hydra::DynamicSceneGraph::Ptr& graph);
    void callInferenceService(const hydra::SceneGraphNode& object_node, const kimera_pgmo_msgs::KimeraPgmoMesh& mesh);
    void loadLabelWhitelist();

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber mesh_sub_;
    ros::ServiceClient service_client_;
    std::unique_ptr<hydra::DsgReceiver> dsg_receiver_;

    std::mutex mesh_mutex_;
    kimera_pgmo_msgs::KimeraPgmoMesh::ConstPtr latest_mesh_;

    std::string label_space_;
    std::unordered_set<int> label_whitelist_;
    std::unordered_set<uint64_t> processed_object_ids_;
};
