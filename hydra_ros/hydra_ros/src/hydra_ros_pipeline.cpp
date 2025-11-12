/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra_ros/hydra_ros_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/active_window/reconstruction_module.h>
#include <hydra/backend/backend_module.h>
#include <hydra/backend/zmq_interfaces.h>
#include <hydra/common/dsg_types.h>
#include <hydra/common/global_info.h>
#include <hydra/frontend/graph_builder.h>
#include <hydra/loop_closure/loop_closure_module.h>
#include <hydra/utils/mesh_utilities.h>
#include <pose_graph_tools_ros/conversions.h>

#include <memory>

#include "hydra_ros/backend/ros_backend_publisher.h"
#include "hydra_ros/frontend/ros_frontend_publisher.h"
#include "hydra_ros/loop_closure/ros_lcd_registration.h"
#include "hydra_ros/utils/bow_subscriber.h"
#include "hydra_ros/utils/external_loop_closure_subscriber.h"

namespace hydra {

void declare_config(HydraRosPipeline::Config& config) {
  using namespace config;
  name("HydraRosConfig");
  field(config.active_window, "active_window");
  field(config.frontend, "frontend");
  field(config.backend, "backend");
  field(config.enable_frontend_output, "enable_frontend_output");
  field(config.enable_zmq_interface, "enable_zmq_interface");
  field(config.input, "input");
  config.features.setOptional();
  field(config.features, "features");
  // Continue mapping
}

HydraRosPipeline::HydraRosPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraPipeline(config::fromRos<PipelineConfig>(nh), robot_id),
      config(config::checkValid(config::fromRos<Config>(nh))),
      nh_(nh) {
  LOG(INFO) << "Starting Hydra-ROS with input configuration\n"
            << config::toString(config.input);
}

HydraRosPipeline::~HydraRosPipeline() {}

void HydraRosPipeline::init() {
  const auto& pipeline_config = GlobalInfo::instance().getConfig();
  const auto logs = GlobalInfo::instance().getLogs();

  // Note: loadMap is called in hydra_node.cpp before init() if continue_mapping is enabled

  backend_ = config.backend.create(backend_dsg_, shared_state_, logs);
  modules_["backend"] = CHECK_NOTNULL(backend_);

  frontend_ = config.frontend.create(frontend_dsg_, shared_state_, logs);
  modules_["frontend"] = CHECK_NOTNULL(frontend_);

  active_window_ = config.active_window.create(frontend_->queue());
  modules_["active_window"] = CHECK_NOTNULL(active_window_);

  if (pipeline_config.enable_lcd) {
    initLCD();
    bow_sub_.reset(new BowSubscriber(nh_));
  }

  external_loop_closure_sub_.reset(new ExternalLoopClosureSubscriber(nh_));

  ros::NodeHandle bnh(nh_, "backend");
  backend_->addSink(std::make_shared<RosBackendPublisher>(bnh));
  if (config.enable_zmq_interface) {
    const auto zmq_config = config::fromRos<ZmqSink::Config>(bnh, "zmq_sink");
    backend_->addSink(std::make_shared<ZmqSink>(zmq_config));
  }

  if (config.enable_frontend_output) {
    CHECK(frontend_) << "Frontend module required!";
    frontend_->addSink(
        std::make_shared<RosFrontendPublisher>(ros::NodeHandle(nh_, "frontend")));
  }

  input_module_ =
      std::make_shared<RosInputModule>(config.input, active_window_->queue());
  if (config.features) {
    modules_["features"] = config.features.create();  // has to come after input module
  }
}

void HydraRosPipeline::stop() {
  // enforce stop order to make sure every data packet is processed
  input_module_->stop();
  // TODO(nathan) push extracting active window objects to module stop
  active_window_->stop();
  frontend_->stop();
  backend_->stop();

  HydraPipeline::stop();
}

void HydraRosPipeline::initLCD() {
  auto lcd_config = config::fromRos<LoopClosureConfig>(nh_);
  lcd_config.detector.num_semantic_classes = GlobalInfo::instance().getTotalLabels();
  VLOG(1) << "Number of classes for LCD: " << lcd_config.detector.num_semantic_classes;
  config::checkValid(lcd_config);

  auto lcd = std::make_shared<LoopClosureModule>(lcd_config, shared_state_);
  modules_["lcd"] = lcd;

  if (lcd_config.detector.enable_agent_registration) {
    lcd->getDetector().setRegistrationSolver(0,
                                             std::make_unique<lcd::DsgAgentSolver>());
  }
}

}  // namespace hydra
#include <kimera_pgmo/utils/mesh_io.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl_conversions/pcl_conversions.h>

#include <filesystem>

namespace fs = std::filesystem;

bool hydra::HydraRosPipeline::loadMap(const std::string& map_load_path) {
  // check if the pose_graph_tracker is set to PoseGraphFromOdom
  auto frontend_config = config.frontend.getUnderlying<hydra::GraphBuilder::Config>();
  if (frontend_config && frontend_config->pose_graph_tracker.getType() != "PoseGraphFromOdom") {
    LOG(ERROR) << "Map loading is currently only supported when 'pose_graph_tracker' is "
                  "set to 'PoseGraphFromOdom' (i.e., 'use_gt_frame' is true).";
    LOG(ERROR)
        << "Visual relocalization has not been implemented yet. Aborting map load.";
    return false;
  }

  LOG(INFO) << "[Hydra Load] Starting map loading process from: " << map_load_path;

  // file existence check
  fs::path root_path(map_load_path);
  fs::path dsg_path = root_path / "backend" / "dsg.json";
  fs::path mesh_path = root_path / "backend" / "mesh.ply";

  if (!fs::exists(dsg_path)) {
    LOG(ERROR) << "[Hydra Load] Scene graph file not found at: " << dsg_path;
    return false;
  }
  if (!fs::exists(mesh_path)) {
    LOG(ERROR) << "[Hydra Load] Mesh file not found at: " << mesh_path;
    return false;
  }

  // DSG loading
  LOG(INFO) << "[Hydra Load] Loading scene graph...";
  backend_dsg_->graph = DynamicSceneGraph::load(dsg_path.string());
  if (!backend_dsg_->graph) {
    LOG(ERROR) << "[Hydra Load] Failed to parse scene graph file!";
    return false;
  }
  LOG(INFO) << "[Hydra Load] Scene graph loaded successfully.";

  // Load mesh via PCL and convert to spark_dsg::Mesh
  LOG(INFO) << "[Hydra Load] Loading 3D mesh from PLY file via PCL...";
  
  // Step 1: Load PLY into PCL mesh
  pcl::PolygonMesh::Ptr pcl_mesh(new pcl::PolygonMesh());
  kimera_pgmo::ReadMeshFromPly(mesh_path.string(), pcl_mesh);

  if (pcl_mesh->cloud.data.empty() || pcl_mesh->polygons.empty()) {
    LOG(ERROR) << "[Hydra Load] Failed to load mesh or mesh is empty!";
    return false;
  }

  LOG(INFO) << "[Hydra Load] PCL mesh loaded: " 
            << pcl_mesh->cloud.width << " points, " 
            << pcl_mesh->polygons.size() << " polygons";

  // Step 2: Convert PCL mesh to spark_dsg::Mesh
  auto dsg_mesh = std::make_shared<spark_dsg::Mesh>();

  // Convert vertices and colors
  pcl::PointCloud<pcl::PointXYZRGBA> vertices;
  pcl::fromPCLPointCloud2(pcl_mesh->cloud, vertices);

  dsg_mesh->points.resize(vertices.size());
  dsg_mesh->colors.resize(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    const auto& point = vertices.at(i);
    dsg_mesh->points[i] = spark_dsg::Mesh::Pos(point.x, point.y, point.z);
    dsg_mesh->colors[i] = spark_dsg::Color(point.r, point.g, point.b, point.a);
  }

  // Convert faces (properly handle non-triangular faces)
  std::vector<spark_dsg::Mesh::Face> valid_faces;
  valid_faces.reserve(pcl_mesh->polygons.size());

  for (const auto& polygon : pcl_mesh->polygons) {
    if (polygon.vertices.size() == 3) {
      valid_faces.push_back({polygon.vertices[0], 
                             polygon.vertices[1], 
                             polygon.vertices[2]});
    }
  }
  dsg_mesh->faces = std::move(valid_faces);

  LOG(INFO) << "[Hydra Load] Mesh conversion complete:";
  LOG(INFO) << "  - Vertices: " << dsg_mesh->points.size();
  LOG(INFO) << "  - Faces: " << dsg_mesh->faces.size();
  LOG(INFO) << "  - Has colors: " << (dsg_mesh->colors.empty() ? "No" : "Yes");

  // connect mesh to DSG
  backend_dsg_->graph->setMesh(dsg_mesh);

  // Pre-fill semantic labels from Object and Place nodes
  // This ensures that loaded mesh has semantic information for continue_mapping mode
  // All vertices not connected to any node will remain as 0 (unlabeled)
  const size_t mesh_size = dsg_mesh->points.size();
  
  // Initialize labels array (all vertices start as 0 = unlabeled)
  if (!dsg_mesh->has_labels) {
    dsg_mesh->labels.resize(mesh_size, 0);
    // Note: has_labels is read-only, but may be automatically computed when labels array is resized
    LOG(INFO) << "[Hydra Load] Initialized labels array with size " << mesh_size;
  } else if (dsg_mesh->labels.size() < mesh_size) {
    // If labels exist but size mismatch, resize and fill with 0
    const size_t old_size = dsg_mesh->labels.size();
    dsg_mesh->labels.resize(mesh_size, 0);
    LOG(WARNING) << "[Hydra Load] Resized labels array from " 
                 << old_size << " to " << mesh_size;
  }
  
  size_t total_nodes_processed = 0;
  size_t total_vertices_labeled = 0;
  size_t total_invalid_indices_skipped = 0;
  
  // Process OBJECTS layer nodes
  if (backend_dsg_->graph->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
    LOG(INFO) << "[Hydra Load] Pre-filling semantic labels from Object nodes...";
    
    const auto& objects_layer = backend_dsg_->graph->getLayer(spark_dsg::DsgLayers::OBJECTS);
    size_t nodes_processed = 0;
    size_t vertices_labeled = 0;
    size_t invalid_indices_skipped = 0;
    
    for (const auto& [node_id, node] : objects_layer.nodes()) {
      try {
        const auto& attrs = node->attributes<ObjectNodeAttributes>();
        
        // Skip if no mesh connections or invalid semantic label
        if (attrs.mesh_connections.empty() || attrs.semantic_label == 0) {
          continue;
        }
        
        const uint32_t label = attrs.semantic_label;
        size_t valid_connections = 0;
        
        // Fill labels for all connected vertices
        for (const auto vertex_idx : attrs.mesh_connections) {
          if (vertex_idx >= mesh_size) {
            ++invalid_indices_skipped;
            continue;
          }
          
          dsg_mesh->labels[vertex_idx] = label;
          ++valid_connections;
          ++vertices_labeled;
        }
        
        if (valid_connections > 0) {
          ++nodes_processed;
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "[Hydra Load] Failed to process Object node " 
                     << spark_dsg::NodeSymbol(node_id).getLabel() 
                     << ": " << e.what();
      }
    }
    
    total_nodes_processed += nodes_processed;
    total_vertices_labeled += vertices_labeled;
    total_invalid_indices_skipped += invalid_indices_skipped;
    
    LOG(INFO) << "[Hydra Load] Objects processed: " << nodes_processed 
              << " nodes, " << vertices_labeled << " vertices labeled";
  }
  
  // Process MESH_PLACES layer nodes (Place2D)
  if (backend_dsg_->graph->hasLayer(spark_dsg::DsgLayers::MESH_PLACES)) {
    LOG(INFO) << "[Hydra Load] Pre-filling semantic labels from Place2D nodes...";
    
    const auto& places_layer = backend_dsg_->graph->getLayer(spark_dsg::DsgLayers::MESH_PLACES);
    size_t nodes_processed = 0;
    size_t vertices_labeled = 0;
    size_t invalid_indices_skipped = 0;
    
    for (const auto& [node_id, node] : places_layer.nodes()) {
      try {
        const auto& attrs = node->attributes<Place2dNodeAttributes>();
        
        // Skip if no mesh connections or invalid semantic label
        if (attrs.pcl_mesh_connections.empty() || attrs.semantic_label == 0) {
          continue;
        }
        
        const uint32_t label = attrs.semantic_label;
        size_t valid_connections = 0;
        
        // Fill labels for all connected vertices
        for (const auto vertex_idx : attrs.pcl_mesh_connections) {
          if (vertex_idx >= mesh_size) {
            ++invalid_indices_skipped;
            continue;
          }
          
          dsg_mesh->labels[vertex_idx] = label;
          ++valid_connections;
          ++vertices_labeled;
        }
        
        if (valid_connections > 0) {
          ++nodes_processed;
        }
      } catch (const std::exception& e) {
        LOG(WARNING) << "[Hydra Load] Failed to process Place2D node " 
                     << spark_dsg::NodeSymbol(node_id).getLabel() 
                     << ": " << e.what();
      }
    }
    
    total_nodes_processed += nodes_processed;
    total_vertices_labeled += vertices_labeled;
    total_invalid_indices_skipped += invalid_indices_skipped;
    
    LOG(INFO) << "[Hydra Load] Places processed: " << nodes_processed 
              << " nodes, " << vertices_labeled << " vertices labeled";
  }
  
  // Log summary
  LOG(INFO) << "[Hydra Load] Semantic pre-fill complete:";
  LOG(INFO) << "  - Total nodes processed: " << total_nodes_processed;
  LOG(INFO) << "  - Total vertices labeled: " << total_vertices_labeled;
  LOG(INFO) << "  - Unlabeled vertices (remain as 0): " 
            << (mesh_size - total_vertices_labeled);
  if (total_invalid_indices_skipped > 0) {
    LOG(WARNING) << "  - Invalid indices skipped: " << total_invalid_indices_skipped 
                 << " (will be cleaned up later)";
  }

  // synchronize the loaded map to other modules
  LOG(INFO) << "[Hydra Load] Synchronizing map state to all pipeline modules...";
  frontend_dsg_->graph->mergeGraph(*backend_dsg_->graph);
  shared_state_->lcd_graph->graph->mergeGraph(*backend_dsg_->graph);

  // synchronize mesh to all DSG instances
  if (backend_dsg_->graph->hasMesh()) {
    auto loaded_mesh = backend_dsg_->graph->mesh();
    frontend_dsg_->graph->setMesh(loaded_mesh);
    shared_state_->lcd_graph->graph->setMesh(loaded_mesh);
    LOG(INFO) << "[Hydra Load] Synchronized mesh to all DSG instances: "
              << loaded_mesh->numVertices() << " vertices, "
              << loaded_mesh->numFaces() << " faces";
  } else {
    LOG(WARNING) << "[Hydra Load] No mesh found in backend DSG, skipping mesh synchronization";
  }

  // Clean up any invalid mesh indices in loaded nodes
  // This can happen if the mesh was optimized/compressed after node creation
  if (dsg_mesh) {
    cleanupInvalidMeshIndices(*backend_dsg_->graph, dsg_mesh->numVertices());
  }

  LOG(INFO) << "[Hydra Load] Map loading and synchronization complete!";
  return true;
}

void hydra::HydraRosPipeline::cleanupInvalidMeshIndices(DynamicSceneGraph& graph,
                                                  size_t max_valid_index) {
  LOG(INFO) << "[Hydra Load] Cleaning up invalid mesh indices (max valid: " 
            << max_valid_index << ")...";
  
  size_t cleaned_count = 0;
  size_t cleaned_faces = 0;
  
  // Clean up mesh faces first - remove any faces that reference invalid vertices
  if (graph.hasMesh()) {
    auto mesh = graph.mesh();
    if (mesh && !mesh->faces.empty()) {
      std::vector<spark_dsg::Mesh::Face> valid_faces;
      valid_faces.reserve(mesh->faces.size());
      
      for (const auto& face : mesh->faces) {
        // Check if all three vertices are valid
        bool is_valid = true;
        for (size_t i = 0; i < 3; ++i) {
          if (face[i] >= max_valid_index) {
            is_valid = false;
            VLOG(2) << "Face has invalid vertex index " << face[i] 
                    << " (max: " << max_valid_index << ")";
            break;
          }
        }
        
        if (is_valid) {
          valid_faces.push_back(face);
        } else {
          cleaned_faces++;
        }
      }
      
      if (cleaned_faces > 0) {
        mesh->faces = std::move(valid_faces);
        LOG(WARNING) << "[Hydra Load] Removed " << cleaned_faces 
                     << " invalid faces from mesh (remaining: " << mesh->faces.size() << ")";
      }
    }
  }
  
  // Clean up Object nodes
  if (graph.hasLayer(DsgLayers::OBJECTS)) {
    for (auto& [id, node] : graph.getLayer(DsgLayers::OBJECTS).nodes()) {
      auto& attrs = node->attributes<ObjectNodeAttributes>();
      
      auto iter = attrs.mesh_connections.begin();
      while (iter != attrs.mesh_connections.end()) {
        if (*iter >= max_valid_index) {
          VLOG(1) << "Removing invalid mesh index " << *iter 
                  << " from Object " << NodeSymbol(id).getLabel();
          iter = attrs.mesh_connections.erase(iter);
          cleaned_count++;
        } else {
          ++iter;
        }
      }
    }
  }
  
  // Clean up Place 2D nodes
  if (graph.hasLayer(DsgLayers::MESH_PLACES)) {
    for (auto& [id, node] : graph.getLayer(DsgLayers::MESH_PLACES).nodes()) {
      auto& attrs = node->attributes<Place2dNodeAttributes>();
      
      // Clean up pcl_mesh_connections
      auto iter = attrs.pcl_mesh_connections.begin();
      while (iter != attrs.pcl_mesh_connections.end()) {
        if (*iter >= max_valid_index) {
          VLOG(1) << "Removing invalid mesh index " << *iter 
                  << " from Place2D " << NodeSymbol(id).getLabel();
          iter = attrs.pcl_mesh_connections.erase(iter);
          cleaned_count++;
        } else {
          ++iter;
        }
      }
      
      // Clean up pcl_boundary_connections (synchronized with boundary vector)
      auto boundary_iter = attrs.pcl_boundary_connections.begin();
      auto boundary_pos_iter = attrs.boundary.begin();
      while (boundary_iter != attrs.pcl_boundary_connections.end() &&
             boundary_pos_iter != attrs.boundary.end()) {
        if (*boundary_iter >= max_valid_index) {
          VLOG(1) << "Removing invalid boundary index " << *boundary_iter 
                  << " from Place2D " << NodeSymbol(id).getLabel();
          boundary_iter = attrs.pcl_boundary_connections.erase(boundary_iter);
          boundary_pos_iter = attrs.boundary.erase(boundary_pos_iter);
          cleaned_count++;
        } else {
          ++boundary_iter;
          ++boundary_pos_iter;
        }
      }
      
      // Update min/max indices after cleanup
      if (!attrs.pcl_mesh_connections.empty()) {
        attrs.pcl_min_index = *std::min_element(attrs.pcl_mesh_connections.begin(), 
                                                attrs.pcl_mesh_connections.end());
        attrs.pcl_max_index = *std::max_element(attrs.pcl_mesh_connections.begin(), 
                                                attrs.pcl_mesh_connections.end());
      }
    }
  }
  
  if (cleaned_count > 0 || cleaned_faces > 0) {
    LOG(WARNING) << "[Hydra Load] Cleaned " << cleaned_count 
                 << " invalid mesh indices and " << cleaned_faces 
                 << " invalid faces (likely due to mesh optimization after save)";
  } else {
    LOG(INFO) << "[Hydra Load] All mesh indices and faces are valid";
  }
}
