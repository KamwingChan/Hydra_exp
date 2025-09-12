#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import open3d as o3d
import numpy as np
import shutil
import json
import time
from datetime import datetime
import rospkg

from physical_inference.physical_inference_lib.inferenceCore import PhysicalInference
from physical_inference.physical_inference_lib.meshProcess import render_mesh_to_image, render_pointcloud_to_image, render_pointcloud_to_image_legacy
from physical_inference.srv import GetProperties, GetPropertiesResponse
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class PhysicalInferenceServer:
    def __init__(self):
        rospy.init_node('physical_inference_server')

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            rospy.logerr("OPENAI_API_KEY environment variable not set.")
            return

        self.inference_tool = PhysicalInference(api_key=api_key)
        rospy.loginfo("Initialized PhysicalInference client.")        
        # Get debug parameter
        self.debug_save_mesh = rospy.get_param('~debug_save_mesh', False)
        if self.debug_save_mesh:
            rospy.loginfo("Debug mode: Will save received meshes as PLY files")
        
        # Get JSON output parameter
        self.save_json_output = rospy.get_param('~save_json_output', True)
        if self.save_json_output:
            self.setup_output_directory()
            rospy.loginfo(f"JSON output will be saved to: {self.output_dir}")
        
        # Object counter for unique filenames
        self.object_counter = 0
        
        self.service = rospy.Service('get_physical_properties', GetProperties, self.handle_get_properties)
        
        rospy.loginfo("Physical inference service is ready.")

    def setup_output_directory(self):
        """Setup the output directory with timestamp"""
        try:
            # Get package path
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('physical_inference')
            
            # Create timestamp-based directory name
            timestamp = datetime.now().strftime("%m-%d_%H-%M")
            self.output_dir = os.path.join(package_path, 'output', timestamp)
            
            # Create the directory
            os.makedirs(self.output_dir, exist_ok=True)
            
        except Exception as e:
            rospy.logerr(f"Failed to setup output directory: {e}")
            self.save_json_output = False

    def save_inference_result(self, label, properties, object_id=None, processing_time_ms=None):
        """Save inference result to JSON file"""
        if not self.save_json_output:
            return
            
        try:
            # Increment object counter
            self.object_counter += 1
            
            # Create filename
            if object_id:
                filename = f"object_{object_id}_{label}.json"
            else:
                filename = f"object_{self.object_counter}_{label}.json"
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Prepare JSON data (only inference results, no timestamp)
            json_data = {
                "object_id": object_id if object_id else f"object_{self.object_counter}",
                "label": label,
                "description": properties.get("description", ""),
                "friction_level": properties.get("friction_level", 0),
                "pushable": properties.get("pushable", 0),
                "processing_time_ms": processing_time_ms if processing_time_ms is not None else 0
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            rospy.loginfo(f"Saved inference result to: {filepath}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save inference result: {e}")

    def handle_get_properties(self, req):
        rospy.loginfo(f"Received request for label: {req.label}")
        
        # Start timing
        start_time = time.time()

        try:
            # 1. Check if we received a KimeraPgmoMesh
            if not req.object_mesh.vertices:
                rospy.logwarn("Received an empty mesh.")
                return GetPropertiesResponse()
            
            rospy.loginfo(f"Received mesh with {len(req.object_mesh.vertices)} vertices and {len(req.object_mesh.triangles)} triangles")

            # Debug: Save mesh and print statistics if enabled
            if self.debug_save_mesh:
                self.save_debug_mesh(req.object_mesh, req.label)

            # 2. Render the mesh to an image using the new mesh rendering method
            rospy.loginfo("Using direct mesh rendering")
            image_path, temp_dir = render_mesh_to_image(req.object_mesh)
            
            if not image_path:
                rospy.logerr("Failed to render mesh to image.")
                return GetPropertiesResponse()

            # 3. Call the inference tool
            rospy.loginfo(f"Sending image {image_path} to inference tool.")
            properties = self.inference_tool.get_properties(image_path, req.label)

            # 4. Clean up the temporary directory and image
            # if temp_dir:
            #     shutil.rmtree(temp_dir)

            # 5. Populate and return the response
            if "error" in properties:
                error_msg = properties['error']
                rospy.logerr(f"Inference failed: {error_msg}")
                rospy.logerr(f"Full properties response: {properties}")
                return GetPropertiesResponse() # Return empty on error

            # Validate that we have all required fields
            required_fields = ["description", "friction_level", "pushable"]
            missing_fields = [field for field in required_fields if field not in properties]
            if missing_fields:
                rospy.logerr(f"Missing required fields in response: {missing_fields}")
                rospy.logerr(f"Full properties response: {properties}")
                return GetPropertiesResponse()

            # Calculate processing time
            end_time = time.time()
            processing_time_ms = int((end_time - start_time) * 1000)

            response = GetPropertiesResponse()
            response.description = properties.get("description", "")
            response.friction_level = properties.get("friction_level", 0)
            response.pushable = bool(properties.get("pushable", 0))
            
            # Save inference result to JSON file
            self.save_inference_result(
                label=req.label,
                properties=properties,
                object_id=None,  # We could extract this from the mesh namespace if needed
                processing_time_ms=processing_time_ms
            )
            
            rospy.loginfo(f"Successfully processed request in {processing_time_ms}ms. Response: description='{response.description}', friction={response.friction_level}, pushable={response.pushable}")
            return response

        except Exception as e:
            rospy.logerr(f"An error occurred in the service handler: {e}")
            return GetPropertiesResponse() # Return empty on exception

    def save_debug_mesh(self, kimera_mesh, label):
        """Save the received mesh as PLY file and print statistics for debugging"""
        try:
            import time
            import rospkg
            
            # Create debug directory
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('physical_inference')
            debug_dir = os.path.join(package_path, 'debug_meshes')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = int(time.time() * 1000)
            ply_filename = f"debug_mesh_{label}_{timestamp}.ply"
            ply_path = os.path.join(debug_dir, ply_filename)
            
            # Convert to Open3D mesh
            vertices = []
            for vertex in kimera_mesh.vertices:
                vertices.append([vertex.x, vertex.y, vertex.z])
            vertices = np.array(vertices)
            
            triangles = []
            for triangle in kimera_mesh.triangles:
                triangles.append([
                    triangle.vertex_indices[0],
                    triangle.vertex_indices[1], 
                    triangle.vertex_indices[2]
                ])
            triangles = np.array(triangles)
            
            colors = None
            if kimera_mesh.vertex_colors and len(kimera_mesh.vertex_colors) == len(vertices):
                colors = []
                for color in kimera_mesh.vertex_colors:
                    colors.append([color.r, color.g, color.b])
                colors = np.array(colors)
            
            # Create Open3D mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            
            if len(triangles) > 0:
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # Save PLY file
            o3d.io.write_triangle_mesh(ply_path, mesh)
            
            # Print detailed statistics
            rospy.loginfo("=== DEBUG MESH STATISTICS ===")
            rospy.loginfo(f"Object Label: {label}")
            rospy.loginfo(f"Saved to: {ply_path}")
            rospy.loginfo(f"Vertices: {len(vertices)}")
            rospy.loginfo(f"Triangles: {len(triangles)}")
            rospy.loginfo(f"Has Colors: {colors is not None}")
            
            if len(vertices) > 0:
                # Calculate bounding box
                min_coords = np.min(vertices, axis=0)
                max_coords = np.max(vertices, axis=0)
                size = max_coords - min_coords
                center = (min_coords + max_coords) / 2
                
                rospy.loginfo(f"Bounding Box:")
                rospy.loginfo(f"  Min: [{min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f}]")
                rospy.loginfo(f"  Max: [{max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f}]")
                rospy.loginfo(f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
                rospy.loginfo(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            
            if colors is not None:
                # Color statistics
                mean_color = np.mean(colors, axis=0)
                rospy.loginfo(f"Average Color: [R:{mean_color[0]:.3f}, G:{mean_color[1]:.3f}, B:{mean_color[2]:.3f}]")
            
            rospy.loginfo("=== END DEBUG MESH STATISTICS ===")
            
        except Exception as e:
            rospy.logerr(f"Failed to save debug mesh: {e}")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        server = PhysicalInferenceServer()
        server.run()
    except rospy.ROSInterruptException:
        pass
