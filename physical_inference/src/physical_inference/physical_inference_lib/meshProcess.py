#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d
import os
import time
import rospkg
import numpy as np

def render_mesh_to_image(kimera_mesh):
    """
    Renders a KimeraPgmoMesh to a temporary image file.
    
    :param kimera_mesh: A kimera_pgmo_msgs/KimeraPgmoMesh object.
    :return: The path to the temporary image file, and the temporary directory handle.
             Returns (None, None) on failure.
    """
    try:
        # Create a temporary directory to store the image
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('physical_inference')
        temp_dir = os.path.join(package_path, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time() * 1000) 
        image_path = os.path.join(temp_dir, f"rendered_mesh_{timestamp}.png")

        # Convert KimeraPgmoMesh to Open3D TriangleMesh
        mesh = _kimera_mesh_to_open3d(kimera_mesh)
        if mesh is None:
            return None, None
        
        # Render the mesh
        success = _render_triangle_mesh(mesh, image_path)
        
        if success:
            return image_path, temp_dir
        else:
            return None, None

    except Exception as e:
        print(f"Failed to render mesh: {e}")
        return None, None


def _kimera_mesh_to_open3d(kimera_mesh):
    """
    Convert KimeraPgmoMesh to Open3D TriangleMesh.
    
    :param kimera_mesh: kimera_pgmo_msgs/KimeraPgmoMesh
    :return: Open3D TriangleMesh or None if conversion fails
    """
    try:
        if not kimera_mesh.vertices:
            print("No vertices in mesh")
            return None
        
        # Extract vertices
        vertices = []
        for vertex in kimera_mesh.vertices:
            vertices.append([vertex.x, vertex.y, vertex.z])
        vertices = np.array(vertices)
        
        # Extract triangles
        triangles = []
        for triangle in kimera_mesh.triangles:
            triangles.append([
                triangle.vertex_indices[0],
                triangle.vertex_indices[1], 
                triangle.vertex_indices[2]
            ])
        triangles = np.array(triangles)
        
        # Extract colors if available
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
        
        # Validate mesh
        if len(mesh.vertices) == 0:
            print("Mesh has no vertices after conversion")
            return None
            
        print(f"Converted mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
        
    except Exception as e:
        print(f"Error converting KimeraPgmoMesh to Open3D: {e}")
        return None


def _render_triangle_mesh(mesh, image_path):
    """
    Render an Open3D TriangleMesh to image.
    
    :param mesh: Open3D TriangleMesh
    :param image_path: Output image path
    :return: True if successful, False otherwise
    """
    try:
        # Compute vertex normals for better lighting
        mesh.compute_vertex_normals()
        
        # Set a neutral color if no colors are present
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        # Remove degenerate triangles and duplicates
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        
        # Check if mesh is still valid
        if len(mesh.triangles) == 0:
            print("No valid triangles after cleanup")
            # Fall back to point cloud rendering
            pcd = mesh.sample_points_uniformly(number_of_points=1000)
            return _render_as_improved_pointcloud(pcd, image_path)
        
        # Off-screen rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        vis.add_geometry(mesh)
        
        # Optimize camera view
        _optimize_camera_view(vis, mesh)
        
        # Improve rendering options
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = True
        render_option.mesh_show_wireframe = False
        
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path, do_render=True)
        vis.destroy_window()
        
        return True
        
    except Exception as e:
        print(f"Mesh rendering error: {e}")
        return False


def render_pointcloud_to_image(pcd):
    """
    Renders an Open3D point cloud to a temporary image file.
    First attempts to reconstruct a mesh surface, falls back to improved point cloud rendering.

    :param pcd: An open3d.geometry.PointCloud object.
    :return: The path to the temporary image file, and the temporary directory handle.
             Returns (None, None) on failure.
    """
    if not pcd.has_points():
        return None, None

    try:
        # Create a temporary directory to store the image
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('physical_inference')
        temp_dir = os.path.join(package_path, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time() * 1000) 
        image_path = os.path.join(temp_dir, f"rendered_object{timestamp}.png")

        # Try to render as mesh first, then fall back to point cloud
        success = _render_as_mesh(pcd, image_path)
        if not success:
            print("Mesh reconstruction failed, falling back to improved point cloud rendering")
            success = _render_as_improved_pointcloud(pcd, image_path)
        
        if success:
            return image_path, temp_dir
        else:
            return None, None

    except Exception as e:
        print(f"Failed to render point cloud: {e}")
        return None, None


def _render_as_mesh(pcd, image_path):
    """
    Attempt to reconstruct a mesh from point cloud and render it.
    
    :param pcd: Open3D point cloud
    :param image_path: Output image path
    :return: True if successful, False otherwise
    """
    try:
        # Ensure we have enough points for mesh reconstruction
        if len(pcd.points) < 10:
            return False
            
        # Estimate normals for mesh reconstruction
        pcd.estimate_normals()
        
        # Try Poisson surface reconstruction first
        try:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            
            # Remove outlier triangles
            if len(mesh.triangles) > 0:
                # Remove vertices that are too far from the original point cloud
                mesh.remove_unreferenced_vertices()
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                
                # Check if we still have a valid mesh
                if len(mesh.triangles) > 0:
                    return _render_triangle_mesh(mesh, image_path)
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")
        
        # Try Alpha Shape reconstruction as fallback
        try:
            # Estimate a good alpha value
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            alpha = 2.0 * avg_dist
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            if len(mesh.triangles) > 0:
                mesh.remove_unreferenced_vertices()
                mesh.remove_duplicated_vertices()
                mesh.remove_duplicated_triangles()
                mesh.remove_degenerate_triangles()
                
                if len(mesh.triangles) > 0:
                    return _render_triangle_mesh(mesh, image_path)
        except Exception as e:
            print(f"Alpha shape reconstruction failed: {e}")
            
        return False
        
    except Exception as e:
        print(f"Mesh reconstruction error: {e}")
        return False


def _render_as_improved_pointcloud(pcd, image_path):
    """
    Render point cloud with improved settings.
    
    :param pcd: Open3D point cloud
    :param image_path: Output image path
    :return: True if successful, False otherwise
    """
    try:
        # Estimate normals for better visualization
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Set default colors if not present
        if not pcd.has_colors():
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        
        # Off-screen rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        vis.add_geometry(pcd)
        
        # Optimize camera view
        _optimize_camera_view(vis, pcd)
        
        # Improve rendering options for point cloud
        render_option = vis.get_render_option()
        render_option.point_size = 3.0  # Increased point size
        render_option.light_on = True
        render_option.point_show_normal = False  # Don't show normal vectors
        
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path, do_render=True)
        vis.destroy_window()
        
        return True
        
    except Exception as e:
        print(f"Point cloud rendering error: {e}")
        return False


def _optimize_camera_view(vis, geometry):
    """
    Optimize camera position for better object visibility.
    
    :param vis: Open3D visualizer
    :param geometry: The geometry to view (point cloud or mesh)
    """
    try:
        # Get bounding box
        bbox = geometry.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = bbox.get_extent()
        max_extent = np.max(extent)
        
        # Set camera to view the object from a good angle
        ctr = vis.get_view_control()
        
        # Reset view to fit the object
        ctr.set_lookat(center)
        ctr.set_front([0.5, -0.5, -0.5])  # View from front-top-right
        ctr.set_up([0, 0, 1])  # Z-axis up
        
        # Set appropriate zoom level
        ctr.set_zoom(0.8)
        
        # Alternative: try multiple viewpoints and select the best one
        # This could be enhanced to automatically select the best viewing angle
        
    except Exception as e:
        print(f"Camera optimization error: {e}")


def render_pointcloud_to_image_legacy(pcd):
    """
    Legacy function - renders point cloud with original simple method.
    Kept for backward compatibility.
    
    :param pcd: An open3d.geometry.PointCloud object.
    :return: The path to the temporary image file, and the temporary directory handle.
             Returns (None, None) on failure.
    """
    if not pcd.has_points():
        return None, None

    try:
        # Create a temporary directory to store the image
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('physical_inference')
        temp_dir = os.path.join(package_path, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time() * 1000) 
        image_path = os.path.join(temp_dir, f"rendered_object{timestamp}_legacy.png")

        # Off-screen rendering
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        # Set a good camera angle
        vis.get_render_option().point_size = 2.0
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path, do_render=True)
        vis.destroy_window()

        return image_path, temp_dir
    except Exception as e:
        print(f"Failed to render point cloud: {e}")
        return None, None
