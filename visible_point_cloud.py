#!/usr/bin/env python
import os
import numpy as np
import open3d as o3d
import cv2
import argparse
from glob import glob

# project_points_to_image_all projects 3D world points onto the 2D camera image plane.
# It returns the pixel coordinates, a visibility flag (1 if visible), and depth values.
def project_points_to_image_all(world_points, camera_pose, intrinsic, width, height):
    """
    Project point cloud onto image plane, adjusted for different coordinate conventions
    """
    # Convert 3D points to homogeneous coordinates
    homogeneous_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

    # Transform from world coordinate system to camera coordinate system
    camera_points = (camera_pose @ homogeneous_points.T).T

    # Modified: Determine camera viewing direction based on Z value range, if all Z values are positive, then points with Z>0 are in front
    z_min = camera_points[:, 2].min()
    z_max = camera_points[:, 2].max()

    # Since Z values are all positive in camera coordinate system, we should use Z>0 as the front determination condition
    in_front = camera_points[:, 2] > 0  # Modified condition
    
    # Initialize arrays for projected pixel coordinates, visibility flags, and depth values.
    pixel_coords = np.zeros((len(world_points), 2))
    visibility = np.zeros(len(world_points))
    depths = camera_points[:, 2]
 
    # Only points in front of the camera have a positive z-coordinate.
    if np.any(in_front):
        # Apply perspective projection using the intrinsic matrix.
        pixel_x = (intrinsic[0, 0] * camera_points[in_front, 0] / camera_points[in_front, 2]) + intrinsic[0, 2]
        pixel_y = (intrinsic[1, 1] * camera_points[in_front, 1] / camera_points[in_front, 2]) + intrinsic[1, 2]
        # Get indices of the points that are in front.
        front_indices = np.where(in_front)[0]
        pixel_coords[front_indices, 0] = pixel_x
        pixel_coords[front_indices, 1] = pixel_y
 
        # Check if the projected pixel positions are within the image frame's boundaries.
        in_frame = ((pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height))
        frame_indices = front_indices[in_frame]
        visibility[frame_indices] = 1
 
    return pixel_coords, visibility, depths
 
# handle_occlusion_raycasting_all uses Open3D's raycasting capabilities to update the visibility
# of points by checking if they are occluded (i.e., not directly visible from the camera).
def handle_occlusion_raycasting_all(world_coords, camera_pos, mesh, visibility, ray_step=0.2):
    # Create a raycasting scene and add the mesh converted to Open3D's tensor format.
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
 
    # Process only points initially marked as visible.
    visible_indices = np.where(visibility == 1)[0]
    if len(visible_indices) == 0:
        return visibility
 
    # Prepare an array of rays. Each ray consists of its origin (camera position) and a normalized direction.
    rays = np.zeros((len(visible_indices), 6))
    rays[:, 0:3] = camera_pos
    ray_dirs = world_coords[visible_indices] - camera_pos
    ray_lens = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs = ray_dirs / ray_lens
    rays[:, 3:6] = ray_dirs
 
    # Convert the rays array to a tensor required by the Open3D raycasting API.
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_t)
    t_hit = ans['t_hit'].numpy()
 
    # Update the visibility flag for points that are occluded.
    updated_visibility = visibility.copy()
    for i, (idx, t, ray_len) in enumerate(zip(visible_indices, t_hit, ray_lens.flatten())):
        if abs(t - ray_len) > 0.5:  # Changed to larger threshold
            updated_visibility[idx] = 0

    return updated_visibility

def visualize_cameras_and_model(mesh, poses, output_path):
    # Create visualization geometries
    vis_geometries = [mesh]

    # Create a small coordinate system visualization for each camera pose
    for i, pose in enumerate(poses[:10]):  # Only visualize first 10 cameras
        # Create camera coordinate system
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_frame.transform(pose)
        vis_geometries.append(camera_frame)

        # Create sphere indicating camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(pose[:3, 3])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        vis_geometries.append(sphere)

    # Save as visualization file
    o3d.io.write_triangle_mesh(output_path, mesh)  # Save model

    return vis_geometries

def process_scene_with_depth_guided_raycasting(scene_path, point_cloud_scene_dir, output_root):
    """
    Use depth image reconstructed point cloud to guide raycasting for visibility determination
    """
    scene_name = os.path.basename(os.path.normpath(scene_path))
    print(f"Processing scene: {scene_name}")

    # 1. Load mesh
    mesh_files = [f for f in os.listdir(scene_path) if f.endswith(".obj")]
    if not mesh_files:
        print(f"Scene {scene_name} mesh file not found, skipping.")
        return
    mesh_file_path = os.path.join(scene_path, mesh_files[0])
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()

    # 2. Read pose data
    pose_file = os.path.join(scene_path, "pose.txt")
    if not os.path.exists(pose_file):
        print(f"Scene {scene_name} pose.txt not found, skipping.")
        return
    
    poses_raw = np.loadtxt(pose_file, delimiter=",")
    poses = []
    for i in range(poses_raw.shape[0]):
        pose_raw_matrix = poses_raw[i].reshape((4,4))
        standard_pose = np.eye(4)
        standard_pose[:3, :3] = pose_raw_matrix[:3, :3]
        standard_pose[:3, 3] = pose_raw_matrix[3, :3]
        poses.append(standard_pose)

    # 3. Create output directory
    scene_output_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)

    # 4. Process each frame
    for i in range(len(poses)):
        print(f"  Processing frame {i}")

        # 4.1 Load reconstructed point cloud (using new naming format: XXXX_s.ply)
        reconstructed_pcd_path = os.path.join(point_cloud_scene_dir, f"{i:04d}_s.ply")
        if not os.path.exists(reconstructed_pcd_path):
            print(f"    Reconstructed point cloud not found: {reconstructed_pcd_path}")
            continue

        reconstructed_pcd = o3d.io.read_point_cloud(reconstructed_pcd_path)
        if not reconstructed_pcd.has_points():
            print(f"    Reconstructed point cloud is empty: {reconstructed_pcd_path}")
            continue

        # 4.2 Calculate camera pose
        camera_pose = np.linalg.inv(poses[i])
        camera_position = poses[i][:3, 3]

        # 4.3 Use depth image reconstructed point cloud to determine view direction and range
        view_direction, view_range = determine_view_from_depth_pcd(
            reconstructed_pcd, camera_position
        )

        # 4.4 Sample points from mesh, but only sample points within field of view
        pcd_mesh = mesh.sample_points_uniformly(number_of_points=50000)
        world_points = np.asarray(pcd_mesh.points)

        # 4.5 Preliminary filtering based on view direction
        initial_visibility = filter_points_by_view_direction(
            world_points, camera_position, view_direction, view_range
        )

        # 4.6 Perform raycasting on preliminarily filtered points
        final_visibility = raycast_with_view_guidance(
            world_points, camera_position, mesh, initial_visibility,
            view_direction, view_range
        )

        # 4.7 Generate final visible point cloud
        visible_points = world_points[final_visibility == 1]
        if len(visible_points) > 0:
            pcd_visible = o3d.geometry.PointCloud()
            pcd_visible.points = o3d.utility.Vector3dVector(visible_points)

            # 4.8 Save result (using new naming format: XXXX_t.ply)
            output_filename = os.path.join(scene_output_dir, f"{i:04d}_t.ply")
            o3d.io.write_point_cloud(output_filename, pcd_visible)
            print(f"    Saved visible point cloud ({len(visible_points)} points): {output_filename}")
        else:
            print(f"    Warning: No visible points!")

def determine_view_from_depth_pcd(depth_pcd, camera_position):
    """
    Determine view direction and range from depth image reconstructed point cloud
    """
    # 1. Calculate point cloud center
    center = depth_pcd.get_center()

    # 2. Calculate view direction (direction from camera to point cloud center)
    view_direction = center - camera_position
    view_direction = view_direction / np.linalg.norm(view_direction)

    # 3. Calculate view range (maximum distance from point cloud to camera)
    points = np.asarray(depth_pcd.points)
    distances = np.linalg.norm(points - camera_position, axis=1)
    view_range = np.max(distances)

    # Print debug information
    print(f"Point cloud center: {center}")
    print(f"Camera position: {camera_position}")
    print(f"View direction: {view_direction}")
    print(f"View range: {view_range}")

    return view_direction, view_range

def filter_points_by_view_direction(points, camera_position, view_direction, view_range):
    """
    Preliminary filtering based on view direction
    """
    visibility = np.zeros(len(points))

    # 1. Calculate direction vectors from all points to camera
    directions = points - camera_position
    distances = np.linalg.norm(directions, axis=1)

    # 2. Normalize direction vectors
    directions = directions / distances[:, np.newaxis]

    # 3. Calculate angle with view direction
    cos_angles = np.dot(directions, view_direction)

    # 4. Set filtering conditions
    angle_threshold = np.cos(np.radians(60))  # 60 degree view cone angle
    distance_threshold = view_range * 1.2  # Allow 20% extra range

    # 5. Apply filtering conditions
    valid_indices = (cos_angles > angle_threshold) & (distances < distance_threshold)
    visibility[valid_indices] = 1

    return visibility

def raycast_with_view_guidance(points, camera_position, mesh, initial_visibility,
                             view_direction, view_range):
    """
    Perform raycasting based on view direction
    """
    # 1. Create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    # 2. Get preliminarily visible points
    visible_indices = np.where(initial_visibility == 1)[0]
    if len(visible_indices) == 0:
        return initial_visibility

    # 3. Prepare rays
    rays = np.zeros((len(visible_indices), 6))
    rays[:, 0:3] = camera_position

    # 4. Calculate ray directions (considering view direction)
    ray_dirs = points[visible_indices] - camera_position
    ray_lens = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs = ray_dirs / ray_lens

    # 5. Adjust rays based on view direction
    for i, ray_dir in enumerate(ray_dirs):
        # Calculate angle with view direction
        cos_angle = np.dot(ray_dir, view_direction)
        if cos_angle < 0:  # If ray direction is opposite to view direction
            ray_dirs[i] = -ray_dirs[i]  # Flip ray direction

    rays[:, 3:6] = ray_dirs

    # 6. Execute raycasting
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_t)
    t_hit = ans['t_hit'].numpy()

    # 7. Update visibility
    updated_visibility = initial_visibility.copy()
    for i, (idx, t, ray_len) in enumerate(zip(visible_indices, t_hit, ray_lens.flatten())):
        if abs(t - ray_len) > 0.5:  # Ray detection threshold
            updated_visibility[idx] = 0

    return updated_visibility

def main():
    parser = argparse.ArgumentParser(description="Generate visible point cloud corresponding to each frame's depth image, process all scenes")
    parser.add_argument("--input", type=str, required=True, help="C3VD folder path, each subfolder represents a scene, containing .obj and pose.txt")
    parser.add_argument("--point_cloud_source", type=str, required=True, help="C3VD_ply_source folder path, containing point cloud files for all scenes")
    parser.add_argument("--output", type=str, required=True, help="Visible point cloud output root directory")
    args = parser.parse_args()

    input_root = args.input
    point_cloud_source = args.point_cloud_source
    output_root = args.output

    # Get all scene folders under input directory
    scene_dirs = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if len(scene_dirs) == 0:
        print("No scene folders found under input path.")
        return

    for scene_path in scene_dirs:
        scene_name = os.path.basename(scene_path)
        # Ensure corresponding scene folder exists in point cloud source directory
        point_cloud_scene_dir = os.path.join(point_cloud_source, scene_name)
        if not os.path.exists(point_cloud_scene_dir):
            print(f"Warning: Scene {scene_name} folder not found in point cloud source directory, skipping processing.")
            continue

        process_scene_with_depth_guided_raycasting(scene_path, point_cloud_scene_dir, output_root)

    print("All scenes' visible point cloud generation tasks completed!")

if __name__ == "__main__":
    main()