import cv2
import numpy as np
import open3d as o3d
import os
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import copy

def depth_image_to_point_cloud(depth_image_path, fx, fy, cx, cy, k1, k2, k3, k4, voxel_size=None, batch_size=10, max_points=30000):
    # Read depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

    # Convert depth image to float and scale
    depth_image = (depth_image.astype(np.float32)/(2**16-1))*100

    # Print depth image information
    print(f"Depth image shape: {depth_image.shape}, data type: {depth_image.dtype}")
    print(f"Depth value range: {np.min(depth_image)} - {np.max(depth_image)}")


    # Get depth image dimensions
    height, width = depth_image.shape

    # Set camera intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Set distortion coefficients - for fisheye camera model
    D = np.array([k1, k2, k3, k4])

    try:
        # Correct distortion using fisheye camera model
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
        undistorted_depth = cv2.remap(depth_image, map1, map2, cv2.INTER_LINEAR)
        # Release original depth image memory
        del depth_image, map1, map2
    except Exception as e:
        print(f"Distortion correction failed, using original depth image: {e}")
        undistorted_depth = depth_image
        del depth_image

    # Create mask for valid depth
    valid_mask = undistorted_depth > 0
    valid_depth_count = np.sum(valid_mask)
    print(f"Valid depth pixel count: {valid_depth_count}")

    if valid_depth_count == 0:
        print("Warning: No valid depth values found")
        return np.array([[0, 0, 0]])

    # Vectorized point cloud calculation (replacing nested loops)
    # Create grid coordinates
    v, u = np.indices((height, width))

    # Only select pixels with valid depth
    z = undistorted_depth[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]

    # Calculate 3D coordinates
    x = (u_valid - cx) / fx * z
    y = (v_valid - cy) / fy * z

    # Merge coordinates to create point cloud
    point_cloud = np.column_stack((x, y, z))

    # Release no longer needed memory
    del undistorted_depth, valid_mask, v, u, z, u_valid, v_valid, x, y

    # If downsampling is needed
    if voxel_size is not None and voxel_size > 0:
        # Use Open3D's built-in voxel downsampling method (faster than manual implementation)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Convert back to NumPy array
        point_cloud = np.asarray(pcd.points)

        # Clean up memory
        del pcd

    print(f"Generated point cloud contains {len(point_cloud)} points")
    return point_cloud

def compute_range(point_cloud):
    """
    Calculate bounding box diagonal length of point cloud.
    If return value is abnormal, it may indicate unit or intrinsic parameter issues.
    """
    if len(point_cloud) == 0:
        return 0.0
    min_xyz = np.min(point_cloud, axis=0)
    max_xyz = np.max(point_cloud, axis=0)
    return np.linalg.norm(max_xyz - min_xyz)

def load_poses(pose_path):
    """Load all poses from pose.txt (stored in column-major order as 4x4 matrices).

    Each row contains 16 numbers (comma-separated), but arranged in column-major order,
    i.e., the first 4 numbers are the first column, the next 4 numbers are the second column, and so on.
    """
    print(f"Loading: {pose_path}")
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = list(map(float, line.strip().split(',')))
            if len(data) != 16:
                raise ValueError(f"Pose row data length error, expected 16 numbers, got {len(data)}, content: {line}")

            # Reconstruct 4x4 matrix in column-major order
            T = np.zeros((4, 4))
            for i in range(4):  # columns
                for j in range(4):  # rows
                    T[j, i] = data[i * 4 + j]

            # Validate matrix format
            if not np.allclose(T[3, :], [0, 0, 0, 1]):
                raise ValueError(f"Invalid transformation matrix, last row should be [0,0,0,1]:\n{T}")

            poses.append(T)

    print(f"Successfully loaded {len(poses)} pose matrices")
    return poses

def extract_rotation_translation_from_pose(pose):
    """Extract rotation matrix and translation vector from pose matrix"""
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return rotation, translation

def apply_scale_to_point_cloud(pcd, scale_factor):
    """Apply scale transformation to point cloud, centered at coordinate origin"""
    pcd_scaled = copy.deepcopy(pcd)

    # Scale centered at coordinate origin (0,0,0)
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    pcd_scaled.transform(scale_matrix)

    return pcd_scaled

def get_rotation_matrix(rotation):
    """Create 4x4 transformation matrix from 3x3 rotation matrix (rotation only)"""
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = rotation
    return rot_matrix

def get_translation_matrix(translation):
    """Create 4x4 transformation matrix from translation vector (translation only)"""
    trans_matrix = np.eye(4)
    trans_matrix[:3, 3] = translation
    return trans_matrix

def extract_scene_name(file_path, root_folder):
    """Extract scene name from file path"""
    # Get relative path
    rel_path = os.path.relpath(file_path, root_folder)
    # First level directory is usually the scene name
    parts = rel_path.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return None

def find_pose_file(pose_root, scene_name):
    """Find pose file corresponding to the scene"""
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    if os.path.exists(pose_path):
        return pose_path
    return None

def process_single_file(file_path, output_folder, root_folder, camera_params, voxel_size, max_points, pose_matrix=None, apply_transform=False):
    try:
        print(file_path)


        fx, fy, cx, cy, k1, k2, k3, k4 = camera_params

        # Generate point cloud data
        point_cloud = depth_image_to_point_cloud(
            file_path, fx, fy, cx, cy, k1, k2, k3, k4,
            voxel_size=voxel_size, max_points=max_points
        )

        # Calculate point cloud range, for debugging reference only
        pc_range = compute_range(point_cloud)
        print(f"Point cloud range: {pc_range:.2f}")

        # Check if point cloud is empty
        if len(point_cloud) <= 1:
            print(f"Warning: {os.path.basename(file_path)} generated point cloud is empty or contains only default points, skipping processing")
            return False

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        # If transformation needs to be applied
        if apply_transform and pose_matrix is not None:
            # Record original point cloud center
            original_center = pcd.get_center()
            print(f"Original point cloud center: {original_center}")

            # Extract rotation and translation parts from pose matrix
            rotation, translation = extract_rotation_translation_from_pose(pose_matrix)

            # Print currently used pose information
            print(f"Using pose information:")
            print(f"Rotation matrix:\n{rotation}")
            print(f"Translation vector: {translation}")

            # Step 1: First apply rotation (centered at coordinate origin)
            rotation_matrix = get_rotation_matrix(rotation)
            rotated_pcd = copy.deepcopy(pcd)
            rotated_pcd.transform(rotation_matrix)
            rotated_center = rotated_pcd.get_center()
            print(f"Point cloud center after rotation: {rotated_center}")

            # Step 3: Finally apply translation transformation
            translation_matrix = get_translation_matrix(translation)
            transformed_pcd = copy.deepcopy(rotated_pcd)
            transformed_pcd.transform(translation_matrix)
            final_center = transformed_pcd.get_center()
            print(f"Final transformed point cloud center: {final_center}")

            # Use transformed point cloud
            pcd = transformed_pcd

        # Create corresponding subfolders in output folder based on original folder structure
        root_dir = os.path.dirname(file_path)
        rel_path = os.path.relpath(root_dir, root_folder)
        target_folder = os.path.join(output_folder, rel_path)
        os.makedirs(target_folder, exist_ok=True)

        # Generate and save point cloud file (PLY format)
        filename = os.path.basename(file_path)
        # Extract frame number (e.g.: 0000_depth.tiff -> 0000)
        frame_number = filename.split('_')[0]
        pc_file_name = f"{frame_number}_s.ply"  # Use new naming format: XXXX_s.ply
        pc_file_path = os.path.join(target_folder, pc_file_name)
        o3d.io.write_point_cloud(pc_file_path, pcd)

        return True
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return False
    finally:
        # Clean up memory after processing each file
        gc.collect()

def main():
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Generate point cloud files from depth images')
    parser.add_argument('--input', type=str, required=True, help='Root directory path of input depth image files')
    parser.add_argument('--output', type=str, required=True, help='Root directory path of output point cloud files')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel size for point cloud downsampling')
    parser.add_argument('--max_points', type=int, default=30000, help='Maximum number of points in point cloud')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for parallel processing')
    parser.add_argument('--pose_path', type=str, help='Path to pose file (single file)')
    parser.add_argument('--pose_root', type=str, help='Root directory of pose files (automatically find pose files for each scene)')
    parser.add_argument('--apply_transform', action='store_true', help='Whether to apply rotation and translation transformation')
    args = parser.parse_args()

    # Camera intrinsic parameters and distortion coefficients (set according to actual data)
    fx = 767.3861511125845
    fy = 767.5058656118406
    cx = 679.054265997005
    cy = 543.646891684636
    k1 = -0.18867185058223412
    k2 = -0.003927337093919806
    k3 = 0.030524814153620117
    k4 = -0.012756926010904904
    camera_params = (fx, fy, cx, cy, k1, k2, k3, k4)

    # Use paths specified by command line arguments
    root_folder = args.input
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    # Set voxel size for point cloud downsampling and maximum point count limit
    voxel_size = args.voxel_size
    max_points = args.max_points

    # Collect all depth image files to be processed
    depth_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_depth.tiff'):
                depth_files.append(os.path.join(root, file))

    total_files = len(depth_files)
    print(f"Found {total_files} depth image files to process")

    # Process files
    successful = 0

    # If transformation needs to be applied
    if args.apply_transform:
        # Create scene->pose data mapping
        scene_poses = {}

        if args.pose_path and os.path.exists(args.pose_path):
            # If only a single pose file is provided, all scenes share this pose
            poses = load_poses(args.pose_path)
            print(f"Loaded {len(poses)} pose matrices from single pose file")
            # Use same pose for all scenes
            for file_path in depth_files:
                scene_name = extract_scene_name(file_path, root_folder)
                if scene_name and scene_name not in scene_poses:
                    scene_poses[scene_name] = poses

        elif args.pose_root and os.path.exists(args.pose_root):
            # If pose root directory is provided, find corresponding pose file for each scene
            print(f"Attempting to load pose files from pose root directory {args.pose_root}")
            # Collect all scene names
            scene_names = set()
            for file_path in depth_files:
                scene_name = extract_scene_name(file_path, root_folder)
                if scene_name:
                    scene_names.add(scene_name)

            # Load pose data for each scene
            for scene_name in scene_names:
                pose_file = find_pose_file(args.pose_root, scene_name)
                if pose_file:
                    print(f"Loading pose file for scene {scene_name}: {pose_file}")
                    scene_poses[scene_name] = load_poses(pose_file)
                else:
                    print(f"Warning: Pose file not found for scene {scene_name}")

        else:
            print("Error: No valid pose file or pose root directory specified")
            return

        # Process each file
        for file_path in depth_files:
            try:
                # Extract scene name and frame index
                scene_name = extract_scene_name(file_path, root_folder)
                frame_idx = int(os.path.basename(file_path).split('_')[0])

                # If pose data for corresponding scene is found
                if scene_name in scene_poses:
                    poses = scene_poses[scene_name]

                    # Ensure frame index is within range
                    if frame_idx < len(poses):
                        print(f"Processing file: {file_path} (scene: {scene_name}, frame: {frame_idx})")
                        result = process_single_file(
                            file_path,
                            output_folder,
                            root_folder,
                            camera_params,
                            voxel_size,
                            max_points,
                            pose_matrix=poses[frame_idx],
                            apply_transform=True
                        )
                        if result:
                            successful += 1
                    else:
                        print(f"Warning: Frame index {frame_idx} exceeds scene {scene_name} pose data range ({len(poses)}), skipping transformation")
                        result = process_single_file(
                            file_path,
                            output_folder,
                            root_folder,
                            camera_params,
                            voxel_size,
                            max_points
                        )
                        if result:
                            successful += 1
                else:
                    print(f"Warning: Pose data not found for scene {scene_name}, skipping transformation")
                    result = process_single_file(
                        file_path,
                        output_folder,
                        root_folder,
                        camera_params,
                        voxel_size,
                        max_points
                    )
                    if result:
                        successful += 1
            except Exception as e:
                print(f"Error processing file {os.path.basename(file_path)}: {e}")
                continue
    else:
        # Create partial function with fixed parameters
        process_file_fn = partial(
            process_single_file,
            output_folder=output_folder,
            root_folder=root_folder,
            camera_params=camera_params,
            voxel_size=voxel_size,
            max_points=max_points
        )

        # Process files in parallel using multithreading
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            results = list(executor.map(process_file_fn, depth_files))
            successful = sum(1 for r in results if r)

    print(f"Processing complete: {successful}/{total_files} files successfully processed")

if __name__ == '__main__':
    main()
