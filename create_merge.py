import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy  # New import

# ================== Utility Functions ==================
def load_poses(pose_path):
    """Load all poses from pose.txt (directly loading T_cam_to_world), and automatically correct the format.

    If the last row of the loaded matrix is not [0, 0, 0, 1],
    but the 4th column of the first three rows are all 0, and the first three elements of the 4th row are non-zero,
    then convert the translation vector from the 4th row to the 4th column.
    """
    print(f"Loading: {pose_path}")  # Output the current file path being loaded
    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = list(map(float, line.strip().split(',')))
            if len(data) != 16:
                raise ValueError(f"Pose row data length error, expected 16 numbers, got {len(data)}, content: {line}")
            T = np.array(data).reshape(4, 4)
            if not np.allclose(T[3, :], [0, 0, 0, 1]):
                if np.allclose(T[:3, 3], np.zeros(3)) and not np.allclose(T[3, :3], np.zeros(3)):
                    T_fixed = np.eye(4)
                    T_fixed[:3, :3] = T[:3, :3]
                    T_fixed[:3, 3] = T[3, :3]
                    T = T_fixed
                else:
                    raise ValueError(f"Invalid transformation matrix in line: {line}")
            poses.append(T)
    return poses

def compute_relative_pose(pose_i, pose_j):
    """Calculate relative pose: T_ij = T_j @ inv(T_i) (applicable to T_cam_to_world)"""
    return pose_j @ np.linalg.inv(pose_i)

def multi_scale_icp(src_pcd, tgt_pcd,
                    voxel_size=[0.1, 0.05, 0.025],
                    max_iter=[100, 50, 30],
                    init_trans=np.eye(4)):
    """Multi-scale ICP refinement, align src_pcd to tgt_pcd, return the final transformation matrix."""
    current_trans = init_trans
    for scale in range(len(voxel_size)):
        src_down = src_pcd.voxel_down_sample(voxel_size[scale])
        tgt_down = tgt_pcd.voxel_down_sample(voxel_size[scale])
        result_icp = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=voxel_size[scale] * 2,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter[scale])
        )
        current_trans = result_icp.transformation
    return current_trans

# ================== Global ICP Registration and Point Cloud Fusion ==================
def align_scene_with_icp(pose_root, ply_root, scene_name,
                         individual_voxel=0.05, final_voxel=0.05,
                         icp_parallel=True):
    """
    Perform global ICP registration for a single subscene (optimized to handle incomplete point cloud datasets)
    """
    # Scan available point cloud files
    pcd_dir = os.path.join(ply_root, scene_name)
    available_indices, file_paths = scan_available_pointclouds(pcd_dir)
    
    if not available_indices:
        print(f"Error: Scene {scene_name} has no valid point cloud files found")
        return

    print(f"Scene {scene_name} found {len(available_indices)} valid point cloud files")

    # Only load the required pose data
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    poses = load_selective_poses(pose_path, available_indices)
    
    # Load point cloud data
    all_pcds = {}
    for idx in available_indices:
        pcd = o3d.io.read_point_cloud(file_paths[idx])
        if not pcd.has_points():
            print(f"Warning: Point cloud {idx} has no points, skipping")
            continue
        all_pcds[idx] = pcd

    # Find continuous index frame pairs for ICP calculation
    continuous_pairs = []
    for i in range(len(available_indices) - 1):
        idx1 = available_indices[i]
        idx2 = available_indices[i + 1]

        # Only process continuous or close frame indices
        if idx2 - idx1 <= 3 and idx1 in all_pcds and idx2 in all_pcds:
            continuous_pairs.append((idx1, idx2))

    # Calculate relative transformations
    relative_transforms = {}
    
    def icp_task(i, j):
        src_pcd = all_pcds[i]
        tgt_pcd = all_pcds[j]
        init_trans = compute_relative_pose(poses[i], poses[j])
        final_trans = multi_scale_icp(src_pcd, tgt_pcd, init_trans=init_trans)
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            src_pcd, tgt_pcd, 0.1, final_trans
        )
        return i, j, final_trans, info
    
    if icp_parallel and continuous_pairs:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(icp_task, i, j): (i, j) for i, j in continuous_pairs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="ICP Registration"):
                i, j, final_trans, info = future.result()
                relative_transforms[(i, j)] = (final_trans, info)
    else:
        for i, j in tqdm(continuous_pairs, desc="ICP Registration"):
            _, _, final_trans, info = icp_task(i, j)
            relative_transforms[(i, j)] = (final_trans, info)

    # Build PoseGraph
    pose_graph = o3d.pipelines.registration.PoseGraph()

    # Add all nodes
    first_idx = available_indices[0]
    abs_poses = {first_idx: np.eye(4)}
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    # Add all edges
    for pair, (trans, info) in relative_transforms.items():
        i, j = pair

        # Add node (if not exists)
        for idx in [i, j]:
            if idx != first_idx and idx not in abs_poses:
                # Use cumulative transformation relative to the first frame
                path_to_idx = find_path(first_idx, idx, relative_transforms)
                if path_to_idx:
                    abs_pose = calculate_absolute_pose(path_to_idx, relative_transforms, first_idx)
                    abs_poses[idx] = abs_pose
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(abs_pose))

        # Add edge
        if i in abs_poses and j in abs_poses:
            i_idx = available_indices.index(i)
            j_idx = available_indices.index(j)
            edge = o3d.pipelines.registration.PoseGraphEdge(i_idx, j_idx, trans, info, uncertain=False)
            pose_graph.edges.append(edge)

    # Globally optimize PoseGraph
    if pose_graph.edges:
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.1,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )

    # Fuse point clouds based on optimized poses
    combined_pcd = o3d.geometry.PointCloud()
    for i, idx in enumerate(available_indices):
        if idx in all_pcds:
            pcd = all_pcds[idx]
            if i < len(pose_graph.nodes):
                pcd.transform(pose_graph.nodes[i].pose)
            else:
                print(f"Warning: Index {i} exceeds optimized node range, using original pose")
                pcd.transform(poses[idx])
            pcd_ds = pcd.voxel_down_sample(individual_voxel)
            combined_pcd += pcd_ds

    combined_pcd = combined_pcd.voxel_down_sample(final_voxel)
    output_ply = os.path.join(ply_root, scene_name, f"{scene_name}_combined_downsampled_{final_voxel:.3f}.ply")
    o3d.io.write_point_cloud(output_ply, combined_pcd)
    print(f"Saved registration result to file: {output_ply}")

def find_path(start, end, relative_transforms):
    """Find path from start frame to target frame"""
    # Simplified breadth-first search
    visited = set()
    queue = [(start, [start])]

    while queue:
        node, path = queue.pop(0)
        if node == end:
            return path

        if node in visited:
            continue

        visited.add(node)

        # Find all adjacent nodes
        for pair in relative_transforms:
            i, j = pair
            if i == node and j not in visited:
                queue.append((j, path + [j]))
            elif j == node and i not in visited:
                queue.append((i, path + [i]))

    return None  # Path not found

def calculate_absolute_pose(path, relative_transforms, first_idx):
    """Calculate absolute pose based on path"""
    abs_pose = np.eye(4)

    for i in range(len(path) - 1):
        curr = path[i]
        next_node = path[i + 1]

        if (curr, next_node) in relative_transforms:
            trans, _ = relative_transforms[(curr, next_node)]
            abs_pose = abs_pose @ trans
        elif (next_node, curr) in relative_transforms:
            trans, _ = relative_transforms[(next_node, curr)]
            abs_pose = abs_pose @ np.linalg.inv(trans)

    return abs_pose

# ================== Subscene Frame Group Fusion (with ICP refine) ==================
def fuse_scene_groups(pose_root, ply_root, scene_name, fuse_frame_count=5,
                      individual_voxel=0.05, final_voxel=0.05, new_dataset_root="new_dataset",
                      max_points=30000):  # Add maximum point count parameter
    """
    Perform frame group fusion for a single subscene - group by fuse_frame_count and limit point cloud size
    """
    # Scan available point cloud files
    pcd_dir = os.path.join(ply_root, scene_name)
    available_indices, file_paths = scan_available_pointclouds(pcd_dir)

    if not available_indices:
        print(f"Error: Scene {scene_name} has no valid point cloud files found")
        return []

    print(f"Scene {scene_name} found {len(available_indices)} valid point cloud files")

    # Only load the required pose data
    pose_path = os.path.join(pose_root, scene_name, "pose.txt")
    poses = load_selective_poses(pose_path, available_indices)

    # Process each subscene separately
    fused_scene_dir = os.path.join(new_dataset_root, scene_name)
    os.makedirs(fused_scene_dir, exist_ok=True)

    fusion_info = []

    # Group by fuse_frame_count, no longer using fixed maximum group size
    available_groups = []
    i = 0
    while i < len(available_indices):
        # Check if there are enough frames to form a complete group
        if i + fuse_frame_count <= len(available_indices):
            # Enough frames, take complete group
            group_indices = available_indices[i:i+fuse_frame_count]
        else:
            # Not enough frames, take all remaining frames as the last group
            group_indices = available_indices[i:]
            # If last group has less than 2 frames, skip processing
            if len(group_indices) < 2:
                break

        available_groups.append(group_indices)
        i += fuse_frame_count

    print(f"Dividing {len(available_indices)} point clouds into {len(available_groups)} groups with {fuse_frame_count} frames per group")

    # Process each group of point clouds
    for group_idx, group_indices in enumerate(available_groups):
        print(f"Processing group {group_idx+1}/{len(available_groups)}: frame indices {group_indices}")
            
        start_idx = group_indices[0]
        end_idx = group_indices[-1]

        # Use empty point cloud as container
        combined_pcd = o3d.geometry.PointCloud()

        # Process each frame
        for idx in group_indices:
            # Load and process point cloud
            pcd = o3d.io.read_point_cloud(file_paths[idx])
            if not pcd.has_points():
                print(f"Warning: Point cloud {idx} has no points, skipping")
                continue

            # Transform to global coordinate system
            pcd.transform(poses[idx])

            # Downsample
            pcd_ds = pcd.voxel_down_sample(individual_voxel)

            # If not the first frame, refine with ICP
            if idx != start_idx and combined_pcd.has_points():
                icp_result = o3d.pipelines.registration.registration_icp(
                    pcd_ds, combined_pcd,
                    max_correspondence_distance=0.1,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                )
                pcd_ds.transform(icp_result.transformation)

            # Directly add to result point cloud
            combined_pcd += pcd_ds

            # Release memory
            del pcd, pcd_ds

        # Final downsampling and control point count
        if combined_pcd.has_points():
            # Initial downsampling
            combined_pcd = combined_pcd.voxel_down_sample(final_voxel)

            # Check point count and further downsample until below maximum point count
            current_voxel_size = final_voxel
            points_array = np.asarray(combined_pcd.points)
            num_points = len(points_array)

            while num_points > max_points:
                # Increase voxel size to reduce point count
                current_voxel_size *= 1.2
                print(f"Point cloud too large ({num_points} > {max_points}), increasing voxel size to {current_voxel_size:.4f}")
                combined_pcd = combined_pcd.voxel_down_sample(current_voxel_size)

                points_array = np.asarray(combined_pcd.points)
                num_points = len(points_array)

            print(f"Final point cloud point count: {num_points}, using voxel size: {current_voxel_size:.4f}")

            # Save result
            frame_count = len(group_indices)
            fused_filename = f"merged_{start_idx:04d}_{end_idx:04d}_{frame_count}frames.ply"
            output_path = os.path.join(fused_scene_dir, fused_filename)
            
            try:
                o3d.io.write_point_cloud(output_path, combined_pcd)
                print(f"Subscene {scene_name}: Saved fused point cloud to {output_path}")

                fusion_info.append({
                    "group_name": fused_filename,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "frame_count": frame_count,
                    "frame_indices": group_indices,
                    "first_pose": poses[start_idx].tolist(),
                    "points_count": num_points,
                    "voxel_size": current_voxel_size
                })
            except Exception as e:
                print(f"Error saving point cloud: {e}")

        # Clean up memory
        del combined_pcd
        import gc
        gc.collect()
    
    return fusion_info

def process_all_scenes(pose_root, ply_root, scene_list, fuse_frame_count=5,
                       individual_voxel=0.05, final_voxel=0.05, new_dataset_root="new_dataset",
                       json_out="fused_info.json", max_points=30000):
    """
    Process point cloud fusion for multiple subscenes, and write fusion information of each subscene into the same JSON file.
    Example output JSON file structure:

    {
        "cecum_t1_a": {
             "5": [{fusion group info 1}, {fusion group info 2}, ...],
             "10": [{...}],
             "15": [{...}],
             "20": [{...}]
        },
        "cecum_t1_b": {
            ...
        },
        ...
    }
    """
    # If JSON file already exists, load existing data, otherwise initialize as empty dictionary
    if os.path.exists(json_out):
        with open(json_out, "r", encoding="utf-8") as f:
            all_fusion_info = json.load(f)
    else:
        all_fusion_info = {}

    for scene_name in scene_list:
        print(f"Processing subscene: {scene_name}")
        fusion_info = fuse_scene_groups(pose_root, ply_root, scene_name, fuse_frame_count,
                                        individual_voxel, final_voxel, new_dataset_root,
                                        max_points=max_points)
        # For each scene, first distinguish by scene
        if scene_name not in all_fusion_info:
            all_fusion_info[scene_name] = {}
        # Then distinguish and save by fuse_frame_count (converted to string)
        all_fusion_info[scene_name][str(fuse_frame_count)] = fusion_info

    # Write to JSON file
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_fusion_info, f, indent=4, ensure_ascii=False)
    print(f"All fusion information has been written to JSON file: {json_out}")

def scan_available_pointclouds(pcd_dir, filename_pattern="{i:04d}_depth_pcd.ply"):
    """
    Scan directory, find all actually existing point cloud files, return frame index list
    Parameters:
      pcd_dir: Point cloud file directory
      filename_pattern: Filename format template
    Returns:
      Existing point cloud file index list and corresponding file path dictionary
    """
    available_indices = []
    file_paths = {}

    # Check if directory exists
    if not os.path.exists(pcd_dir):
        print(f"Warning: Directory {pcd_dir} does not exist")
        return available_indices, file_paths

    # Scan all files in directory
    for filename in os.listdir(pcd_dir):
        if filename.endswith("_depth_pcd.ply"):
            try:
                # Extract index from filename, e.g., extract 4 from "0004_depth_pcd.ply"
                index = int(filename.split("_")[0])
                path = os.path.join(pcd_dir, filename)
                available_indices.append(index)
                file_paths[index] = path
            except ValueError:
                print(f"Warning: Cannot extract index from filename {filename}")

    # Sort indices to ensure sequential processing
    available_indices.sort()
    return available_indices, file_paths

def load_selective_poses(pose_path, indices):
    """
    Only load pose data for specified indices
    Parameters:
      pose_path: Pose file path
      indices: List of frame indices to load
    Returns:
      List containing poses for specified indices
    """
    print(f"Selectively loading poses: {pose_path}")
    poses = {}  # Use dictionary to store, key is index, value is pose matrix

    with open(pose_path, 'r') as f:
        lines = f.readlines()

        # Ensure enough lines
        max_index = max(indices) if indices else 0
        if max_index >= len(lines):
            print(f"Warning: Requested maximum index {max_index} exceeds pose file line count {len(lines)}")

        # Only load poses for needed indices
        for idx in indices:
            if idx < len(lines):
                line = lines[idx].strip()
                if not line:
                    print(f"Warning: Line {idx} in pose file is empty")
                    poses[idx] = np.eye(4)  # Use identity matrix as default
                    continue

                data = list(map(float, line.split(',')))
                if len(data) != 16:
                    print(f"Warning: Pose row {idx} data length error, using identity matrix")
                    poses[idx] = np.eye(4)
                    continue

                T = np.array(data).reshape(4, 4)
                if not np.allclose(T[3, :], [0, 0, 0, 1]):
                    if np.allclose(T[:3, 3], np.zeros(3)) and not np.allclose(T[3, :3], np.zeros(3)):
                        T_fixed = np.eye(4)
                        T_fixed[:3, :3] = T[:3, :3]
                        T_fixed[:3, 3] = T[3, :3]
                        T = T_fixed
                    else:
                        print(f"Warning: Transformation matrix on line {idx} is invalid, using identity matrix")
                        T = np.eye(4)

                poses[idx] = T
            else:
                print(f"Warning: Index {idx} exceeds pose file range, using identity matrix")
                poses[idx] = np.eye(4)

    return poses

# ================== Main Entry Point ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point cloud registration and fusion processing")
    parser.add_argument("--mode", type=str, choices=["total", "partial"], default="total",
                        help="Select processing mode: 'total' for global ICP registration and fusion, 'partial' for subscene frame group fusion (with ICP refine)")
    # Specify pose and ply dataset root directories separately
    parser.add_argument("--pose_root", type=str, default="/path/to/pose_dataset",
                        help="Pose dataset root directory, each scene folder contains pose.txt")
    parser.add_argument("--ply_root", type=str, default="/path/to/ply_dataset",
                        help="PLY dataset root directory, each scene folder contains fragments_ply")
    parser.add_argument("--scene", type=str, default="cecum_t1_a",
                        help="Single subscene name (for total mode)")
    parser.add_argument("--scene_list", nargs="+", default=["scene1", "scene2", "scene3"],
                        help="List of multiple subscene names (for partial mode)")
    parser.add_argument("--fuse_frame_count", type=int, default=5,
                        help="Number of frames to fuse each time")
    parser.add_argument("--individual_voxel", type=float, default=0.05,
                        help="Voxel size for individual point cloud pre-downsampling")
    parser.add_argument("--final_voxel", type=float, default=0.05,
                        help="Voxel size for overall point cloud downsampling after fusion")
    parser.add_argument("--new_dataset_root", type=str, default="/path/to/your/new_dataset",
                        help="New dataset save directory (for partial mode)")
    parser.add_argument("--json_out", type=str, default="fused_info.json",
                        help="JSON filename to save fusion information")
    parser.add_argument("--max_points", type=int, default=30000,
                       help="Maximum number of points in fused point cloud")
    args = parser.parse_args()

    if args.mode == "total":
        align_scene_with_icp(args.pose_root, args.ply_root, args.scene,
                             individual_voxel=args.individual_voxel,
                             final_voxel=args.final_voxel,
                             icp_parallel=True)
    elif args.mode == "partial":
        process_all_scenes(args.pose_root, args.ply_root, args.scene_list, args.fuse_frame_count,
                           individual_voxel=args.individual_voxel,
                           final_voxel=args.final_voxel,
                           new_dataset_root=args.new_dataset_root,
                           json_out=args.json_out,
                           max_points=args.max_points)