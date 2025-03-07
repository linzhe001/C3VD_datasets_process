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
    # Convert the 3D points into homogeneous coordinates ([x, y, z, 1]).
    homogeneous_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    # Transform world coordinates into camera coordinates using the extrinsic matrix.
    camera_points = (camera_pose @ homogeneous_points.T).T
 
    # Initialize arrays for projected pixel coordinates, visibility flags, and depth values.
    pixel_coords = np.zeros((len(world_points), 2))
    visibility = np.zeros(len(world_points))
    depths = camera_points[:, 2]
 
    # Only points in front of the camera have a positive z-coordinate.
    in_front = camera_points[:, 2] > 0
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
        # If the actual hit distance differs significantly from the expected point distance, mark as occluded.
        if abs(t - ray_len) > 0.1:
            updated_visibility[idx] = 0
 
    return updated_visibility

def process_scene(scene_path, output_root, intrinsic_params, distortion_params, depth_scale=0.001, depth_threshold=0.1):
    """
    处理单个场景文件夹：加载 mesh、pose.txt 及所有深度图，并生成每帧的可见点云
    """
    scene_name = os.path.basename(os.path.normpath(scene_path))
    print(f"正在处理场景: {scene_name}")
    
    # 查找 mesh 文件（假设后缀为 .obj）
    mesh_files = [f for f in os.listdir(scene_path) if f.endswith(".obj")]
    if not mesh_files:
        print(f"场景 {scene_name} 未找到 mesh 文件，跳过。")
        return
    mesh_file_path = os.path.join(scene_path, mesh_files[0])
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()
    
    # 从 mesh 上均匀采样候选点（方法 B）
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=100000)
    world_points = np.asarray(pcd_mesh.points)
    
    # 读取 pose.txt 文件（假设每一行为 16 个数字，表示 4x4 矩阵）
    pose_file = os.path.join(scene_path, "pose.txt")
    if not os.path.exists(pose_file):
        print(f"场景 {scene_name} 未找到 pose.txt，跳过。")
        return
    poses_raw = np.loadtxt(pose_file)
    if poses_raw.ndim == 1 and poses_raw.size == 16:
        poses = [poses_raw.reshape((4,4))]
    elif poses_raw.ndim == 2 and poses_raw.shape[1] == 16:
        poses = [poses_raw[i].reshape((4,4)) for i in range(poses_raw.shape[0])]
    else:
        print(f"场景 {scene_name} pose.txt 格式不符合预期，跳过。")
        return
    
    # 查找所有的深度图文件（假设命名模式 *_depth.tiff）
    depth_files = sorted(glob(os.path.join(scene_path, "*_depth.tiff")))
    if len(depth_files) == 0:
        print(f"场景 {scene_name} 未找到深度图文件，跳过。")
        return
    
    num_frames = min(len(depth_files), len(poses))
    print(f"场景 {scene_name} 共找到 {num_frames} 帧深度图与对应 pose。")
    
    # 创建输出目录：output_root/scene_name/
    scene_output_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 构造内参矩阵和畸变系数数组
    fx, fy, cx, cy = intrinsic_params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    k1, k2, k3, k4 = distortion_params
    D = np.array([k1, k2, k3, k4])
    
    for i in range(num_frames):
        depth_path = depth_files[i]
        print(f"  处理帧 {i}: {os.path.basename(depth_path)}")
        # 加载深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"    无法读取深度图 {depth_path}，跳过此帧。")
            continue
        height, width = depth_img.shape
        
        # 执行去畸变处理
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
        undistorted_depth = cv2.remap(depth_img, map1, map2, cv2.INTER_LINEAR)
        
        # 读取第 i 帧对应的 pose（假设 pose 为相机在世界坐标系下的表示）
        frame_pose = poses[i]
        # 构建从世界到相机的变换矩阵
        camera_pose = np.linalg.inv(frame_pose)
        
        # 投影模型采样点到图像平面
        pixel_coords, visibility, depths = project_points_to_image_all(
            world_points, camera_pose, K, width, height
        )
        
        # 对比投影深度和校正后的深度图值，剔除不匹配的点
        for j, (coord, d_model) in enumerate(zip(pixel_coords, depths)):
            u, v = int(round(coord[0])), int(round(coord[1]))
            if u < 0 or u >= width or v < 0 or v >= height:
                continue
            d_actual = undistorted_depth[v, u] * depth_scale  # 单位统一为米（假设深度图单位为毫米）
            if abs(d_actual - d_model) > depth_threshold:
                visibility[j] = 0
        
        # 利用射线检测进一步更新可见性
        camera_pos = frame_pose[:3, 3]
        visibility = handle_occlusion_raycasting_all(world_points, camera_pos, mesh, visibility)
        
        # 筛选可见点
        visible_points = world_points[visibility == 1]
        
        # 构造点云并保存为 PLY 文件
        pcd_visible = o3d.geometry.PointCloud()
        pcd_visible.points = o3d.utility.Vector3dVector(visible_points)
        output_filename = os.path.join(scene_output_dir, f"frame_{i:04d}_visible.ply")
        o3d.io.write_point_cloud(output_filename, pcd_visible)
        print(f"    保存可见点云: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="生成每一帧深度图对应的可见点云，处理所有场景")
    parser.add_argument("--input", type=str, required=True, help="C3VD 文件夹路径，每个子文件夹表示一个场景")
    parser.add_argument("--output", type=str, required=True, help="可见点云输出根目录")
    args = parser.parse_args()
    
    input_root = args.input
    output_root = args.output
    
    # 定义相机内参和畸变系数（参见 creat_check_pcd.py）
    intrinsic_params = (767.3861511125845, 767.5058656118406, 679.054265997005, 543.646891684636)
    distortion_params = (-0.18867185058223412, -0.003927337093919806, 0.030524814153620117, -0.012756926010904904)
    
    # 获取输入目录下的所有场景文件夹
    scene_dirs = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if len(scene_dirs) == 0:
        print("未在输入路径下找到任何场景文件夹。")
        return
    
    for scene_path in scene_dirs:
        process_scene(scene_path, output_root, intrinsic_params, distortion_params)
    
    print("所有场景的可见点云生成任务已完成！")

if __name__ == "__main__":
    main()