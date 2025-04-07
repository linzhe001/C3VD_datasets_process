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
    投影点云到图像平面，针对不同的坐标约定做了调整
    """
    # 将3D点转换为齐次坐标
    homogeneous_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    
    # 将世界坐标系转到相机坐标系
    camera_points = (camera_pose @ homogeneous_points.T).T
    
    # 修改：根据Z值范围判断相机看向的方向，如果所有Z值都为正，则判断Z>0的点在前方
    z_min = camera_points[:, 2].min()
    z_max = camera_points[:, 2].max()
    
    # 由于相机坐标系下Z值都为正，我们应该使用Z>0作为前方判断条件
    in_front = camera_points[:, 2] > 0  # 修改条件
    
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
        if abs(t - ray_len) > 0.5:  # 改为更大的阈值
            updated_visibility[idx] = 0
 
    return updated_visibility

def visualize_cameras_and_model(mesh, poses, output_path):
    # 创建可视化几何体
    vis_geometries = [mesh]
    
    # 为每个相机位姿创建一个小的坐标系可视化
    for i, pose in enumerate(poses[:10]):  # 仅可视化前10个相机
        # 创建相机坐标系
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_frame.transform(pose)
        vis_geometries.append(camera_frame)
        
        # 创建指示相机位置的球体
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(pose[:3, 3])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        vis_geometries.append(sphere)
    
    # 保存为可视化文件
    o3d.io.write_triangle_mesh(output_path, mesh)  # 保存模型
    
    return vis_geometries

def process_scene_with_depth_guided_raycasting(scene_path, point_cloud_scene_dir, output_root):
    """
    使用深度图重建点云指导射线追踪判断可见性
    """
    scene_name = os.path.basename(os.path.normpath(scene_path))
    print(f"正在处理场景: {scene_name}")
    
    # 1. 加载mesh
    mesh_files = [f for f in os.listdir(scene_path) if f.endswith(".obj")]
    if not mesh_files:
        print(f"场景 {scene_name} 未找到 mesh 文件，跳过。")
        return
    mesh_file_path = os.path.join(scene_path, mesh_files[0])
    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_vertex_normals()
    
    # 2. 读取位姿数据
    pose_file = os.path.join(scene_path, "pose.txt")
    if not os.path.exists(pose_file):
        print(f"场景 {scene_name} 未找到 pose.txt，跳过。")
        return
    
    poses_raw = np.loadtxt(pose_file, delimiter=",")
    poses = []
    for i in range(poses_raw.shape[0]):
        pose_raw_matrix = poses_raw[i].reshape((4,4))
        standard_pose = np.eye(4)
        standard_pose[:3, :3] = pose_raw_matrix[:3, :3]
        standard_pose[:3, 3] = pose_raw_matrix[3, :3]
        poses.append(standard_pose)
    
    # 3. 创建输出目录
    scene_output_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # 4. 处理每一帧
    for i in range(len(poses)):
        print(f"  处理帧 {i}")
        
        # 4.1 加载重建点云
        reconstructed_pcd_path = os.path.join(point_cloud_scene_dir, f"{i:04d}_depth_pcd.ply")
        if not os.path.exists(reconstructed_pcd_path):
            print(f"    找不到重建点云: {reconstructed_pcd_path}")
            continue
            
        reconstructed_pcd = o3d.io.read_point_cloud(reconstructed_pcd_path)
        if not reconstructed_pcd.has_points():
            print(f"    重建点云为空: {reconstructed_pcd_path}")
            continue
        
        # 4.2 计算相机位姿
        camera_pose = np.linalg.inv(poses[i])
        camera_position = poses[i][:3, 3]
        
        # 4.3 使用深度图重建点云确定观察方向和范围
        view_direction, view_range = determine_view_from_depth_pcd(
            reconstructed_pcd, camera_position
        )
        
        # 4.4 从mesh采样点，但只采样在视野范围内的点
        pcd_mesh = mesh.sample_points_uniformly(number_of_points=50000)
        world_points = np.asarray(pcd_mesh.points)
        
        # 4.5 基于观察方向进行初步筛选
        initial_visibility = filter_points_by_view_direction(
            world_points, camera_position, view_direction, view_range
        )
        
        # 4.6 对初步筛选的点进行射线追踪
        final_visibility = raycast_with_view_guidance(
            world_points, camera_position, mesh, initial_visibility,
            view_direction, view_range
        )
        
        # 4.7 生成最终可见点云
        visible_points = world_points[final_visibility == 1]
        if len(visible_points) > 0:
            pcd_visible = o3d.geometry.PointCloud()
            pcd_visible.points = o3d.utility.Vector3dVector(visible_points)
            
            # 4.8 保存结果
            output_filename = os.path.join(scene_output_dir, f"frame_{i:04d}_visible.ply")
            o3d.io.write_point_cloud(output_filename, pcd_visible)
            print(f"    保存可见点云 ({len(visible_points)}点): {output_filename}")
        else:
            print(f"    警告: 无可见点!")

def determine_view_from_depth_pcd(depth_pcd, camera_position):
    """
    从深度图重建点云确定观察方向和范围
    """
    # 1. 计算点云中心
    center = depth_pcd.get_center()
    
    # 2. 计算观察方向（从相机到点云中心的方向）
    view_direction = center - camera_position
    view_direction = view_direction / np.linalg.norm(view_direction)
    
    # 3. 计算观察范围（点云到相机的最大距离）
    points = np.asarray(depth_pcd.points)
    distances = np.linalg.norm(points - camera_position, axis=1)
    view_range = np.max(distances)
    
    # 打印调试信息
    print(f"点云中心: {center}")
    print(f"相机位置: {camera_position}")
    print(f"观察方向: {view_direction}")
    print(f"观察范围: {view_range}")
    
    return view_direction, view_range

def filter_points_by_view_direction(points, camera_position, view_direction, view_range):
    """
    基于观察方向进行初步筛选
    """
    visibility = np.zeros(len(points))
    
    # 1. 计算所有点到相机的方向向量
    directions = points - camera_position
    distances = np.linalg.norm(directions, axis=1)
    
    # 2. 归一化方向向量
    directions = directions / distances[:, np.newaxis]
    
    # 3. 计算与观察方向的夹角
    cos_angles = np.dot(directions, view_direction)
    
    # 4. 设置筛选条件
    angle_threshold = np.cos(np.radians(60))  # 60度视锥角
    distance_threshold = view_range * 1.2  # 允许20%的额外范围
    
    # 5. 应用筛选条件
    valid_indices = (cos_angles > angle_threshold) & (distances < distance_threshold)
    visibility[valid_indices] = 1
    
    return visibility

def raycast_with_view_guidance(points, camera_position, mesh, initial_visibility,
                             view_direction, view_range):
    """
    基于观察方向进行射线追踪
    """
    # 1. 创建射线追踪场景
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    # 2. 获取初步可见的点
    visible_indices = np.where(initial_visibility == 1)[0]
    if len(visible_indices) == 0:
        return initial_visibility
    
    # 3. 准备射线
    rays = np.zeros((len(visible_indices), 6))
    rays[:, 0:3] = camera_position
    
    # 4. 计算射线方向（考虑观察方向）
    ray_dirs = points[visible_indices] - camera_position
    ray_lens = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs = ray_dirs / ray_lens
    
    # 5. 根据观察方向调整射线
    for i, ray_dir in enumerate(ray_dirs):
        # 计算与观察方向的夹角
        cos_angle = np.dot(ray_dir, view_direction)
        if cos_angle < 0:  # 如果射线方向与观察方向相反
            ray_dirs[i] = -ray_dirs[i]  # 翻转射线方向
    
    rays[:, 3:6] = ray_dirs
    
    # 6. 执行射线追踪
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_t)
    t_hit = ans['t_hit'].numpy()
    
    # 7. 更新可见性
    updated_visibility = initial_visibility.copy()
    for i, (idx, t, ray_len) in enumerate(zip(visible_indices, t_hit, ray_lens.flatten())):
        if abs(t - ray_len) > 0.5:  # 射线检测阈值
            updated_visibility[idx] = 0
    
    return updated_visibility

def main():
    parser = argparse.ArgumentParser(description="生成每一帧深度图对应的可见点云，处理所有场景")
    parser.add_argument("--input", type=str, required=True, help="C3VD 文件夹路径，每个子文件夹表示一个场景，包含.obj和pose.txt")
    parser.add_argument("--point_cloud_source", type=str, required=True, help="C3VD_ply_source 文件夹路径，包含所有场景的点云文件")
    parser.add_argument("--output", type=str, required=True, help="可见点云输出根目录")
    args = parser.parse_args()
    
    input_root = args.input
    point_cloud_source = args.point_cloud_source
    output_root = args.output
    
    # 获取输入目录下的所有场景文件夹
    scene_dirs = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    if len(scene_dirs) == 0:
        print("未在输入路径下找到任何场景文件夹。")
        return
    
    for scene_path in scene_dirs:
        scene_name = os.path.basename(scene_path)
        # 确保点云源目录中存在对应的场景文件夹
        point_cloud_scene_dir = os.path.join(point_cloud_source, scene_name)
        if not os.path.exists(point_cloud_scene_dir):
            print(f"警告：在点云源目录中未找到场景 {scene_name} 的文件夹，跳过处理。")
            continue
            
        process_scene_with_depth_guided_raycasting(scene_path, point_cloud_scene_dir, output_root)
    
    print("所有场景的可见点云生成任务已完成！")

if __name__ == "__main__":
    main()